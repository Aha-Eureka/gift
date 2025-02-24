from scipy.io import loadmat
import tensorflow as tf
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(0)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0:],'GPU')
import tensorflow_probability as tfp
from math import pi

class rel_norm_step(tf.keras.losses.Loss):
    '''
    Compute the average relative l2 loss between a batch of targets and predictions, step wise
    '''
    def __init__(self, steps):
        super(rel_norm_step, self).__init__()
        self.steps = steps
        
    def call(self, true, pred):
        rel_error  = tf.math.divide(tf.norm(tf.keras.layers.Reshape((-1, self.steps))(true-pred), axis=1), tf.norm(tf.keras.layers.Reshape((-1, self.steps))(true), axis=1)) # step-wise relative l2 loss#
        return tf.math.reduce_mean(rel_error)
        
    def get_config(self):
        config = {
        'steps': self.steps
        }
        return config

def rel_norm_traj(true, pred):
    '''
    Compute the average relative l2 loss between a batch of targets and predictions, full trajectory
    '''
    rel_error  = tf.math.divide(tf.norm(tf.keras.layers.Reshape((-1,))(true-pred), axis=1), tf.norm(tf.keras.layers.Reshape((-1,))(true), axis=1)) 
    return tf.math.reduce_mean(rel_error)
        
        
def pairwise_dist(res1x, res1y, res2x, res2y):
    
    gridx = tf.reshape(tf.linspace(0, 1, res1x+1)[:-1], (1,-1,1))
    gridx = tf.tile(gridx, [res1y,1,1])
    gridy = tf.reshape(tf.linspace(0, 1, res1y+1)[:-1], (-1,1,1))
    gridy = tf.tile(gridy, [1,res1x,1])
    grid1 = tf.concat([gridx, gridy], axis=-1)
    grid1 = tf.reshape(grid1, (res1x*res1y,2)) #(res1*res1,2)
    
    gridx = tf.reshape(tf.linspace(0, 1, res2x+1)[:-1], (1,-1,1))
    gridx = tf.tile(gridx, [res2y,1,1])
    gridy = tf.reshape(tf.linspace(0, 1, res2y+1)[:-1], (-1,1,1))
    gridy = tf.tile(gridy, [1,res2x,1])
    grid2 = tf.concat([gridx, gridy], axis=-1)
    grid2 = tf.reshape(grid2, (res2x*res2y,2)) #(res2*res2,2)
    
    print(grid1.shape, grid2.shape)
    grid1 = tf.tile(tf.expand_dims(grid1, 1), [1,grid2.shape[0],1])
    grid2 = tf.tile(tf.expand_dims(grid2, 0), [grid1.shape[0],1,1])

    dist  = tf.math.minimum(tf.norm(grid1-grid2, axis=-1), tf.norm(grid1+tf.constant([[1,0]], dtype='float64')-grid2, axis=-1))
    dist  = tf.math.minimum(dist, tf.norm(grid1+tf.constant([[-1,0]], dtype='float64')-grid2, axis=-1))
    dist  = tf.math.minimum(dist, tf.norm(grid1+tf.constant([[0,1]], dtype='float64')-grid2, axis=-1))
    dist  = tf.math.minimum(dist, tf.norm(grid1+tf.constant([[0,-1]], dtype='float64')-grid2, axis=-1))
    dist2 = tf.cast(dist**2, 'float32')
    return dist2

def load_data(file_path, ntrain, ntest):

    try:
        data  = loadmat(file_path)
    except:
        import mat73
        data = mat73.loadmat(file_path)
    flow  = data['u'].astype('float32')
    print(flow.shape)
    del data
    
    trainX = flow[:ntrain,:,:,:10] # ntrain 1000
    trainY = flow[:ntrain,:,:,10:30]
    testX = flow[-ntest:,:,:,:10]
    testY = flow[-ntest:,:,:,10:30]

    del flow
    return trainX, trainY, testX, testY


class mlp(tf.keras.layers.Layer):
    '''
    A two-layer MLP with GELU activation.
    '''
    def __init__(self, n_filters1, n_filters2):
        super(mlp, self).__init__()

        self.width1 = n_filters1
        self.width2 = n_filters2
        self.mlp1 = tf.keras.layers.Dense(self.width1, activation='gelu', kernel_initializer="he_normal")
        self.mlp2 = tf.keras.layers.Dense(self.width2, kernel_initializer="he_normal")

    def call(self, inputs):
        x = self.mlp1(inputs)
        x = self.mlp2(x)
        return x
        
    def get_config(self):
        config = {
        'n_filters1': self.width1,
        'n_filters2': self.width2
        }
        return config

class reccurent_PiT(tf.keras.Model):
    def __init__(self, network, steps):
        super(reccurent_PiT, self).__init__()
        self.PiT   = network
        self.steps = steps
        
    def call(self, inputs):
        x = tf.identity(inputs)
        pred = x[...,-1:]
        for t in range(self.steps):
            y = self.PiT(x)
            pred = tf.concat([pred,y], axis=-1)
            x = tf.concat([x[...,1:],y], axis=-1)
        return pred[...,1:]
    def get_config(self):
        config = super(reccurent_PiT, self).get_config()
        config.update({
            'network': self.PiT.get_config(),  # Store the config of the network instead of the network itself
            'steps': self.steps,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        network_config = config.pop('network')
        network = tf.keras.Model.from_config(network_config)
        return cls(network=network, **config)

class MultiHeadPosAtt(tf.keras.layers.Layer):
    '''
    Global, local and cross variants of the multi-head position-attention mechanism.
    '''
    def __init__(self, m_dist, n_head, hid_dim, locality):
        super(MultiHeadPosAtt, self).__init__()
        '''
        m_dist: distance matrix
        n_head: number of attention heads
        hid_dim: encoding dimension
        locality: quantile parameter to customize receptive field in position-attention
        ''' 
        self.dist         = m_dist  
        self.locality     = locality
        self.hid_dim      = hid_dim
        self.n_head       = n_head
        self.v_dim        = round(self.hid_dim/self.n_head)

    def build(self, input_shape):

        self.r = self.add_weight(
            shape=(self.n_head, 1, 1),
            trainable=True,
            name="band_width",
        )

        self.weight = self.add_weight(
            shape=(self.n_head, input_shape[-1], self.v_dim),
            initializer="he_normal",
            trainable=True,
            name="weight",
        )
        self.built = True

    def call(self, inputs):
        scaled_dist = self.dist * tf.math.tan(0.25*pi*(1-1e-7)*(1.0+tf.math.sin(self.r)))# tan(0.25*pi*(1+sin(r))) leads to higher accuracy than using r^2 and tan(r) # (n_head, L, L)
        if self.locality <= 100:
            mask = tfp.stats.percentile(scaled_dist, self.locality, interpolation="linear", axis=-1, keepdims=True)
            scaled_dist = tf.where(scaled_dist<=mask, scaled_dist, tf.float32.max)
        else:
            pass
        scaled_dist = - scaled_dist # (n_head, L, L)
        att         = tf.nn.softmax(scaled_dist, axis=2) # (n_heads, L, L)

        value      = tf.einsum("bnj,hjk->bhnk", inputs, self.weight) # (batch_size, n_head, L, v_dim)
        
        concat     = tf.einsum("hnj,bhjd->bhnd", att, value) # (batch_size, n_head, L, v_dim)
        concat     = tf.transpose(concat, (0,2,1,3)) # (batch_size, L, n_head, v_dim)
        concat     = tf.keras.layers.Reshape((-1,self.hid_dim))(concat) # (batch_size, L, hid_dim)
        return tf.keras.activations.gelu(concat)
        
    def get_config(self):
        config = {
        'm_dist': self.dist,
        'hid_dim': self.hid_dim,
        'n_head': self.n_head,
        'locality': self.locality
        }
        return config

class PiT(tf.keras.layers.Layer):
    '''
    Position-induced Transfomer, built upon the multi-head position-attention mechanism.
    PiT can be trained to decompose and learn the global and local dependcencies of operators in partial differential equations.
    '''
    def __init__(self, m_cross, m_ltt, out_dim, hid_dim, n_head, locality_encoder, locality_decoder):
        super(PiT, self).__init__()
        '''
        m_cross: distance matrix between X_query and X_latent; (L_qry,L_ltt)
        m_ltt: distance matrix between X_latent and X_latent; (L_ltt,L_ltt)
        out_dim: number of variables
        hid_dim: encoding dimension (network width)
        n_head: number of heads in multi-head attention modules
        locality_encoder: quantile parameter of local position-attention in the Encoder, allowing to customize the size of receptive filed
        locality_decoder: quantile parameter of local position-attention in the Decoder, allowing to customize the size of receptive filed
        '''
        self.m_cross  = m_cross
        self.m_ltt    = m_ltt
        self.res      = int(m_cross.shape[0]**0.5)
        self.out_dim  = out_dim
        self.hid_dim  = hid_dim
        self.n_head   = n_head
        self.en_local = locality_encoder
        self.de_local = locality_decoder
        self.n_blocks = 4 # number of position-attention modules in the Processor

        # Encoder
        self.en_layer = tf.keras.layers.Dense(self.hid_dim, activation="gelu", kernel_initializer="he_normal")
        self.down     = MultiHeadPosAtt(tf.transpose(self.m_cross), self.n_head, self.hid_dim, locality=self.en_local)

        # Processor
        self.MHPA     = [MultiHeadPosAtt(self.m_ltt, self.n_head, self.hid_dim, locality=200) for i in range(self.n_blocks)]
        self.MLP      = [mlp(self.hid_dim, self.hid_dim) for i in range(self.n_blocks)]
        self.W        = [tf.keras.layers.Dense(self.hid_dim, kernel_initializer="he_normal") for i in range(self.n_blocks)]

        # Decoder
        self.up       = MultiHeadPosAtt(self.m_cross, self.n_head, self.hid_dim, locality=self.de_local)
        self.de_layer = mlp(self.hid_dim, self.out_dim)

    def call(self, inputs):

        # Encoder
        grid  = self.get_mesh(inputs) #(batch_size, res_qry, res_qry, 2)
        en    = tf.concat([tf.cast(grid, dtype="float32"), inputs], axis=-1) # (batch_size, res_qry, res_qry, input_dim+2)
        en    = tf.keras.layers.Reshape((-1, en.shape[3]))(en) # (batch_size, L_qry, hid_dim)
        en    = self.en_layer(en) # (batch_size, L_qry, hid_dim)
        x     = self.down(en) # (batch_size, L_ltt, hid_dim)

        # Processor
        for i in range(self.n_blocks):
            x = self.MLP[i](self.MHPA[i](x)) + self.W[i](x) # (batch_size, L_ltt, hid_dim)
            x = tf.keras.activations.gelu(x) # (batch_size, L_ltt, hid_dim)

        # Decoder
        de = self.up(x) # (batch_size, L_qry, hid_dim)
        de = self.de_layer(de) # (batch_size, L_qry, out_dim)
        de = tf.keras.layers.Reshape((self.res, self.res, self.out_dim))(de) # (batch_size, res_qry, res_qry, out_dim)
        return de
    
    def get_mesh(self, inputs):
        size_x = size_y = self.res
        gridx = tf.reshape(tf.linspace(0, 1, size_x+1)[:-1], (1,-1,1,1))
        gridx = tf.tile(gridx, [1,1,size_y,1])
        gridy = tf.reshape(tf.linspace(0, 1, size_y+1)[:-1], (1,1,-1,1))
        gridy = tf.tile(gridy, [1,size_x,1,1])
        grid  = tf.concat([gridx, gridy], axis=-1)
        return tf.tile(grid, [tf.shape(inputs)[0],1,1,1])
        
    def get_config(self):
        config = {
        'm_cross': self.m_cross,
        'm_ltt': self.m_ltt,
        'out_dim': self.out_dim,
        'hid_dim': self.hid_dim,
        'n_head': self.n_head,
        'locality_encoder': self.en_local,
        'locality_decoder': self.de_local
        }
        return config

class MultiHeadSelfAtt(tf.keras.layers.Layer):
    '''
    Scaled dot-product multi-head self-attention
    '''
    def __init__(self, n_head, hid_dim):
        super(MultiHeadSelfAtt, self).__init__()

        self.hid_dim      = hid_dim
        self.n_head       = n_head
        self.v_dim        = round(self.hid_dim/self.n_head)

    def build(self, input_shape):

        self.q = self.add_weight(
            shape=(self.n_head, input_shape[-1], self.v_dim),
            initializer="he_normal",
            trainable=True,
            name="query",
        )

        self.k = self.add_weight(
            shape=(self.n_head, input_shape[-1], self.v_dim),
            initializer="he_normal",
            trainable=True,
            name="key",
        )

        self.v = self.add_weight(
            shape=(self.n_head, input_shape[-1], self.v_dim),
            initializer="he_normal",
            trainable=True,
            name="value",
        )
        self.built = True

    def call(self, inputs):

        query       = tf.einsum("bnj,hjk->bhnk", inputs, self.q) # (batch_size, n_head, L, v_dim)
        key         = tf.einsum("bnj,hjk->bhnk", inputs, self.k) # (batch_size, n_head, L, v_dim)
        att         = tf.nn.softmax(tf.einsum("...ij,...kj->...ik", query, key)/self.v_dim**0.5, axis=-1) # (batch_size, n_heads, L, L)

        value      = tf.einsum("bnj,hjk->bhnk", inputs, self.v) #(batch_size, n_head, L, v_dim)

        concat     = tf.einsum("...nj,...jd->...nd", att, value) #(batch_size, n_head, L, v_dim)
        concat     = tf.transpose(concat, (0,2,1,3)) # (batch_size, L, n_head, v_dim)
        concat     = tf.keras.layers.Reshape((-1,self.hid_dim))(concat) # (batch_size, L, hid_dim)
        return tf.keras.activations.gelu(concat)

    def get_config(self):
        config = {
        'n_head': self.n_head,
        'hid_dim': self.hid_dim
        }
        return config

class LiteTransformer(tf.keras.Model):
    '''
    Replace position-attention of the Processor in a PiT with self-attention
    '''
    def __init__(self, m_cross, res, out_dim, hid_dim, n_head, en_local, de_local):
        super(LiteTransformer, self).__init__()

        self.m_cross  = m_cross
        self.res      = res
        self.out_dim  = out_dim
        self.hid_dim  = hid_dim
        self.n_head   = n_head
        self.en_local = en_local
        self.de_local = de_local
        self.n_blocks = 4

        # Encoder
        self.en_layer = tf.keras.layers.Dense(self.hid_dim, activation="gelu", kernel_initializer="he_normal")
        self.down     = MultiHeadPosAtt(tf.transpose(self.m_cross), self.n_head, self.hid_dim, locality=self.en_local)
        
        # Processor
        self.PA       = [MultiHeadSelfAtt(self.n_head, self.hid_dim) for i in range(self.n_blocks)]
        self.MLP      = [mlp(self.hid_dim, self.hid_dim) for i in range(self.n_blocks)]
        self.W        = [tf.keras.layers.Dense(self.hid_dim, kernel_initializer="he_normal") for i in range(self.n_blocks)]

        # Decoder
        self.up       = MultiHeadPosAtt(self.m_cross, self.n_head, self.hid_dim, locality=self.de_local)
        self.de_layer = mlp(self.hid_dim, self.out_dim)
        
    def call(self, inputs):

        # Encoder
        grid  = self.get_mesh(inputs) #(b, s1, s2, 2)
        en    = tf.concat([tf.cast(grid, dtype="float32"), inputs], axis=-1)
        en    = tf.keras.layers.Reshape((-1, en.shape[3]))(en)
        en    = self.en_layer(en)
        x     = self.down(en)
        
        # Processor
        for i in range(self.n_blocks):
            x = self.MLP[i](self.PA[i](x)) + self.W[i](x)
            x = tf.keras.activations.gelu(x)

        # Decoder
        de = self.up(x)
        de = self.de_layer(de)
        de = tf.keras.layers.Reshape((inputs.shape[1], inputs.shape[2], self.out_dim))(de)
        return de

    def get_mesh(self, inputs):
        size_x = size_y = self.res
        gridx = tf.reshape(tf.linspace(0, 1, size_x+1)[:-1], (1,-1,1,1))
        gridx = tf.tile(gridx, [1,1,size_y,1])
        gridy = tf.reshape(tf.linspace(0, 1, size_y+1)[:-1], (1,1,-1,1))
        gridy = tf.tile(gridy, [1,size_x,1,1])
        grid  = tf.concat([gridx, gridy], axis=-1)
        return tf.tile(grid, [tf.shape(inputs)[0],1,1,1])
        
    def get_config(self):
        config = {
        'm_cross': self.m_cross,
        'res': self.res,
        'out_dim': self.out_dim,
        'hid_dim': self.hid_dim,
        'n_head': self.n_head,
        'locality_encoder': self.en_local,
        'locality_decoder': self.de_local,
        }
        return config

class Transformer(tf.keras.Model):
    '''
    Replace position-attention of a PiT with self-attention.
    '''
    def __init__(self, res, out_dim, hid_dim, n_head):
        super(Transformer, self).__init__()

        self.res      = res
        self.out_dim  = out_dim
        self.hid_dim  = hid_dim
        self.n_head   = n_head
        self.n_blocks = 4

        # Encoder
        self.en_layer = tf.keras.layers.Dense(self.hid_dim, activation="gelu", kernel_initializer="he_normal")
        self.down     = MultiHeadSelfAtt(self.n_head, self.hid_dim)

        # Processor
        self.PA       = [MultiHeadSelfAtt(self.n_head, self.hid_dim) for i in range(self.n_blocks)]
        self.MLP      = [mlp(self.hid_dim, self.hid_dim) for i in range(self.n_blocks)]
        self.W        = [tf.keras.layers.Dense(self.hid_dim, kernel_initializer="he_normal") for i in range(self.n_blocks)]

        # Decoder
        self.up       = MultiHeadSelfAtt(self.n_head, self.hid_dim)
        self.de_layer = mlp(self.hid_dim, self.out_dim)

    def call(self, inputs):

        # Encoder
        grid  = self.get_mesh(inputs) #(b, s1, s2, 2)
        en    = tf.concat([tf.cast(grid, dtype="float32"), inputs], axis=-1)
        en    = tf.keras.layers.Reshape((-1, en.shape[3]))(en)
        en    = self.en_layer(en)
        x     = self.down(en)

        # Processor
        for i in range(self.n_blocks):
            x = self.MLP[i](self.PA[i](x)) + self.W[i](x)
            x = tf.keras.activations.gelu(x)

        # Decoder
        de = self.up(x)
        de = self.de_layer(de)
        de = tf.keras.layers.Reshape((inputs.shape[1], inputs.shape[2], self.out_dim))(de)
        return de

    def get_mesh(self, inputs):
        size_x = size_y = self.res
        gridx = tf.reshape(tf.linspace(0, 1, size_x+1)[:-1], (1,-1,1,1))
        gridx = tf.tile(gridx, [1,1,size_y,1])
        gridy = tf.reshape(tf.linspace(0, 1, size_y+1)[:-1], (1,1,-1,1))
        gridy = tf.tile(gridy, [1,size_x,1,1])
        grid  = tf.concat([gridx, gridy], axis=-1)
        return tf.tile(grid, [tf.shape(inputs)[0],1,1,1])

    def get_config(self):
        config = {
        'res':self.res,
        'out_dim': self.out_dim,
        'hid_dim': self.hid_dim,
        'n_head': self.n_head
        }
        return config


        
