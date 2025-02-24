
checkpoint = torch.load('./sod/model.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state'])

from pit import *
# from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from utils import *

ntrain = 1024
ntest = 128
batch_size = 8
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)

def load_data(path_data, ntrain = 1024, ntest=128):

data = loadmat(path_data)

X_data = data["x"].astype('float32')
X_data[...,2] = (X_data[...,2]-0.5*X_data[...,1]**2/X_data[...,0])*(1.4-1) # the primitive variable: pressure 原始变量：压力
X_data[...,1] = X_data[...,1]/X_data[...,0] # the primitive variable: velocity 原始变量：速度
Y_data = data["y"].astype('float32')
Y_data[...,2] = (Y_data[...,2]-0.5*Y_data[...,1]**2/Y_data[...,0])*(1.4-1)
Y_data[...,1] = Y_data[...,1]/Y_data[...,0]
X_train = X_data[:ntrain,:]
Y_train = Y_data[:ntrain,:]
X_test = X_data[-ntest:,:]
Y_test = Y_data[-ntest:,:]
return torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(X_test), torch.from_numpy(Y_test)


x_train, y_train, x_test, y_test = load_data('./supplementary_data/data_sod.mat', ntrain, ntest)
mesh = torch.linspace(-5,5,x_train.shape[1]+1)[:-1].reshape(-1,1).cuda()
mesh_ltt = torch.linspace(-5,5,256+1)[:-1].reshape(-1,1).cuda()
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)


model.eval()
pred = np.zeros_like(y_test.numpy())
count = 0
with torch.no_grad():
for x, y in test_loader:
x, y = x.cuda(), y.cuda()
out = model(mesh, x, mesh)
pred[count*batch_size:(count+1)*batch_size,...] = out.detach().cpu().numpy()
count += 1

y_test = y_test.numpy()
print("relative l1 error", (np.linalg.norm(y_test-pred, axis=1, ord=1) / np.linalg.norm(y_test, axis=1, ord=1)).mean())
print("relative l2 error", (np.linalg.norm(y_test-pred, axis=1, ord=2) / np.linalg.norm(y_test, axis=1, ord=2)).mean())
print("relative l_inf error", (abs(y_test-pred).max(axis=1) / abs(y_test).max(axis=1)).mean() )
savemat("pred.mat", mdict={'pred':pred, 'trueX':x_test, 'trueY':y_test})


index = -1
true = y_test[index,...].reshape(-1,3)
pred = pred[index,...].reshape(-1,3)
mesh = mesh.detach().cpu().numpy().reshape(-1,)
for i in range(3):
plt.figure(figsize=(12,12),dpi=100)
plt.plot(mesh, true[:,i], label='true', c='b')
plt.plot(mesh, pred[:,i], label='pred', c='r')
plt.savefig("{}_pred_{}.pdf".format(index, i+1))
plt.close()