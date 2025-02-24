
checkpoint = torch.load('model_and_optimizer.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state'])
######### 
rel1err   = RelLpNorm(out_dim=4, p=1)
rel2err   = RelLpNorm(out_dim=4, p=2)
relMaxerr = RelMaxNorm(out_dim=4)
pred      = torch.zeros_like(y_test, device='cpu')
count     = 0

model.eval()
with torch.no_grad():
    for x, ext, y in test_loader:
        x, ext, y = x.cuda(), ext.cuda(), y.cuda()

        out = model(x, x, ext)
        pred[count*batch_size:(count+1)*batch_size,...] = out.detach().cpu()
        count += 1
print("relative l1 error", rel1err(y_test, pred) / ntest)
print("relative l2 error", rel2err(y_test, pred) / ntest)
print("relative l_inf error", relMaxerr(y_test, pred) / ntest)
savemat("pred.mat", mdict={'pred':pred.numpy(), 'trueX':x_test.numpy(), 'ext':ext_test.numpy(), 'trueY':y_test.numpy()})
##############################
index = -1
nvariables = 4
true = y_test.numpy()[index,40:-40,:20,:].reshape(-1,nvariables)
pred = pred.numpy()[index,40:-40,:20,:].reshape(-1,nvariables) #
err  = abs(true-pred)
emax = err.max(axis=0)
emin = err.min(axis=0)
vmax = true.max(axis=0)
vmin = true.min(axis=0)
print(vmax, vmin, emax, emin)

x = ext_test.numpy()[index,40:-40,:20,0].reshape(-1,1)
y = ext_test.numpy()[index,40:-40,:20,1].reshape(-1,1)
print(x.max(), x.min(), y.max(), y.min())

for i in range(nvariables):
    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=pred[:,i], cmap="plasma", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.title('Prediction')
    plt.savefig("{}_pred_{}.pdf".format(index, i+1))
    plt.close()

    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=true[:,i], cmap="plasma", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.title('Referencce')
    plt.savefig("{}_true_{}.pdf".format(index, i+1))
    plt.close()

    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=err[:,i], cmap="plasma", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.title('Error')
    plt.savefig("{}_error_{}.pdf".format(index, i+1))
    plt.close()
