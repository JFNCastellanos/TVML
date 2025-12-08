#junk which I might use in the future for testing
#Assembling P and P^+
Pcols = []
for tv in range(var.NV):
    for nx in range(var.BLOCKS_X):
        for nt in range(var.BLOCKS_T):
            for s in range(2):
                vc = np.zeros((var.NV,2,var.BLOCKS_T,var.BLOCKS_X),dtype=complex)
                vc[tv,s,nt,nx] = 1
                Pcols.append(utils.flatten_col(operadores.P_vc(vc)))
Pdaggcols = []
for nx in range(var.NX):
    for nt in range(var.NT):
        for s in range(2):   
            v = np.zeros((2,var.NT,var.NX),dtype=complex)
            v[s,nt,nx] = 1
            Pdaggcols.append(utils.flatten_colV2(operadores.Pdagg_v(v)))
Pcols = np.array(Pcols)
Ptranspose = np.conj(np.transpose(Pcols))
Pdaggcols = np.array(Pdaggcols)
np.all(Pdaggcols == Ptranspose)
#for i in range(len(Pdaggcols)):
    #for j in range(len(Pdaggcols[i])):
        #print(Pdaggcols[i,j],Ptranspose[i,j])

v = np.random.rand(2,var.NT,var.NX) + 1j*np.random.rand(2,var.NT,var.NX)
out = operadores.P_Pdagg(v)
print(v[0,5,0],out[0,5,0])

vc = np.random.rand(var.NV,2,var.BLOCKS_T,var.BLOCKS_X) + 1j*np.random.rand(var.NV,2,var.BLOCKS_T,var.BLOCKS_X)
outvc = operadores.Pdagg_P(vc)
print(vc[0,0,0,0],outvc[0,0,0,0])



#For testing torch vs numpy
import operators as op #Interpolator and prolongator given a set of test vectors
import operators_torch as opt #Same implementations but with pytorch
test = first_batch[1][0]
print("set of test vectors",type(test))
oper = op.Operators(var.BLOCKS_X, var.BLOCKS_T,test)
tvecs = oper.getTestVectors()
print("numpy vectors",type(tvecs))
oper_torch = opt.Operators(var.BLOCKS_X, var.BLOCKS_T,test)
tvecs_torch = oper_torch.getTestVectors()
print("torch vectors",type(tvecs_torch))
print(tvecs.shape,tvecs_torch.shape)
print("test vectors",tvecs[4,1,2,2],tvecs_torch[4,1,2,2])

v = np.random.rand(2,var.NT,var.NX) + 1j*np.random.rand(2,var.NT,var.NX)
vc = np.random.rand(var.NV,2,var.BLOCKS_T,var.BLOCKS_X) + 1j*np.random.rand(var.NV,2,var.BLOCKS_T,var.BLOCKS_X)

vtorch = torch.tensor(v)
vctorch = torch.tensor(vc)


out = oper.P_vc(vc)
print("Pvc (np)",out[0,5,0],type(out))
outt = oper_torch.P_vc(vctorch)
print("Pvc (torch)",outt[0,5,0],type(outt))

out = oper.P_Pdagg(v)
print("PP^+ (np)",v[0,5,0],out[0,5,0])
outvc = oper.Pdagg_P(vc)
print("P^+P (np)",vc[0,0,0,0],outvc[0,0,0,0])

outt = oper_torch.P_Pdagg(vtorch)
print("PP^+ (torch)",vtorch[0,5,0],outt[0,5,0])
outvct = oper_torch.Pdagg_P(vctorch)
print("P^+P (torch)",vctorch[0,0,0,0],outvct[0,0,0,0])


#----Reshaping

for batchID, batch in enumerate(dataloader):
        confs_batch = batch[0]
        near_kernel = batch[1]
        # Predicted test vectors
        # pred_tv.shape = [batch_size,4*NV,NT,NX]
        pred_tv = model(confs_batch)
        # Reshape the test vectors (needed for P, P^+)
        pred_tv = pred_tv.detach().numpy()
        pred_tv = pred_tv.reshape(batch_size,var.NV,4,var.NT,var.NX) 
        pred_tv_complex = np.empty((batch_size,var.NV,2,var.NT,var.NX), dtype=np.complex128)
        pred_tv_complex.real[:,:,0], pred_tv_complex.real[:,:,1] = pred_tv[:,:,0], pred_tv[:,:,1]
        pred_tv_complex.imag[:,:,0], pred_tv_complex.imag[:,:,1] = pred_tv[:,:,2], pred_tv[:,:,3]



#--------------
#The naïve double‑loop over the batch and the NV index in the loss works, but you can usually get a 10‑× speed‑up by vectorising the whole #thing:

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred, target : (B, NV, 2, NT, NX) complex
        """
        B, NV = pred.shape[:2]

        # Build a single Operators instance per *batch element*.
        # If the operator does not depend on the specific sample,
        # you could even create ONE instance outside the loop.
        losses = []
        for i in range(B):
            ops = op_mod.Operators(var.BLOCKS_X, var.BLOCKS_T, pred[i])
            # ops.P_Pdagg works on a tensor of shape (NV, 2, NT, NX)
            corrected = ops.P_Pdagg(target[i])                # (NV,2,NT,NX)
            diff = target[i] - corrected
            losses.append(torch.norm(diff, p='fro') ** 2)      # Frobenius norm per sample

        # stack → (B,) then sum → scalar
        return torch.stack(losses).sum()


#----------------#
#For reading the test vectors

#path = 'sap/near_kernel/b{0}_{1}x{2}/{3}/tvector_{1}x{2}_b{0}0000_m{4}_nconf{5}_tv{6}.tv'.format(
#    int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,0,0)
#read_binary_conf(None,path)

operadores = op.Operators.rand_tv(var.BLOCKS_X,var.BLOCKS_T)
operadores.check_orth()
tvectors = np.zeros((var.NO_CONFS,var.NV,2,var.NX,var.NT),dtype=complex)
for confID in range(1000):
    for tv in range(var.NV):
        path = 'sap/near_kernel/b{0}_{1}x{2}/{3}/tvector_{1}x{2}_b{0}0000_m{4}_nconf{5}_tv{6}.tv'.format(
        int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID,tv)
        tvectors[confID,tv] = read_binary_conf(None,path)
        #Calling the loss function for current operators and test vectors
    loss_fn = lf.CustomLoss(operadores,tvectors[confID])
    loss = loss_fn()
    print(loss)


#-------------------------------------------------------
#Checking dimensions
m = nn.Conv2d(2, 64, 2, 2, 0,dtype=complex)
output = m(first_batch[0][0])
print("Output shape",output.shape)
print("Predicted shape",ut.output_size(32,2,2,0,transpose=False))
m2 = nn.Conv2d(64,64,2,2,0,dtype=complex)
output = m2(output)
print("Output shape",output.shape)
print("Predicted shape",ut.output_size(16,2,2,0,transpose=False))

output_size(16,2,2,0,transpose=True)


#-------------------Check that the dataloader preserves the indexing of my training examples-------------#
A = []
for batch_id, batch in enumerate(train_loader):                          
    confsID = batch[2].detach().cpu().numpy()
    A.append(list(confsID))
for batch_id, batch in enumerate(test_loader):                        
    confsID = batch[2].detach().cpu().numpy()
    A.append(list(confsID))
B = []
for i in range(len(A)):
    B = B + A[i]
B = np.array(B)
np.sort(B)


#--------- Cross checking test vectors from python to binary format --------- # 
confs_batch = first_batch[0].to(device)
pred = model(confs_batch)                  # (B, 4*NV, NT, NX)
B = pred.shape[0]
pred = pred.view(B, var.NV, 4, var.NT, var.NX)   # (B,NV,4,NT,NX)
# Build complex tensor (B,NV,2,NT,NX)
real = torch.stack([pred[:, :, 0], pred[:, :, 1]], dim=2)   # (B,NV,2,NT,NX)
imag = torch.stack([pred[:, :, 2], pred[:, :, 3]], dim=2)   # (B,NV,2,NT,NX)
pred_complex = torch.complex(real, imag).detach().cpu().numpy()
value = pred_complex[0,0,mu,t,x]

file_path = "fake_tv/file.bin"
t, x, mu = 0,0,0
Re = np.real(value)
Im = np.imag(value)

print(t,x,mu,Re,Im)
fmt = "<3i2d"                     # little‑endian, 3 ints + 2 doubles
data = struct.pack(fmt, int(x), int(t), int(mu), float(Re), float(Im))
with open(file_path, "wb") as f:
    f.write(data)
# ---------- reading ----------
int_fmt = "<3i"
dbl_fmt = "<2d"

int_size = struct.calcsize(int_fmt)
dbl_size = struct.calcsize(dbl_fmt)

with open(file_path, "rb") as f:
    x_r, t_r, mu_r = struct.unpack(int_fmt, f.read(int_size))
    re_r, im_r    = struct.unpack(dbl_fmt, f.read(dbl_size))

print(x_r, t_r, mu_r, re_r, im_r)