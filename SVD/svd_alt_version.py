# SVD 2
* Restrict test vectors and apply SVD, but this time we keep the original dimensions of the vectors and only zero-out those components outside of the domain.

This is equivalent to SVD 1, I just wanted to verify.

params_list = [beta, m0_str, m0_folder, nconf, NV, Nx, Nt]
dtv_spin0, dtv_spin1 = read_and_decompose_v2(blockID,block_x,block_t,params_list)
print("\nSpin component 0")
low_rank_tv0 = apply_SVD(dtv_spin0,k_rank)
low_rank_tv0 = low_rank_tv0.reshape(Nt,Nx,NV);
print("Low rank test vectors shape after reshaping",low_rank_tv0.shape)
print("\nSpin component 1")
low_rank_tv1 = apply_SVD(dtv_spin1,k_rank)
low_rank_tv1 =low_rank_tv1.reshape(Nt,Nx,NV);
print("Low rank test vectors shape after reshaping",low_rank_tv1.shape)


tvID = 0
xlims = [xini,xfin-1]
tlims = [tini,tfin-1]
make_heatmap(low_rank_tv1,xlims,tlims,tvID)
#Plot of a cut
cut = xini
X = np.arange(0,low_rank_tv1.shape[0])
plt.plot(X,np.abs(low_rank_tv1[:,cut,tvID]),marker='o',linestyle='--',label='SVD vector {0}'.format(tvID))
plt.legend()
plt.ylabel(r"$|\psi_\alpha|$",size=15)
plt.xlabel(r"$t$, cut at $x=${0}".format(cut),size=15)
plt.xlim(tlims)
plt.show()


fig_name = "heatmaps_blockID{0}_bsize{1}x{1}_beta{2}_v2.pdf".format(blockID,x_elements,beta)
metadata = [Nx, Nt, beta, m0_diff, x_elements,blockID, k_rank, NV, SAP_block_size]
make_heatmaps(low_rank_tv0, low_rank_tv1,xlims, tlims,metadata,fig_name,save=False)


# Plot test vectors without SVD

#Open test vectors of a particular gauge configuration and store them in columns
beta, N, m0_str = 2, 64, "-01884"
nconf = 0
tv = 0 
NV = 30
spin = 0 #0 or 1
test_vectors = np.zeros((NV,N*N),dtype=complex)
for tv in range(NV):
    path = "../real_tv/b{0}_{1}x{1}/m-018/tvector_{1}x{1}_b{0}0000_m{2}_nconf{3}_tv{4}.tv".format(
        beta, N, m0_str, nconf, tv
        )
    test_vector = read_binary_conf(N,N,path)
    flatten_vector_spin0 = test_vector[spin].flatten()
    test_vectors[tv] = flatten_vector_spin0
#We plot the test vectors by taking cuts at different x
X = np.arange(0,N,1)
for i in range(1,4):
    cut1 = np.abs(test_vector[spin,:,i])
    plt.plot(X,cut1,marker='o',linestyle='--',label=r'$x=${0}'.format(i))
plt.xlabel(r"$t$",size=15)
plt.ylabel(r"$|\psi_\alpha|$",size=15)
plt.legend()
plt.show()
tvec = np.abs(test_vector[spin].reshape(-1))
#Linearize everything
X = np.arange(0,tvec.shape[0])
plt.plot(X,tvec,marker='o',linestyle='--',label='cut {0}'.format(i))
plt.legend()
plt.xlim([0,58])
plt.show()