import numpy as np
import matplotlib.pyplot as plt

def read_binary_conf(NX,NT,path):
    """
    Function used for opening a gauge configuration in binary format.
    It can also be used to open a .tv file with a test vector.
    """
    N = 2*NX*NT
    x, t, mu, vals = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N,dtype=complex)
    #U[μ,t,x]
    #conf = np.zeros((2,var.NT,var.NX),dtype=complex)
    data = np.fromfile(path, dtype=[('x', 'i4'),
                ('t', 'i4'),
                ('mu', 'i4'),
                ('re','f8'),
                ('im','f8')])
    conf = np.zeros((2, NT, NX), dtype=np.complex128)
    conf[data['mu'], data['t'], data['x']] = data['re'] + 1j * data['im']            
    return conf

def decomposed_vectors(blockID,block_x,block_t,spin,test_vectors):
    """
    Restrict test vectors to a lattice block
    """
    Nv, lalala, Nt, Nx = test_vectors.shape
    Nblocks =  block_x*block_t

    x_elements = Nx // block_x
    t_elements = Nt // block_t

    assert blockID < Nblocks, "blockID should be within the range of the number of lattice blocks"
    #v = np.zeros((dim,Nv*Nblocks),dtype=complex)
    
    bx = blockID // block_x
    bt = blockID % block_t 
    #----Coordinates of elements inside block----#
    xini, tini = x_elements * bx, t_elements * bt
    xfin = xini + x_elements
    tfin = tini + t_elements
    #--------------------------------------------#
    return test_vectors[:, spin, tini:tfin, xini:xfin]  #[tv,s,t,x]

def read_and_decompose(blockID,block_x,block_t,params_list):
    beta, m0_str, m0_folder, nconf, NV, Nx, Nt = params_list
    test_vectors = np.zeros((NV,2,Nx,Nt),dtype=complex)
    print("beta={0}, conf={1}, Nv={2}, Nx={3}, Nt={4}\nblocks_x={5}, blocks_t={6}, bx_size={7}, bt_size={8}".format
         (beta,nconf,NV,Nx,Nt,block_x,block_t,Nx//block_x,Nt//block_t)
         )
    
    for tv in range(NV):
        path = "../real_tv/b{0}_{1}x{1}/{2}/tvector_{1}x{1}_b{0}0000_m{3}_nconf{4}_tv{5}.tv".format(
            beta, Nx, m0_folder,m0_str, nconf, tv
            )
        test_vector = read_binary_conf(Nx,Nt,path)
        #flatten_vector_spin0 = test_vector[0].flatten()
        test_vectors[tv] = test_vector

    spin = 0
    dtv_spin0 = decomposed_vectors(blockID,block_x,block_t,spin,test_vectors) #[]
    dtv_spin0 =  np.transpose(dtv_spin0.reshape(NV,-1))
    spin = 1
    dtv_spin1 = decomposed_vectors(blockID,block_x,block_t,spin,test_vectors) #[]
    dtv_spin1 =  np.transpose(dtv_spin1.reshape(NV,-1))
    print("Test vectors matrix shape",dtv_spin0.shape)
    return dtv_spin0, dtv_spin1

def apply_SVD(tvectors,k_rank):
    U, s, Vh = np.linalg.svd(tvectors, full_matrices=False, compute_uv=True, hermitian=False)

    if np.allclose(np.matmul(np.matmul(U,np.diag(s)),Vh),tvectors):
        print("matrix succesfully reconstructed")
    else:
        print("something wrong with the SVD")
        print("U shape",U.shape)
        print("s shape",s.shape)
        print("Vh shape",Vh.shape)
    #assert k_rank<NV, "k has to be smaller than NV"
    Uk, sk, Vk = U[:,:k_rank], s[:k_rank], Vh[:k_rank,:]
    print("-------------------")
    print("Uk shape",Uk.shape)
    print("sk shape",sk.shape)
    print("Vk shape",Vk.shape)
    low_rank_tv = np.matmul(np.matmul(Uk,np.diag(sk)),Vk)
    print("Low rank test vectors shape",low_rank_tv.shape)
    return low_rank_tv
    
def make_heatmap(low_rank_tv,xlims,tlims,tvID,fig_name="",save=False):
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs((low_rank_tv[:,:,tvID])))
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"t")
    ax.set_xlim(xlims)
    ax.set_ylim(tlims)
    ax.set_title("Heatmap of test vector after SVD")
    ax.figure.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
    if save == True:
        fig.savefig(fig_name)

def make_heatmaps(low_rank_tv0, low_rank_tv1, xlims, tlims,metadata,fig_name="",save=False):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.25])
    
    data_list = [low_rank_tv0, low_rank_tv1]
    for row in range(2):
        for tvID in range(4):
            ax = fig.add_subplot(gs[row, tvID])
            vmax = np.max(np.abs(data_list[row][:, :, tvID]))
            im = ax.imshow(np.abs(data_list[row][:, :, tvID]),vmin=0.0,vmax=vmax, )
            ax.set_title(f"tvID = {tvID}")
            ax.set_xlabel("x")
            if tvID == 0:
                ax.set_ylabel("t")

            ax.set_xlim(xlims)
            ax.set_ylim(tlims)
            
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    row_titles = [
        "Heatmap of test vectors after SVD (spin 0)",
        "Heatmap of test vectors after SVD (spin 1)"
    ]

    fig.text(0.5, 0.95, row_titles[0], ha='center', fontsize=14)
    fig.text(0.5, 0.53, row_titles[1], ha='center', fontsize=14)

     # Metadata panel (bottom row spanning all columns)
    ax_meta = fig.add_subplot(gs[2, :])
    ax_meta.axis('off')  # hide axes

    # Draw a box with text inside
    Nx, Nt, beta, m0_diff, blockSize,blockID, krank, Nv = metadata
    metadata_text = r"$N_x$={0}, $N_t=${1}, $\beta=${2}, $|m_0-m_c|=${3}".format(Nx, Nt, beta, m0_diff)+\
    "\nBlocks size = {0}x{0}".format(blockSize) +\
    "\nResults shown for block number {0}".format(blockID) +\
    "\n{0}-rank truncation for a total of {1} test vectors\n".format(krank, Nv) +\
    "Test vectors were generated with 4 SAP iterations"
    ax_meta.text(
        0.5, 0.5, metadata_text,
        ha='center', va='center',
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightgray", edgecolor="black")
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    plt.show()
    if save == True:
        fig.savefig(fig_name)


def decomposed_vectors_v2(blockID,block_x,block_t,spin,test_vectors):
    """
    Restrict test vectors over a lattice block
    """
    Nv, lalala, Nt, Nx = test_vectors.shape
    Nblocks =  block_x*block_t

    x_elements = Nx // block_x
    t_elements = Nt // block_t

    assert blockID < Nblocks, "blockID should be within the range of the number of lattice blocks"
    #v = np.zeros((dim,Nv*Nblocks),dtype=complex)
    
    bx = blockID // block_x
    bt = blockID % block_t 
    #----Coordinates of elements inside block----#
    xini, tini = x_elements * bx, t_elements * bt
    xfin = xini + x_elements
    tfin = tini + t_elements
    #--------------------------------------------#
    tvec = np.zeros((Nv,Nt,Nx),dtype=complex)
    for x in range(Nx):
        for t in range(Nt):
            if xini<=x<xfin and tini<=t<tfin:
                tvec[:,t,x] = test_vectors[:,spin,t,x]
    
    return tvec 

def read_and_decompose_v2(blockID,block_x,block_t,params_list):
    beta, m0_str, m0_folder, nconf, NV, Nx, Nt = params_list
    test_vectors = np.zeros((NV,2,Nx,Nt),dtype=complex)
    print("beta={0}, conf={1}, Nv={2}, Nx={3}, Nt={4}\nblocks_x={5}, blocks_t={6}, bx_size={7}, bt_size={8}".format
         (beta,nconf,NV,Nx,Nt,block_x,block_t,Nx//block_x,Nt//block_t)
         )
    
    for tv in range(NV):
        path = "../real_tv/b{0}_{1}x{1}/{2}/tvector_{1}x{1}_b{0}0000_m{3}_nconf{4}_tv{5}.tv".format(
            beta, Nx, m0_folder,m0_str, nconf, tv
            )
        test_vector = read_binary_conf(Nx,Nt,path)
        #flatten_vector_spin0 = test_vector[0].flatten()
        test_vectors[tv] = test_vector

    spin = 0
    dtv_spin0 = decomposed_vectors_v2(blockID,block_x,block_t,spin,test_vectors) #[]
    dtv_spin0 =  np.transpose(dtv_spin0.reshape(NV,-1))
    spin = 1
    dtv_spin1 = decomposed_vectors_v2(blockID,block_x,block_t,spin,test_vectors) #[]
    dtv_spin1 =  np.transpose(dtv_spin1.reshape(NV,-1))
    print("Test vectors matrix shape",dtv_spin0.shape)
    return dtv_spin0, dtv_spin1