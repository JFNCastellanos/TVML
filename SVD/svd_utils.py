import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import struct

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

def read_vectors(params_list):
    beta, m0_str, m0_folder, nconf, NV, Nx, Nt = params_list
    test_vectors = np.zeros((NV,2,Nx,Nt),dtype=complex)    
    for tv in range(NV):
        path = "real_tv/b{0}_{1}x{1}/{2}/tvector_{1}x{1}_b{0}0000_m{3}_nconf{4}_tv{5}.tv".format(
            beta, Nx, m0_folder,m0_str, nconf, tv
            )
        test_vector = read_binary_conf(Nx,Nt,path)
        test_vectors[tv] = test_vector
    #(NV,2,Nx,Nt)
    return test_vectors

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
    test_vectors = read_vectors(params_list)
    spin = 0
    dtv_spin0 = decomposed_vectors(blockID,block_x,block_t,spin,test_vectors) #[]
    dtv_spin0 =  np.transpose(dtv_spin0.reshape(NV,-1))
    spin = 1
    dtv_spin1 = decomposed_vectors(blockID,block_x,block_t,spin,test_vectors) #[]
    dtv_spin1 =  np.transpose(dtv_spin1.reshape(NV,-1))
    #print("Test vectors matrix shape",dtv_spin0.shape)
    return dtv_spin0, dtv_spin1

def apply_SVD(tvectors,k_rank,printMessage=True):
    U, s, Vh = np.linalg.svd(tvectors, full_matrices=False, compute_uv=True, hermitian=False)

    if np.allclose(np.matmul(np.matmul(U,np.diag(s)),Vh),tvectors):
        if printMessage == True:
            print("matrix succesfully reconstructed")
    else:
        print("something wrong with the SVD")
        print("U shape",U.shape)
        print("s shape",s.shape)
        print("Vh shape",Vh.shape)
    #assert k_rank<NV, "k has to be smaller than NV"
    Uk, sk, Vk = U[:,:k_rank], s[:k_rank], Vh[:k_rank,:]
    low_rank_tv = np.matmul(np.matmul(Uk,np.diag(sk)),Vk)
    if printMessage == True:
        print("-------------------")
        print("Uk shape",Uk.shape)
        print("sk shape",sk.shape)
        print("Vk shape",Vk.shape)
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
    Nx, Nt, beta, m0_diff, blockSize,blockID, krank, Nv, SAP_block_size = metadata
    metadata_text = r"$N_x$={0}, $N_t=${1}, $\beta=${2}, $|m_0-m_c|=${3}".format(Nx, Nt, beta, m0_diff)+\
    "\nBlocks size for SVD = {0}x{0}".format(blockSize) +\
    "\nResults shown for block number {0}".format(blockID) +\
    "\n{0}-rank truncation for a total of {1} test vectors\n".format(krank, Nv) +\
    "Test vectors were generated with 4 SAP iterations with block size {0}".format(SAP_block_size)
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

def k_cluster(low_rank_tv,tvID):
    # Flatten data to (n_samples, 1)
    data = np.abs(low_rank_tv[:,:,tvID])
    X = data.reshape(-1, 1)
    # K-means
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    
    #Sort labels according to value. High value -> label 1, low value -> label 0
    # Get cluster centers
    centers = kmeans.cluster_centers_.flatten()
    # Sort clusters by center value
    order = np.argsort(centers)  # smallest first
    # Build mapping: old_label → new_label
    mapping = {old: new for new, old in enumerate(order)}
    # Apply mapping
    labels = np.vectorize(mapping.get)(labels)
    labels_2d = labels.reshape(data.shape)
    return labels_2d
    

def make_heatmap_and_k_cluster(low_rank_tv,xlims,tlims,tvID,fig_name="",save=False):
    data = np.abs(low_rank_tv[:,:,tvID])
    labels_2d = k_cluster(low_rank_tv,tvID)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Test vector {0}".format(tvID))
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("K-means clusters")
    plt.imshow(labels_2d, cmap='coolwarm', origin='lower')
    plt.colorbar(label='Cluster')

    plt.xlabel(r"x")
    plt.ylabel(r"t")
    plt.xlim(xlims)
    plt.ylim(tlims)
    
    plt.tight_layout()
    plt.show()
    if save==True:
        fig.savefig(fig_name)

def mask_vectors(k_rank,block_x,block_t,params_list):
    """
    Inputs: 
        block_i -> Number of blocks on i for the SVD. 
                 This should coincide with the number of blocks for the aggregates.
    returns: 
        masked test vectors
    """
    beta, m0_str, m0_folder, nconf, NV, Nx, Nt = params_list
    Nblocks = block_x*block_t
    x_elements, t_elements = Nx//block_x, Nt//block_t
    test_vectors = np.zeros((NV,2,Nx,Nt),dtype=complex)  
    for blockID in range(Nblocks):
        dtv_spin0, dtv_spin1 = read_and_decompose(blockID,block_x,block_t,params_list)
        #Spin component 0
        low_rank_tv0 = apply_SVD(dtv_spin0,k_rank,False)
        low_rank_tv0 = low_rank_tv0.reshape(t_elements,x_elements,NV)
        #Spin component 1
        low_rank_tv1 = apply_SVD(dtv_spin1,k_rank,False)
        low_rank_tv1 =low_rank_tv1.reshape(t_elements,x_elements,NV)

        svd_vectors = [low_rank_tv0,low_rank_tv1]

        #----Coordinates of elements inside block----#
        bx = blockID // block_x
        bt = blockID % block_t  
        xini, tini = x_elements * bx, t_elements * bt
        xfin = xini + x_elements
        tfin = tini + t_elements
        #--------------------------------------------#
        for spin in range(len(svd_vectors)):
            for tvID in range(NV):
                labels_2d = k_cluster(svd_vectors[spin],tvID)
                for x in range(x_elements):
                    for t in range(t_elements):
                        svd_vectors[spin][t,x,tvID] *= labels_2d[t,x]
                test_vectors[tvID,spin,tini:tfin,xini:xfin] = svd_vectors[spin][:,:,tvID]
    return test_vectors


def save_vectors(masked_test_vectors,beta,Nx,Nt,m0_folder,nconf,k_rank):
    """
    Save masked vectors
    """
    dataname = "train" 
    for tv in range(k_rank):
        file_path = "../fake_tv/b{0}_{1}x{2}/{3}/{4}/conf{5}_fake_tv{6}.tv".format(beta,Nx,Nt,m0_folder,
                        dataname,nconf,tv)
        fmt = "<3i2d"
        with open(file_path, "wb") as f:
            for x in range(Nx):
                for t in range(Nt):
                    for mu in range(2):
                        value = masked_test_vectors[tv,mu,t,x]
                        Re = np.real(value)
                        Im = np.imag(value)
                        data = struct.pack(fmt, int(x), int(t), int(mu), float(Re), float(Im))
                        f.write(data)