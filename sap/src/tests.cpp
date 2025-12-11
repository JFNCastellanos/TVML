#include "tests.h"


void Tests::BiCG(spinor& x,const int max_it,const bool save,const bool print){   
    std::cout << "--------------Bi-CGstab inversion--------------" << std::endl;
    start = clock();
    x = bi_cgstab(&D_phi,LV::Ntot,2,GConf.Conf, rhs, x0, m0, max_it, 1e-10, save,print);
    end = clock();
    elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for Bi-CGstab = " << elapsed_time << " seconds" << std::endl;
}

void Tests::GMRES(spinor& x, const int len, const int restarts,const bool save, const bool print){
    std::cout << "--------------GMRES without preconditioning--------------" << std::endl;
    FGMRES_fine_level fgmres_fine_level(LV::Ntot, 2, len, restarts,1e-10,GConf.Conf, m0);
    start = clock();
    fgmres_fine_level.fgmres(rhs,x0,x,save,print);
    end = clock();
    elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for GMRES = " << elapsed_time << " seconds" << std::endl; 
}

void Tests::CG(spinor& x){
    std::cout << "--------Inverting the normal equations with CG----------" << std::endl; 
    start = clock();
    conjugate_gradient(GConf.Conf, rhs, x, m0);
    end = clock();
    elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for CG = " << elapsed_time << " seconds" << std::endl;  
}

void Tests::FGMRES_sap(spinor& x, const bool save,const bool print){
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "--------------Flexible GMRES with SAP preconditioning version --------------" << std::endl;
     
    MPI_Barrier(MPI_COMM_WORLD);
    FGMRES_SAP fgmres_sap(LV::Ntot, 2, FGMRESV::fgmres_restart_length, FGMRESV::fgmres_restarts,FGMRESV::fgmres_tolerance,GConf.Conf, m0);
    startT = MPI_Wtime();
    fgmres_sap.fgmres(rhs,x0,x,save,print);
    endT = MPI_Wtime();
    if (rank == 0)
        printf("[rank %d] time elapsed during FGMRES_SAP implementation: %.4fs.\n", rank, endT - startT);
    fflush(stdout);

}

void Tests::SAP(spinor& x,const int iterations, const bool print){
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "--------------SAP as stand-alone solver --------------" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    startT = MPI_Wtime();
    sap.SAP(rhs,x,iterations, SAPV::sap_blocks_per_proc,print);
    endT = MPI_Wtime();
    if (rank == 0)
        printf("[rank %d] time elapsed during SAP implementation: %.4fs.\n", rank, endT - startT);
    fflush(stdout);

}

void Tests::check_solution(const spinor& x_sol){
    spinor xini(LV::Ntot, c_vector(2, 0)); //Initial guess
    D_phi(GConf.Conf, x_sol, xini, m0); //D_phi U x
    for(int i = 0; i< LV::Ntot; i++){
        if (std::abs(xini[i][0] - rhs[i][0]) > 1e-8 || std::abs(xini[i][1] - rhs[i][1]) > 1e-8) {
            std::cout << "Solution not correct at index " << i << ": " << xini[i][0] << " != " << rhs[i][0] << " or " << xini[i][1] << " != " << rhs[i][1] << std::endl;
        }
    }
    std::cout << "solution good " << std::endl;
    
}