#include <time.h> 
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include "params.h" //Read parameters for lattice blocks, test vectors and SAP blocks
#include "bi_cgstab.h" //BiCGstab for comparison
#include "conjugate_gradient.h" //Conjugate gradient for inverting the normal equations
#include "boundary.h" //Build boundary conditions at every grid level
#include "amg.h" //Algebraic Multigrid Method
#include "mpi.h" //MPI
#include "tests.h" //Class for testing



#include <cstdint>
#include <cstring>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    readParameters("../parameters.dat");
    srand(19);
    //srand(time(0));
    
    Coordinates(); //Builds array with coordinates of the lattice points x * Nt + t
    boundary(); //Boundaries for every level

    AMGV::cycle = 0; //K-cycle = 1, V-cycle = 0
    AMGV::Nit = 0;
    AMGV::SAP_test_vectors_iterations = 4;
    //-0.1023;//-0.0933;//-0.18840579710144945; //0.0709
    double m0 = -0.18840579710144945; 
    double beta = 2;
    mass::m0 = m0;
    beta::beta = beta;

    
    std::string confFile;
    std::string rhsFile;

    
    MPI_Bcast(&beta, 1, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    MPI_Bcast(&m0, 1, MPI_DOUBLE,  0, MPI_COMM_WORLD);

    //Parameters in variables.cpp
    if (rank == 0){
        printParameters();
    }

    std::vector<int> confsID;
    std::ostringstream confsIDfile;
    confsIDfile << "../../fake_tv/b" << beta << "_" << LV::Nx << "x" << LV::Nt 
		<< "/m-018/confFiles.txt";
    readConfsID(confsID,confsIDfile.str());

    //mlearning::confID = confsID[0];
    //mlearning::confID = 256;
    

    for (int id = 0; id < 5; id++){
        mlearning::confID = confsID[id];
        
    
    std::ostringstream gauge_conf_file;
    gauge_conf_file << "/wsgjsc/home/nietocastellanos1/Documents/SchwingerModel/fermions/SchwingerModel/confs/b"
     << beta << "_" << LV::Nx << "x" << LV::Nt 
		<< "/m-018/"
        <<
        "2D_U1_Ns"<< LV::Nx <<"_Nt" << LV::Nt << "_b" 
        << format(beta::beta).c_str() << "_m" 
        << format(mass::m0).c_str() << "_" 
        << mlearning::confID << ".ctxt";



    GaugeConf GConf = GaugeConf(LV::Nx, LV::Nt);
    GConf.readBinary(gauge_conf_file.str());


    
    
    const spinor x0(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0)); //Intial guesss
    spinor rhs(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));

    
    
    random_rhs(rhs,0);
    //readBinaryRhs(rhs,rhsFile);
    
    

    //Solution buffers
    spinor x_bi(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    spinor x_cg(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    spinor xFAMG(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    spinor xFAMGSetup2(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    //spinor xAMG(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));

    if (rank == 0) std::cout << "/////////////// Testing with confID " << mlearning::confID << "///////////////" << std::endl;
    
    Tests test(GConf, rhs, x0 ,m0);
    if (rank == 0){
        test.BiCG(x_bi, 10000,true); //BiCGstab for comparison  
        test.CG(x_cg); //Conjugate Gradient for inverting the normal equations
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int setup = 0;
    test.fgmresAMG(xFAMG, true,setup);
    
    if (rank == 0){
        std::cout << "********************************************************************" << std::endl;
        std::cout << " Reading test vectors from file " << std::endl;
    }

    setup = 1;
    test.fgmresAMG(xFAMGSetup2, true,setup);
    if (rank == 0) std::cout << "********************************************************************" << std::endl;
    
    }

    MPI_Finalize();

    

    return 0;
}

//Four levels multigrid
//The parameters file has the following information on each row
//level, block_x, block_t, ntest, sap_block_x, sap_block_t
//0 8 8 10 4 4
//1 4 4 10 4 4
//2 2 2 10 2 2

//Three levels multigrid
//0 8 8 10 4 4
//1 4 4 10 4 4

//or

//0 4 4 10 4 4
//1 2 2 10 2 2



/*
 //Reading Conf
    
    {
        std::ostringstream NameData;
        NameData << "../../SchwingerModel/fermions/SchwingerModel/confs/b" << beta << "_" << LV::Nx << "x" << LV::Nt << "/m-01023/2D_U1_Ns" << LV::Nx << "_Nt" << LV::Nt << "_b" << 
        format(beta).c_str() << "_m" << format(m0).c_str() << "_" << nconf << ".ctxt";
               
        if (rank == 0)
            std::cout << "Reading conf from file: " << NameData.str() << std::endl;
        //GConf.read_conf(NameData.str());
        GConf.readBinary(NameData.str());
    }
*/

/*
    double Iter[3]; double exTime[3];
    double dIter[3]; double dexTime[3];
    const int Meas = 10;
    for(int i = 0; i < 3; i++){

    AMGV::Nit = 2*i;
    if (rank == 0) std::cout << "Number of iterations for improving the interpolator: " << AMGV::Nit << std::endl;

    
    std::vector<double> iterations(Meas,0);
    std::vector<double> times(Meas,0);
    if (rank == 0) std::cout << "--------------Flexible GMRES with AMG preconditioning--------------" << std::endl;

    for(int i = 0; i < Meas; i++){
        if (rank == 0) std::cout << "Meas " << i << std::endl;
        iterations[i] = test.fgmresAMG(xFAMG, true);
        times[i] = total_time;
    }
    if (rank == 0){
        std::cout << "Average iteration number over " << Meas << " runs: " << mean(iterations) << " +- " 
        << standard_deviation(iterations)/sqrt(1.0*Meas) << std::endl;
    
    }

    Iter[i] = mean(iterations);
    dIter[i] = standard_deviation(iterations)/sqrt(1.0*Meas);
    exTime[i] = mean(times);
    dexTime[i] = standard_deviation(times)/sqrt(1.0*Meas);

    }

    for(int i = 0; i < 3; i++){
        if (rank == 0) std::cout << "Nit: " << 2*i << " Iter: " << Iter[i] << " +- " << dIter[i] << std::endl;

    }
    if (rank == 0)
        saveParameters(Iter, dIter, exTime, dexTime, 3,nconf);
*/

    
    //test.multigrid(xAMG,true); //Multigrid as stand-alone solver
    //test.check_solution(xFAMG); //Check that the solution is correct


    /*
        std::ostringstream FileName;
    FileName << "../../SchwingerModel/fermions/SchwingerModel/confs/rhs/rhs_conf" << 
    nconf << "_" << LV::Nx << "x" << LV::Nt << "_b" << format(beta) << "_m" << format(m0) << ".rhs";
    
    //read_rhs(rhs,FileName.str());
    readBinaryRhs(rhs,FileName.str());
    //random_rhs(rhs,10);

    // Save rhs to a .txt file
    if (rank == 0){
        std::ostringstream FileName;
        FileName << "rhs_conf" << nconf << "_" << LV::Nx << "x" << LV::Nt
                 << "_b" << format(beta) << "_m" << format(m0)
                 << ".rhs";
      //  save_rhs(rhs,FileName.str());
    }
    */