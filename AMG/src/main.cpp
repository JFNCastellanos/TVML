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
#include "tests.h" //Class for testing

#include <cstdint>
#include <cstring>


int main() {
    std::cout << "Nx " << LV::Nx << " Nt " << LV::Nt << std::endl;
    readParameters("../parameters.dat");
    srand(19);
    //srand(time(0));
    
    Coordinates(); //Builds array with coordinates of the lattice points x * Nt + t
    boundary(); //Boundaries for every level

    AMGV::cycle = 0; //K-cycle = 1, V-cycle = 0
    AMGV::Nit = 0;
    int SAP_test_vec_iter = 4;
    AMGV::SAP_test_vectors_iterations = SAP_test_vec_iter;
    //-0.1023;//-0.0933;//-0.18840579710144945; //0.0709
    double m0 = -0.18840579710144945; 
    double beta = 2;
    mass::m0 = m0;
    beta::beta = beta;

    int nconf;
    std::cout << "Train or test set (0 or 1) ";
    std::cin >> mlearning::set;
    std::cout << "Number of confs ";
    std::cin >> nconf;
    //Open conf from file//
    GaugeConf GConf = GaugeConf(LV::Nx, LV::Nt);
    GConf.initialize();

    std::string confFile;
    std::string rhsFile;
    printParameters();

    std::vector<int> confsID;
    std::ostringstream confsIDfile;
    if (mlearning::set == 0){
        std::cout << "Analyzing training set tv" << std::endl;
        confsIDfile << "../../fake_tv/b" << beta << "_" << LV::Nx << "x" << LV::Nt << "/m-018/train/confFiles.txt";
    }
    else if (mlearning::set == 1){
        std::cout << "Analyzing testing set tv" << std::endl;
        confsIDfile << "../../fake_tv/b" << beta << "_" << LV::Nx << "x" << LV::Nt << "/m-018/test/confFiles.txt";
    }
    else{ 
        std::cout << "Introduce a valid number for the set of configurations (train or test)" << std::endl;
        exit(1);
    }


    readConfsID(confsID,confsIDfile.str());
    
    
    
    std::vector<double> smooth_tv_amg_iter(nconf,0);
    std::vector<double> random_tv_amg_iter(nconf,0);
    std::vector<double> learned_tv_amg_iter(nconf,0);
    for (int id = 0; id < nconf; id++){
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
        spinor xFAMGRandom(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
        spinor xFAMGSetup2(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
        

        std::cout << "/////////////// Testing with confID " << mlearning::confID << "///////////////" << std::endl;
    
        Tests test(GConf, rhs, x0 ,m0);
        //test.BiCG(x_bi, 10000,true); //BiCGstab for comparison  
        //test.CG(x_cg); //Conjugate Gradient for inverting the normal equations
        AMGV::SAP_test_vectors_iterations = SAP_test_vec_iter;
        std::cout << "Smoothing test vectors with " << AMGV::SAP_test_vectors_iterations << " SAP iterations" << std::endl;
        int setup = 0;  //Usual set up (smoothing test vectors)
        smooth_tv_amg_iter[id] = test.fgmresAMG(xFAMG, true,setup);
    
        std::cout << "Random test vectors " << std::endl;
        AMGV::SAP_test_vectors_iterations = 0;
        random_tv_amg_iter[id] = test.fgmresAMG(xFAMGRandom, true,setup);

        std::cout << "********************************************************************" << std::endl;
        std::cout << " Reading test vectors from file " << std::endl;
   
        setup = 1; //machine learning generated test vectors
        learned_tv_amg_iter[id] = test.fgmresAMG(xFAMGSetup2, true,setup);
   
    
    }

    std::cout << "Mean iteration count for smoothed test vectors " << mean(smooth_tv_amg_iter) << " +- " << standard_deviation(smooth_tv_amg_iter) << std::endl;
    std::cout << "Mean iteration count for random test vectors   " << mean(random_tv_amg_iter) << " +- " << standard_deviation(random_tv_amg_iter) << std::endl;
    std::cout << "Mean iteration count for learned test vectors  " << mean(learned_tv_amg_iter) << " +- " << standard_deviation(learned_tv_amg_iter) << std::endl;

    return 0;
}
