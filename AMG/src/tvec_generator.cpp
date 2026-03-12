#include <time.h> 
#include <ctime>
#include <string>
#include <sstream>
#include "utils.h"
#include "params.h" //Read parameters for lattice blocks, test vectors and SAP blocks
#include "boundary.h"
#include "gauge_conf.h"
#include "sap.h"

int main() {
    srand(time(0));
    readParameters("../parameters.dat");
    Coordinates(); //Builds array with coordinates of the lattice points x * Nt + t
    boundary(); //Builds LeftPB and RightPB (periodic boundary for U_mu(n))
    
    mass::m0 = -0.18840579710144945; //Globally declared
    GaugeConf GConf = GaugeConf(LV::Nx, LV::Nt);
    GConf.initialize(); //Initialize a random gauge configuration

    double beta = 2;
    int Nv = 30;    //Number of test vectors to be generated
    int number_of_confs = 1000; //Number of confs to consider.
    int sap_iterations = 4; //Number of smoothing iterations
    
    std::cout << "Generating Nv=" << Nv << " test vectors for " << number_of_confs << " gauge conf" << std::endl;
    std::cout << "Test vectors are smoothed with " << sap_iterations << " SAP iterations" << std::endl;
    std::cout << "SAP_X_BLOCKS " << LevelV::SAP_Block_x[0] << " SAP_T_BLOCKS " << LevelV::SAP_Block_t[0] << std::endl;

    for(int nconf = 0; nconf<number_of_confs; nconf++){
        //Reading Conf
        std::cout << "***************** Conf " << nconf << " **************************\n" << std::endl;
    
        {
            std::ostringstream NameData;
            NameData << "/wsgjsc/home/nietocastellanos1/Documents/SchwingerModel/fermions/SchwingerModel/confs/b" <<
            beta << "_" << LV::Nx << "x" << LV::Nt << "/m-018/2D_U1_Ns" << LV::Nx << "_Nt" << LV::Nt << "_b" << 
            format(beta).c_str() << "_m" << format(mass::m0).c_str() << "_" << nconf << ".ctxt";
            GConf.readBinary(NameData.str());
        }

        SAP_fine_level sap_method(LV::Ntot,  2, SAPV::sap_tolerance, LV::Nt, LV::Nx,LevelV::SAP_Block_x[0],LevelV::SAP_Block_t[0],2,1);
        sap_method.set_params(GConf.Conf, mass::m0); //Setting gauge conf and m0 for SAP 

        bool print = false;
        spinor rhs(LV::Ntot, c_vector(2, 0)); //right hand side = 0
        for(int tv = 0; tv < Nv; tv++){
        
            std::cout << "tv " << tv << std::endl;
         
            spinor test_vector(LV::Ntot, c_vector(2, 0)); //initial guess
            random_rhs(test_vector,nconf*Nv+tv); //random test vector

            clock_t start, end;
            double elapsed_time;
            double startT, endT;

            sap_method.SAP(rhs,test_vector,sap_iterations, 1,print);
        
            //save testvector
            {
                std::ostringstream NameData;
                NameData << "tvector_" << LV::Nx << "x" << LV::Nt << "_b" << 
                format(beta).c_str() << "_m" << format(mass::m0).c_str() << "_nconf" << nconf << "_tv" << tv << ".tv";
                save_rhs(test_vector,NameData.str());
            }
        
    
        }

        std::cout << "\n";

    }
    return 0;
}