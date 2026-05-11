#include <time.h> 
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include "params.h" //Read parameters for lattice blocks, test vectors and SAP blocks
#include "boundary.h"
#include "dirac_operator.h"
#include "gauge_conf.h"
#include "utils.h"
#include "app_config.h"


int main(int argc, char **argv) {
    const std::string configFile = (argc > 1) ? argv[1] : locateConfigFile("config.txt");
    AppConfig appConfig = readAppConfig(configFile);

    srand(time(0));    
    readParameters("../parameters.dat");
    Coordinates(); //Builds array with coordinates of the lattice points x * Nt + t
    boundary(); //Builds LeftPB and RightPB (periodic boundary for U_mu(n))
    
    mass::m0 = appConfig.m0; //Globally declared
    GaugeConf GConf = GaugeConf(LV::Nx, LV::Nt);
    GConf.initialize(); //Initialize a random gauge configuration
        
    double beta = appConfig.beta;
    int number_of_confs = appConfig.number_of_confs;
    std::string conf_dir = appConfig.gauge_conf_dir;
    std::string m_dir = appConfig.m_dir;
    
    std::cout << "Computing and storing the plaquettes for beta=" << beta << " m0=" << appConfig.m0 << " and " << number_of_confs << " configs" << std::endl;
    for(int nconf = 0; nconf<number_of_confs; nconf++){    
        std::ostringstream NameData;
        NameData << conf_dir << "/b" << beta << "_" << LV::Nx << "x" << LV::Nt << "/" << m_dir
                 << "/2D_U1_Ns" << LV::Nx << "_Nt" << LV::Nt << "_b" << format(beta).c_str()
                 << "_m" << format(mass::m0).c_str() << "_" << nconf << ".ctxt";
        GConf.readBinary(NameData.str());

        GConf.Compute_Plaquette01();
        if (nconf < 20){
            c_vector Plaquette01 = GConf.getPlaq();
            std::cout << "U_01(0,0) " << Plaquette01[0] << std::endl;
            int nx = 0, nt = 1;
            int N = nx * LV::Nt + nt; 
            std::cout << "U_01(1,0) " << Plaquette01[N] << std::endl;
            nx = 1, nt = 0;
            N = nx * LV::Nt + nt; 
            std::cout << "U_01(0,1) " << Plaquette01[N] << std::endl;
            nx = 1, nt = 1;
            N = nx * LV::Nt + nt; 
            std::cout << "U_01(1,1) " << Plaquette01[N] << std::endl;
            nx = 10, nt = 15;
            N = nx * LV::Nt + nt; 
            std::cout << "U_01(15,10) " << Plaquette01[N] << std::endl;

        }
        
        std::ostringstream NamePlaq;
        NamePlaq << "plaquette_" << LV::Nx << "x" << LV::Nt << "_b" << 
        format(beta).c_str() << "_m" << format(mass::m0).c_str() << "_nconf" << nconf << ".plaq";
        GConf.savePlaquette(NamePlaq.str());
    }
    std::cout << "Done " << std::endl;


    return 0;
}