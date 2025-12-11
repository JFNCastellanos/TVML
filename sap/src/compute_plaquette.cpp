#include <time.h> 
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include "dirac_operator.h"
#include "gauge_conf.h"

//Formats decimal numbers
//For opening file with confs 
static std::string format(const double& number) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << number;
    std::string str = oss.str();
    str.erase(str.find('.'), 1); //Removes decimal dot 
    return str;
}

int main(int argc, char **argv) {

    //srand(19);
    srand(time(0));    
    Coordinates(); //Builds array with coordinates of the lattice points x * Nt + t
    periodic_boundary(); //Builds LeftPB and RightPB (periodic boundary for U_mu(n))
    CheckBlocks(); //Check blocks dimensions
    
    m0 = -0.18840579710144945; //Globally declared

    
    GaugeConf GConf = GaugeConf(LV::Nx, LV::Nt);
    GConf.initialize(); //Initialize a random gauge configuration

    
    double beta = 2;
    int number_of_confs = 1000;
    
    std::cout << "Computing and storing the plaquettes... " << std::endl;
    for(int nconf = 0; nconf<number_of_confs; nconf++){
   
    
        std::ostringstream NameData;
        NameData << "/wsgjsc/home/nietocastellanos1/Documents/SchwingerModel/fermions/SchwingerModel/confs/b" <<
        beta << "_" << LV::Nx << "x" << LV::Nt << "/m-018/2D_U1_Ns" << LV::Nx << "_Nt" << LV::Nt << "_b" << 
        format(beta).c_str() << "_m" << format(m0).c_str() << "_" << nconf << ".ctxt";
        GConf.readBinary(NameData.str());

        GConf.Compute_Plaquette01();
        if (nconf == 0){
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
        format(beta).c_str() << "_m" << format(m0).c_str() << "_nconf" << nconf << ".plaq";
        //GConf.savePlaquette(NamePlaq.str());
    }
    std::cout << "Done " << std::endl;


    return 0;
}