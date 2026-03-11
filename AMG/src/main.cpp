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

//mean of a vector
template <typename T>
double mean(std::vector<T> x){ 
    double prom = 0;
    for (T i : x) {
        prom += i*1.0;
    }   
    prom = prom / x.size();
    return prom;
}

template <typename T>
double standard_deviation(const std::vector<T>& data) {
    if (data.empty()) return 0.0;
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sq_sum = 0.0;
    for (const auto& val : data) {
        sq_sum += (static_cast<double>(val) - mean) * (static_cast<double>(val) - mean);
    }
    return std::sqrt(sq_sum / data.size());
}

//Formats decimal numbers
//For opening file with confs 
static std::string format(const double& number) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << number;
    std::string str = oss.str();
    str.erase(str.find('.'), 1); //Removes decimal dot 
    return str;
}

int main() {
    readParameters("../parameters.dat");
    srand(19);
    //srand(time(0));
    
    Coordinates(); //Builds array with coordinates of the lattice points x * Nt + t
    boundary(); //Boundaries for every level

    AMGV::cycle = 0; //K-cycle = 1, V-cycle = 0
    AMGV::Nit = 0;
    AMGV::SAP_test_vectors_iterations = 4;
    //-0.1023;//-0.0933;//-0.18840579710144945; //0.0709
    double m0 = mass::m0; 


    //Open conf from file//
    GaugeConf GConf = GaugeConf(LV::Nx, LV::Nt);
    GConf.initialize();

    double beta;
    int nconf;
    std::string confFile;
    std::string rhsFile;

    //---Input data---//
    std::cout << "Nx " << LV::Nx << " Nt " << LV::Nt << std::endl;
    std::cout << "beta : ";
    std::cin >> beta;
    std::cout << "m0: ";
    std::cin >> m0;
    std::cout << "Configuration id: ";
    std::cin >> nconf;
    std::cout << "Configuration file path: ";
    std::cin >> confFile;
    std::cout << "RHS file path: ";
    std::cin >> rhsFile;
    std::cout << " " << std::endl;
    
   
    mass::m0 = m0;

    //Parameters in variables.cpp

    printParameters();
    std::cout << "Conf read from " << confFile << std::endl;
    std::cout << "rhs read from " << rhsFile << std::endl;
    
    
    const spinor x0(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0)); //Intial guesss
    spinor rhs(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));

    GConf.readBinary(confFile);
    readBinaryRhs(rhs,rhsFile);

    sap.set_params(GConf.Conf, m0); //Setting gauge conf and m0 for SAP 

    //Solution buffers
    spinor x_bi(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    spinor x_cg(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    spinor xSAP(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    spinor XFGMRES_SAP(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    spinor xFAMG(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    spinor xGMRES(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));
    //spinor xAMG(LevelV::Nsites[0],c_vector(LevelV::DOF[0],0));


    
    Tests test(GConf, rhs, x0 ,m0);
    test.BiCG(x_bi, 10000,true); //BiCGstab for comparison  
    test.CG(x_cg); //Conjugate Gradient for inverting the normal equations
    test.SAP(xSAP,400,true);
    test.FGMRES_sap(XFGMRES_SAP,true);

    test.GMRES(xGMRES, 50, 100,true);
    test.fgmresAMG(xFAMG, true);
    test.check_solution(xFAMG);

    return 0;
}
