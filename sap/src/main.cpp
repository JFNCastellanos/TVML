#include <time.h> 
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include "tests.h"
#include "mpi.h"

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

    using namespace SAPV;
    MPI_Init(&argc, &argv);
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //srand(19);

    srand(time(0));
    
    Coordinates(); //Builds array with coordinates of the lattice points x * Nt + t
    periodic_boundary(); //Builds LeftPB and RightPB (periodic boundary for U_mu(n))
    CheckBlocks(); //Check blocks dimensions
    
    m0 = -0.18840579710144945; //Globally declared
    FGMRESV::fgmres_restart_length = 25;
    //Parameters in variables.cpp
    if (rank == 0)
        print_parameters();
    
    GaugeConf GConf = GaugeConf(Nx, Nt);
    GConf.initialize(); //Initialize a random gauge configuration

    
    double beta = 2;
    int Nv = 30;
    int number_of_confs = 1000;
    
    for(int nconf = 0; nconf<number_of_confs; nconf++){
    //Reading Conf
    if (rank == 0){
    std::cout << "********************************************************************************************\n";
    std::cout << "Conf " << nconf << std::endl;
    }
    {
        std::ostringstream NameData;
        NameData << "/wsgjsc/home/nietocastellanos1/Documents/SchwingerModel/fermions/SchwingerModel/confs/b" <<
        beta << "_" << LV::Nx << "x" << LV::Nt << "/m-018/2D_U1_Ns" << LV::Nx << "_Nt" << LV::Nt << "_b" << 
        format(beta).c_str() << "_m" << format(m0).c_str() << "_" << nconf << ".ctxt";
        GConf.readBinary(NameData.str());
    }

    sap.set_params(GConf.Conf, m0); //Setting gauge conf and m0 for SAP 

    for(int tv = 0; tv < Nv; tv++){
        if (rank == 0){
            std::cout << "tv " << tv << std::endl;
        }
    spinor rhs(Ntot, c_vector(2, 0)); //right hand side = 0
    spinor test_vector(Ntot, c_vector(2, 0)); //initial guess
    random_rhs(test_vector,nconf*Nv+tv); //random test vector

    clock_t start, end;
    double elapsed_time;
    double startT, endT;
    //spinor x_bi(Ntot, c_vector(2, 0));
    //spinor x_cg(Ntot, c_vector(2, 0));
    //spinor x_gmres(Ntot, c_vector(2, 0));
    //spinor x_fsap(Ntot, c_vector(2, 0));
    //spinor x_sap(Ntot, c_vector(2, 0));

    Tests tests(GConf,rhs,test_vector, m0);
    //if (rank == 0){
        //tests.BiCG(x_bi,10000,false,true); 
        //tests.GMRES(x_gmres,25, 100,false,true);
        //tests.CG(x_cg);
    //}
    //tests.FGMRES_sap(x_fsap,false,true);
    tests.SAP(test_vector,4,true);
    //save testvector
    if (rank == 0){
        std::ostringstream NameData;
        NameData << "tvector_" << LV::Nx << "x" << LV::Nt << "_b" << 
        format(beta).c_str() << "_m" << format(m0).c_str() << "_nconf" << nconf << "_tv" << tv << ".tv";
        save_rhs(test_vector,NameData.str());
    }
    
}
    if (rank == 0){
        std::cout << "\n";
    }
}

     
    MPI_Barrier(MPI_COMM_WORLD);

    //if (rank == 0)
        //tests.check_solution(x_sap);
    
    MPI_Finalize();

    return 0;
}