#ifndef VARIABLES_H_INCLUDED
#define VARIABLES_H_INCLUDED
#include "config.h"
#include <iostream>
#include <complex>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <random>

typedef std::complex<double> c_double;
constexpr double pi=3.14159265359;

extern double m0;

//------------Lattice parameters--------------//
namespace LV {
    constexpr int Nx=NS; 
    constexpr int Nt = NT; 
    constexpr int Ntot = Nx*Nt; //Total number of lattice points
}

//------------Schwarz alternating procedure parameters--------------//
namespace SAPV {
    using namespace LV; 
    constexpr int sap_block_x = SAP_BLOCK_X; 
    constexpr int sap_block_t = SAP_BLOCK_T; 
    constexpr int sap_x_elements = Nx/sap_block_x; //Number of lattice points in the x direction (without the spin index)
    constexpr int sap_t_elements = Nt/sap_block_t; //Number of lattice points in the t direction (without the spin index)
    constexpr int N_sap_blocks = sap_block_x * sap_block_t; //Number of Schwarz blocks
    constexpr int sap_lattice_sites_per_block = sap_x_elements * sap_t_elements; //Number of lattice points in the block
    constexpr int sap_variables_per_block = 2 * sap_lattice_sites_per_block; //Number of variables in the block
    constexpr int sap_coloring_blocks = N_sap_blocks/2; //Number of red or black blocks 
    extern bool schwarz_blocks; //True if the Schwarz blocks are initialized
    extern int sap_gmres_restart_length; //GMRES restart length for the Schwarz blocks.
    extern int sap_gmres_restarts; //GMRES iterations for the Schwarz blocks
    extern double sap_gmres_tolerance; //GMRES tolerance for the Schwarz blocks 
    extern double sap_tolerance; //tolerance for the SAP method
    extern int sap_blocks_per_proc; //Number of blocks per process for the parallel SAP method
}


//--------------Parameters for FGMRES--------------//
namespace FGMRESV {
    extern double fgmres_tolerance; //Tolerance for FGMRES
    extern int fgmres_restart_length; //Restart length for FGMRES
    extern int fgmres_restarts; //Number of restarts for FGMRES
}

namespace CG{
    extern int max_iter;
    extern double tol;
}


//Coordinates vectorization for the lattice points
extern std::vector<std::vector<int>>Coords; 
void Coordinates();


//--Coordinates of the neighbors to avoid recomputing them each time the operator D is called--//
//Check dirac_operator.h for the definition of RightPB and LeftPB
extern std::vector<std::vector<int>>RightPB; //Right periodic boundary
extern std::vector<std::vector<int>>LeftPB; //Left periodic boundary
extern std::vector<std::vector<c_double>>SignR; //Right fermionic boundary
extern std::vector<std::vector<c_double>>SignL; //Left fermionic boundary

extern std::vector<std::vector<c_double>>D_TEMP;

void CheckBlocks(); //Check that Nx/block_x and Nt/block_t are integers, the same for Schwarz blocks


void save_vec(const std::vector<double>& vec,const std::string& name); //save vector to .txt file 
void read_rhs(std::vector<std::vector<c_double>>& vec,const std::string& name);
void save_rhs(std::vector<std::vector<c_double>>& vec,const std::string& name);
void random_rhs(std::vector<std::vector<c_double>>& vec,const int seed);

void print_parameters();

#endif 