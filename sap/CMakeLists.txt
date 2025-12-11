cmake_minimum_required(VERSION 3.26.5)
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED True)
project(SAP VERSION 1.0)

#Source files
set(SRC_FILES 
    src/variables.cpp           #Relevant parameters for the method
    src/gauge_conf.cpp          #Gauge configuration class
    src/dirac_operator.cpp      #Implementation of the Dirac operator
    src/bi_cgstab.cpp           #Bi-CGStab solver for comparison
    src/conjugate_gradient.cpp  #Conjugate gradient for solving normal equations
    src/sap.cpp                 #Schwarz alternating method
    src/main.cpp                #Main program
    src/fgmres.cpp              #Flexible GMRES solver
    src/tests.cpp
)

#--------------------------Lattice dimensions---------------------------#
set(NS "32") #Space
set(NT "32") #Time
#-----------------------------------------------------------------------#


#-------------Lattice blocks for Schwarz alternating method-------------# 
set(SAP_BLOCK_X "4")
set(SAP_BLOCK_T "4")
#-----------------------------------------------------------------------#

#Number of test vectors
set(NTEST "20") 
#The coarse grid matrix dimension is NTEST * 2 * BLOCK_X * BLOCK_T

configure_file(${CMAKE_SOURCE_DIR}/include/config.h.in ${CMAKE_SOURCE_DIR}/include/config.h)

add_executable("SAP_${NS}x${NT}" ${SRC_FILES})

target_include_directories("SAP_${NS}x${NT}"
        PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options("SAP_${NS}x${NT}" PRIVATE
            #-Wall
			-O3
            #-g
			#-fopenmp
)

find_package(MPI REQUIRED)

target_link_libraries("SAP_${NS}x${NT}" PUBLIC MPI::MPI_CXX)


#---------- For computing plaquettes -----------#
set(SOURCES_PLAQ
    src/variables.cpp
    src/gauge_conf.cpp 
    src/dirac_operator.cpp 
    src/compute_plaquette.cpp
)

add_executable("Plaq_${NS}x${NT}" ${SOURCES_PLAQ})

target_include_directories("Plaq_${NS}x${NT}"
        PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options("Plaq_${NS}x${NT}" PRIVATE
			-O3
)


#For windows with MinGW, use the following command to generate the makefile:
#cmake -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER="C:/msys64/ucrt64/bin/g++.exe" -DCMAKE_C_COMPILER="C:/msys64/ucrt64/bin/gcc.exe" ../


