#include "gauge_conf.h"


std::complex<double> RandomU1() {
	double cociente = ((double) rand() / (RAND_MAX));
    double theta = 2.0*pi * cociente;
	std::complex<double> z(cos(theta), sin(theta));
	return z;
}

void GaugeConf::initialize() {
	for (int i = 0; i < Ntot; i++) {
		for (int mu = 0; mu < 2; mu++) {
			Conf[i][mu] = RandomU1(); //Conf[Ns x Nt][mu in {0,1}]
		}
	}
}

void GaugeConf::read_conf(const std::string& name){
    std::ifstream infile(name);
    if (!infile) {
        std::cerr << "File " << name << " not found " << std::endl;
        exit(1);
    }
    int x, t, mu;
    double re, im; 
    while (infile >> x >> t >> mu >> re >> im) {
        Conf[Coords[x][t]][mu] = c_double(re, im); 
    }
    infile.close();
    std::cout << "Conf read from " << name << std::endl;
    
}

void GaugeConf::readBinary(const std::string& name){
    using namespace LV;
    std::ifstream infile(name, std::ios::binary);
    if (!infile) {
        std::cerr << "File " << name << " not found " << std::endl;
        exit(1);
    }
    
    for (int x = 0; x < Nx; x++) {
    for (int t = 0; t < Nt; t++) {
        int n = x * Nx + t;
        for (int mu = 0; mu < 2; mu++) {
            int x_read, t_read, mu_read;
            double re, im;
            infile.read(reinterpret_cast<char*>(&x_read), sizeof(int));
            infile.read(reinterpret_cast<char*>(&t_read), sizeof(int));
            infile.read(reinterpret_cast<char*>(&mu_read), sizeof(int));
            infile.read(reinterpret_cast<char*>(&re), sizeof(double));
            infile.read(reinterpret_cast<char*>(&im), sizeof(double));
                Conf[n][mu_read] = c_double(re, im);
           
        }
    }
    }
    infile.close();
      
}

void GaugeConf::Compute_Plaquette01() {
	//U_mv(x) = U_m(x) U_v(x+m) U*_m(x+v) U*_v(x)
	//mu = 0 time direction, mu = 1 space direction
    for (int n = 0; n<Ntot; n++){
        //int Coord0 = Coords[x][t], Coord1 = Coords[x][modulo(t + 1, Nt)], Coord2 = Coords[modulo(x + 1, Ns)][t];
		Plaquette01[n] = Conf[n][0] * Conf[RightPB[n][0]][1] * std::conj(Conf[RightPB[n][1]][0]) * std::conj(Conf[n][1]);
    }		
}


void GaugeConf::savePlaquette(const std::string& name){
    std::ofstream rhsfile(name,std::ios::binary);
    if (!rhsfile.is_open()) {
        std::cerr << "Error opening plaquette file for writing." << std::endl;
    } 
    else {
        int x,t;
        //x, t, mu, real part, imaginary part
        for (int n = 0; n < LV::Ntot; ++n) {
            x = n/LV::Nt;
            t = n%LV::Nt;
            const double& re = std::real(Plaquette01[n]);
            const double& im = std::imag(Plaquette01[n]);
            rhsfile.write(reinterpret_cast<char*>(&x), sizeof(int));
            rhsfile.write(reinterpret_cast<char*>(&t), sizeof(int));
            rhsfile.write(reinterpret_cast<const char*>(&re), sizeof(double));
            rhsfile.write(reinterpret_cast<const char*>(&im), sizeof(double));             
        }
        rhsfile.close();
    }

}