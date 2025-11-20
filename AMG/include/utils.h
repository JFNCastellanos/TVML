#ifndef UTILS_H
#define UTILS_H
 //Utility functions
 #include <vector>
 #include "variables.h"
 #include "lin_alg_op.h"

 inline void readBinaryTv(const std::string& name,std::vector<spinor>& test_vectors,const int tv,const int level){
    /*
    We read a file with a test vectors for level l
    */
    std::ifstream infile(name, std::ios::binary);
    if (!infile) {
        std::cerr << "File " << name << " not found " << std::endl;
        exit(1);
    }
    
    int n;
    int x, t, mu, c;
    c = 1;
    double re, im;
	for (int i = 0; i < LevelV::Nsites[level] * 2 * LevelV::Colors[level]; i++) {
        infile.read(reinterpret_cast<char*>(&x), sizeof(int));
        infile.read(reinterpret_cast<char*>(&t), sizeof(int));
        infile.read(reinterpret_cast<char*>(&mu), sizeof(int)); //In case that I include the color I have to modify this
        infile.read(reinterpret_cast<char*>(&re), sizeof(double));
        infile.read(reinterpret_cast<char*>(&im), sizeof(double));
		test_vectors[tv][x * LevelV::NtSites[level] + t][mu] = c_double(re, im);
    
	}
    
    infile.close();
}
      
inline void checkTv(std::vector<spinor>& test_vectors,const int level){
    //Check that test vectors here coincide with what I generated in Python
    int n;
    for (int tv = 0; tv < LevelV::Ntest[level]; tv++) {
    //for (int tv = 0; tv < 1; tv++) {
		for (int x = 0; x < LevelV::NxSites[level]; x++) {
        for (int t = 0; t < LevelV::NtSites[level]; t++) {
            n = x * LevelV::NtSites[level] + t;
		    for (int dof = 0; dof < LevelV::DOF[level]; dof++) {
                std::cout << "tv " << tv << " x " << x << " t " << t << " dof " << dof << " value " << test_vectors[tv][n][dof] << std::endl;
		    }
		}
	    }
    }
}


inline void readConfsID(std::vector<int>& confsID,std::string name){
    std::ifstream infile(name);
    if (!infile) {
        std::cerr << "File " << name << " not found " << std::endl;
        exit(1);
    }
    int ID;
    while (infile >> ID ) {
        confsID.push_back(ID); 
    }
    infile.close();
}



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



#endif