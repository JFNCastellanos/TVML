#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

struct AppConfig {
    double beta = 2;
    double m0 = -0.1868;
    int Nv = 30;
    int number_of_confs = 1000;
    std::string gauge_conf_dir;
    std::string fake_tv_base_dir;
    std::string m_dir;
};

extern AppConfig globalAppConfig;

inline const AppConfig& getAppConfig() {
    return globalAppConfig;
}

inline std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

inline std::string locateConfigFile(const std::string& name) {
    std::ifstream infile(name);
    if (infile) return name;
    infile.close();

    std::string alt = std::string("../") + name;
    infile.open(alt);
    if (infile) return alt;
    infile.close();

    alt = std::string("./") + name;
    infile.open(alt);
    if (infile) return alt;
    infile.close();

    std::cerr << "Config file '" << name << "' not found in the current or parent directory." << std::endl;
    exit(1);
}

inline AppConfig readAppConfig(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Config file " << filename << " not found" << std::endl;
        exit(1);
    }

    AppConfig config;
    std::string line;
    int line_no = 0;
    while (std::getline(infile, line)) {
        line_no++;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;

        const auto sep = line.find('=');
        if (sep == std::string::npos) {
            std::cerr << filename << ":" << line_no << ": invalid line, expected key = value" << std::endl;
            exit(1);
        }

        std::string key = trim(line.substr(0, sep));
        std::string value = trim(line.substr(sep + 1));
        if (key.empty() || value.empty()) {
            std::cerr << filename << ":" << line_no << ": invalid key or value" << std::endl;
            exit(1);
        }

        key = toLower(key);
        if (key == "beta") {
            config.beta = std::stod(value);
        } else if (key == "m0") {
            config.m0 = std::stod(value);
        } else if (key == "nv") {
            config.Nv = std::stoi(value);
        } else if (key == "number_of_confs") {
            config.number_of_confs = std::stoi(value);
        } else if (key == "gauge_conf_dir") {
            config.gauge_conf_dir = value;
        } else if (key == "fake_tv_base_dir") {
            config.fake_tv_base_dir = value;
        } else if (key == "m_dir") {
            config.m_dir = value;
        } else {
            std::cerr << filename << ":" << line_no << ": unknown config key '" << key << "'" << std::endl;
            exit(1);
        }
    }

    if (config.gauge_conf_dir.empty()) {
        std::cerr << filename << ": required key 'gauge_conf_dir' missing" << std::endl;
        exit(1);
    }
    if (config.fake_tv_base_dir.empty()) {
        std::cerr << filename << ": required key 'fake_tv_base_dir' missing" << std::endl;
        exit(1);
    }
    if (config.m_dir.empty()) {
        std::cerr << filename << ": required key 'm_dir' missing" << std::endl;
        exit(1);
    }

    return config;
}

#endif
