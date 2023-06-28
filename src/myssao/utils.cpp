#include "utils.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <string>

float lerp(float a, float b, float f){
    return a + f * (b - a);
}

std::vector<glm::vec3> get_ssdo_kernel(){
    std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0);
    std::default_random_engine generator;
    std::vector<glm::vec3> ssdoKernel;
    for (unsigned int i = 0; i < 64; ++i)
    {
        glm::vec3 sample(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator));
        sample = glm::normalize(sample);
        sample *= randomFloats(generator);
        float scale = float(i) / 64.0f;
        scale = lerp(0.1f, 1.0f, scale * scale);
        sample *= scale;
        ssdoKernel.push_back(sample);
    }
    return ssdoKernel;
}

std::vector<glm::vec3> get_ssdo_noise(){
    std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0);
    std::default_random_engine generator;
    std::vector<glm::vec3> ssdoNoise;
    for (unsigned int i = 0; i < 16; i++)
    {
        glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f); // rotate around z-axis (in tangent space)
        ssdoNoise.push_back(noise);
    }
    return ssdoNoise;
}

std::map<std::string, std::string> load_config(const std::string& path) {
    std::map<std::string, std::string> config;

    std::ifstream file(path);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            config[key] = value;
        }
    }

    file.close();
    return config;
}