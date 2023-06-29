
#ifndef MYSSAO_UTILS_HPP
#define MYSSAO_UTILS_HPP
#include <config.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <map>

std::vector<glm::vec3> get_ssdo_kernel(int kernel_size);
std::vector<glm::vec3> get_ssdo_noise(int noise_size);
std::map<std::string, std::string> load_config(const std::string& path);

std::vector<std::string> split(const std::string& input, const std::string& delimiter);

#endif  // MYSSAO_UTILS_HPP