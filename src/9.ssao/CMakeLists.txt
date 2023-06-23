add_executable(ssao
        ssao.cpp
)
target_link_libraries(ssao PRIVATE assimp::assimp glew_s glm stb glfw imgui glad)
target_include_directories(ssao PRIVATE
    ../../third_party/glew/include/
    ../../third_party/glad/include/
    ../../third_party/learnopengl/
    ${CMAKE_BINARY_DIR}/src/)

target_compile_features(ssao PRIVATE cxx_std_11)
