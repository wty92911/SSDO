add_executable(myssao
        main.cpp
        utils.cpp
        utils.hpp
)
target_link_libraries(myssao PRIVATE assimp::assimp glew_s glm stb glfw imgui glad)
target_include_directories(myssao PRIVATE
    ../../third_party/glew/include/
    ../../third_party/glad/include/
    ../../third_party/learnopengl/
    ${CMAKE_BINARY_DIR}/src/)

target_compile_features(myssao PRIVATE cxx_std_11)
