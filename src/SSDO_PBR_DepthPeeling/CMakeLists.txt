add_executable(SSDO_PBR_DepthPeeling
        main.cpp
)
target_link_libraries(SSDO_PBR_DepthPeeling PRIVATE assimp::assimp glew_s glm stb glfw imgui glad)
target_include_directories(SSDO_PBR_DepthPeeling PRIVATE
    ../../third_party/glew/include/
    ../../third_party/glad/include/
    ../../third_party/learnopengl/
    ${CMAKE_BINARY_DIR}/src/)

target_compile_features(SSDO_PBR_DepthPeeling PRIVATE cxx_std_11)
