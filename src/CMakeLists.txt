add_subdirectory(SSDO_PBR)
add_subdirectory(9.ssao)
# add_subdirectory(SSDO_PBR_DepthPeeling)
configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)