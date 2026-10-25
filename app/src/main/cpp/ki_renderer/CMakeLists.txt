# Fork (KI upscaler): builds only libvulkan_renderer.so, mirroring the vulkan_renderer
# target of ../CMakeLists.txt (whose other targets reference sources that are not in the
# repo because those libraries ship prebuilt).
cmake_minimum_required(VERSION 3.22.1)
project(GameNativeKiRenderer C CXX)

set(GN_CPP ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wno-unused-function -Wimplicit-function-declaration")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -s")

add_subdirectory(${GN_CPP}/lsfg-vk-android/thirdparty/dxbc ${CMAKE_CURRENT_BINARY_DIR}/dxbc EXCLUDE_FROM_ALL)
add_subdirectory(${GN_CPP}/extras/adrenotools ${CMAKE_CURRENT_BINARY_DIR}/adrenotools EXCLUDE_FROM_ALL)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/window_vert.h
    COMMAND glslangValidator -V --vn window_vert_code -x
            -o ${CMAKE_CURRENT_BINARY_DIR}/window_vert.h
            ${GN_CPP}/winlator/window.vert
    DEPENDS ${GN_CPP}/winlator/window.vert
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/window_frag.h
    COMMAND glslangValidator -V --vn window_frag_code -x
            -o ${CMAKE_CURRENT_BINARY_DIR}/window_frag.h
            ${GN_CPP}/winlator/window.frag
    DEPENDS ${GN_CPP}/winlator/window.frag
)

add_library(vulkan_renderer SHARED
            ${GN_CPP}/winlator/vulkan_jni.cpp
            ${GN_CPP}/winlator/VulkanRendererContext.cpp
            ${GN_CPP}/winlator/VulkanRendererScanout.cpp
            ${GN_CPP}/winlator/VulkanRendererLsfg.cpp
            ${GN_CPP}/winlator/VulkanRendererKi.cpp
            ${GN_CPP}/lsfg/vk_dispatch.c
            ${GN_CPP}/lsfg/lsfg_dll.c
            ${GN_CPP}/lsfg/lsfg_dxbc.cpp
            ${GN_CPP}/lsfg/lsfg_probe.c
            ${GN_CPP}/lsfg/lsfg_jni.c
            ${GN_CPP}/lsfg/lsfg_common.cpp
            ${GN_CPP}/lsfg/lsfg_shaders.cpp
            ${GN_CPP}/lsfg/lsfg_mipmaps.cpp
            ${GN_CPP}/lsfg/lsfg_generate.cpp
            ${GN_CPP}/lsfg/lsfg_alpha.cpp
            ${GN_CPP}/lsfg/lsfg_beta.cpp
            ${GN_CPP}/lsfg/lsfg_gamma.cpp
            ${GN_CPP}/lsfg/lsfg_delta.cpp
            ${GN_CPP}/lsfg/lsfg_chain.cpp
            ${GN_CPP}/lsfg/lsfg_pacer.cpp
            ${GN_CPP}/lsfg/vkr_lsfg.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/window_vert.h
            ${CMAKE_CURRENT_BINARY_DIR}/window_frag.h)

target_include_directories(vulkan_renderer PRIVATE
                           ${CMAKE_CURRENT_BINARY_DIR}
                           ${GN_CPP}/winlator
                           ${GN_CPP}/lsfg)

set_target_properties(vulkan_renderer PROPERTIES
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED ON)

target_compile_options(vulkan_renderer PRIVATE
                       -Wall -Wextra -fvisibility=hidden)

target_link_libraries(vulkan_renderer
                      log
                      android
                      vulkan
                      adrenotools
                      dxbc
                      dl
                      atomic)

target_link_options(vulkan_renderer PRIVATE -Wl,-z,max-page-size=16384)
