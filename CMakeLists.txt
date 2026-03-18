cmake_minimum_required(VERSION 3.20)

if (WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if (DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    elseif (EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    endif()
endif()

project(BlockGame VERSION 0.1.0 LANGUAGES C CXX)

set(BLOCKGAME_DEFAULT_DXC_EXE "D:/dxc_2026_02_20/bin/x64/dxc.exe")
find_program(BLOCKGAME_DXC_FALLBACK_EXE
    NAMES dxc.exe dxc
    HINTS "D:/dxc_2026_02_20/bin/x64"
)
if (NOT DEFINED BLOCKGAME_DXC_EXE OR BLOCKGAME_DXC_EXE STREQUAL "")
    if (EXISTS "${BLOCKGAME_DEFAULT_DXC_EXE}")
        set(BLOCKGAME_DXC_EXE "${BLOCKGAME_DEFAULT_DXC_EXE}" CACHE FILEPATH "Path to dxc.exe used by shader precompilation" FORCE)
    elseif (BLOCKGAME_DXC_FALLBACK_EXE)
        set(BLOCKGAME_DXC_EXE "${BLOCKGAME_DXC_FALLBACK_EXE}" CACHE FILEPATH "Path to dxc.exe used by shader precompilation" FORCE)
    else()
        set(BLOCKGAME_DXC_EXE "dxc.exe" CACHE FILEPATH "Path to dxc.exe used by shader precompilation" FORCE)
    endif()
endif()
message(STATUS "BlockGame shader compiler backend: ${BLOCKGAME_DXC_EXE}")

if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    message(STATUS "Generator is '${CMAKE_GENERATOR}'. For fastest incremental builds consider running: cmake -G Ninja -S . -B build")
endif()

if(NOT CMAKE_CONFIGURATION_TYPES)
    set(DEFAULT_BUILD_TYPE "Release")
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "No build type selected, defaulting to ${DEFAULT_BUILD_TYPE}.")
        set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(SRC_ROOT ${CMAKE_SOURCE_DIR}/src)

set(BLOCKGAME_SOURCES
    ${SRC_ROOT}/main.cpp
    ${SRC_ROOT}/camera.cpp
    ${SRC_ROOT}/chunk_manager.cpp
    ${SRC_ROOT}/input_context.cpp
    ${SRC_ROOT}/renderer.cpp
    ${SRC_ROOT}/terrain/biome_database.cpp
    ${SRC_ROOT}/terrain/climate_map.cpp
    ${SRC_ROOT}/terrain/far_lod_worldgen.cpp
    ${SRC_ROOT}/terrain/surface_map.cpp
    ${SRC_ROOT}/terrain/terrain_generator.cpp
    ${SRC_ROOT}/terrain/worldgen_profile.cpp
    ${CMAKE_SOURCE_DIR}/blockgame.rc
)

add_executable(blockgame ${BLOCKGAME_SOURCES})

add_executable(blockgame_shader_precompiler
    ${CMAKE_SOURCE_DIR}/tools/shader_precompiler.cpp
)

target_include_directories(blockgame
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${SRC_ROOT}
)

target_include_directories(blockgame_shader_precompiler
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${SRC_ROOT}
)

target_compile_definitions(blockgame
    PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
        $<$<CONFIG:Release>:BLOCKGAME_USE_PRECOMPILED_SHADERS=1>
)

target_compile_definitions(blockgame_shader_precompiler
    PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
)

target_compile_options(blockgame
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /EHsc /Zc:preprocessor /FS>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)

target_compile_options(blockgame_shader_precompiler
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /EHsc /Zc:preprocessor /FS>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)

if (MSVC)
    target_compile_options(blockgame
        PRIVATE
            $<$<CONFIG:Release>:/Zi>
            $<$<CONFIG:RelWithDebInfo>:/Zi>
    )

    target_link_options(blockgame
        PRIVATE
            $<$<CONFIG:Release>:/DEBUG:FULL /OPT:REF /OPT:ICF>
            $<$<CONFIG:RelWithDebInfo>:/DEBUG:FULL>
    )

    set_target_properties(blockgame PROPERTIES
        PDB_NAME "blockgame"
        COMPILE_PDB_NAME "blockgame"
        PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
        COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )
endif()

if (NOT WIN32)
    message(FATAL_ERROR "The D3D12 port is Windows-only.")
endif()

if (NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "Vcpkg target triplet")
endif()

find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

target_link_libraries(blockgame
    PRIVATE
        glfw
        imgui::imgui
        d3d12
        dxgi
        dxguid
        Dbghelp
        d3dcompiler
        user32
        gdi32
        shell32
        advapi32
)

target_link_libraries(blockgame_shader_precompiler
    PRIVATE
        d3dcompiler
)

add_dependencies(blockgame blockgame_shader_precompiler)

add_custom_command(TARGET blockgame POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:blockgame>
        $<TARGET_FILE_DIR:blockgame>
    COMMAND_EXPAND_LISTS
    COMMENT "Copying runtime DLLs"
)

if (EXISTS "${CMAKE_SOURCE_DIR}/block_atlas.png")
    add_custom_command(TARGET blockgame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/block_atlas.png"
            $<TARGET_FILE_DIR:blockgame>
        COMMENT "Copying block atlas"
    )
endif()

if (EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(TARGET blockgame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:blockgame>/assets"
        COMMENT "Copying runtime assets"
    )
endif()

add_custom_command(TARGET blockgame POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        -DBUILD_CONFIG=$<CONFIG>
        -DSHADER_COMPILER_EXE=$<TARGET_FILE:blockgame_shader_precompiler>
        -DSHADER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/assets/shaders
        -DSHADER_OUTPUT_DIR=$<TARGET_FILE_DIR:blockgame>/assets/shaders/compiled
        -DSHADER_DXC_EXE=${BLOCKGAME_DXC_EXE}
        -P "${CMAKE_SOURCE_DIR}/cmake/precompile_shaders.cmake"
    COMMENT "Precompiling Release shaders"
)

source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${BLOCKGAME_SOURCES})
