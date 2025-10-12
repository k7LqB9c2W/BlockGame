cmake_minimum_required(VERSION 3.20)

project(BlockGame VERSION 0.1.0 LANGUAGES C CXX)

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
    ${SRC_ROOT}/glad.c
    ${SRC_ROOT}/camera.cpp
    ${SRC_ROOT}/chunk_manager.cpp
    ${SRC_ROOT}/input_context.cpp
    ${SRC_ROOT}/renderer.cpp
    ${SRC_ROOT}/TextureLoader.cpp
    ${SRC_ROOT}/text_overlay.cpp
    ${SRC_ROOT}/terrain/biome_database.cpp
    ${SRC_ROOT}/terrain/climate_map.cpp
    ${SRC_ROOT}/terrain/surface_map.cpp
    ${SRC_ROOT}/terrain/terrain_generator.cpp
    ${SRC_ROOT}/terrain/worldgen_profile.cpp
)

add_executable(blockgame ${BLOCKGAME_SOURCES})

target_include_directories(blockgame
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${SRC_ROOT}
)

target_compile_options(blockgame
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /EHsc /Zc:preprocessor>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)

if (WIN32)
    target_link_libraries(blockgame
        PRIVATE
            "${CMAKE_SOURCE_DIR}/libs/glfw3.lib"
            opengl32
            user32
            gdi32
            shell32
            advapi32
    )

    if (EXISTS "${CMAKE_SOURCE_DIR}/libs/glfw3.dll")
        add_custom_command(TARGET blockgame POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/libs/glfw3.dll"
                $<TARGET_FILE_DIR:blockgame>
            COMMENT "Copying GLFW runtime"
        )
    else()
        message(STATUS "glfw3.dll not found; skipping runtime copy.")
    endif()
elseif (APPLE)
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
    find_library(IOKIT_LIBRARY IOKit REQUIRED)
    find_library(COREVIDEO_LIBRARY CoreVideo REQUIRED)
    target_link_libraries(blockgame PRIVATE "-framework OpenGL" ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY})

    find_package(glfw3 REQUIRED)
    target_link_libraries(blockgame PRIVATE glfw)
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(blockgame PRIVATE OpenGL::GL pthread dl)

    find_package(glfw3 REQUIRED)
    target_link_libraries(blockgame PRIVATE glfw)
endif()

source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${BLOCKGAME_SOURCES})
