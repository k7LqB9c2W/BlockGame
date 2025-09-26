cmake_minimum_required(VERSION 3.16)

project(BlockGame VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(blockgame
    src/main.cpp
    src/glad.c
)

target_include_directories(blockgame
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
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

source_group(TREE ${CMAKE_SOURCE_DIR} FILES src/main.cpp src/glad.c)
