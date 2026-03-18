if (NOT DEFINED BUILD_CONFIG OR NOT BUILD_CONFIG STREQUAL "Release")
    return()
endif()

if (NOT DEFINED SHADER_COMPILER_EXE OR NOT EXISTS "${SHADER_COMPILER_EXE}")
    message(FATAL_ERROR "Missing SHADER_COMPILER_EXE: ${SHADER_COMPILER_EXE}")
endif()

if (NOT DEFINED SHADER_SOURCE_DIR OR NOT EXISTS "${SHADER_SOURCE_DIR}")
    message(FATAL_ERROR "Missing SHADER_SOURCE_DIR: ${SHADER_SOURCE_DIR}")
endif()

if (NOT DEFINED SHADER_OUTPUT_DIR)
    message(FATAL_ERROR "Missing SHADER_OUTPUT_DIR")
endif()

file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

set(shader_precompile_command
    "${SHADER_COMPILER_EXE}"
    "${SHADER_SOURCE_DIR}"
    "${SHADER_OUTPUT_DIR}"
)

if (DEFINED SHADER_DXC_EXE AND NOT SHADER_DXC_EXE STREQUAL "")
    list(APPEND shader_precompile_command "${SHADER_DXC_EXE}")
endif()

execute_process(
    COMMAND ${shader_precompile_command}
    RESULT_VARIABLE shader_compile_result
    COMMAND_ECHO STDOUT
)

if (NOT shader_compile_result EQUAL 0)
    message(FATAL_ERROR "Release shader precompile step failed with code ${shader_compile_result}")
endif()
