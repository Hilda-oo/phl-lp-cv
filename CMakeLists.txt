cmake_minimum_required(VERSION 3.0)

project(designauto)

set(CMAKE_CXX_STANDARD 17)

set(CPPLINT_PATH ${CMAKE_SOURCE_DIR}/3rd/cpplint.py)
set(CPPLINT_COMMAND ${CPPLINT_PATH};
        --linelength=100;
        --filter=-legal/copyright,-build/include_subdir,-readability/alt_tokens
        )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/exe/${CMAKE_BUILD_TYPE})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

function(EnableCppLint target processing_files)
    foreach (processing_file ${processing_files})
        add_custom_command(TARGET ${target} PRE_BUILD
                COMMAND ${CPPLINT_COMMAND} ${processing_file}
                )
    endforeach ()
endfunction()
if (POLICY CMP0057)
cmake_policy(SET CMP0057 NEW)
endif()
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
# find_package(Python REQUIRED COMPONENTS Interpreter Development)
# find_package(PyBind11 REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem system filesystem date_time thread graph program_options)
find_package(TBB REQUIRED)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

set(sys_include_directories
        /usr/local/include/eigen3
        /usr/include/eigen3
        /usr/local/include/
        /usr/include/
        ${Python_INCLUDE_DIRS}
)

set(3rd_include_directories "")

add_subdirectory(${CMAKE_SOURCE_DIR}/3rd)

message(STATUS "3rd_include_directories: " ${3rd_include_directories})

include_directories(
        ${3rd_include_directories}
        ${sys_include_directories}
        ${CMAKE_SOURCE_DIR}/da-cpt
        ${CMAKE_SOURCE_DIR}/da-sha
)

link_directories(
        /usr/local/lib
        /usr/lib
)

add_subdirectory(da-sha)  # Shared Libraries
add_subdirectory(da-cpt)
add_subdirectory(da-ent)  # Entry Programs
add_subdirectory(da-cmd)  # Command Line Workflows
add_subdirectory(da-bld)  # Blender