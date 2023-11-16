cmake_minimum_required(VERSION 3.1)
project(nested_cages)
set(CMAKE_MODULE_PATH /usr/local/lib/cmake ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(${CMAKE_MODULE_PATH})
# libigl + cgal + tetgen
# option(LIBIGL_USE_PREBUILT_LIBRARIES "Use prebuilt libraries"       OFF)
# option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" ${LIBIGL_USE_PREBUILT_LIBRARIES})
# option(LIBIGL_WITH_CGAL              "Use CGAL"                     ON)
# option(LIBIGL_WITH_TETGEN            "Use Tetgen"                   ON)
# find_package(LIBIGL REQUIRED QUIET)

# MeshFix
set(WITH_MESHFIX 1)
# option(WITH_MESHFIX "Use meshfix" ON)
if(WITH_MESHFIX)
  set(MESHFIX_LIBRARY ON CACHE BOOL "Build MeshFix as library" FORCE)
  set(MESHFIX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/MeshFix-v2.1")
  add_subdirectory("${MESHFIX_DIR}")
endif()

# build eltopo library
set(ELTOPO_DIR "${PROJECT_SOURCE_DIR}/eltopo")
file(GLOB ELTOPO_SRCFILES             "${ELTOPO_DIR}/eltopo3d/*.cpp"
                                      "${ELTOPO_DIR}/common/*.cpp"
                                      "${ELTOPO_DIR}/common/newsparse/*.cpp"
                                      "${ELTOPO_DIR}/tunicate/*.cpp")
# Don't build the eltopo viewers that depends on OpenGL and GLUT
list(REMOVE_ITEM ELTOPO_SRCFILES "${ELTOPO_DIR}/common/gluvi.cpp")
list(REMOVE_ITEM ELTOPO_SRCFILES "${ELTOPO_DIR}/eltopo3d/meshrenderer.cpp")
add_library(eltopo ${ELTOPO_SRCFILES})
target_compile_definitions(eltopo PUBLIC -D__LITTLE_ENDIAN__ -DUSE_FORTRAN_BLAS -DNO_GUI -DGRID_100 -DEXTRA_PASSES -DREMOVE_RIZ)
set_property(TARGET eltopo PROPERTY POSITION_INDEPENDENT_CODE ON)
# eltopo uses #include <myfile.h> instead of #include "myfile.h"
target_include_directories(eltopo PUBLIC "${ELTOPO_DIR}/common/" )
target_include_directories(eltopo PUBLIC "${ELTOPO_DIR}/common/tunicate" )
target_include_directories(eltopo PUBLIC "${ELTOPO_DIR}/common/newsparse" )
target_include_directories(eltopo PUBLIC "${ELTOPO_DIR}/eltopo3d/" )
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(eltopo PUBLIC ${LAPACK_LIBRARIES})
target_link_libraries(eltopo PUBLIC ${BLAS_LIBRARIES})


# if(APPLE)
#   set(CMAKE_CXX_LINK_FLAGS "-framework Accelerate")
#   # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
# endif()

# Build collisiondetection library
set(COLLISIONDETECTION_DIR "${PROJECT_SOURCE_DIR}/collisiondetection")
file(GLOB COLLISIONDETECTION_SRCFILES "${COLLISIONDETECTION_DIR}/src/*.cpp")
add_library(collisiondetection ${COLLISIONDETECTION_SRCFILES})
target_include_directories(collisiondetection PUBLIC "${COLLISIONDETECTION_DIR}/include" )
target_include_directories(collisiondetection PRIVATE ${sys_include_directories})

# target_link_libraries(collisiondetection igl::core)

file(GLOB SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# add_executable(${PROJECT_NAME} ${SRCFILES})
add_library(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/include/nested_cages.h ${SRCFILES})

option(VERBOSE_DEBUG "VERBOSE_DEBUG" OFF)
if (VERBOSE_DEBUG)
  target_compile_definitions(${PROJECT_NAME} PUBLIC "VERBOSE_DEBUG")
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(WITH_MESHFIX)
  target_link_libraries(${PROJECT_NAME} MeshFix)
  target_compile_definitions(${PROJECT_NAME} PUBLIC "WITH_MESHFIX")
  # for meshfix_eigen.h
  target_include_directories(${PROJECT_NAME} PUBLIC "${MESHFIX_DIR}/examples")
endif()
target_link_libraries(${PROJECT_NAME} eltopo)
target_link_libraries(${PROJECT_NAME} collisiondetection)
target_link_libraries(${PROJECT_NAME} tet)
target_link_libraries(${PROJECT_NAME} ${GMP_LIBRARIES} ${MPFR_LIBRARIES})