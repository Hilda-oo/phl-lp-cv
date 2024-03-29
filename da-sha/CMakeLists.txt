function(AddShaProject project_name)
    file(GLOB_RECURSE source_files ${project_name}/*.h ${project_name}/*.cpp)
    add_library(${project_name} ${source_files})
    #EnableCppLint(${project_name} ${source_files})
    target_link_libraries(${project_name} ${ARGN})
endfunction(AddShaProject)

add_compile_definitions(BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)

AddShaProject(sha-base-framework      ${Boost_LIBRARIES} fmt TBB::tbb)
AddShaProject(sha-entry-framework     sha-base-framework)
AddShaProject(sha-surface-mesh     gmp mpfr)
AddShaProject(sha-volume-mesh         sha-surface-mesh OpenVolumeMesh)
AddShaProject(sha-io-foundation       sha-surface-mesh sha-volume-mesh)
AddShaProject(sha-implicit-modeling)
AddShaProject(sha-hexahedron-generation sha-surface-mesh sha-volume-mesh cpt-l1-norm-polycube)

AddShaProject(sha-voronoi-foundation sha-surface-mesh sha-volume-mesh geogram)
target_include_directories(sha-voronoi-foundation PUBLIC /usr/local/include/geogram1)

AddShaProject(sha-simulation-utils cpt-linear-solver mshio::mshio)
AddShaProject(sha-simulation-3d  cpt-linear-solver  sha-base-framework sha-io-foundation TBB::tbb sha-simulation-utils)
AddShaProject(sha-fem-quasistatic  cpt-linear-solver  sha-base-framework sha-io-foundation TBB::tbb sha-simulation-utils)

AddShaProject(sha-topology-optimization-3d  sha-base-framework sha-volume-mesh sha-simulation-utils mma::mma mma::gcmma cpt-linear-solver sha-io-foundation)
target_compile_definitions(sha-topology-optimization-3d PUBLIC USE_SUITESPARSE)
target_compile_definitions(sha-topology-optimization-3d PUBLIC DEBUG)
# target_compile_definitions(sha-topology-optimization-3d PUBLIC WRITE_TENSOR_IN_LOOP)# for debug

AddShaProject(sha-fem-thermoelastic sha-base-framework sha-io-foundation TBB::tbb amgcl::amgcl sha-simulation-utils)
# chose linear solver
option(MECH_ONLY "Only mech simulate" OFF)
option(ENABLE_AMGCL "Use AMGCL" OFF)
if (ENABLE_AMGCL)
    option(ENABLE_AMGCL_CUDA "use Cuda to speed up AMGCL" ON)
else ()
    option(ENABLE_SUITESPARSE "Use SuiteSparse" ON)
endif ()

if (ENABLE_SUITESPARSE)
    # SuiteSparse
    find_package(SuiteSparse REQUIRED)
    target_compile_definitions(sha-fem-thermoelastic PUBLIC USE_SUITESPARSE)
endif ()

if (ENABLE_AMGCL)
    # AMGCL
    find_package(amgcl REQUIRED)
    target_compile_definitions(sha-fem-thermoelastic PUBLIC USE_AMGCL)
    if (ENABLE_AMGCL_CUDA)
        find_package(CUDA REQUIRED)
        find_package(CUDAToolkit REQUIRED)
        target_compile_definitions(sha-fem-thermoelastic PUBLIC USE_AMGCL_CUDA)
        target_compile_options(sha-fem-thermoelastic PRIVATE -Xcompiler -fopenmp)
    endif ()
endif ()

