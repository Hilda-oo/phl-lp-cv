function(AddCptProject project_name)
    file(GLOB source_files ${project_name}/*.h
            ${project_name}/*.cpp)
    add_library(${project_name} ${source_files})
    EnableCppLint(${project_name} "${source_files}")
    target_link_libraries(${project_name} ${ARGN})
endfunction(AddCptProject)

find_package(ifopt)
find_package(TBB REQUIRED)
find_package(SuiteSparse REQUIRED)

AddCptProject(cpt-l1-norm-polycube    ifopt::ifopt_ipopt)
AddCptProject(cpt-linear-solver       ${SUITESPARSE_LIBRARIES} TBB::tbb)

# include
target_include_directories(cpt-linear-solver PUBLIC ${SUITESPARSE_INCLUDE_DIRS})
