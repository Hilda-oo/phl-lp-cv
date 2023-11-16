function(AddEntProject project_name)
    file(GLOB_RECURSE source_files ${project_name}/*.h
            ${project_name}/*.cpp)
    add_executable(${project_name} ${source_files})
    # EnableCppLint(${project_name} "${source_files}")
    target_link_libraries(${project_name} ${ARGN} sha-entry-framework)
    target_compile_definitions(${project_name} PRIVATE DA_CMD=1)

    string(REPLACE "-" "_" py_module "${project_name}")
    message("${py_module}")
    pybind11_add_module("dapy_${py_module}" ${source_files})
    target_link_libraries("dapy_${py_module}" PRIVATE ${ARGN} sha-entry-framework)
    target_compile_definitions("dapy_${py_module}" PRIVATE DA_PY=1)

endfunction(AddEntProject)

add_executable(da da.cpp)
target_link_libraries(da ${BOOST_Libraries})
EnableCppLint(da ${CMAKE_CURRENT_SOURCE_DIR}/da.cpp)

add_subdirectory(ent-lattice-design)
add_subdirectory(ent-texture-3d)
add_subdirectory(ent-physical-lattice)
add_subdirectory(ent-quasistatic-simulation)
add_subdirectory(ent-topology-optimaztion)
