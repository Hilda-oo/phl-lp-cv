set(3rd_include_directories "")

function(Include3rd 3rd_include_directory)
  set(3rd_include_directories
    ${3rd_include_directories}
    ${3rd_include_directory}
    PARENT_SCOPE
    )
  include_directories(
    ${3rd_include_directories}
    ${sys_include_directories}
  )
endfunction(Include3rd)

function(Add3rdProject project_path)
  add_subdirectory(${project_path})
  foreach(3rd_include_directory ${ARGN})
    Include3rd(${3rd_include_directory})
  endforeach()
  set(3rd_include_directories ${3rd_include_directories} PARENT_SCOPE)

endfunction(Add3rdProject)

function(Add3rdHeaderOnlyProject)
  foreach(3rd_include_directory ${ARGN})
    Include3rd(${3rd_include_directory})
  endforeach()
  set(3rd_include_directories ${3rd_include_directories} PARENT_SCOPE)
endfunction(Add3rdHeaderOnlyProject)


Add3rdHeaderOnlyProject(${CMAKE_CURRENT_SOURCE_DIR}/CGAL/include)
Add3rdProject(${CMAKE_CURRENT_SOURCE_DIR}/tetgen   
                ${CMAKE_CURRENT_SOURCE_DIR}/tetgen)
Add3rdHeaderOnlyProject(${CMAKE_CURRENT_SOURCE_DIR}/libigl/include)
Add3rdProject(${CMAKE_CURRENT_SOURCE_DIR}/OpenVolumeMesh   
                ${CMAKE_CURRENT_SOURCE_DIR}/OpenVolumeMesh/src
                ${CMAKE_CURRENT_BINARY_DIR}/OpenVolumeMesh/src)
Add3rdProject(${CMAKE_CURRENT_SOURCE_DIR}/mma
                ${CMAKE_CURRENT_SOURCE_DIR}/mma/src)
Add3rdProject(${CMAKE_CURRENT_SOURCE_DIR}/LpCVT
                ${CMAKE_CURRENT_SOURCE_DIR}/LpCVT)
Add3rdHeaderOnlyProject(${CMAKE_CURRENT_SOURCE_DIR}/nlohmann)
Add3rdProject(${CMAKE_CURRENT_SOURCE_DIR}/MshIO
                ${CMAKE_CURRENT_SOURCE_DIR}/MshIO/include
)
Add3rdProject(${CMAKE_CURRENT_SOURCE_DIR}/nested_cages
                ${CMAKE_CURRENT_SOURCE_DIR}/nested_cages/include
)

set(3rd_include_directories "${3rd_include_directories}" PARENT_SCOPE)
