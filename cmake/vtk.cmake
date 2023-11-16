include(FetchContent)

FetchContent_Declare(
        vtk
        GIT_REPOSITORY https://github.com/Kitware/VTK.git
        GIT_TAG v9.1.0
)
FetchContent_MakeAvailable(vtk)