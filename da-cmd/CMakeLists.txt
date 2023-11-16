set_property(GLOBAL PROPERTY cmd_projects_property)
set(cmd_frame_file frame.sh)

function(AddCmdProject project_name)
    set(project_path ${CMAKE_CURRENT_SOURCE_DIR}/${project_name})
    if (NOT EXISTS "${project_path}/cmd.sh")
        message(FATAL_ERROR "${project_name}/cmd.sh can not be found.")
    endif ()
    get_property(cmd_projects GLOBAL PROPERTY cmd_projects_property)
    list(APPEND cmd_projects ${project_name})
    set_property(GLOBAL PROPERTY cmd_projects_property "${cmd_projects}")
endfunction(AddCmdProject)

function(SetUpCmdTarget)
    add_custom_target(da-cmd)
    get_property(cmd_project_paths GLOBAL PROPERTY cmd_projects_property)
    foreach (project_name ${cmd_project_paths})
        set(project_path ${CMAKE_CURRENT_SOURCE_DIR}/${project_name})
        add_custom_command(TARGET da-cmd
                COMMAND echo "Generating '${EXECUTABLE_OUTPUT_PATH}/${project_name}'")
        add_custom_command(TARGET da-cmd
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${project_path}/cmd.sh
                ${EXECUTABLE_OUTPUT_PATH}/${project_name})
        add_custom_command(TARGET da-cmd
                COMMAND chmod +x
                ${EXECUTABLE_OUTPUT_PATH}/${project_name})

    endforeach ()
endfunction(SetUpCmdTarget)

AddCmdProject(cmd-lattice-design)
AddCmdProject(cmd-quasistatic-simulation)
AddCmdProject(cmd-topology-optimation)

SetUpCmdTarget()
