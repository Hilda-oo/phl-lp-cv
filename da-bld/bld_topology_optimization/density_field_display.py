import bpy
import bmesh
import os
import sys
import numpy as np
import shutil
from mathutils import Vector
import matplotlib.pyplot as plt
from ..project_config import PROJECT_CONFIG


class CBNPolyhedronOperator(bpy.types.Operator):
    bl_idname: str = "designauto.top_density_field_display"
    bl_label: str = "密度场展示"
    bl_options = {"REGISTER", "UNDO"}

    mesh_file: bpy.props.StringProperty(name="Mesh Path")
    mesh_property_file: bpy.props.StringProperty(name="Density Path")

    def execute(self, context):
        working_object = context.active_object
        bpyscene = bpy.context.scene
        workplace = PROJECT_CONFIG.workplace_dir_path
        executable_path = PROJECT_CONFIG.executable_dir_path

        import dapy_top_density_field_display
        os.chdir(executable_path)
        surface_mesh, vtx_property = \
            dapy_top_density_field_display.DensityFieldDisplay(self.mesh_file, self.mesh_property_file)

        new_density_mesh = bpy.data.meshes.new("mesh-density")
        new_density_mesh.from_pydata(surface_mesh.mat_coordinates.tolist(), [], surface_mesh.mat_faces.tolist())
        new_density_mesh.update()
        new_density_object = bpy.data.objects.new("density", new_density_mesh)
        vtx_property_colors = plt.cm.jet(vtx_property)
        if not new_density_mesh.vertex_colors:
            new_density_mesh.vertex_colors.new()
        for polygon in new_density_mesh.polygons:
            for i, index in enumerate(polygon.vertices):
                loop_index = polygon.loop_indices[i]
                new_density_mesh.vertex_colors.active.data[loop_index].color = \
                    vtx_property_colors[index, :]
        
        found = False
        for c in bpy.data.collections:
            if c.name == 'designauto':
                found = True
                new_collection = c
        if not found:
            new_collection = bpy.data.collections.new('designauto')
            bpy.context.scene.collection.children.link(new_collection)
        new_collection.objects.link(new_density_object)

        # working_object.select_set(True)
        return {'FINISHED'}
        
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
        # if context.active_object.type == 'MESH':
            
        #     return context.window_manager.invoke_props_dialog(self)
        # else:
        #     self.report({'ERROR'}, "Selected object is not a mesh!")
        #     return {'CANCELLED'}
