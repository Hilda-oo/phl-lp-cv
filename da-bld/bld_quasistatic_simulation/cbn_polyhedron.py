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
    bl_idname: str = "designauto.quasistatic_simulation_cbn_polyhedron"
    bl_label: str = "CBN仿真"
    bl_options = {"REGISTER", "UNDO"}

    config_file: bpy.props.StringProperty(name="Config Path")
    YM: bpy.props.FloatProperty(name="Young's modulus", default=1e5)
    PR: bpy.props.FloatProperty(name="Poisson's ratio", default=0.3)
    density: bpy.props.FloatProperty(name="density", default=1e3)

    def execute(self, context):
        working_object = context.active_object
        working_name = 'cbn'
        bpyscene = bpy.context.scene
        workplace = PROJECT_CONFIG.workplace_dir_path
        executable_path = PROJECT_CONFIG.executable_dir_path

        import dapy_qss_cbn_polyhedron
        os.chdir(executable_path)
        surface_mesh, mat_deformed_coordinates, \
            vtx_displacement, vtx_stress = \
            dapy_qss_cbn_polyhedron.QuasistaticSimulationByCBN(self.config_file, self.YM, self.PR, self.density)
        self.report({"INFO"}, f"deformed v #{len(mat_deformed_coordinates)}")
        # print(mat_deformed_coordinates)
        self.report({"INFO"}, f"displacement {len(vtx_displacement)}")
        # print(mat_vtx_displacement)
        self.report({"INFO"}, f"stress {len(vtx_stress)}")
        # print(mat_vtx_stress)
        
        new_deformed_mesh = bpy.data.meshes.new("mesh-" + working_name + "-deformed")
        new_deformed_mesh.from_pydata(mat_deformed_coordinates.tolist(), [], surface_mesh.mat_faces.tolist())
        new_deformed_mesh.update()
        new_deformed_object = bpy.data.objects.new(working_name + "-deformed", new_deformed_mesh)
        
        new_displacement_mesh = bpy.data.meshes.new("mesh-" + working_name + "-displacement")
        new_displacement_mesh.from_pydata(surface_mesh.mat_coordinates.tolist(), [], surface_mesh.mat_faces.tolist())
        new_displacement_mesh.update()
        new_displacement_object = bpy.data.objects.new(working_name + "-displacement", new_displacement_mesh)
        vtx_displacement_colors = plt.cm.jet(vtx_displacement)
        if not new_displacement_mesh.vertex_colors:
            new_displacement_mesh.vertex_colors.new()
        for polygon in new_displacement_mesh.polygons:
            for i, index in enumerate(polygon.vertices):
                loop_index = polygon.loop_indices[i]
                new_displacement_mesh.vertex_colors.active.data[loop_index].color = \
                    vtx_displacement_colors[index, :]
        
        new_stress_mesh = bpy.data.meshes.new("mesh-" + working_name + "-stress")
        new_stress_mesh.from_pydata(surface_mesh.mat_coordinates.tolist(), [], surface_mesh.mat_faces.tolist())
        new_stress_mesh.update()
        new_stress_object = bpy.data.objects.new(working_name + "-stress", new_stress_mesh)
        vtx_stress_colors = plt.cm.jet(vtx_stress)
        if not new_stress_mesh.vertex_colors:
            new_stress_mesh.vertex_colors.new()
        for polygon in new_stress_mesh.polygons:
            for i, index in enumerate(polygon.vertices):
                loop_index = polygon.loop_indices[i]
                new_stress_mesh.vertex_colors.active.data[loop_index].color = \
                    vtx_stress_colors[index, :]

        found = False
        for c in bpy.data.collections:
            if c.name == 'designauto':
                found = True
                new_collection = c
        if not found:
            new_collection = bpy.data.collections.new('designauto')
            bpy.context.scene.collection.children.link(new_collection)
        new_collection.objects.link(new_deformed_object)
        new_collection.objects.link(new_displacement_object)
        new_collection.objects.link(new_stress_object)
        
        # working_object.select_set(True)
        return {'FINISHED'}
        
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
        # if context.active_object.type == 'MESH':
            
        #     return context.window_manager.invoke_props_dialog(self)
        # else:
        #     self.report({'ERROR'}, "Selected object is not a mesh!")
        #     return {'CANCELLED'}
