import bpy
import bmesh
import os
import sys
import numpy as np
import shutil
from mathutils import Vector
from ..project_config import PROJECT_CONFIG


def boundary_condition_mesh_callback(scene, context):
  items = [("","empty","")]
  object_list = bpy.context.scene.objects
  for i in object_list:
      if i.type == 'MESH':
        items.append((i.name, i.name, i.name))
  return items
  

class FEMTetrahedralOperator(bpy.types.Operator):
    bl_idname: str = "designauto.quasistatic_simulation_fem_tetrahedral"
    bl_label: str = "FEM仿真"
    bl_options = {"REGISTER", "UNDO"}
    
    # YM: bpy.props.FloatProperty(name="Young's modulus", default=1e5)
    # PR: bpy.props.FloatProperty(name="Poisson's ratio", default=0.3)
    # density: bpy.props.FloatProperty(name="density", default=1e3)
    
    config_path: bpy.props.StringProperty(name="Config Path")
    
    # nbc1: bpy.props.EnumProperty(name="NBC 1", items=boundary_condition_mesh_callback)
    # nbcv1: bpy.props.FloatProperty(name="NBC va1 1", default=0)
    
    # nbc2: bpy.props.EnumProperty(name="NBC 2", items=boundary_condition_mesh_callback)
    # nbcv2: bpy.props.FloatProperty(name="NBC val 2", default=0)
    
    # nbc3: bpy.props.EnumProperty(name="NBC 3", items=boundary_condition_mesh_callback)
    # nbcv3: bpy.props.FloatProperty(name="NBC val 3", default=0)
    
    # dbc1: bpy.props.EnumProperty(name="DBC 1", items=boundary_condition_mesh_callback)
    # dbc2: bpy.props.EnumProperty(name="DBC 2", items=boundary_condition_mesh_callback)
    # dbc3: bpy.props.EnumProperty(name="DBC 3", items=boundary_condition_mesh_callback)
    
    
    def execute(self, context):
        working_object = context.active_object
        working_name = 'fem'
        bpyscene = bpy.context.scene
        workplace = PROJECT_CONFIG.workplace_dir_path
        executable_path = PROJECT_CONFIG.executable_dir_path

        shutil.copyfile(self.config_path, os.path.join(workplace, "assets", "config.json"))
        
        import dapy_qss_fem_tetrahedral
        os.chdir(executable_path)
        surface_mesh, mat_deformed_coordinates, \
            vtx_displacement, vtx_stress = \
                dapy_qss_fem_tetrahedral.QuasistaticSimulationByFEMTetrahedral()
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
        if not new_displacement_mesh.vertex_colors:
            new_displacement_mesh.vertex_colors.new()
        for polygon in new_displacement_mesh.polygons:
            for i, index in enumerate(polygon.vertices):
                loop_index = polygon.loop_indices[i]
                new_displacement_mesh.vertex_colors.active.data[loop_index].color = \
                    (vtx_displacement[index], 0, 0, 1)
        
        new_stress_mesh = bpy.data.meshes.new("mesh-" + working_name + "-stress")
        new_stress_mesh.from_pydata(surface_mesh.mat_coordinates.tolist(), [], surface_mesh.mat_faces.tolist())
        new_stress_mesh.update()
        new_stress_object = bpy.data.objects.new(working_name + "-stress", new_stress_mesh)
        if not new_stress_mesh.vertex_colors:
            new_stress_mesh.vertex_colors.new()
        for polygon in new_stress_mesh.polygons:
            for i, index in enumerate(polygon.vertices):
                loop_index = polygon.loop_indices[i]
                new_stress_mesh.vertex_colors.active.data[loop_index].color = \
                    (vtx_stress[index], 0, 0, 1)
        
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
