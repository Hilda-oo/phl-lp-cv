import bpy
import bmesh
import os
import sys
import numpy as np
from mathutils import Vector
from ..project_config import PROJECT_CONFIG

def boundary_condition_mesh_callback(scene, context):
  items = []
  object_list = bpy.context.scene.objects
  for i in object_list:
      if i.type == 'MESH':
        items.append((i.name, i.name, i.name))
  return items

class SurfaceHoleOperator(bpy.types.Operator):
    bl_idname: str = "designauto.texture_3d_surface_hole"
    bl_label: str = "表面打孔纹理"
    bl_options = {"REGISTER", "UNDO"}

    part_mesh: bpy.props.EnumProperty(name="Part", items=boundary_condition_mesh_callback)
    hole_height: bpy.props.FloatProperty(name="Height of hole", default=0.05)
    hole_radius: bpy.props.FloatProperty(name="Radius of hole", default=0.05)
    use_curvature_field: bpy.props.BoolProperty(name="Use curvature field", default=True)
    num_samples: bpy.props.IntProperty(name="Number of samples",
                                       description="Sample numbers along longest axis", default=10)

    def execute(self, context):
        working_object = context.active_object
        bpyscene = bpy.context.scene
        part_object = bpy.context.scene.objects.get(self.part_mesh)

        workplace = PROJECT_CONFIG.workplace_dir_path
        executable_path = PROJECT_CONFIG.executable_dir_path
        tri_mesh_path = os.path.join(workplace, "assets", "model.obj")
        part_mesh_path = os.path.join(workplace, "assets", "part.obj")
        
        bpy.ops.export_scene.obj(filepath=tri_mesh_path, use_selection=True, axis_forward='Y', axis_up='Z', use_edges=False, use_animation=False, use_materials=False, use_uvs=False, use_normals=False,
                                 use_mesh_modifiers=False, use_nurbs=False, use_smooth_groups=False, use_vertex_groups=False, use_blen_objects=False, use_smooth_groups_bitflags=False)
        working_object.select_set(False)
        part_object.select_set(True)
        bpy.ops.export_scene.obj(filepath=part_mesh_path, use_selection=True, axis_forward='Y', axis_up='Z', use_edges=False, use_animation=False, use_materials=False, use_uvs=False, use_normals=False,
                                 use_mesh_modifiers=False, use_nurbs=False, use_smooth_groups=False, use_vertex_groups=False, use_blen_objects=False, use_smooth_groups_bitflags=False)
        working_object.select_set(True)
        part_object.select_set(False)
        
        import dapy_t3d_surface_hole
        os.chdir(executable_path)
        mesh_with_hole = dapy_t3d_surface_hole.GenerateSurfaceHoles(self.num_samples, self.hole_height, self.hole_radius, self.use_curvature_field)
        # print(lattice_mesh_beams)
        new_mesh = bpy.data.meshes.new("mesh-" + working_object.name + "-hole")
        new_mesh.from_pydata(mesh_with_hole.mat_coordinates.tolist(), [], mesh_with_hole.mat_faces.tolist())
        new_mesh.update()
        new_object = bpy.data.objects.new(working_object.name + "-hole", new_mesh)
        found = False
        for c in bpy.data.collections:
            if c.name == 'designauto':
                found = True
                new_collection = c
        if not found:
            new_collection = bpy.data.collections.new('designauto')
            bpy.context.scene.collection.children.link(new_collection)
        new_collection.objects.link(new_object)
        
        working_object.select_set(True)
        return {'FINISHED'}

    def invoke(self, context, event):
        if context.active_object.type == 'MESH':
            return context.window_manager.invoke_props_dialog(self)
        else:
            self.report({'ERROR'}, "Selected object is not a mesh!")
            return {'CANCELLED'}
