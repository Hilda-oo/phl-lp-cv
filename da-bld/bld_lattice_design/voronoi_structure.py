import bpy
import bmesh
import os
import sys
import math
import numpy as np
from mathutils import Vector
from ..project_config import PROJECT_CONFIG


class VoronoiStructureOperator(bpy.types.Operator):
    bl_idname: str = "designauto.lattice_design_voronoi_structure"
    bl_label: str = "三维 Voronoi 晶格设计"
    bl_options = {"REGISTER", "UNDO"}

    lattice_radius: bpy.props.FloatProperty(name="Lattice(inner) Beam Radius", default=1)

    sharp_angle: bpy.props.FloatProperty(
        name="Sharp Angle of Edges", default=45)

    num_seeds: bpy.props.IntProperty(name="Number of voronoi seeds", default=80)

    num_samples: bpy.props.IntProperty(name="Number of samples",
                                       description="Sample numbers along x,y,z axes", default=100)

    def execute(self, context):
        working_object = context.active_object
        bpyscene = bpy.context.scene

        workplace = PROJECT_CONFIG.workplace_dir_path
        executable_path = PROJECT_CONFIG.executable_dir_path
        mesh_path = os.path.join(workplace, "assets", "model.obj")

        bpy.ops.export_scene.obj(filepath=mesh_path, use_selection=True, axis_forward='Y', axis_up='Z',use_edges=False, use_animation=False, use_materials=False, use_uvs=False, use_normals=False,
                                 use_mesh_modifiers=False, use_nurbs=False, use_smooth_groups=False, use_vertex_groups=False, use_blen_objects=False, use_smooth_groups_bitflags=False)
        working_object.hide_set(True)

        import dapy_lad_voronoi_structure
        os.chdir(executable_path)
        lattice_mesh = dapy_lad_voronoi_structure.GenerateVoronoiLatticeStructure(
            self.num_samples, self.num_seeds, self.lattice_radius, self.sharp_angle)

        self.report({"INFO"}, "Command Done.")

        bpy.ops.object.select_all(action='DESELECT')

        new_mesh = bpy.data.meshes.new("mesh-" + working_object.name + "-voronoi-lattice")
        new_mesh.from_pydata(lattice_mesh.mat_coordinates.tolist(), [], lattice_mesh.mat_faces.tolist())
        new_mesh.update()
        new_object = bpy.data.objects.new(working_object.name + "-voronoi-lattice", new_mesh)
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
            if not self.is_repeat():
                box = context.active_object.bound_box
                size = (Vector(box[6]) - Vector(box[0])).length
                self.lattice_radius = size / math.pow(self.num_seeds, 1.0 / 3.0) / 20
            return context.window_manager.invoke_props_dialog(self)
        else:
            self.report({'ERROR'}, "Selected object is not a mesh!")
            return {'CANCELLED'}
