import bpy
import bmesh
import os
import sys
import numpy as np
from mathutils import Vector
from ..project_config import PROJECT_CONFIG


supported_tpms_types = ["G", "G_rec", "Schwarzp", "DoubleP", "Schwarzd", "DoubleD"]
supported_field_types = ["Matrix", "F1", "F2", "F3", "NoField"]

class FieldDrivenTPMSStructureOperator(bpy.types.Operator):
    bl_idname: str = "designauto.lattice_design_field_driven_tpms"
    bl_label: str = "场引导TPMS晶格设计"
    bl_options = {"REGISTER", "UNDO"}

    tpms_type: bpy.props.EnumProperty(
        items=[(f"{type}", f"{type}", f"{type}") for type in supported_tpms_types])

    tpms_coeff: bpy.props.FloatProperty(name="Coeffient for TPMS", default=12)

    offset: bpy.props.FloatProperty(name="Offset", default=0)

    field_type: bpy.props.EnumProperty(
        items=[(f"{type}", f"{type}", f"{type}") for type in supported_field_types])

    field_coeff: bpy.props.FloatProperty(name="Coeffient for Field", default=0.1)   
    
    num_samples: bpy.props.IntProperty(name="Number of samples",
                                       description="Sample numbers along x,y,z axes", default=100)
    last_num_samples = None
    last_object = None
    def execute(self, context):
        working_object = context.active_object
        bpyscene = bpy.context.scene
        
        if self.is_repeat():
            return {'FINISHED'}
        
        workplace = PROJECT_CONFIG.workplace_dir_path
        executable_path = PROJECT_CONFIG.executable_dir_path
        mesh_path = os.path.join(workplace, "assets", "model.obj")

        bpy.ops.export_scene.obj(filepath=mesh_path, use_selection=True,axis_forward='Y', axis_up='Z', use_edges=False, use_animation=False, use_materials=False, use_uvs=False, use_normals=False,
                                 use_mesh_modifiers=False, use_nurbs=False, use_smooth_groups=False, use_vertex_groups=False, use_blen_objects=False, use_smooth_groups_bitflags=False)
        working_object.hide_set(True)
        import dapy_lad_field_driven_tpms
        os.chdir(executable_path)
        lattice_mesh = dapy_lad_field_driven_tpms.GenerateTPMSLatticeStructureByField(
            self.num_samples, self.tpms_type, self.tpms_coeff, self.offset, 
            self.field_type, self.field_coeff, working_object.name == self.last_object and self.last_num_samples == int(self.num_samples))
        self.report({"INFO"}, f"{self.last_num_samples}, {int(self.num_samples)}")
        self.report({"INFO"}, f"{self.last_object}, {working_object.name_full}")
        
        self.last_num_samples = int(self.num_samples)
        self.last_object = working_object.name
        self.report({"INFO"}, "Command Done.")

        bpy.ops.object.select_all(action='DESELECT')

        # print(lattice_mesh_beams)
        new_mesh = bpy.data.meshes.new("mesh-" + working_object.name + "-tpms-lattice")
        new_mesh.from_pydata(lattice_mesh.mat_coordinates.tolist(), [], lattice_mesh.mat_faces.tolist())
        new_mesh.update()
        new_object = bpy.data.objects.new(working_object.name + "-field-driven-tpms-lattice", new_mesh)
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
