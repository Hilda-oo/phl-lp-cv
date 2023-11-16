import bpy
import bmesh
import os
import sys
import numpy as np
from mathutils import Vector
from ..project_config import PROJECT_CONFIG

supported_types = [0, 2, 3, 8, 12, 14, 22, 27, 501, 511]

class HexahedronStructureOperator(bpy.types.Operator):
    bl_idname: str = "designauto.lattice_design_hexahedron_structure"
    bl_label: str = "六面体晶格设计"
    bl_options = {"REGISTER", "UNDO"}

    alpha: bpy.props.FloatProperty(name="Parameter `alpha` for l1 norm optimizing", default=4)
    beta: bpy.props.FloatProperty(name="Parameter `beta` for l1 norm optimizing", default=1)
    num_iterations: bpy.props.IntProperty(name="Number of iterations of optimizer", default=20)
    
    lattice_type: bpy.props.EnumProperty(
        items=[(f"{type}", f"Type {type}", f" Type{type}") for type in supported_types])

    lattice_radius: bpy.props.FloatProperty(name="Lattice(inner) Beam Radius", default=1)

    tetgen_switches: bpy.props.StringProperty(name="TetGen switches", default="pQq1")
    
    num_xyz_cells: bpy.props.IntVectorProperty(name="Number of cells",
                                               description="Cell numbers along x,y,z axes", default=[4, 4, 4])

    num_samples: bpy.props.IntProperty(name="Number of samples",
                                       description="Sample numbers along x,y,z axes", default=100)

    def execute(self, context):
        working_object = context.active_object
        bpyscene = bpy.context.scene
        
        workplace = PROJECT_CONFIG.workplace_dir_path
        executable_path = PROJECT_CONFIG.executable_dir_path
        mesh_path = os.path.join(workplace, "assets", "model.obj")

        bpy.ops.export_scene.obj(filepath=mesh_path, use_selection=True,axis_forward='Y', axis_up='Z', use_edges=False, use_animation=False, use_materials=False, use_uvs=False, use_normals=False,
                                 use_mesh_modifiers=False, use_nurbs=False, use_smooth_groups=False, use_vertex_groups=False, use_blen_objects=False, use_smooth_groups_bitflags=False)
        working_object.hide_set(True)
        
        import dapy_lad_hexahedron_structure
        os.chdir(executable_path)
        lattice_mesh = dapy_lad_hexahedron_structure.GenerateHexahedronLatticeStructure(
            self.num_samples, self.alpha, self.beta, self.num_iterations, np.array([self.num_xyz_cells[idx] for idx in range(3)], np.int_), 
            self.lattice_radius, int(self.lattice_type), self.tetgen_switches
        )

        self.report({"INFO"}, "Command Done.")

        bpy.ops.object.select_all(action='DESELECT')
        
        new_mesh = bpy.data.meshes.new("mesh-" + working_object.name + "-hex-lattice")
        new_mesh.from_pydata(lattice_mesh.mat_coordinates.tolist(), [], lattice_mesh.mat_faces.tolist())
        new_mesh.update()
        new_object = bpy.data.objects.new(working_object.name + "-hex-lattice", new_mesh)
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
                self.lattice_radius = size / min([self.num_xyz_cells[v] for v in range(3)]) / 20
            return context.window_manager.invoke_props_dialog(self)
        else:
            self.report({'ERROR'}, "Selected object is not a mesh!")
            return {'CANCELLED'}
