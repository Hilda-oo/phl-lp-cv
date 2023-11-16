import numpy as np
import bpy
import os
from .p2voxel import *

class LoadLatticeVoxelOperator(bpy.types.Operator):
    bl_idname: str = "designauto.load_lattice_voxel"
    bl_label: str = "加载晶格体素"
    bl_options = {"REGISTER", "UNDO"}

    lattice_name:bpy.props.StringProperty(name = "lattice")
    lattice_id : bpy.props.IntProperty(name = "id")
    lattice_volumn : bpy.props.IntProperty(name = "体积")
    lattice_volumn_ratio : bpy.props.FloatProperty(name="体积比",precision=7)
    radius : bpy.props.FloatProperty(name="杆半径",default=0.02,min=0.01,max=0.09,step=0.01)
    offset : bpy.props.FloatProperty(name="偏移参数",default=0.5,min=0.1,max=0.9,step=0.1)
    sample : bpy.props.IntProperty(name="边采样数",default=30,min=10,max=60,step=1)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    voxel_array = np.load(cur_dir+'/mircostruct.npy')
    def execute(self, context):
        print(self.lattice_name)
        volumn = 0
        sample = self.voxel_array.shape[0]
        for x in range(0,sample):
            for y in range(0,sample):
                for z in range(0,sample):
                    hasCube = self.voxel_array[x][y][z]
                    if hasCube == 1:
                        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=True, align='WORLD', location=(x, y, z), scale=(1, 1, 1))
                        volumn = volumn + 1
        self.lattice_volumn = volumn
        self.lattice_volumn_ratio = volumn / 64000.000
        bpy.ops.object.editmode_toggle()
        return {'FINISHED'}

    def invoke(self, context, event):
        path = self.cur_dir +"/user_setting.txt"
        with open(path,'r') as f:
            lines=f.readlines()
            self.radius = float(lines[0].split(",")[0])
            self.offset = float(lines[1].split(",")[0])
            self.sample = int(lines[2].split(",")[0])
        main_f(self.lattice_id - 1,self.radius,self.offset,self.sample)
        self.voxel_array = np.load(self.cur_dir + '/mircostruct.npy')

        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self,"lattice_volumn")
        col.prop(self,"lattice_volumn_ratio")
        col.prop(self,"radius")
        col.prop(self,"offset")
        col.prop(self,"sample")