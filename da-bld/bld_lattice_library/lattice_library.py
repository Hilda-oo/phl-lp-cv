import bpy
import os
import json

supported_family = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

class LatticeLibraryOperator(bpy.types.Operator):

    bl_idname: str = "designauto.lattice_library"
    bl_label: str = "晶格库"
    bl_options = {"REGISTER", "UNDO"}

    family_items = [("-1","全部晶格","")]
    for family in supported_family :
        family_items.append((str(family),"族" + str(family),""))

    lattice_family: bpy.props.EnumProperty(name = "lattice_family", items=family_items)
    lattice_type : bpy.props.StringProperty(name = "lattice")

    start_id = 1
    end_id = 61
    out_type = -1

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def execute(self, context):

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.ui_units_x = 40
        row = layout.row()
        # row.prop(self,"lattice_type",text="Type")
        row.prop(self,"lattice_family",text="相连性类别")
        button = row.operator("designauto.search_lattice",text="查询晶格")
        button.lattice_family = self.lattice_family # string
        
        row = layout.row()
        button = row.operator("designauto.edit_lattice_para",text="修改生成晶格参数")

        box =layout.box()
        box.ui_units_y = 20
        row = box.row()


        #从数据库中取的晶格列表
        lattices_id_list=self.getFromJson(self.start_id,self.end_id,self.out_type)
        #i记录一行中的按钮数
        i = 0 
        for lattices_id in lattices_id_list:
            # 取到晶格名称
            name = "Type " + str(lattices_id)
            # 在界面上显示按钮
            button = row.operator("designauto.load_lattice_voxel",text = name)
            # 传参
            button.lattice_name = name
            button.lattice_id = lattices_id
            #五个一行，超过换行
            i = i + 1
            if i % 5 == 0 :
                col =box.column()
                row = col.row()

        row = box.row()
        row = box.row()
        row.operator("designauto.pre_page",text="上一页")
        row.operator("designauto.nex_page",text="下一页")

    def getFromJson(self,id_start,id_end,out_type):
        lattices_id_list = []
        with open(self.cur_dir+'/line3_valid.json', encoding='utf-8') as f:
            all_lattices = json.load(f)
            for lattice in all_lattices :
                if(out_type == -1):
                    lattices_id_list.append(lattice.get("id"))
                elif(lattice.get("out_type") == str(out_type)):
                    lattices_id_list.append(lattice.get("id"))
        return lattices_id_list[id_start-1 :id_end-1]

class PrePageOperator(bpy.types.Operator):
    bl_idname: str = "designauto.pre_page"
    bl_label: str = "上一页"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        LatticeLibraryOperator.start_id -= 60
        LatticeLibraryOperator.end_id -= 60
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(self)

class NexPageOperator(bpy.types.Operator):
    bl_idname: str = "designauto.nex_page"
    bl_label: str = "下一页"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        LatticeLibraryOperator.start_id += 60
        LatticeLibraryOperator.end_id += 60
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(self)

class EditLatticeParaOperator(bpy.types.Operator):
    bl_idname: str = "designauto.edit_lattice_para"
    bl_label: str = "修改生成晶格参数"
    bl_options = {"REGISTER", "UNDO"}

    radius : bpy.props.FloatProperty(name="杆半径",default=0.02,min=0.01,max=0.09,step=0.01)
    offset : bpy.props.FloatProperty(name="偏移参数",default=0.5,min=0.1,max=0.9,step=0.1)
    sample : bpy.props.IntProperty(name="边采样数",default=30,min=16,max=40,step=1)

    def execute(self, context):
        path = LatticeLibraryOperator.cur_dir + "/user_setting.txt"
        with open(path,'w') as f:
            f.truncate()
            round(self.radius,2)
            radius_lines = str(round(self.radius,2))+",lattice_radius;\n"
            offset_lines = str(round(self.offset,2))+",lattice_offset;\n"
            sample_lines = str(self.sample)+",lattice_sample;\n"
            txtlist = [radius_lines,offset_lines,sample_lines]
            f.writelines(txtlist)
        return {'FINISHED'}

    def invoke(self, context, event):

        path = LatticeLibraryOperator.cur_dir + "/user_setting.txt"
        with open(path,'r') as f:
            lines=f.readlines()
            self.radius = float(lines[0].split(",")[0])
            self.offset = float(lines[1].split(",")[0])
            self.sample = int(lines[2].split(",")[0])
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
    
class SearchLatticeOperator(bpy.types.Operator):
    bl_idname: str = "designauto.search_lattice"
    bl_label: str = "搜索晶格"
    bl_options = {"REGISTER", "UNDO"}


    lattice_family : bpy.props.StringProperty(name = 'family')

    def execute(self, context):
        LatticeLibraryOperator.start_id = 1
        LatticeLibraryOperator.end_id = 61
        LatticeLibraryOperator.out_type = int(self.lattice_family)
        return {'FINISHED'}

    def invoke(self, context, event):

        return self.execute(self)