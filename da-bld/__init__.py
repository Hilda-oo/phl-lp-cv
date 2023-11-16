bl_info = {
    "name" : "LatticeDesign",
    "author" : "kwp",
    "description" : "晶格设计软件",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "View3D > Sidebar > Edit Tab",
    "warning" : "",
    "category": "Object"
}

import bpy
from .menu import DesignautoMenu

def menu_func(self, context):
    self.layout.menu(DesignautoMenu.bl_idname)

from . import auto_load

auto_load.init()

def register():
    auto_load.register()
    bpy.types.VIEW3D_MT_editor_menus.append(menu_func)

def unregister():
    auto_load.unregister()
    bpy.types.VIEW3D_MT_editor_menus.remove(menu_func)
