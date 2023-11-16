import bpy
from .manifest import MANIFEST

class DesignautoMenu(bpy.types.Menu):
    bl_label = "Design Auto"
    bl_idname = "VIEW3D_MT_DesignautoMenu"

    def draw(self, context):
        layout = self.layout
        for item in MANIFEST:
            if item.operation is not None:
                item.operation(layout)
