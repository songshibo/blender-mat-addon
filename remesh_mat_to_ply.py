import bpy
import time

print(bpy.app.version_string)
bpy.ops.import_mesh.mat(filepath="E:/PROJECTS/blender-mat-addon/bug.ma")
objects = []

for obj in bpy.context.scene.objects:
    obj.select_set(obj.type == "MESH")
    objects.append(obj)

c = {}
c["object"] = c["active_object"] = bpy.context.object
c["selected_objects"] = c["selected_editable_objects"] = objects
bpy.ops.object.join(c)

start_time = time.time()
bpy.ops.object.modifier_add(type='REMESH')
bpy.context.object.modifiers['Remesh'].mode='VOXEL'
bpy.context.object.modifiers['Remesh'].voxel_size=0.01
bpy.ops.object.modifier_apply(modifier='Remesh')
print("--- %.2f seconds ---" % (time.time() - start_time))

bpy.ops.export_mesh.ply(filepath="./remeshed.ply", use_selection=True)
bpy.ops.object.delete()