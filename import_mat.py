import os
import bpy
import bmesh
import math
import mathutils
from bpy.props import (
    StringProperty,
    EnumProperty,
)
from bpy_extras.io_utils import (ImportHelper)

bl_info = {
    "name": "MAT format",
    "description": "Import-Export medial mesh & interpolated MAT.",
    "author": "Shibo Song",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "File > Import-Export",
    "warning": "",
    "support": "COMMUNITY",
    "category": "Import-Export"
}


# Import MAT class
# Import .ma as medial mesh & medial sphere group & medial cone group & medial slab group
class ImportMAT(bpy.types.Operator, ImportHelper):
    """Load an MAT mesh file"""
    bl_idname = "import_mesh.mat"
    bl_label = "Import MAT mesh"
    file_ext = ".ma"
    #
    filter_glob: StringProperty(
        default="*.ma",
        options={'HIDDEN'},
    )
    axis_forward: EnumProperty(
        name="Forward",
        items=(
            ('X', "X Forward", ""),
            ('Y', "Y Forward", ""),
            ('Z', "Z Forward", ""),
            ('-X', "-X Forward", ""),
            ('-Y', "-Y Forward", ""),
            ('-Z', "-Z Forward", ""),
        ),
        default='Y',
    )
    axis_up: EnumProperty(
        name="Up",
        items=(
            ('X', "X Up", ""),
            ('Y', "Y Up", ""),
            ('Z', "Z Up", ""),
            ('-X', "-X Up", ""),
            ('-Y', "-Y Up", ""),
            ('-Z', "-Z Up", ""),
        ),
        default='Z',
    )

    def execute(self, context):
        keywords = self.as_keywords(ignore=(
            'axis_forward',
            'axis_up',
            'filter_glob',
        ))

        load(self, context, keywords['filepath'])

        context.view_layer.update()

        return {'FINISHED'}


def load(operator, context, filepath):
    filepath = os.fsencode(filepath)
    file = open(filepath, 'r')
    first_line = file.readline().rstrip()
    # number of vertices/edges/faces
    vcount, ecount, fcount = [int(x) for x in first_line.split()]
    lineno = 1

    verts = []
    radii = []
    faces = []
    edges = []
    # read vertices
    i = 0
    while i < vcount:
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue  # skip empty lines or comment lines
        v = line.split()
        assert v[0] == 'v', "vertex line " + str(lineno) + " symbol error!"
        assert len(v) == 5, "line " + str(
            lineno
        ) + " should have 1 symbol & 4d vector as center & radius of sphere!"
        x = float(v[1])
        y = float(v[2])
        z = float(v[3])
        radii.append(float(v[4]))
        verts.append((x, y, z))
        i = i + 1
        lineno = lineno + 1
    # read edges
    i = 0
    while i < ecount:
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert len(ef) == 3 and ef[0] == 'e', "line " + str(
            lineno
        ) + " should have \'e\' symbol & 2d vector as end point of the edge!"

        ids = list(map(int, ef[1:]))
        edges.append(tuple(ids))
        i = i + 1
        lineno = lineno + 1
    # read faces
    i = 0
    while i < fcount:
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert len(ef) == 4 and ef[0] == 'f', "line " + str(
            lineno
        ) + " should have \'f\' symbol & 3d vector as vertices of the face!"

        ids = list(map(int, ef[1:]))
        faces.append(tuple(ids))
        i = i + 1
        lineno = lineno + 1

    # Assemble mesh
    mesh_name = bpy.path.display_name_from_filepath(filepath)
    mesh = bpy.data.meshes.new(name=mesh_name)
    mesh.from_pydata(verts, edges, faces)
    mesh.validate()
    mesh.update()

    scene = context.scene
    layer = context.view_layer

    # Genreate medial mesh object
    obj_medial_mesh = bpy.data.objects.new(mesh.name, mesh)
    scene.collection.objects.link(obj_medial_mesh)

    medial_sphere_parent = bpy.data.objects.new(mesh.name + ".SphereGroup",
                                                None)
    scene.collection.objects.link(medial_sphere_parent)

    # Generete medial sphere at each vertex
    for i in range(len(radii)):
        bm = bmesh.new()
        name = "v" + str(i)
        sphere_mesh = bpy.data.meshes.new(name)
        mat = mathutils.Matrix.Translation(verts[i]) @ mathutils.Matrix.Scale(
            radii[i], 4)
        bmesh.ops.create_uvsphere(bm, u_segments=36, v_segments=16, diameter=1)
        bm.to_mesh(sphere_mesh)
        bm.free()
        # create medial sphere object
        sphere_obj = bpy.data.objects.new(name, sphere_mesh)
        scene.collection.objects.link(sphere_obj)  # link to scene
        layer.objects.active = sphere_obj  # set active
        sphere_obj.select_set(True)  # select sphere object
        # transform & scale object
        sphere_obj.matrix_world = mat
        sphere_obj.parent = medial_sphere_parent  # assign to parent object
        bpy.ops.object.shade_smooth()  # set shading to smooth


def menu_func_import(self, context):
    self.layout.operator(ImportMAT.bl_idname, text="MAT Mesh (.ma)")


classes = (ImportMAT, )


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
