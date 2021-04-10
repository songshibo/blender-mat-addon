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
import numpy as np

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

    # Generate medial cone at each edge
    cone_verts = []
    cone_faces = []
    for e in edges:
        v1 = np.array(verts[e[0]])
        r1 = radii[e[0]]
        v2 = np.array(verts[e[1]])
        r2 = radii[e[1]]
        generate_conical_surface(
            v1, r1, v2, r2, 64, cone_verts, cone_faces)

    cone_mesh = bpy.data.meshes.new(name=mesh.name + ".ConeGroup")
    cone_mesh.from_pydata(cone_verts, [], cone_faces)
    cone_mesh.validate()
    cone_mesh.update()
    obj_medial_cone = bpy.data.objects.new(cone_mesh.name, cone_mesh)
    scene.collection.objects.link(obj_medial_cone)
    layer.objects.active = obj_medial_cone  # set active
    obj_medial_cone.select_set(True)  # select cone object
    bpy.ops.object.shade_smooth()  # smooth shading


# generate the conical surface for a medial cone
# the vertices & face indices are stored in cone_verts & cone_faces respectively
def generate_conical_surface(v1, r1, v2, r2, resolution, cone_verts, cone_faces):
    c12 = v2 - v1
    phi = compute_angle(r1, r2, c12)
    phi = np.pi - phi if r1 < r2 else phi
    c12 = c12 / np.sqrt(np.dot(c12, c12))

    # start direction
    start_dir = np.array([0.0, 1.0, 0.0])  # pick randomly
    if abs(np.dot(c12, start_dir)) > 0.999:  # if parallel to c12, then pick a new direction
        start_dir = np.array([0.0, 1.0, 0.0])
    # correct the start_dir (perpendicular to c12)
    start_dir = np.cross(c12, start_dir)
    start_dir = start_dir / \
        np.sqrt(np.dot(start_dir, start_dir))  # normalization

    start_index = len(cone_verts)
    local_vcount = 0
    for i in range(resolution):
        cos, sin = np.cos(phi), np.sin(phi)
        pos = v1 + (c12 * cos + start_dir * sin) * \
            r1  # vertex on sphere {v1,r1}
        cone_verts.append(tuple(pos))
        pos = v2 + (c12 * cos + start_dir * sin) * \
            r2  # vertex on sphere {v2,r2}
        cone_verts.append(tuple(pos))
        local_vcount = local_vcount + 2
        # rotate
        mat = rotate_mat(v1, c12, 2 * np.pi / resolution)
        start_dir = mat.dot(np.append(start_dir, [0]))[:3]

    for i in range(0, local_vcount-2, 2):
        cone_faces.append(
            (start_index+i, start_index+i+3, start_index+i+1))
        cone_faces.append(
            (start_index+i, start_index+i+2, start_index+i+3))
    # close conical surface
    cone_faces.append((start_index+local_vcount-2,
                       start_index+1, start_index+local_vcount-1))
    cone_faces.append((start_index+local_vcount-2,
                       start_index, start_index+1))


# compute angle between two medial sphere from a medial cone
def compute_angle(r1, r2, c21):
    r21 = abs(r1-r2)
    r21_2 = pow(r21, 2)
    return np.arctan(np.sqrt(np.dot(c21, c21) - r21_2) / r21)


# create a matrix representing rotation around vector at point
def rotate_mat(point, vector, angle):
    u, v, w = vector[0], vector[1], vector[2]
    a, b, c = point[0], point[1], point[2]
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([
        [u * u + (v * v + w * w) * cos, u * v * (1 - cos) - w * sin, u * w * (1 - cos) + v *
         sin, (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - cos) + (b * w - c * v) * sin],
        [u * v * (1 - cos) + w * sin, v * v + (u * u + w * w) * cos, v * w * (1 - cos) - u *
         sin, (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - cos) + (c * u - a * w) * sin],
        [u * w * (1 - cos) - v * sin, v * w * (1 - cos) + u * sin, w * w + (u * u + v * v) *
         cos, (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - cos) + (a * v - b * u) * sin],
        [0, 0, 0, 1]
    ])


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
