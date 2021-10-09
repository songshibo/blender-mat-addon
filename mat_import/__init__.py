# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
import time
from . import auto_load
import os
import sys
import re
from time import sleep

from numpy.lib.function_base import select
import bpy
import bmesh
import math
import mathutils
from bpy.props import (
    StringProperty,
    EnumProperty,
    IntProperty,
    BoolProperty,
    FloatProperty,
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

medial_sphere_color = (0.922, 0.250, 0.204, 1.0)
medial_slab_color = (0.204, 0.694, 0.922, 1.0)
medial_cone_color = (0.204, 0.694, 0.922, 1.0)


# ProgressBar class
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self,
                 total,
                 width=40,
                 fmt=DEFAULT,
                 symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
                          r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


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
    ico_subdivide: IntProperty(
        name="subdivide of Ico Sphere",
        default=2,
        min=1,
        max=4,
    )
    init_radius: FloatProperty(
        name="initial diameter",
        default=1.0,
    )
    separate_sphere: BoolProperty(
        name="Separate Sphere",
        default=False,
    )
    individual_primitive: BoolProperty(
        name="Individual Primitive",
        default=False,
    )

    def execute(self, context):
        keywords = self.as_keywords(ignore=(
            'axis_forward',
            'axis_up',
            'filter_glob',
        ))

        load(self, context, keywords['filepath'], self.ico_subdivide,
             self.init_radius, self.separate_sphere, self.individual_primitive)

        context.view_layer.update()

        return {'FINISHED'}


def create_material(name, color, nodes_name='Principled BSDF'):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        # print("Create new material:", name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes[nodes_name]
    if bsdf is not None:
        bsdf.inputs[0].default_value = color
    return material


def assign_material(obj, material):
    if obj.type == 'MESH':
        if len(obj.data.materials) < 1:
            # print(obj.name, "Assign material")
            obj.data.materials.append(material)
        else:
            # print(obj.name, "Override material")
            obj.data.materials[0] = material


def load(operator, context, filepath, ico_subdivide, radius, separate,
         individual_primitive):
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

    if len(radii) == 0:
        print("No medial vertices!")
        return

    # Assemble mesh
    mesh_name = bpy.path.display_name_from_filepath(filepath)
    mesh = bpy.data.meshes.new(name=mesh_name)
    mesh.from_pydata(verts, edges, faces)
    mesh.validate()
    mesh.update()

    scene = context.scene
    layer = context.view_layer

    if individual_primitive:  # each medial slab or medial cone will be a single object
        # medial slab object generation
        if len(faces) == 0:
            print("No Medial Slab Primitives")
        else:
            print("Generating medial slabs:")
            start_time = time.time()
            progress = ProgressBar(len(faces), fmt=ProgressBar.FULL)
            # TODO: Create materials for each part
            # medial_sphere_mat = create_material("sphere_mat", medial_sphere_color)
            # medial_slab_mat = create_material("slab_mat", medial_slab_color)
            # medial_cone_mat = create_material("cone_mat", medial_cone_color)
            for index, f in enumerate(faces):
                progress.current += 1
                progress()
                slab_verts = []
                slab_faces = []
                v1 = np.array(verts[f[0]])
                r1 = radii[f[0]]
                v2 = np.array(verts[f[1]])
                r2 = radii[f[1]]
                v3 = np.array(verts[f[2]])
                r3 = radii[f[2]]
                generate_slab(v1, r1, v2, r2, v3, r3, slab_verts, slab_faces)
                # Conical surface of 3 medial cones
                # for thoses edge are not in edges, add to edges to generate medial cones
                flag_12, flag_23, flag_13 = True, True, True
                for e in edges:
                    if tuple_compare_2d(e, f[:2]) and not flag_12:
                        flag_12 = False
                    if tuple_compare_2d(e, f[1:3]) and not flag_23:
                        flag_23 = False
                    if tuple_compare_2d(e, (f[0], f[2])) and not flag_13:
                        flag_13 = False
                if flag_12:
                    # edges.append(f[:2])
                    generate_conical_surface(v1, r1, v2, r2, 64, slab_verts,
                                             slab_faces)
                if flag_23:
                    # edges.append(f[1:3])
                    generate_conical_surface(v2, r2, v3, r3, 64, slab_verts,
                                             slab_faces)
                if flag_13:
                    # edges.append((f[0], f[2]))
                    generate_conical_surface(v1, r1, v3, r3, 64, slab_verts,
                                             slab_faces)
                # Combine all sub-meshes into one single mesh via bmesh
                bm = bmesh.new()
                # slab mesh
                slab_mesh = bpy.data.meshes.new(name="TMP_Slab")
                slab_mesh.from_pydata(slab_verts, [], slab_faces)
                slab_mesh.validate()
                slab_mesh.update()
                bm.from_mesh(slab_mesh)
                del slab_verts, slab_faces
                # 3 spheres at 3 vertices
                for i in range(3):
                    matrix = mathutils.Matrix.Translation(
                        verts[f[i]]) @ mathutils.Matrix.Scale(radii[f[i]], 4)
                    bmesh.ops.create_icosphere(bm,
                                               subdivisions=ico_subdivide,
                                               diameter=radius,
                                               matrix=matrix)
                slab_mesh = bpy.data.meshes.new(name="Slab:" + str(index))
                bm.to_mesh(slab_mesh)
                bm.free()
                obj_slab = bpy.data.objects.new(slab_mesh.name, slab_mesh)
                scene.collection.objects.link(obj_slab)
                layer.objects.active = obj_slab  # set active
                obj_slab.select_set(True)  # select slab object
                bpy.ops.object.shade_smooth()  # smooth shading
            progress.done()
        # medial cone object generation
        if len(edges) == 0:
            print("No Medial Cone Primitives")
        else:
            print("Generating medial cones:")
            start_time = time.time()
            progress = ProgressBar(len(edges), fmt=ProgressBar.FULL)
            for index, e in enumerate(edges):
                progress.current += 1
                progress()
                cone_verts = []
                cone_faces = []
                v1 = np.array(verts[e[0]])
                r1 = radii[e[0]]
                v2 = np.array(verts[e[1]])
                r2 = radii[e[1]]
                generate_conical_surface(v1, r1, v2, r2, 64, cone_verts,
                                         cone_faces)
                # Combine all sub-meshes into one single mesh via bmesh
                bm = bmesh.new()
                # cone mesh
                cone_mesh = bpy.data.meshes.new(name="TMP_Cone")
                cone_mesh.from_pydata(cone_verts, [], cone_faces)
                cone_mesh.validate()
                cone_mesh.update()
                bm.from_mesh(cone_mesh)
                del cone_verts, cone_faces
                # 2 spheres at 2 vertices
                for i in range(2):
                    matrix = mathutils.Matrix.Translation(
                        verts[e[i]]) @ mathutils.Matrix.Scale(radii[e[i]], 4)
                    bmesh.ops.create_icosphere(bm,
                                               subdivisions=ico_subdivide,
                                               diameter=radius,
                                               matrix=matrix)
                cone_mesh = bpy.data.meshes.new(name="Cone:" + str(index))
                bm.to_mesh(cone_mesh)
                bm.free()
                obj_cone = bpy.data.objects.new(cone_mesh.name, cone_mesh)
                scene.collection.objects.link(obj_cone)
                layer.objects.active = obj_cone  # set active
                obj_cone.select_set(True)  # select slab object
                bpy.ops.object.shade_smooth()  # smooth shading
            progress.done()
    else:  # 3 objects based on materials: SphereGroup, SlabGroup, ConeGroup
        # Genreate medial mesh object
        print("Generating medial spheres:")
        # record time
        start_time = time.time()
        obj_medial_mesh = bpy.data.objects.new(mesh.name, mesh)
        scene.collection.objects.link(obj_medial_mesh)
        # progress bar
        progress = ProgressBar(len(radii), fmt=ProgressBar.FULL)
        # Create medial cone material
        medial_sphere_mat = create_material("sphere_mat", medial_sphere_color)
        if separate:
            # Pack all spheres into a group
            medial_sphere_parent = bpy.data.objects.new(
                mesh.name + ".SphereGroup", None)
            scene.collection.objects.link(medial_sphere_parent)
            # Generete medial sphere at each vertex
            bm = bmesh.new()
            sphere_mesh = bpy.data.meshes.new("UnitMedialSphere")
            # bmesh.ops.create_uvsphere(bm, u_segments=u_segments, v_segments=v_segments, diameter=radius)
            bmesh.ops.create_icosphere(bm,
                                       subdivisions=ico_subdivide,
                                       diameter=radius)
            bm.to_mesh(sphere_mesh)
            bm.free()
            for i in range(len(radii)):
                progress.current += 1
                progress()
                name = "v" + str(i)
                mat = mathutils.Matrix.Translation(
                    verts[i]) @ mathutils.Matrix.Scale(radii[i], 4)
                # create medial sphere object
                obj_medial_sphere = bpy.data.objects.new(name, sphere_mesh)
                scene.collection.objects.link(
                    obj_medial_sphere)  # link to scene
                layer.objects.active = obj_medial_sphere  # set active
                obj_medial_sphere.select_set(True)  # select sphere object
                # transform & scale object
                obj_medial_sphere.matrix_world = mat
                obj_medial_sphere.parent = medial_sphere_parent  # assign to parent object
                assign_material(obj_medial_sphere, medial_sphere_mat)
                bpy.ops.object.shade_smooth()  # set shading to smooth
            progress.done()
        else:
            # Generate a single mesh contains all spheres
            medial_sphere_mesh = bpy.data.meshes.new(
                'CombinedMedialSphereMesh')
            obj_medial_spheres = bpy.data.objects.new(
                mesh.name + ".SphereGroup", medial_sphere_mesh)
            scene.collection.objects.link(obj_medial_spheres)
            layer.objects.active = obj_medial_spheres
            obj_medial_spheres.select_set(True)
            bm = bmesh.new()
            for i in range(len(radii)):
                progress.current += 1
                progress()
                matrix = mathutils.Matrix.Translation(
                    verts[i]) @ mathutils.Matrix.Scale(radii[i], 4)
                # bmesh.ops.create_uvsphere(bm, u_segments=u_segments, v_segments=v_segments, diameter=radius, matrix=matrix)
                bmesh.ops.create_icosphere(bm,
                                           subdivisions=ico_subdivide,
                                           diameter=radius,
                                           matrix=matrix)
            progress.done()
            bm.to_mesh(medial_sphere_mesh)
            bm.free()
            assign_material(obj_medial_spheres, medial_sphere_mat)
            bpy.ops.object.shade_smooth()
            print("--- %.2f seconds ---" % (time.time() - start_time))

        if len(faces) == 0:
            print("No medial slab")
        else:
            print("Generating medial slabs:")
            start_time = time.time()
            progress = ProgressBar(len(faces), fmt=ProgressBar.FULL)
            # Create medial slab material
            medial_slab_mat = create_material("slab_mat", medial_slab_color)
            # Generate medial slab at each face
            slab_verts = []
            slab_faces = []
            for f in faces:
                progress.current += 1
                progress()
                v1 = np.array(verts[f[0]])
                r1 = radii[f[0]]
                v2 = np.array(verts[f[1]])
                r2 = radii[f[1]]
                v3 = np.array(verts[f[2]])
                r3 = radii[f[2]]
                bm = bmesh.new()
                generate_slab(v1, r1, v2, r2, v3, r3, slab_verts, slab_faces)
                # for thoses edge are not in edges, add to edges to generate medial cones
                flag_12, flag_23, flag_13 = True, True, True
                for e in edges:
                    if tuple_compare_2d(e, f[:2]) and not flag_12:
                        flag_12 = False
                    if tuple_compare_2d(e, f[1:3]) and not flag_23:
                        flag_23 = False
                    if tuple_compare_2d(e, (f[0], f[2])) and not flag_13:
                        flag_13 = False
                if flag_12:
                    edges.append(f[:2])
                if flag_23:
                    edges.append(f[1:3])
                if flag_13:
                    edges.append((f[0], f[2]))
            progress.done()
            slab_mesh = bpy.data.meshes.new(name=mesh.name + ".SlabGroup")
            slab_mesh.from_pydata(slab_verts, [], slab_faces)
            slab_mesh.validate()
            slab_mesh.update()
            obj_medial_slab = bpy.data.objects.new(slab_mesh.name, slab_mesh)
            scene.collection.objects.link(obj_medial_slab)
            assign_material(obj_medial_slab, medial_slab_mat)
            del slab_verts, slab_faces
            print("--- %.2f seconds ---" % (time.time() - start_time))

        if len(edges) == 0:
            print("No medial cone")
        else:
            print("Generating medial cones:")
            start_time = time.time()
            progress = ProgressBar(len(edges), fmt=ProgressBar.FULL)
            # Create medial cone material
            medial_cone_mat = create_material("cone_mat", medial_cone_color)
            # Generate medial cone at each edge
            cone_verts = []
            cone_faces = []
            for e in edges:
                progress.current += 1
                progress()
                v1 = np.array(verts[e[0]])
                r1 = radii[e[0]]
                v2 = np.array(verts[e[1]])
                r2 = radii[e[1]]
                generate_conical_surface(v1, r1, v2, r2, 64, cone_verts,
                                         cone_faces)
            progress.done()
            cone_mesh = bpy.data.meshes.new(name=mesh.name + ".ConeGroup")
            cone_mesh.from_pydata(cone_verts, [], cone_faces)
            cone_mesh.validate()
            cone_mesh.update()
            obj_medial_cone = bpy.data.objects.new(cone_mesh.name, cone_mesh)
            scene.collection.objects.link(obj_medial_cone)
            layer.objects.active = obj_medial_cone  # set active
            obj_medial_cone.select_set(True)  # select cone object
            assign_material(obj_medial_cone, medial_cone_mat)
            bpy.ops.object.shade_smooth()  # smooth shading
            del cone_verts, cone_faces
            print("--- %.2f seconds ---" % (time.time() - start_time))


# generate the conical surface for a medial cone
# the vertices & face indices are stored in cone_verts & cone_faces respectively
def generate_conical_surface(v1, r1, v2, r2, resolution, cone_verts,
                             cone_faces):
    c12 = v2 - v1
    phi = compute_angle(r1, r2, c12)

    c12 = normalize(c12)

    # start direction
    start_dir = np.array([0.0, 1.0, 0.0])  # pick randomly
    if abs(np.dot(c12, start_dir)
           ) > 0.999:  # if parallel to c12, then pick a new direction
        start_dir = np.array([0.0, 1.0, 0.0])
    # correct the start_dir (perpendicular to c12)
    start_dir = normalize(np.cross(c12, start_dir))

    start_index = len(cone_verts)
    local_vcount = 0
    mat = rotate_mat(v1, c12, 2 * np.pi / resolution)
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
        start_dir = mat.dot(np.append(start_dir, [0]))[:3]

    for i in range(0, local_vcount - 2, 2):
        cone_faces.append(
            (start_index + i, start_index + i + 3, start_index + i + 1))
        cone_faces.append(
            (start_index + i, start_index + i + 2, start_index + i + 3))
    # close conical surface
    cone_faces.append((start_index + local_vcount - 2, start_index + 1,
                       start_index + local_vcount - 1))
    cone_faces.append(
        (start_index + local_vcount - 2, start_index, start_index + 1))


# generate two triangle slabs for a medial slab
def generate_slab(v1, r1, v2, r2, v3, r3, slab_verts, slab_faces):
    n = normalize(np.cross(v1 - v3, v1 - v2))
    # v1-v2,v1-v3, tangent point on {v1,r1}
    tangent_p1 = intersect_point_of_cones(v1, r1, v2, r2, v3, r3, n)
    # v2-v1,v2-v3, tangent point on {v2,r2}
    tangent_p2 = intersect_point_of_cones(v2, r2, v1, r1, v3, r3, n)
    # v3-v1,v3-v2, tangent point on {v3,r3}
    tangent_p3 = intersect_point_of_cones(v3, r3, v1, r1, v2, r2, n)
    # first triangle face
    slab_verts.append(tuple(tangent_p1[0]))
    slab_verts.append(tuple(tangent_p2[0]))
    slab_verts.append(tuple(tangent_p3[0]))
    # second triangle face
    slab_verts.append(tuple(tangent_p1[1]))
    slab_verts.append(tuple(tangent_p2[1]))
    slab_verts.append(tuple(tangent_p3[1]))
    # current vertex array length
    vcount = len(slab_verts)
    slab_faces.append((vcount - 1, vcount - 3, vcount - 2))
    slab_faces.append((vcount - 4, vcount - 5, vcount - 6))


# compute the intersect points of medial cone:{v1,v2} and medial cone:{v1,v3} on sphere (v1,r1)
def intersect_point_of_cones(v1, r1, v2, r2, v3, r3, norm):
    v12 = v2 - v1
    phi_12 = compute_angle(r1, r2, v12)

    v13 = v3 - v1
    phi_13 = compute_angle(r1, r3, v13)

    p12 = v1 + normalize(v12) * np.cos(phi_12) * r1
    p13 = v1 + normalize(v13) * np.cos(phi_13) * r1
    mat = rotate_mat(p12, v12, np.pi / 2)
    dir_12 = mat.dot(np.append(norm, [0]))[:3]
    intersect_p = plane_line_intersection(v13, p13, dir_12, p12)

    v1p = intersect_p - v1
    scaled_n = np.sqrt(max(r1 * r1 - np.dot(v1p, v1p), 1e-5)) * norm
    return [intersect_p + scaled_n, intersect_p - scaled_n]


# ray-plane intersection point
# plane {normal: n, point on plane: p}
# ray {direction: d, start point: a}
def plane_line_intersection(n, p, d, a):
    vpt = d[0] * n[0] + d[1] * n[1] + d[2] * n[2]

    if vpt == 0:
        print("SLAB GENERATION::Parallel Error")
        return np.zeros(3)

    t = ((p[0] - a[0]) * n[0] + (p[1] - a[1]) * n[1] +
         (p[2] - a[2]) * n[2]) / vpt
    return a + d * t


# normalize vector
def normalize(v):
    return v / np.sqrt((v**2).sum())


# compare two 2d tuples(does not require same order)
def tuple_compare_2d(a, b):
    return a == b or (a[0] == b[1] and a[1] == b[0])


# compute angle between two medial sphere from a medial cone
def compute_angle(r1, r2, c21):
    r21_2 = max(pow(r1 - r2, 2), 1e-5)
    phi = np.arctan(np.sqrt(max(np.dot(c21, c21) - r21_2, 0) / r21_2))
    phi = np.pi - phi if r1 < r2 else phi
    return phi


# create a matrix representing rotation around vector at point
def rotate_mat(point, vector, angle):
    u, v, w = vector[0], vector[1], vector[2]
    a, b, c = point[0], point[1], point[2]
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[
        u * u + (v * v + w * w) * cos,
        u * v * (1 - cos) - w * sin, u * w * (1 - cos) + v * sin,
        (a * (v * v + w * w) - u *
         (b * v + c * w)) * (1 - cos) + (b * w - c * v) * sin
    ],
                     [
                         u * v * (1 - cos) + w * sin,
                         v * v + (u * u + w * w) * cos,
                         v * w * (1 - cos) - u * sin,
                         (b * (u * u + w * w) - v *
                          (a * u + c * w)) * (1 - cos) + (c * u - a * w) * sin
                     ],
                     [
                         u * w * (1 - cos) - v * sin,
                         v * w * (1 - cos) + u * sin,
                         w * w + (u * u + v * v) * cos,
                         (c * (u * u + v * v) - w *
                          (a * u + b * v)) * (1 - cos) + (a * v - b * u) * sin
                     ], [0, 0, 0, 1]])


def menu_func_import(self, context):
    self.layout.operator(ImportMAT.bl_idname, text="MAT Mesh (.ma)")


auto_load.init()

classes = (ImportMAT, )


def register():
    auto_load.register()
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    auto_load.unregister()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
