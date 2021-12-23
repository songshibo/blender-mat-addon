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
from bpy.props import (
    StringProperty,
    EnumProperty,
    IntProperty,
    BoolProperty,
    FloatProperty,
)
from bpy_extras.io_utils import (ImportHelper)

from .matutil import *
from multiprocessing import Process

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

print("Blender Version:{}".format(bpy.app.version))
bpy_new_version = (3, 0, 0) <= bpy.app.version


def create_icosphere(bm, subdivisions, radius, matrix):
    if bpy_new_version:
        bmesh.ops.create_icosphere(
            bm, subdivisions=subdivisions, radius=radius, matrix=matrix)
    else:
        bmesh.ops.create_icosphere(
            bm, subdivisions=subdivisions, diameter=radius, matrix=matrix)


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
        default="*.ma;*.woff",
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
        name="subdivision of Ico Sphere",
        default=2,
        min=1,
        max=4,
    )
    init_radius: FloatProperty(
        name="initial diameter",
        default=1.0,
    )
    cone_resolution: IntProperty(
        name="resolution of medial cone",
        default=64,
        min=16,
        max=64
    )
    mat_type: EnumProperty(
        name="MAT Type",
        items=(
            ('std', "Standard MAT", ""),
            ('matwild', "MAT with features", ""),
            ('sat', "scale axis WOFF", ""),
        ),
        default="std",
    )
    import_type: EnumProperty(
        name="Import Type",
        items=(
            ('mm', "Only Medial Mesh", ""),
            ('fast', "Sphere-Cone-Slab", ""),
            ('prim', "Individual Primtive", ""),
        ),
        default="fast"
    )
    filter_threshold: FloatProperty(
        name="Threshold for Degenerated Slab",
        default=1e-3,
    )

    def execute(self, context):
        keywords = self.as_keywords(ignore=(
            'axis_forward',
            'axis_up',
            'filter_glob',
        ))

        load(self, context, keywords['filepath'], self.ico_subdivide,
             self.init_radius, self.mat_type, self.import_type, self.cone_resolution, self.filter_threshold)

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


def assemble_mesh(meshname, verts, edges, faces):
    mesh = bpy.data.meshes.new(name=meshname)
    mesh.from_pydata(verts, edges, faces)
    mesh.validate()
    mesh.update()
    return bpy.data.objects.new(mesh.name, mesh)


def generate_medial_mesh(scene, name, verts, radii, faces, edges, *args):
    if len(args) == 0:  # stndard MAT
        # Simply assemble Medial Mesh
        medial_mesh = assemble_mesh(name, verts, edges, faces)
        scene.collection.objects.link(medial_mesh)
    else:  # matwild
        in_edges = args[0]
        out_edges = args[1]
        # Assemble Medial Mesh
        # !firstly create an object with all vertice and type 0 edges
        # !No faces, since face will include edges with other types
        medial_mesh = assemble_mesh(
            name, verts, edges, [])
        # Medial features handling
        bm = bmesh.new()
        bm.from_mesh(medial_mesh.data)
        if hasattr(bm.verts, "ensure_lookup_table"):
            bm.verts.ensure_lookup_table()
        #! Adding inside medial feature edges and mark them as sharp edges
        for e in in_edges:
            ne = bm.edges.new((bm.verts[e[0]], bm.verts[e[1]]))
            ne.smooth = False
        #! Adding outside medial feature edges and mark them as seam edges
        for e in out_edges:
            ne = bm.edges.new((bm.verts[e[0]], bm.verts[e[1]]))
            ne.seam = True
        #! Adding all faces
        for f in faces:
            bm.faces.new((bm.verts[f[0]], bm.verts[f[1]], bm.verts[f[2]]))
        bm.to_mesh(medial_mesh.data)
        bm.free()
        medial_mesh.data.update()
        scene.collection.objects.link(medial_mesh)


def fast_mesh_duplicate_with_TS(bm, mesh, translation, scale):
    matrix = mathutils.Matrix.Translation(
        translation) @ mathutils.Matrix.Scale(scale, 4)
    mesh.transform(matrix)
    bm.from_mesh(mesh)
    # Inversed oder should be T->R->S
    mesh.transform(mathutils.Matrix.Translation(
        tuple(-1.0 * x for x in translation)))
    mesh.transform(mathutils.Matrix.Scale(1.0 / scale, 4))


def load(operator, context, filepath, ico_subdivide, radius, mat_type, import_type, resolution, filter_threshold):
    scene = context.scene
    layer = context.view_layer
    mat_name = bpy.path.display_name_from_filepath(filepath)
    filepath = os.fsencode(filepath)

    ##################################
    ####  Medial Mesh Generation  ####
    ##################################
    if mat_type == 'matwild':
        print("MAT Type: MAT with features")
        # read .ma file with features
        vcount, ecount, fcount, verts, radii, faces, edges, in_edges, out_edges = load_ma_file_with_feature(
            filepath)
        generate_medial_mesh(scene, mat_name, verts, radii,
                             faces, edges, in_edges, out_edges)
        # combined all edges for generation
        edges.extend(in_edges)
        edges.extend(out_edges)
        del in_edges, out_edges
    elif mat_type == 'std':
        print("MAT Type: Standard MAT")
        # read .ma file
        vcount, ecount, fcount, verts, radii, faces, edges = load_ma_file(
            filepath)
        generate_medial_mesh(scene, mat_name, verts, radii, faces, edges)
    elif mat_type == 'sat':
        print("MAT Type: WOFF from scale axis")
        vcount, ecount, fcount, verts, radii, faces, edges = load_woff_file(
            filepath)
    else:
        assert False, "Unknow MAT Type"

    ##################################
    ## Interpolated MAT Generation  ##
    ##################################
    if import_type == 'mm':
        print("Import Type: Only Medial Mesh")
        return
    else:
        # generate materials
        medial_sphere_mat = create_material("sphere_mat", medial_sphere_color)
        medial_slab_mat = create_material("slab_mat", medial_slab_color)
        medial_cone_mat = create_material("cone_mat", medial_cone_color)
        if import_type == 'fast':
            print("Import Type: Sphere/Cone/Slab")
            # Genreate medial mesh object
            print("Generating medial spheres:")
            # record time
            start_time = time.time()
            # progress bar
            progress = ProgressBar(len(radii), fmt=ProgressBar.FULL)
            # Generate a single mesh contains all spheres
            medial_sphere_mesh = bpy.data.meshes.new(
                'CombinedMedialSphereMesh')
            obj_medial_spheres = bpy.data.objects.new(
                mat_name + ".SphereGroup", medial_sphere_mesh)
            scene.collection.objects.link(obj_medial_spheres)
            layer.objects.active = obj_medial_spheres
            obj_medial_spheres.select_set(True)

            bm = bmesh.new()
            create_icosphere(
                bm, subdivisions=ico_subdivide, radius=1.0, matrix=mathutils.Matrix.Identity(4))
            bm.to_mesh(medial_sphere_mesh)
            bm.clear()
            for i in range(len(radii)):
                progress.current += 1
                progress()
                # only generate sphere whose radii > 0
                if radii[i] != 0:
                    fast_mesh_duplicate_with_TS(
                        bm, medial_sphere_mesh, verts[i], radii[i])

            #####!DEPRECATED######
            # TOO SLOW
            # for i in range(len(radii)):
            #     progress.current += 1
            #     progress()
            #     matrix = mathutils.Matrix.Translation(
            #         verts[i]) @ mathutils.Matrix.Scale(radii[i], 4)
            #     create_icosphere(bm,
            #                                subdivisions=ico_subdivide,
            #                                radius=radius,
            #                                matrix=matrix)

            progress.done()
            print("--- %.2f seconds ---" % (time.time() - start_time))
            print("Visualize Medial Sphere Mesh...")
            bm.to_mesh(medial_sphere_mesh)
            bm.free()
            assign_material(obj_medial_spheres, medial_sphere_mat)
            bpy.ops.object.shade_smooth()

            if len(faces) == 0:
                print("No medial slab")
            else:
                print("Generating medial slabs:")
                start_time = time.time()
                progress = ProgressBar(len(faces), fmt=ProgressBar.FULL)
                # Create medial slab material
                # Generate medial slab at each face
                slab_verts = []
                slab_faces = []
                degenerated = 0
                for index, f in enumerate(faces):
                    progress.current += 1
                    progress()
                    v1 = np.array(verts[f[0]])
                    r1 = radii[f[0]]
                    v2 = np.array(verts[f[1]])
                    r2 = radii[f[1]]
                    v3 = np.array(verts[f[2]])
                    r3 = radii[f[2]]
                    bm = bmesh.new()
                    result = generate_slab(
                        v1, r1, v2, r2, v3, r3, slab_verts, slab_faces, filter_threshold)
                    if result == -1:
                        degenerated += 1
                    else:
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
                print("Degenerated Slabs:{}".format(degenerated))
                slab_mesh = bpy.data.meshes.new(name=mat_name + ".SlabGroup")
                slab_mesh.from_pydata(slab_verts, [], slab_faces)
                slab_mesh.validate()
                slab_mesh.update()
                obj_medial_slab = bpy.data.objects.new(
                    slab_mesh.name, slab_mesh)
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
                    generate_conical_surface(v1, r1, v2, r2, resolution, cone_verts,
                                             cone_faces)
                progress.done()
                cone_mesh = bpy.data.meshes.new(name=mat_name + ".ConeGroup")
                cone_mesh.from_pydata(cone_verts, [], cone_faces)
                cone_mesh.validate()
                cone_mesh.update()
                obj_medial_cone = bpy.data.objects.new(
                    cone_mesh.name, cone_mesh)
                scene.collection.objects.link(obj_medial_cone)
                layer.objects.active = obj_medial_cone  # set active
                obj_medial_cone.select_set(True)  # select cone object
                assign_material(obj_medial_cone, medial_cone_mat)
                bpy.ops.object.shade_smooth()  # smooth shading
                del cone_verts, cone_faces
                print("--- %.2f seconds ---" % (time.time() - start_time))
        elif import_type == 'prim':
            print("Import Type: Individual medial primtive")
            if len(faces) == 0:
                print("No Medial Slab Primitives")
            else:
                print("Generating medial slabs:")
                start_time = time.time()
                progress = ProgressBar(len(faces), fmt=ProgressBar.FULL)
                # TODO: Create materials for each part
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
                    generate_slab(v1, r1, v2, r2, v3, r3,
                                  slab_verts, slab_faces, filter_threshold)
                    slab_face_num = len(slab_faces)
                    # Conical surface of 3 medial cones
                    generate_conical_surface(v1, r1, v2, r2, resolution, slab_verts,
                                             slab_faces)
                    generate_conical_surface(v2, r2, v3, r3, resolution, slab_verts,
                                             slab_faces)
                    generate_conical_surface(v1, r1, v3, r3, resolution, slab_verts,
                                             slab_faces)
                    cone_face_num = len(slab_faces)
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
                        create_icosphere(bm, ico_subdivide, radius, matrix)
                    slab_mesh = bpy.data.meshes.new(name="Slab:" + str(index))
                    bm.to_mesh(slab_mesh)
                    bm.free()
                    obj_slab = bpy.data.objects.new(slab_mesh.name, slab_mesh)
                    scene.collection.objects.link(obj_slab)
                    layer.objects.active = obj_slab  # set active
                    obj_slab.select_set(True)  # select slab object
                    bpy.ops.object.shade_smooth()  # smooth shading
                    # appending materials
                    obj_slab.data.materials.append(medial_slab_mat)
                    obj_slab.data.materials.append(medial_cone_mat)
                    obj_slab.data.materials.append(medial_sphere_mat)
                    for index, face in enumerate(obj_slab.data.polygons):
                        if index < slab_face_num:
                            face.material_index = 0
                        elif index < cone_face_num:
                            face.material_index = 1
                        else:
                            face.material_index = 2
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
                    generate_conical_surface(v1, r1, v2, r2, resolution, cone_verts,
                                             cone_faces)
                    # Combine all sub-meshes into one single mesh via bmesh
                    cone_face_num = len(cone_faces)
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
                        create_icosphere(bm, ico_subdivide, radius, matrix)
                    cone_mesh = bpy.data.meshes.new(name="Cone:" + str(index))
                    bm.to_mesh(cone_mesh)
                    bm.free()
                    obj_cone = bpy.data.objects.new(cone_mesh.name, cone_mesh)
                    scene.collection.objects.link(obj_cone)
                    layer.objects.active = obj_cone  # set active
                    obj_cone.select_set(True)  # select slab object
                    bpy.ops.object.shade_smooth()  # smooth shading
                    # appending materials
                    obj_cone.data.materials.append(medial_cone_mat)
                    obj_cone.data.materials.append(medial_sphere_mat)
                    for index, face in enumerate(obj_cone.data.polygons):
                        if index < cone_face_num:
                            face.material_index = 0
                        else:
                            face.material_index = 1
                progress.done()
        else:
            assert False, "Unknow Import Type"


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
