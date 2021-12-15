import os
import math
import mathutils
import numpy as np


def load_ma_file(filepath):
    file = open(filepath, 'r')
    # read first line number of vertices/edges/faces
    first_line = file.readline().rstrip()
    vcount, ecount, fcount = [int(x) for x in first_line.split()]
    # No Medial Vertices
    assert vcount != 0, "No Medial Vertices!"
    # line number
    lineno = 1

    # Medial Mesh Info
    verts, radii, faces, edges = [], [], [], []

    # read vertices
    for i in range(vcount):
        line = file.readline()
        # skip empty lines or comment lines
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        v = line.split()
        # Handle exception
        assert v[0] == 'v', "vertex line:" + str(
            lineno) + " should start with \'v\'!"
        x = float(v[1])
        y = float(v[2])
        z = float(v[3])
        radii.append(float(v[4]))
        verts.append((x, y, z))
        lineno += 1

    # read edges
    for i in range(ecount):
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert ef[0] == 'e', "line:" + str(
            lineno) + " should start with \'e\'!"
        ids = list(map(int, ef[1:3]))
        edges.append(tuple(ids))
        lineno += 1

    # read faces
    for i in range(fcount):
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert ef[0] == 'f', "line:" + str(
            lineno) + " should start with \'f\'!"
        ids = list(map(int, ef[1:4]))
        faces.append(tuple(ids))
        lineno = lineno + 1

    return vcount, fcount, ecount, verts, radii, faces, edges


def load_ma_file_with_feature(filepath):
    file = open(filepath, 'r')
    # read first line number of vertices/edges/faces
    first_line = file.readline().rstrip()
    vcount, ecount, fcount = [int(x) for x in first_line.split()]
    # No Medial Vertices
    assert vcount != 0, "No Medial Vertices!"
    # line number
    lineno = 1

    # Medial Mesh Info
    verts, radii, faces = [], [], []
    # deleted face number/ deleted edge number
    d_fcount, d_ecount = 0, 0
    edges, i_edges, o_edges = [], [], []

    # read vertices
    # format: v x y z r unknown_tags
    for i in range(vcount):
        line = file.readline()
        # skip empty lines or comment lines
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        v = line.split()
        # Handle exception
        assert v[0] == 'v', "vertex line:" + str(
            lineno) + " should start with \'v\'!"
        x = float(v[1])
        y = float(v[2])
        z = float(v[3])
        radii.append(float(v[4]))
        verts.append((x, y, z))

        lineno += 1

    # read edges
    # format: e idx1 idx2 delete_tag featuer_tag
    for i in range(ecount):
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert ef[0] == 'e', "line:" + str(
            lineno) + " should start with \'e\'!"

        if int(ef[3]) == 0:
            ids = list(map(int, ef[1:3]))
            medial_feature = int(ef[4])
            if medial_feature == 0:  # normal edge
                edges.append(tuple(ids))
            elif medial_feature == 1:  # inside feature
                i_edges.append(tuple(ids))
            elif medial_feature == 2:  # outside feature
                o_edges.append(tuple(ids))
            else:
                assert False, "edge line:" + \
                    str(lineno) + " Unknown Feature Type!"
        else:  # edge deleted
            d_ecount += 1
        lineno += 1
    print("Deleted Edges: {}\nNormal Edges: {}\tInside Features: {}\tOutside Features: {}".format(
        d_ecount, len(edges), len(i_edges), len(o_edges)))

    # read faces
    # format: f idx1 idx2 delete_tag
    for i in range(fcount):
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert ef[0] == 'f', "line:" + str(
            lineno) + " should start with \'f\'!"
        if int(ef[4]) == 0:
            ids = list(map(int, ef[1:4]))
            faces.append(tuple(ids))
        else:  # face deleted
            d_fcount += 1
        lineno = lineno + 1
    print("Deleted Face: {}".format(d_fcount))

    return vcount, fcount - d_fcount, ecount - d_ecount, verts, radii, faces, edges, i_edges, o_edges


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
        start_dir = np.array([1.0, 0.0, 0.0])
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