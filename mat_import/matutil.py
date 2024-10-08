from distutils.command.build_scripts import first_line_re
import math
# import mathutils
import numpy as np


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def load_hd_file(filepath):
    file = open(filepath, 'r')
    first_line = file.readline().rstrip()
    vcount, _, fcount = [int(x) for x in first_line.split()]
    verts, HDs, faces = [], [], []
    lineno = 1

    # read vertices
    i = 0
    while i < vcount:
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
        HDs.append(float(v[4]))
        verts.append((x, y, z))
        lineno += 1
        i += 1

    i = 0
    # read faces
    while i < fcount:
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert ef[0] == 'f', "line:" + str(
            lineno) + " should start with \'f\'!"
        f = tuple(list(map(int, ef[1:4])))
        faces.append(f)

        lineno += 1
        i += 1

    return vcount, fcount, verts, HDs, faces


def load_woff_file(filepath):
    file = open(filepath, 'r')
    first_line = file.readline().rstrip().split()
    vcount = int(first_line[1])
    ecount, fcount = 0, 0
    # No Spheres
    assert vcount != 0, "No Medial Vertices!"
    lineno = 1
    verts, radii, faces, edges = [], [], [], []
    # read vertices
    for i in range(vcount):
        line = file.readline().rstrip()
        # skip empty lines or comment lines
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        v = line.split()
        # Handle exception
        assert len(v) == 4, "vertex line:" + str(
            lineno) + " 4 floats for x/y/z/r!"
        x = float(v[0])
        y = float(v[1])
        z = float(v[2])
        radii.append(float(v[3]))
        verts.append((x, y, z))
        lineno += 1

    return vcount, fcount, ecount, verts, radii, faces, edges


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
    i = 0
    while i < vcount:
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
        i += 1

    i = 0
    # read edges
    while i < ecount:
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
        i += 1

    i = 0
    # read faces
    while i < fcount:
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert ef[0] == 'f', "line:" + str(
            lineno) + " should start with \'f\'!"
        f = tuple(list(map(int, ef[1:4])))
        faces.append(f)
        edges.append(f[:2])
        edges.append(f[1:3])
        edges.append((f[0], f[2]))

        lineno += 1
        i += 1

    unique_edges = list(unique_everseen(edges, key=frozenset))
    return vcount, fcount, ecount, verts, radii, faces, unique_edges


def load_mesh_with_fcolor(filepath):
    file = open(filepath, 'r')
    first_line = file.readline().rstrip()
    vcount, _, fcount = [int(x) for x in first_line.split()]
    verts, faces, fcolor = [], [], []
    lineno = 1

    # read vertices
    i = 0
    while i < vcount:
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
        verts.append((x, y, z))
        lineno += 1
        i += 1

    i = 0
    # read faces
    while i < fcount:
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert ef[0] == 'f', "line:" + str(
            lineno) + " should start with \'f\'!"
        f = tuple(list(map(int, ef[1:4])))
        r = float(ef[4])
        g = float(ef[5])
        b = float(ef[6])
        faces.append(f)
        fcolor.append((r, g, b))
        lineno += 1
        i += 1

    return verts, faces, fcolor


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
        # handle delete flag
        if not (len(v) == 7 and v[6] == 1):
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

        # Handle extra flags
        if len(ef) == 4:  # only has type flag
            type_flag = int(ef[3])
            ids = list(map(int, ef[1:3]))
            if type_flag == 0:  # normal edge
                edges.append(tuple(ids))
            elif type_flag == 1:  # inside feature
                i_edges.append(tuple(ids))
            elif type_flag == 2:  # outside feature
                o_edges.append(tuple(ids))
            else:
                assert False, "edge line:" + \
                    str(lineno) + " Unknown Feature Type!"
        elif len(ef) == 5:  # has delete and type flags
            delete_flag = int(ef[4])
            if delete_flag == 0:  # has not been deleted
                type_flag = int(ef[3])
                ids = list(map(int, ef[1:3]))
                if type_flag == 0:  # normal edge
                    edges.append(tuple(ids))
                elif type_flag == 1:  # inside feature
                    i_edges.append(tuple(ids))
                elif type_flag == 2:  # outside feature
                    o_edges.append(tuple(ids))
                else:
                    assert False, "edge line:" + \
                        str(lineno) + " Unknown Feature Type!"
            else:  # edge deleted
                d_ecount += 1
        elif len(ef) < 4:
            ids = list(map(int, ef[1:3]))
            edges.append(tuple(ids))
        else:
            assert False, "Incorrect number of flags!"
        # if int(ef[3]) == 0:
        #     ids = list(map(int, ef[1:3]))
        #     medial_feature = int(ef[4])
        #     if medial_feature == 0:  # normal edge
        #         edges.append(tuple(ids))
        #     elif medial_feature == 1:  # inside feature
        #         i_edges.append(tuple(ids))
        #     elif medial_feature == 2:  # outside feature
        #         o_edges.append(tuple(ids))
        #     else:
        #         assert False, "edge line:" + \
        #             str(lineno) + " Unknown Feature Type!"
        # else:  # edge deleted
        #     d_ecount += 1
        lineno += 1
    print("Deleted Edges: {}\nNormal Edges: {}\tInside Features: {}\tOutside Features: {}".format(
        d_ecount, len(edges), len(i_edges), len(o_edges)))

    # read faces
    # format: f idx1 idx2 delete_tag
    feature_edges = i_edges + o_edges
    for i in range(fcount):
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno = lineno + 1
            continue
        ef = line.split()
        # Handle exception
        assert ef[0] == 'f', "line:" + str(
            lineno) + " should start with \'f\'!"

        delete_flag = 0
        if len(ef) == 5:  # has delete flag
            delete_flag = int(ef[4])

        if delete_flag == 0:
            f = tuple(list(map(int, ef[1:4])))
            faces.append(f)
            # add to edges
            feature_edges.append(f[:2])
            feature_edges.append(f[1:3])
            feature_edges.append((f[0], f[2]))
        else:  # face deleted
            d_fcount += 1

        lineno = lineno + 1
    print("Deleted Face: {}, Faces left:{}".format(d_fcount, len(faces)))

    in_out_e_num = len(i_edges) + len(o_edges)
    edges += list(unique_everseen(
        feature_edges, key=frozenset))[in_out_e_num:]
    del feature_edges
    unique_edges = list(unique_everseen(edges, key=frozenset))
    return vcount, fcount - d_fcount, ecount - d_ecount, verts, radii, faces, unique_edges, i_edges, o_edges


# generate the conical surface for a medial cone
# the vertices & face indices are stored in cone_verts & cone_faces respectively
def generate_conical_surface(v1, r1, v2, r2, resolution, cone_verts,
                             cone_faces):
    if r1 < 1e-3 and r2 < 1e-3:
        return
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
        if r1 < 1e-3:
            pos = v1
        else:
            pos = v1 + (c12 * cos + start_dir * sin) * \
                r1  # vertex on sphere {v1,r1}
        cone_verts.append(tuple(pos))
        if r2 < 1e-3:
            pos = v2
        else:
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


def degenerated_cone(v1, v2, threshold):
    pass


# special degenerated slab
# ra > rb > rp
def distance_to_tangent(A, ra, B, rb, P):
    AB = B - A
    phi = compute_angle(ra, rb, AB)
    # linear interpolation parameter & distance from P to AB
    t, d_to_AB = closest_point_on_line(A, B, P)
    len_AB = length(AB)
    real_t = t - d_to_AB * np.cos(phi) / len_AB
    r = ra + (rb - ra) * real_t
    dist = r - length(A + AB * real_t - P)
    return dist


# generate two triangle slabs for a medial slab
def generate_slab(v1, r1, v2, r2, v3, r3, slab_verts, slab_faces, threshold=1e-4):
    # invalid slab, degenerated case
    # if degenerated_slab(v1, r1, v2, r2, v3, r3, threshold):
    #     return -1
    v12 = v1-v2
    v13 = v1-v3
    n = normalize(np.cross(v12, v13))
    # v1-v2,v1-v3, tangent point on {v1,r1}
    tangent_p1 = intersect_point_of_cones(v1, r1, v2, r2, v3, r3, n)
    # print(tangent_p1)
    d2v1 = length(tangent_p1[0] - v1) - r1
    # v2-v1,v2-v3, tangent point on {v2,r2}
    tangent_p2 = intersect_point_of_cones(v2, r2, v1, r1, v3, r3, n)
    d2v2 = length(tangent_p2[0] - v2) - r2
    # print(tangent_p2)
    # v3-v1,v3-v2, tangent point on {v3,r3}
    tangent_p3 = intersect_point_of_cones(v3, r3, v1, r1, v2, r2, n)
    d2v3 = length(tangent_p3[0] - v3) - r3
    # print(tangent_p3)
    # print(d2v1,d2v2,d2v3)
    if d2v1 > threshold or d2v2 > threshold or d2v3 > threshold:
        return -1
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
    return 1


# compute the intersect points of medial cone:{v1,v2} and medial cone:{v1,v3} on sphere (v1,r1)
def intersect_point_of_cones(v1, r1, v2, r2, v3, r3, norm):
    if r1 < 1e-3:
        return [v1, v1]
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


# closest point on line AB to P
def closest_point_on_line(A, B, P):
    dir = B - A
    t0 = np.dot(dir, P - A) / (dir**2).sum()
    intersectPnt = A + t0 * dir
    return t0, length(P - intersectPnt)


# normalize vector
def normalize(v):
    return v / np.sqrt((v**2).sum())


def length(v):
    return np.sqrt((v**2).sum())


# compare two 2d tuples(does not require same order)
def tuple_compare_2d(a, b):
    return a == b or (a[0] == b[1] and a[1] == b[0])


# compute angle between two medial sphere from a medial cone
def compute_angle(r1, r2, c21):
    r21_2 = max(pow(r1 - r2, 2), 0.0)
    if r21_2 == 0:
        return np.pi / 2
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


v1 = np.array([936.875,69.0509,43.893])
r1 = 3.34577
v2 = np.array([932.95,68.9973,43.7378])
r2 = 6.85103
v3 = np.array([935.3,65.5693,53.4417])
r3 = 10.279
generate_slab(v1, r1, v2, r2, v3, r3, [], [])