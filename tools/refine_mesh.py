import os
import sys
import re
import numpy as np
import time


def compute_angle(r1, r2, c21):
    r21_2 = max(pow(r1 - r2, 2), 0.0)
    if r21_2 == 0:
        return np.pi / 2
    phi = np.arctan(np.sqrt(max(np.dot(c21, c21) - r21_2, 0) / r21_2))
    phi = np.pi - phi if r1 < r2 else phi
    return phi


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


def normalize(v):
    return v / np.sqrt((v**2).sum())


def length(v):
    return np.sqrt((v**2).sum())


def plane_line_intersection(n, p, d, a):
    vpt = d[0] * n[0] + d[1] * n[1] + d[2] * n[2]

    if vpt == 0:
        print("SLAB GENERATION::Parallel Error")
        return np.zeros(3)

    t = ((p[0] - a[0]) * n[0] + (p[1] - a[1]) * n[1] +
         (p[2] - a[2]) * n[2]) / vpt
    return a + d * t


def intersect_point_of_cones(v1, r1, v2, r2, v3, r3, norm):
    if r1 < 1e-3:
        return [v1, v1]
    v12 = v2 - v1
    phi_12 = compute_angle(r1, r2, v12)

    v13 = v3 - v1
    phi_13 = compute_angle(r1, r3, v13)
    # print("\nAngles:{},{}".format(np.rad2deg(phi_12), np.rad2deg(phi_13)))

    p12 = v1 + normalize(v12) * np.cos(phi_12) * r1
    p13 = v1 + normalize(v13) * np.cos(phi_13) * r1
    # print("\nP12:{}\nP13:{}".format(p12, p13))
    mat = rotate_mat(p12, v12, np.pi / 2)
    dir_12 = mat.dot(np.append(norm, [0]))[:3]
    # print("\ndir_12:{},norm:{}".format(dir_12, norm))
    intersect_p = plane_line_intersection(v13, p13, dir_12, p12)
    # print("\nintersect_p:{}".format(intersect_p))
    v1p = intersect_p - v1
    scaled_n = np.sqrt(max(r1 * r1 - np.dot(v1p, v1p), 0.0)) * norm
    # print("\nscaled_n:{}".format(np.linalg.norm(scaled_n)))
    return [intersect_p + scaled_n, intersect_p - scaled_n, np.linalg.norm(scaled_n)]


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
    verts, faces, edges = [], [], []

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
        r = float(v[4])
        verts.append((x, y, z, r))
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

        lineno += 1
        i += 1

    return vcount, fcount, ecount, verts, faces, edges


def degenerated_slab(v1, r1, v2, r2, v3, r3, threshold=1e-3):
    n = normalize(np.cross(v1-v2, v1-v3))
    # v1-v2,v1-v3, tangent point on {v1,r1}
    tangent_p1 = intersect_point_of_cones(v1, r1, v2, r2, v3, r3, n)

    # v2-v1,v2-v3, tangent point on {v2,r2}
    tangent_p2 = intersect_point_of_cones(v2, r2, v1, r1, v3, r3, n)

    # v3-v1,v3-v2, tangent point on {v3,r3}
    tangent_p3 = intersect_point_of_cones(v3, r3, v1, r1, v2, r2, n)

    d2v1 = length(tangent_p1[0] - v1) - r1
    d2v2 = length(tangent_p2[0] - v2) - r2
    d2v3 = length(tangent_p3[0] - v3) - r3

    return d2v1 > threshold, d2v2 > threshold, d2v3 > threshold
    # return d2v1 > threshold or tangent_p1[2] > threshold, False, False


def degenerated_cone(v1, r1, v2, r2):
    if r1 < r2:
        v1, v2 = v2, v1
        r1, r2 = r2, r1
    lc = length(v1-v2)
    if lc + r2 < r1:
        return True
    return False


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"
    FULL = "%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go"

    def __init__(self, total, width=40, fmt=DEFAULT, symbol="=", output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d",
                          r"\g<name>%dd" % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "remaining": remaining,
        }
        print("\r" + self.fmt % args, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)


filepath = sys.argv[1]
filename = filepath.split("/")[-1][0:-4]
dirname = os.path.dirname(filepath)
threshold = float(sys.argv[2])

filename_out = f"{dirname}/{filename}_refined.ma"

vcount, fcount, ecount, verts, faces, edges = load_ma_file(filepath)

progress = ProgressBar(len(faces), fmt=ProgressBar.FULL)

df, de = 0, 0
new_edges = []
new_faces = []
start_t = time.time()
for index, f in enumerate(faces):
    progress.current += 1
    progress()
    v1 = np.array(verts[f[0]][:3])
    r1 = verts[f[0]][3]
    v2 = np.array(verts[f[1]][:3])
    r2 = verts[f[1]][3]
    v3 = np.array(verts[f[2]][:3])
    r3 = verts[f[2]][3]
    dv1, dv2, dv3 = degenerated_slab(v1, r1, v2, r2, v3, r3, threshold)
    if dv1 or dv2 or dv3:
        if dv1 and not dv2 and not dv3:
            new_edges.append([f[1], f[2]])
        elif dv2 and not dv1 and not dv3:
            new_edges.append([f[0], f[2]])
        elif dv3 and not dv1 and not dv2:
            new_edges.append([f[0], f[1]])
        df += 1
        # new_faces.append([f[0], f[1], f[2]])
    else:
        new_faces.append([f[0], f[1], f[2]])
        # pass

progress.done()
print("Degenerated Slabs:{}".format(df))
print("--- %.2f seconds ---" % (time.time() - start_t))

if len(edges) != 0:
    start_t = time.time()
    de = 0
    progress = ProgressBar(len(edges), fmt=ProgressBar.FULL)
    for e in edges:
        progress.current += 1
        progress()
        v1 = np.array(verts[e[0]][:3])
        r1 = verts[e[0]][3]
        v2 = np.array(verts[e[1]][:3])
        r2 = verts[e[1]][3]
        if not degenerated_cone(v1, r1, v2, r2):
            new_edges.append([e[0], e[1]])
            # pass
        else:
            de += 1
            # new_edges.append([e[0], e[1]])
    progress.done()
    print("Degenerated Cones:{}".format(de))
    print("--- %.2f seconds ---" % (time.time() - start_t))

# file_out = open(filename_out, 'w')
# file_out.write(str(len(verts)) + " " + str(len(new_edges)) +
#                " " + str(len(new_faces)) + "\n")
# for v in range(len(verts)):
#     file_out.write("v")
#     for j in range(4):
#         file_out.write(" " + str(verts[v][j]))
#     file_out.write("\n")

# for e in range(len(new_edges)):
#     file_out.write("e")
#     for j in range(2):
#         file_out.write(" " + str(new_edges[e][j]))
#     file_out.write("\n")

# for f in range(len(new_faces)):
#     file_out.write("f")
#     for j in range(3):
#         file_out.write(" " + str(new_faces[f][j]))
#     file_out.write("\n")

# file_out.close()
