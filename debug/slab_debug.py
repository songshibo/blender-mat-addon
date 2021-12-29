import numpy as np
from matutil import *


def degenerated_slab(v1, r1, v2, r2, v3, r3, threshold):
    v12 = v1-v2
    v13 = v1-v3
    v23 = v2-v3

    l_v12 = length(v12)
    l_v13 = length(v13)
    l_v23 = length(v23)

    r12 = l_v12 * 0.5 / max(r1, r2)
    r13 = l_v13 * 0.5 / max(r1, r3)
    r23 = l_v23 * 0.5 / max(r2, r3)

    print("v12:{},{},{}".format(v12, l_v12, r12))
    print("v13:{},{},{}".format(v13, l_v13, r13))
    print("v23:{},{},{}".format(v23, l_v23, r23))

    # if any two vertices are too closed
    if r12 < 0.05 or r13 < 0.05 or r23 < 0.05:
        return True

    n_v23 = v23 / l_v23
    n_v12 = v12 / l_v12
    n_v13 = v13 / l_v13

    # if any two edges are close to parallel
    d1 = 1.0 - abs(np.dot(n_v12, n_v13))
    d2 = 1.0 - abs(np.dot(n_v12, n_v23))
    d3 = 1.0 - abs(np.dot(n_v13, n_v23))
    print("parallel v12, v13:{}".format(d1))
    print("parallel v12, v23:{}".format(d2))
    print("parallel v13, v23:{}".format(d3))

    if d1 < threshold or d2 < threshold or d3 < threshold:
        return True
    # if the smallest sphere is inside the cone
    if r1 < r2:
        v1, v2 = v2, v1
        r1, r2 = r2, r1
    if r1 < r3:
        v1, v3 = v3, v1
        r1, r3 = r3, r1
    if r2 < r3:
        v2, v3 = v3, v2
        r2, r3 = r3, r2

    dist = distance_to_tangent(v1, r1, v2, r2, v3)
    return dist > r3

    return False


_, _, _, verts, radii, faces, _ = load_ma_file(
    "/Users/saber/Desktop/matwild/ma_results/ma_medial_features/00000329_12a935cd9b684f879c0505fe_trimesh_000.ma")
idx = 13691
f = faces[idx]
v1 = np.array(verts[f[0]])
v2 = np.array(verts[f[1]])
v3 = np.array(verts[f[2]])
r1 = radii[f[0]]
r2 = radii[f[1]]
r3 = radii[f[2]]
print("Slab ({}):\n\tV1:{},{}\n\tV2:{},{}\n\tV3:{},{}".format(
    idx, v1, r1, v2, r2, v3, r3))
print(degenerated_slab(v1, r1, v2, r2, v3, r3, 0.001))
