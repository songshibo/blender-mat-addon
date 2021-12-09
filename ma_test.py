import os


def load_ma(filepath):
    path = os.fsencode(filepath)
    file = open(path, 'r')
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
        assert v[0] == 'v', "vertex line:" + str(
            lineno) + " should start with \'v\'!"
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
        assert ef[0] == 'e', "line:" + str(
            lineno) + " should start with \'e\'!"

        ids = list(map(int, ef[1:3]))
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
        assert ef[0] == 'f', "line:" + str(
            lineno) + " should start with \'f\'!"

        ids = list(map(int, ef[1:4]))
        faces.append(tuple(ids))
        i = i + 1
        lineno = lineno + 1


load_ma("/Users/saber/Desktop/matwild/ma_results/ma_medial_features/59767.ma")