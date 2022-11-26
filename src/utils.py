import numpy as np


def depth_to_xyz(K, img_depth, resolution, z_max):
    '''convert depth image to xyz point cloud'''

    img_h, img_w = img_depth.shape
    img_xyz = np.zeros([img_h, img_w, 3])

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    for v in range(img_h):
        for u in range(img_w):
            z = img_depth[v, u] * resolution
            if z < 0 or z > z_max:
                z = 0
            x = z * (u - cx) / fx
            y = z * (v - cy) / fy
            img_xyz[v, u, 0] = x
            img_xyz[v, u, 1] = y
            img_xyz[v, u, 2] = z

    return img_xyz


def is_good_triangle(a, b, c):
    '''
    check the quality of the triangle base on ratio of max and min side
    reject if ratio is larger than 5.
    '''

    ab = np.sqrt(np.sum(np.square(np.subtract(a, b))))
    bc = np.sqrt(np.sum(np.square(np.subtract(b, c))))
    ca = np.sqrt(np.sum(np.square(np.subtract(c, a))))

    sides = [ab, bc, ca]
    side_max = np.max(sides)
    side_min = np.min(sides)

    if side_max > 5 * side_min:
        return False
    return True


def triangulate(img_xyz):
    '''convert organized xyz point cloud into triangles with flattened vertex index'''

    img_h, img_w, _ = img_xyz.shape
    triangles = []

    for v in range(img_h - 1):
        for u in range(img_w - 1):
            # a1: (u    , v    )
            # b : (u + 1, v    )
            # c : (u    , v + 1)
            # a1: (u + 1, v + 1)
            if img_xyz[v, u + 1, 2] == 0 or \
               img_xyz[v + 1, u, 2] == 0:
               continue

            # triangle 1 (a1, b, c)
            if img_xyz[v, u, 2] != 0 and \
               is_good_triangle(img_xyz[v, u], img_xyz[v, u + 1], img_xyz[v + 1, u]):
                triangles.append([v * img_w + u, (v + 1) * img_w + u, v * img_w + u + 1])

            # triangle 2 (a2, c, b)
            if img_xyz[v + 1, u + 1, 2] != 0 and \
               is_good_triangle(img_xyz[v + 1, u + 1], img_xyz[v, u + 1], img_xyz[v + 1, u]):
                triangles.append([(v + 1) * img_w + u + 1, v * img_w + u + 1, (v + 1) * img_w + u])

    return np.array(triangles, dtype=int)


def load_camera_matrix(filepath):
    '''load camera matrix of the depth camera'''

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        if   "fx_l" in line:
            fx_l = np.float64(line.strip().split("=")[1])
        elif "fy_l" in line:
            fy_l = np.float64(line.strip().split("=")[1])
        elif "cx_l" in line:
            cx_l = np.float64(line.strip().split("=")[1])
        elif "cy_l" in line:
            cy_l = np.float64(line.strip().split("=")[1])
        else:
            continue

    camera_matrix = np.array([
        [fx_l,  0.0, cx_l],
        [ 0.0, fy_l, cy_l],
        [ 0.0,  0.0,  1.0]])

    return camera_matrix


def load_poses(filepath):
    '''load poses (matrix) from pose.txt'''

    poses = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    if n != len(lines) - 1:
        raise ValueError("[FAIL]: Mismatch in number of poses.")

    for i in range(1, n + 1):
        pose = np.fromstring(lines[i].strip(), dtype=np.float64, count=16, sep=" ")
        poses.append(pose.reshape([4, 4]))

    return poses


def load_points(filepath):
    '''load points (xyz) from points.txt'''

    points = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    if n != len(lines) - 1:
        raise ValueError("[FAIL]: Mismatch in number of points.")

    for i in range(1, n + 1):
        point = np.fromstring(lines[i].strip(), dtype=np.float64, count=3, sep=" ")
        points.append(point)

    return points


def load_constraints(filepath):
    '''load constraints from constraints.txt'''

    constraints = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    if n != len(lines) - 1:
        raise ValueError("[FAIL]: Mismatch in number of constraints.")

    for i in range(1, n + 1):
        constraint = dict()
        arr = lines[i].strip().split(" ")
        constraint["pose_id"] = int(arr[0])
        constraint["point_id"] = int(arr[1])
        constraint["u_l"] = float(arr[2])
        constraint["v_l"] = float(arr[3])
        constraint["u_r"] = float(arr[4])
        constraint["v_r"] = float(arr[5])
        constraints.append(constraint)

    return constraints


def save_ply(filename, verts, norms=None, colors=None, faces=None):
    '''save 3D mesh to ply file'''

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    if norms is not None:
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
    if colors is not None:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
    if faces is not None:
        ply_file.write("element face %d\n"%(faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        str_vertex = "{} {} {}".format(verts[i, 0], verts[i, 1], verts[i, 2])
        if norms  is not None:
            str_vertex += " {} {} {}".format(norms[i, 0], norms[i, 1], norms[i, 2])
        if colors is not None:
            str_vertex += " {} {} {}".format(colors[i, 0], colors[i, 1], colors[i, 2])
        str_vertex += "\n"
        ply_file.write(str_vertex)

    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n"%(faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()