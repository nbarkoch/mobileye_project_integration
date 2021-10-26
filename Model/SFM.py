import numpy as np


class FrameContainer(object):
    def __init__(self):
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind=[]
        self.valid=[]


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_curr_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot, valid_p = find_corresponding_points(p_curr, norm_rot_pts, foe)

        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0) and valid_p
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return np.array(list(map(lambda x: [(x[0] - pp[0]) / focal, (x[1] - pp[1]) / focal], pts)))


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return np.array(list(map(lambda x: [(x[0] * focal) + pp[0], (x[1] * focal) + pp[1]], pts)))


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    # using https://gamedev.stackexchange.com/questions/72044/why-do-we-use-4x4-matrices-to-transform-things-in-3d
    R = EM[:3, :3]  # matrix 3X3 with det of 1
    t = EM[:3, -1]  # getting last row 1X3

    # foe = [tX / tZ, tY / tZ]
    foe = [t[0] / t[2], t[1] / t[2]]
    return R, foe, t[2]


def rotate(pts, R):
    # rotate the points - pts using R
    res = list()
    for point in pts:
        point_rotate = R.dot(np.array([point[0], point[1], 1]))
        # 𝑥_𝑟 = 𝑎/𝑐
        # 𝑦_𝑟 = 𝑏/𝑐
        point_rotate = (point_rotate[0], point_rotate[1]) / point_rotate[2]
        res.append(point_rotate)
    return np.array(res)


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    min_distance = float("inf")
    point_index = -1
    valid = True
    for i, pt in enumerate(norm_pts_rot):
        current_distance = np.linalg.norm(np.cross(foe - p, p - pt)) / np.linalg.norm(foe - p)
        if current_distance < min_distance:
            min_distance = current_distance
            point_index = i
    if min_distance > 0.0022:
        valid = False
    return point_index, norm_pts_rot[point_index], valid


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    zX = (tZ * (foe[0] - p_rot[0]))  #/ p_curr[0]
    # curr_rotate x
    crX = p_curr[0] - p_rot[0]

    if crX != 0:
        zX /= crX

    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    zY = (tZ * (foe[1] - p_rot[1])) #/ p_curr[1]
    # curr_rotate y
    crY = p_curr[1] - p_rot[1]
    if crY != 0:
        zY /= crY

    # calculate z distance using curr and rotate positions
    sumW = abs(crX) + abs(crY)

    if sumW == 0:
        return 0

    Z = np.array([(abs(crX) / sumW) * zX, (abs(crY) / sumW) * zY])
    return np.sum(Z)
