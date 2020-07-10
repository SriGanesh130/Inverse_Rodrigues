

'''
Inverse Rodrigues: Generating joint locations from rotation matrix of every joint
Ref:
https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio

'''
import numpy as np


def inv_rodrigues(Rotation):
    '''
    input: Rotation Matrix of (None, 3, 3)
    output: Joint Locations (None, 3)
    '''
    inv_r = []
    trace_r = []
    angle_w = []
    for i in range(Rotation.shape[0]):
        invr = Rotation[i][2, 1] - Rotation[i][1, 2], Rotation[i][0,
                                                                  2] - Rotation[i][2, 0], Rotation[i][1, 0] - Rotation[i][0, 1]
        tacer = Rotation[i][0, 0] + Rotation[i][1, 1] + Rotation[i][2, 2]
        s = np.sqrt(tacer + 1)
        inv_r.append(invr)
        trace_r.append(tacer)
        if tacer >= 3-np.finfo(np.float64).eps:
            angelw = (0.5 - ((tacer-3) / 12)) * np.array(invr)
        elif tacer < 3 - np.finfo(np.float64).eps and tacer > -1 + np.finfo(np.float64).eps:
            theta_w = np.arccos((tacer - 1) / 2)
            angelw = (theta_w / (2 * np.sin(theta_w))) * np.array(invr)
        elif tacer <= -1 + np.finfo(np.float64).eps:
            v_a = s/2
            v_b = (1/2*s) * (Rotation[i][1, 0] + Rotation[i][0][1])
            v_c = (1/2*s) * (Rotation[i][2, 0] + Rotation[i][0, 2])
            angelw = np.pi * (np.array((v_a, v_b, v_c)) *
                              np.linalg.norm(np.array((v_a, v_b, v_c))))
        angle_w.append(angelw)
    inv_r = np.array(inv_r)
    trace_r = np.array(trace_r)
    angle_w = np.array(angle_w)
    return angle_w


if __name__ == "__main__":
    rotation_mat = "/path/to/rotation_matrix.npy"
    DATA = np.load(rotation_mat)
    JOINT_LOC = inv_rodrigues(DATA)
    print(JOINT_LOC)
