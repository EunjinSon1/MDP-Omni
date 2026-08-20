import math

import numpy as np
import torch

from utils.common import *
from utils.geometry import *


class OcamModel:
    def __init__(self):
        pass

    def setConfig(self, cfg):
        self.view = cfg["camera"]
        self.xc = cfg["intrinsic_parameters"][2]["cx"]
        self.yc = cfg["intrinsic_parameters"][3]["cy"]
        self.fx = cfg["intrinsic_parameters"][0]["fx"]
        self.fy = cfg["intrinsic_parameters"][1]["fy"]
        self.height = cfg["image_height"]
        self.width = cfg["image_width"]
        self.max_fov = 180
        self.max_theta = np.deg2rad(self.max_fov) / 2.0

        self.k1 = cfg["distortion_parameters"][0]["k1"]
        self.k2 = cfg["distortion_parameters"][1]["k2"]
        self.k3 = cfg["distortion_parameters"][2]["k3"]
        self.k4 = cfg["distortion_parameters"][3]["k4"]

        roll = cfg["extrinsic_parameters"][0]["roll"]
        pitch = cfg["extrinsic_parameters"][1]["pitch"]
        yaw = cfg["extrinsic_parameters"][2]["yaw"]
        tx = cfg["extrinsic_parameters"][3]["x"]
        ty = cfg["extrinsic_parameters"][4]["y"]
        tz = cfg["extrinsic_parameters"][5]["z"]

        r_x = np.array(
            [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
        )
        r_y = np.array(
            [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
        )
        r_z = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        )
        tr_cam_to_lidar = r_z @ r_y @ r_x
        trans = np.array([tx, ty, tz])

        tr_lidar_to_rig = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        tr_cam_to_rig = tr_lidar_to_rig @ tr_cam_to_lidar
        trans_new = tr_lidar_to_rig @ trans
        trans_inv = -(np.linalg.inv(tr_cam_to_rig) @ trans_new.T)

        tr_rig_to_cam = np.column_stack((np.linalg.inv(tr_cam_to_rig), trans_inv.T))
        self.rig2cam = np.row_stack((tr_rig_to_cam, [0.0, 0.0, 0.0, 1.0]))

    def rayToPixel(self, P, out_theta=False, max_theta=None):
        if max_theta is None:
            max_theta = self.max_theta

        norm = np.linalg.norm(P, axis=0)
        x = P[0, :] / norm
        y = P[1, :] / norm
        z = P[2, :] / norm

        a = x / z
        b = y / z
        r = (a**2 + b**2) ** 0.5
        theta = np.arctan(r)
        theta_d = theta * (1 + self.k1 * theta**2 + self.k2 * theta**4 + self.k3 * theta**6 + self.k4 * theta**8)
        x_d = (theta_d / r) * a
        y_d = (theta_d / r) * b

        u = self.fx * x_d + self.xc
        v = self.fy * y_d + self.yc
        x2 = u.reshape((1, -1))
        y2 = v.reshape((1, -1))
        norm_cor = sqrt(P[0, :] ** 2 + P[1, :] ** 2) + EPS
        theta_cor = atan2(-P[2, :], norm_cor) + np.pi / 2
        out = concat((x2, y2), axis=0)
        out[:, theta_cor.squeeze() > max_theta] = -1e5

        if out_theta:
            return out, theta_cor
        return out

    def rayToPixel_volume(self, P, out_theta=False, max_theta=None):
        if max_theta is None:
            max_theta = self.max_theta

        B, _, N, HW = P.shape
        norm = sqrt(P[:, 0, :] ** 2 + P[:, 1, :] ** 2 + P[:, 2, :] ** 2)

        x = P[:, 0] / norm
        y = P[:, 1] / norm
        z = P[:, 2] / norm

        a = x / z
        b = y / z
        r = (a**2 + b**2) ** 0.5
        theta = torch.arctan(r)
        theta_d = theta * (1 + self.k1 * theta**2 + self.k2 * theta**4 + self.k3 * theta**6 + self.k4 * theta**8)
        x_d = (theta_d / r) * a
        y_d = (theta_d / r) * b

        u = self.fx * x_d + self.xc
        v = self.fy * y_d + self.yc
        x2 = u.reshape((B, 1, -1))
        y2 = v.reshape((B, 1, -1))
        out = concat((x2, y2), axis=1).view(B, -1, N, HW)

        norm_cor = sqrt(P[:, 0, :] ** 2 + P[:, 1, :] ** 2) + EPS
        theta_cor = atan2(-P[:, 2, :], norm_cor) + np.pi / 2
        out[theta_cor.unsqueeze(1).expand_as(out) > max_theta] = -1e5
        if out_theta:
            return out, theta_cor.float()
        return out

    def makeInvisibleMask(self) -> np.ndarray:
        threshold = 455
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        mask = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        mask = mask & (((u - self.xc) ** 2 + (v - self.yc) ** 2) ** 0.5 < threshold)
        return (~mask).astype(bool)
