# dataset_omnipus.py

import os.path as osp

import numpy as np
import torch
from easydict import EasyDict as Edict

from dataset import Dataset, makeSphericalRays
from utils.array_utils import *
from utils.common import *
from utils.geometry import *
from utils.image import *
from utils.log import *
import utils.dbhelper


class DatasetOmniPus(Dataset):
    def __init__(self, dbname: str, db_opts=None, load_lut=False, train=True, db_root="../omnidata"):
        torch.utils.data.Dataset.__init__(self)
        self.dbname = dbname.lower()
        self.db_path = osp.join(db_root, self.dbname)

        opts = Edict()
        opts.img_fmt = "resize_cam%d/%06d.png"
        opts.gt_depth_fmt = "omnidepth_gt/%06d.tiff"
        opts.equirect_size, opts.num_invdepth = [160, 640], 192
        opts.num_downsample = 1
        opts.phi_deg, opts.phi2_deg = 42.4, -1.0
        opts.min_depth = 1.0
        opts.max_depth = 250.0
        opts.max_fov = 180.0
        opts.read_input_image = True
        opts.train_idx, opts.test_idx = [], []
        opts.gt_phi = 0.0
        opts.dtype = "nogt"

        opts, self.ocams = utils.dbhelper.loadDBConfigs(self.dbname, self.db_path, opts)
        opts.lut_fmt = "RoS_ds%d_lt_(%d,%d,%d).hwd"
        opts = argparse(opts, db_opts)

        self.opts = opts
        self.img_fmt, self.lut_fmt = opts.img_fmt, opts.lut_fmt
        self.gt_depth_fmt = opts.gt_depth_fmt
        self.train_idx, self.test_idx = opts.train_idx, opts.test_idx
        self.gt_phi = opts.gt_phi
        self.dtype = opts.dtype
        self.use_rgb = opts.use_rgb

        self.equirect_size = opts.equirect_size
        self.min_depth, self.max_depth = opts.min_depth, opts.max_depth
        self.max_theta = np.deg2rad(opts.max_fov) / 2.0
        self.phi_deg, self.phi2_deg = opts.phi_deg, opts.phi2_deg
        self.num_invdepth = opts.num_invdepth
        self.num_downsample = opts.num_downsample
        self.read_input_image = opts.read_input_image

        self.__initSweep(load_lut)
        self.train = train

        seq_dict = opts.train_sequences if train else opts.test_sequences
        self.frame_pairs = []
        for seq_name, frame_ids in seq_dict.items():
            available_frame_ids = self.collectAvailableFrameIds(seq_name, frame_ids)
            for frame_id in available_frame_ids:
                self.frame_pairs.append((seq_name, frame_id))
        self.size = len(self.frame_pairs)

    def __initSweep(self, load_lut=True):
        self.rays = makeSphericalRays(self.equirect_size, self.phi_deg, self.phi2_deg)
        self.min_invdepth = 1.0 / self.max_depth
        self.max_invdepth = 1.0 / self.min_depth
        self.sample_step_invdepth = (self.max_invdepth - self.min_invdepth) / (self.num_invdepth - 1.0)
        self.invdepths = np.arange(
            self.min_invdepth,
            self.max_invdepth + self.sample_step_invdepth,
            self.sample_step_invdepth,
            dtype=np.float64,
        )
        if load_lut:
            self.__loadOrBuildLookupTable()

    def __loadOrBuildLookupTable(self) -> None:
        h, w = self.equirect_size
        path = osp.join(self.db_path, self.lut_fmt % (self.num_downsample, h, w, self.num_invdepth))
        h, w = h // 2**self.num_downsample, w // 2**self.num_downsample
        if not osp.exists(path):
            LOG_INFO('Lookup table not found: "%s"' % path)
            LOG_INFO("Build lookup table...")
            self.grids = self.buildLookupTable()
            np.concatenate([toNumpy(g)[np.newaxis, ...] for g in self.grids], axis=0).tofile(path)
            LOG_INFO('Lookup table saved: "%s"' % path)
        else:
            LOG_INFO('Load lookup table: "%s"' % path)
            grids = np.fromfile(path, dtype=np.float32).reshape(
                [4, h, w, int(self.num_invdepth / 2**self.num_downsample), 2]
            )
            self.grids = [grids[i, ...].squeeze() for i in range(4)]

    def get_invdepths(self):
        return self.invdepths

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        seq_name, frame_id = self.frame_pairs[i]
        return self.loadSample(seq_name, frame_id, read_input_image=self.read_input_image, return_meta=True)

    def collectAvailableFrameIds(self, seq_name, frame_ids):
        available_frame_ids = []
        missing_count = 0

        for frame_id in sorted(frame_ids):
            required_files = [
                osp.join(self.db_path, seq_name, self.gt_depth_fmt % frame_id),
                osp.join(self.db_path, seq_name, self.img_fmt % (1, frame_id)),
                osp.join(self.db_path, seq_name, self.img_fmt % (2, frame_id)),
                osp.join(self.db_path, seq_name, self.img_fmt % (3, frame_id)),
                osp.join(self.db_path, seq_name, self.img_fmt % (4, frame_id)),
            ]
            if all(osp.exists(path) for path in required_files):
                available_frame_ids.append(frame_id)
            else:
                missing_count += 1

        if missing_count > 0:
            LOG_WARNING('Skipped %d frames with missing files in "%s"' % (missing_count, seq_name))

        return available_frame_ids

    def buildLookupTable(self, transform=None, phi_deg=None, phi2_deg=None, output_gpu_tensor=False) -> list:
        num_invdepth = int(self.num_invdepth / 2**self.num_downsample)
        h, w = self.equirect_size
        h, w = h // 2 ** self.num_downsample, w // 2 ** self.num_downsample
        equirect_size = [h, w]

        if phi_deg is None:
            phi_deg = self.phi_deg
        if phi2_deg is None:
            phi2_deg = self.phi2_deg
        rays = makeSphericalRays(equirect_size, phi_deg, phi2_deg)
        if output_gpu_tensor:
            grids = [torch.zeros((h, w, num_invdepth, 2), requires_grad=False).cuda() for _ in range(4)]
        else:
            grids = [np.zeros((h, w, num_invdepth, 2), dtype=np.float32) for _ in range(4)]

        for d in range(num_invdepth):
            depth = 1.0 / self.invdepths[2**self.num_downsample * d]
            pts = depth * rays
            if output_gpu_tensor:
                pts = torch.tensor(pts.astype(np.float32), requires_grad=False).cuda()
            if transform is not None:
                pts = applyTransform(transform, pts)
            for i in range(4):
                P = applyTransform(self.ocams[i].rig2cam, pts)
                p = self.ocams[i].rayToPixel(P)
                grid = pixelToGrid(p, equirect_size, (self.ocams[i].height, self.ocams[i].width))
                grid = np.clip(grid, -2, 1)
                if output_gpu_tensor:
                    grids[i][..., d, :] = grid
                else:
                    grids[i][..., d, :] = grid.astype(np.float32)
        return grids

    def loadImages(self, seq_name, frame_id, out_raw_imgs=False, use_rgb=False):
        imgs = []
        raw_imgs = []
        for cam_idx in range(4):
            file_path = osp.join(self.db_path, seq_name, self.img_fmt % (cam_idx + 1, frame_id))
            image = toNumpy(readImage(file_path))
            if out_raw_imgs:
                raw_imgs.append(image)

        for cam_idx, image in enumerate(raw_imgs):
            if not use_rgb and len(image.shape) == 3 and image.shape[2] == 3:
                image = rgb2gray(image, channel_wise_mean=True)
            image = normalizeImage(image, self.ocams[cam_idx].invalid_mask)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)
                if use_rgb:
                    image = np.tile(image, (3, 1, 1))
            else:
                image = np.transpose(image, (2, 0, 1))
            imgs.append(image)
        return (imgs, raw_imgs) if out_raw_imgs else imgs

    def readInvdepth(self, path: str) -> np.ndarray:
        _, ext = osp.splitext(path)
        if ext == ".png":
            step_invdepth = (self.max_invdepth - self.min_invdepth) / 65500.0
            quantized_inv_index = readImage(path).astype(np.float32)
            return self.min_invdepth + quantized_inv_index * step_invdepth
        if ext == ".tif" or ext == ".tiff":
            return readImageFloat(path)
        return np.fromfile(path, dtype=np.float32)

    def writeInvdepth(self, invdepth: np.ndarray, path: str) -> None:
        _, ext = osp.splitext(path)
        if ext == ".png":
            step_invdepth = (self.max_invdepth - self.min_invdepth) / 65500.0
            quantized_inv_index = (invdepth - self.min_invdepth) / step_invdepth
            writeImage(quantized_inv_index.round().astype(np.uint16), path)
        elif ext == ".tif" or ext == ".tiff":
            thumbnail = colorMap("oliver", invdepth, self.min_invdepth, self.max_invdepth)
            thumbnail = imrescale(thumbnail, 0.5)
            writeImageFloat(invdepth.astype(np.float32), path, thumbnail)
        else:
            invdepth.astype(np.float32).tofile(path)

    def indexToInvdepth(self, idx, start_index=0):
        return self.min_invdepth + (idx - start_index) * self.sample_step_invdepth

    def invdepthToIndex(self, inv_depth, start_index=0):
        return (inv_depth - self.min_invdepth) / self.sample_step_invdepth + start_index

    def loadGTInvdepthIndex(self, seq_name, frame_id, remove_gt_noise=True):
        h, _ = self.equirect_size
        gt_depth_file = osp.join(self.db_path, seq_name, self.gt_depth_fmt % frame_id)
        gt = self.readInvdepth(gt_depth_file)
        gt_h = gt.shape[0]
        if h < gt_h:
            sh = int(round((gt_h - h) / 2.0))
            gt = gt[sh : sh + h, :]

        gt_idx = self.invdepthToIndex(gt)
        if not remove_gt_noise:
            return gt_idx
        gt_idx[gt > self.max_invdepth] = -1
        gt_idx[gt < self.min_invdepth] = -1
        return gt_idx

    def loadSample(self, seq_name, frame_id, read_input_image=True, varargin=None, return_meta=False):
        opts = Edict()
        opts = argparse(opts, varargin)
        imgs, raw_imgs = [], []
        if read_input_image:
            imgs, raw_imgs = self.loadImages(seq_name, frame_id, True, use_rgb=self.use_rgb)
        gt, valid = [], []
        if self.dtype == "gt":
            gt = self.loadGTInvdepthIndex(seq_name, frame_id)
            valid = np.logical_and(gt >= 0, gt <= self.num_invdepth).astype(bool)

        sample = (imgs, gt, valid, raw_imgs)
        if return_meta:
            return sample + (seq_name, frame_id)
        return sample

    def evalError_metric(self, pred_depth, gt_depth, valid):
        pred_depth = toNumpy(pred_depth).flatten()
        gt_depth = toNumpy(gt_depth).flatten()
        valid = toNumpy(valid).flatten().astype(bool)
        valid = np.logical_and(valid, np.logical_not(np.isnan(pred_depth)))
        pred_depth = pred_depth[valid]
        gt_depth = gt_depth[valid]

        thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()

        rms = (gt_depth - pred_depth) ** 2
        rms = np.sqrt(rms.mean())

        log_rms = (np.log(gt_depth) - np.log(pred_depth)) ** 2
        log_rms = np.sqrt(log_rms.mean())

        abs_rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
        sq_rel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)

        err = np.log(pred_depth) - np.log(gt_depth)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        err = np.abs(np.log10(pred_depth) - np.log10(gt_depth))
        log10 = np.mean(err)

        return silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3
