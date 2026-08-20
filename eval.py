# File author: Hualie Jiang (jianghualie0@gmail.com)
#
# Modifications: Eunjin Son (eunjinson@jbnu.ac.kr)
# - Added support for OmniPus.
#

from __future__ import print_function, division

from argparse import ArgumentParser
import logging
import time
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Torch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# Internal modules
from dataset import Dataset, is_omnipus_dbname
from dataset_omnipus import DatasetOmniPus
from utils.common import *
from utils.image import *
from module.network import MDP_Omni

# Initialize
torch.backends.cudnn.benchmark = True
torch.backends.cuda.benchmark = True

parser = ArgumentParser(description='Evaluation for MDP_Omni')
parser.add_argument('--name', default='MDP_Omni', help="name of your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")

parser.add_argument('--db_root', default='../omnidata', type=str, help='path to dataset')
parser.add_argument('--dbname', default='omnithings', type=str,help='databases to evaluation')

# data options
parser.add_argument('--phi_deg', type=float, default=45.0, help='phi_deg')
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640], help="size of out ERP.")

parser.add_argument('--vis', action='store_true', help='oneline visualization')
parser.add_argument('--save_result', action='store_true', help='save inverse depth prediction results')
parser.add_argument('--save_misc', action='store_true', help='save misc')
parser.add_argument('--save_point_cloud', action='store_true', help='save point cloud')

parser.add_argument('--threshold', type=float, default=0.01, help='threshold')
parser.add_argument('--eval_metric', default=False, action='store_true',
                    help='compute eval metrics on metric depth scale for omnipus datasets')

args = parser.parse_args()

opts = Edict()
opts.snapshot_path = args.restore_ckpt
opts.name = args.name

opts.dbname = args.dbname
opts.db_root = args.db_root

opts.data_opts = Edict()
opts.data_opts.color_aug = False
opts.data_opts.phi_deg = args.phi_deg
opts.data_opts.equirect_size = args.equirect_size

opts.net_opts = Edict()

# Results
opts.vis = args.vis
opts.save_result = args.save_result
opts.save_misc = args.save_misc
opts.save_point_cloud = args.save_point_cloud
snapshot_name = osp.splitext(osp.basename(opts.snapshot_path))[0]
opts.result_dir = osp.join('./results', opts.dbname, snapshot_name)

if opts.vis:
    fig = plt.figure(frameon=False, figsize=(25, 10), dpi=40)
    plt.ion()
    plt.show()


def build_eval_list(data):
    if hasattr(data.opts, 'test_sequences'):
        eval_list = []
        for seq_name, frame_ids in data.opts.test_sequences.items():
            for frame_id in frame_ids:
                eval_list.append((seq_name, frame_id))
        return eval_list
    return data.opts.test_idx


def build_output_formats(base_dir, snapshot_name):
    misc_fmt = osp.join(base_dir, f'misc_%05d_{snapshot_name}.png')
    return Edict(
        misc=misc_fmt,
        input=misc_fmt.replace('misc', 'input'),
        pano=misc_fmt.replace('misc', 'pano'),
        idepth=misc_fmt.replace('misc', 'idepth'),
        err=misc_fmt.replace('misc', 'err'),
        invdepth=osp.join(base_dir, 'invdepth_%05d.png'),
        point_cloud=osp.join(base_dir, 'pc_%05d.ply'),
    )

def main():
    if not osp.exists(opts.snapshot_path):
        sys.exit('%s does not exsits' % (opts.snapshot_path))
    snapshot = torch.load(opts.snapshot_path)

    opts.net_opts = snapshot['net_opts']
    if 'threshold' not in opts.net_opts:
        opts.net_opts.threshold = args.threshold
    opts.net_opts.phi_deg = args.phi_deg
    opts.data_opts.use_rgb = opts.net_opts.use_rgb
    opts.data_opts.num_invdepth = opts.net_opts.num_invdepth

    if is_omnipus_dbname(opts.dbname):
        data = DatasetOmniPus(opts.dbname, opts.data_opts, db_root=opts.db_root, train=False)
    else:
        data = Dataset(opts.dbname, opts.data_opts, db_root=opts.db_root, train=False)

    net = torch.nn.DataParallel(MDP_Omni(data.ocams, opts.net_opts), device_ids=[0])
    net.load_state_dict(snapshot['net_state_dict'])

    disp_hypo_init = torch.tensor(data.get_invdepths(), requires_grad=False).unsqueeze(0)

    if not osp.exists(opts.result_dir):
        os.makedirs(opts.result_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.result_dir))

    eval_list = build_eval_list(data)

    use_metric_eval = is_omnipus_dbname(opts.dbname) and args.eval_metric
    if use_metric_eval:
        errors = np.zeros((len(eval_list), 9))
    else:
        errors = np.zeros((len(eval_list), 5))
    acc_toc = 0
    for d in range(len(eval_list)):
        eval_item = eval_list[d]
        if isinstance(eval_item, tuple):
            seq_name, fidx = eval_item
            imgs, gt_idx, gt_valid, raw_imgs = data.loadSample(*eval_item)
            out_dir = osp.join(opts.result_dir, seq_name)
        else:
            fidx = eval_item
            imgs, gt_idx, gt_valid, raw_imgs = data.loadSample(fidx)
            out_dir = opts.result_dir
        os.makedirs(out_dir, exist_ok=True)
        out_fmt = build_output_formats(out_dir, snapshot_name)
        toc, toc2 = 0, 0
        net.eval()
        tic = time.time()
        imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]
        imgs = torch.stack(imgs, dim=1)

        with torch.no_grad():
            outputs = net(imgs, disp_hypo_init)

        hypos = outputs["hypos"]
        prob_grids = outputs["prob_grids"]

        final_prob = prob_grids[2]
        final_hypo = hypos[2]
        pred_index = torch.sum(final_prob * final_hypo, dim=2).squeeze(1)
        invdepth = data.indexToInvdepth(toNumpy(pred_index))
        toc = time.time() - tic
        acc_toc += toc

        # Compute errors
        if len(gt_idx) > 0:
            if use_metric_eval:
                pred_depth = 1 / invdepth
                gt_depth = 1 / data.indexToInvdepth(toNumpy(gt_idx))
                errors[d, :] = data.evalError_metric(pred_depth, gt_depth, gt_valid)
            else:
                errors[d, :] = data.evalError(pred_index, gt_idx, gt_valid)

        # Visualization
        if opts.vis or opts.save_misc or opts.save_point_cloud:
            tic2 = time.time()
            vis_img, inputs_rgb, pano_rgb, invdepth_rgb, err = data.makeVisImage(raw_imgs, invdepth, gt_idx, return_all=True)
            if opts.vis:
                fig.clf()
                plt.imshow(vis_img)
                plt.axis('off')
                plt.tight_layout()
                plt.draw()
                plt.pause(0.5)
            if opts.save_misc:
                writeImage(vis_img, out_fmt.misc % fidx)
                writeImage(inputs_rgb, out_fmt.input % fidx)
                writeImage(pano_rgb, out_fmt.pano % fidx)
                writeImage(invdepth_rgb, out_fmt.idepth % fidx)
                if err is not None:
                    writeImage(err, out_fmt.err % fidx)
            if opts.save_point_cloud:
                data.writePointCloud(pano_rgb, invdepth, out_fmt.point_cloud % fidx)
            toc2 = toc2 + time.time() - tic2

        # Save result
        if opts.save_result:
            tic2 = time.time()
            data.writeInvdepth_color(invdepth, out_fmt.invdepth % fidx)
            toc2 = toc2 + time.time() - tic2

        if use_metric_eval:
            LOG_INFO('Process %d/%d, abs_rel: %.3f, %.3f s, misc: %.3f s' %
                     (d + 1, len(eval_list), errors[d, 1], toc, toc2))
        else:
            LOG_INFO('Process %d/%d, MAE: %.3f, %.3f s, misc: %.3f s' %
                     (d + 1, len(eval_list), errors[d, 3], toc, toc2))

    mean_errors = errors.mean(axis=0)
    if use_metric_eval:
        LOG_INFO('silog: %.3f, abs_rel: %.3f, log10: %.3f, rms: %.3f, sq_rel: %.3f, '
                 'log_rms: %.3f, d1: %.3f, d2: %.3f, d3: %.3f, Avg time: %.3f' %
                 (mean_errors[0], mean_errors[1], mean_errors[2],
                  mean_errors[3], mean_errors[4], mean_errors[5],
                  mean_errors[6], mean_errors[7], mean_errors[8],
                  acc_toc / len(eval_list)))
    else:
        LOG_INFO('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f, Avg time: %.3f' %
                 (mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4], acc_toc / len(eval_list)))

if __name__ == "__main__":
    main()
