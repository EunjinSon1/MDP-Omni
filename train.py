# File author: Hualie Jiang (jianghualie0@gmail.com)
#
# Modifications: Eunjin Son (eunjinson@jbnu.ac.kr)
# - Added support for OmniPus.
#

from __future__ import print_function, division

import os

import time
import random
from argparse import ArgumentParser

# Torch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import LambdaLR
from PIL import Image

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# Internal modules
from dataset import Dataset, MultiDataset, is_omnipus_dbname
from dataset_omnipus import DatasetOmniPus
from utils.common import *
from utils.image import *
from module.network import MDP_Omni
from module.losses import compute_loss

from datetime import datetime

import math

# Initializef
torch.backends.cudnn.benchmark = True
torch.backends.cuda.benchmark = True

parser = ArgumentParser(description='Training for MDP_Omni')

parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--pretrain_ckpt', help="pretrained checkpoint for finetuning")

parser.add_argument('--db_root', default='../omnidata', type=str, help='path to dataset')
parser.add_argument('--dbname', nargs='+', default=['omnithings'], type=str,help='databases to train')

# data options
parser.add_argument('--phi_deg', type=float, default=45.0, help='phi_deg')
parser.add_argument('--num_invdepth', type=int, default=192, metavar='N', help='number of disparity')
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640], help="size of out ERP.")
parser.add_argument('--use_rgb', default=True, action='store_true', help='use 3-channel rgb color images as input')
parser.add_argument('--ndepths', type=int, nargs='+', default=[48, 32, 8], help='number of disparity at each stage')

# net options
parser.add_argument('--base_channel', type=int, default=32, help='base channel of the network')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--fix_bn', action='store_true', help='fix batch normalization')

# training options
parser.add_argument('--total_epochs', type=int, default=30, help='total epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')

parser.add_argument('--lr', type=float, default=0.00354, help="max learning rate.")
parser.add_argument('--wdecay', type=float, default=.01, help="Weight decay in optimizer.")
parser.add_argument('--min_lr', type=float, default=0.01, help="min_lr in optimizer.")
parser.add_argument('--warmup_steps', type=float, default=1000, help="warmup_steps in optimizer.")

parser.add_argument('--sigmoid_param', type=float, default=3,help="sigmoid_param.")
parser.add_argument('--threshold', type=float, default=0.01,help="threshold.")

args = parser.parse_args()

opts = Edict()
# Dataset & sweep arguments

args.name = datetime.now().strftime("%y%m%d_%H%M")

opts.name = args.name
opts.model_dir = os.path.join('./checkpoints', args.name)
opts.runs_dir = os.path.join('./runs', args.name)

opts.snapshot_path = args.restore_ckpt
opts.pretrain_path = args.pretrain_ckpt
opts.dbname = args.dbname
opts.db_root = args.db_root

opts.data_opts = Edict()
opts.data_opts.phi_deg = args.phi_deg
opts.data_opts.num_invdepth = args.num_invdepth
opts.data_opts.equirect_size = args.equirect_size

opts.data_opts.use_rgb = args.use_rgb
opts.data_opts.ndepths = args.ndepths

opts.net_opts = Edict()
opts.net_opts.base_channel = args.base_channel
opts.net_opts.num_invdepth = opts.data_opts.num_invdepth
opts.net_opts.use_rgb = opts.data_opts.use_rgb

opts.net_opts.mixed_precision = args.mixed_precision
opts.net_opts.fix_bn = args.fix_bn
opts.net_opts.ndepths = opts.data_opts.ndepths
opts.net_opts.equirect_size = args.equirect_size
opts.net_opts.phi_deg = opts.data_opts.phi_deg

opts.net_opts.sigmoid_param = args.sigmoid_param
opts.net_opts.threshold = args.threshold

opts.total_epochs = args.total_epochs
opts.batch_size = args.batch_size

opts.lr = args.lr
opts.wdecay = args.wdecay


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr_schedule_with_warmup(optimizer, num_warmup_steps, total_steps, min_lr, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            lr_weight = min_lr + (1. - min_lr) * 0.5 * (
                        1. + math.cos(math.pi * (current_step - num_warmup_steps) / (total_steps - num_warmup_steps)))
        return lr_weight

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def fetch_optimizer(model, train_data_num):
    optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.wdecay, eps=1e-3)

    scheduler = get_lr_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, min_lr=args.min_lr,
                                            total_steps=train_data_num * opts.total_epochs)

    return optimizer, scheduler


def unpack_data_blob(data_blob):
    if len(data_blob) == 4:
        return data_blob
    if len(data_blob) >= 6:
        return data_blob[:4]
    raise ValueError("Unexpected batch format")


def build_eval_list(data):
    if hasattr(data.opts, 'test_sequences'):
        eval_list = []
        for seq_name, frame_ids in data.opts.test_sequences.items():
            for frame_id in frame_ids:
                eval_list.append((seq_name, frame_id))
        return eval_list
    return data.opts.test_idx


def train(epoch_total, load_state):
    if len(opts.dbname) > 1:
        if any(is_omnipus_dbname(dbname) for dbname in opts.dbname):
            raise ValueError("MultiDataset currently supports synthetic datasets only.")
        data = MultiDataset(opts.dbname, opts.data_opts, db_root=opts.db_root)
    else:
        dbname = opts.dbname[0]
        if is_omnipus_dbname(dbname):
            data = DatasetOmniPus(dbname, opts.data_opts, db_root=opts.db_root)
        else:
            data = Dataset(dbname, opts.data_opts, db_root=opts.db_root)
    dbloader = torch.utils.data.DataLoader(data, batch_size=opts.batch_size,
                                           pin_memory=True, shuffle=True,
                                           num_workers=4, drop_last=True)
    ocams = data.ocams

    net = nn.DataParallel(MDP_Omni(ocams, opts.net_opts)).cuda()

    if opts.net_opts.fix_bn:
        net.module.freeze_bn()
    LOG_INFO("Parameter Count: %d" % count_parameters(net))

    optimizer, scheduler = fetch_optimizer(net, len(data))
    scaler = GradScaler(enabled=opts.net_opts.mixed_precision)

    start_epoch = 0
    if load_state:
        if opts.snapshot_path and osp.exists(opts.snapshot_path):
            snapshot = torch.load(opts.snapshot_path)
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.snapshot_path))
            if 'epoch' in snapshot.keys():
                start_epoch = snapshot['epoch'] + 1
            if 'epoch_loss' in snapshot.keys():
                epoch_loss = snapshot['epoch_loss']
            if 'optimizer' in snapshot.keys():
                optimizer.load_state_dict(snapshot['optimizer'])
            if 'epoch' in snapshot.keys() and 'epoch_loss' in snapshot.keys():
                LOG_INFO('startepoch:%d epoch_loss:%f' % (start_epoch, epoch_loss))
        elif opts.pretrain_path is None:
            sys.exit('%s do not exsits' % (opts.snapshot_path))

        if opts.pretrain_path and osp.exists(opts.pretrain_path):
            snapshot = torch.load(opts.pretrain_path)
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.pretrain_path))
        elif opts.snapshot_path is None:
            sys.exit('%s do not exsits' % (opts.snapshot_path))

    invdepth_hypo_init = torch.tensor(data.get_invdepths(), requires_grad=False).unsqueeze(0)

    if not osp.exists(opts.model_dir):
        os.makedirs(opts.model_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.model_dir))
    if not osp.exists(opts.runs_dir):
        os.makedirs(opts.runs_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.runs_dir))
    writer = SummaryWriter(log_dir=opts.runs_dir)

    total_iters = len(data) * start_epoch // opts.batch_size

    for epoch in range(start_epoch, epoch_total):
        # training
        net.train()
        train_loss = 0
        epoch_loss = 0
        LOG_INFO('\nEpoch: %d' % epoch)

        for step, data_blob in enumerate(dbloader):
            start_time = time.time()
            imgs, gt_idx, gt_valid, raw_imgs = unpack_data_blob(data_blob)

            imgs = [img.cuda() for img in imgs]
            optimizer.zero_grad()
            imgs = torch.stack(imgs, dim=1)

            outputs = net(imgs, invdepth_hypo_init)

            loss = compute_loss(outputs, gt_idx, gt_valid, args.ndepths)

            train_loss += loss.data
            epoch_loss = train_loss / (step + 1)

            if step % 1000 == 0:
                LOG_INFO("Iter %d training loss = %.3f, average training loss for every step = %.3f, \
                                    time = %.2f" % (total_iters, loss, epoch_loss, time.time() - start_time))
                
                hypos = outputs["hypos"]
                prob_grids = outputs["prob_grids"]
                intervals = outputs["intervals"]

                final_prob = prob_grids[2]
                final_hypo = hypos[2]
                pred_idx = torch.sum(final_prob * final_hypo, dim=2)
     
                # logging
                writer.add_scalar("train/epoch_loss", epoch_loss, total_iters)
                pred_invdepth = data.indexToInvdepth(toNumpy(pred_idx.squeeze(1)))
                raw_imgs = [toNumpy(raw[0]) for raw in raw_imgs]
                vis_img = data.makeVisImage(raw_imgs, pred_invdepth, gt=gt_idx)
                writer.add_image("train/vis", vis_img.transpose(2, 0, 1), total_iters)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_iters += 1

        # evaluation
        net.eval()
        eval_list = build_eval_list(data)
        errors = np.zeros((len(eval_list), 5))
        for d in range(len(eval_list)):
            eval_item = eval_list[d]
            if isinstance(eval_item, tuple):
                seq_name, frame_id = eval_item
                imgs, gt_idx, gt_valid, raw_imgs = data.loadSample(seq_name, frame_id)
            else:
                imgs, gt_idx, gt_valid, raw_imgs = data.loadSample(eval_item)
            imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]
            imgs = torch.stack(imgs, dim=1)

            with torch.no_grad():
                outputs = net(imgs, invdepth_hypo_init)

            hypos = outputs["hypos"]
            prob_grids = outputs["prob_grids"]

            final_prob = prob_grids[2]
            final_hypo = hypos[2]
            pred_idx = torch.sum(final_prob * final_hypo, dim=2).squeeze(1)

            errors[d, :] = data.evalError(toNumpy(pred_idx), gt_idx, gt_valid)

        # logging
        pred_invdepth = data.indexToInvdepth(toNumpy(pred_idx))

        raw_imgs = [toNumpy(raw) for raw in raw_imgs]
        vis_img = data.makeVisImage(raw_imgs, toNumpy(pred_invdepth), gt=gt_idx)
        writer.add_image("val/vis", vis_img.transpose(2, 0, 1), total_iters)

        mean_errors = errors.mean(axis=0)
        writer.add_scalar("val/>1", mean_errors[0], total_iters)
        writer.add_scalar("val/>3", mean_errors[1], total_iters)
        writer.add_scalar("val/>5", mean_errors[2], total_iters)
        writer.add_scalar("val/MAE", mean_errors[3], total_iters)
        writer.add_scalar("val/RMS", mean_errors[4], total_iters)
        LOG_INFO('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f' %
                 (mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4]))

        with open(osp.join(opts.model_dir, 'valid_record.txt'), 'a') as valid_record:
            valid_record.write("epoch: %d\n" % epoch)
            valid_record.write('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f\n' %
                               (mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4]))
            valid_record.write("================================================\n")

        # save
        savefilename = opts.model_dir + '/%s_e%d.pth' % (opts.name, epoch)
        torch.save({
            'net_state_dict': net.state_dict(),
            'net_opts': opts.net_opts,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'epoch_loss': epoch_loss,
        }, savefilename)


def main():
    load_state = opts.snapshot_path is not None or opts.pretrain_path is not None
    train(opts.total_epochs, load_state)


if __name__ == "__main__":
    main()
