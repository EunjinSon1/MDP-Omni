import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.common import *
from module.modules import *
from module.featurelayer import FeatureLayers

class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


class MDP_Omni(torch.nn.Module):
    def __init__(self, ocams, varargin=None):
        super(MDP_Omni, self).__init__()
        opts = Edict()
        self.opts = argparse(opts, varargin)
        self.mf = self.opts.mixed_precision

        self.equi_size = self.opts.equirect_size
        self.ndepths = self.opts.ndepths
        self.sigmoid_param = self.opts.sigmoid_param
        self.threshold = self.opts.threshold
        self.phi_deg = self.opts.phi_deg

        self.ocams = ocams

        self.encoder = FeatureLayers(self.opts.base_channel, self.opts.use_rgb)

        self.cost_reg_net0 = CostRegNet(self.opts.base_channel, self.opts.base_channel)
        self.cost_reg_net1 = CostRegNet(self.opts.base_channel, self.opts.base_channel)
        self.cost_reg_net2 = CostRegNet(self.opts.base_channel, self.opts.base_channel)

    def forward(self, imgs, disp_hypo_init):
        B, V, C, H, W = imgs.shape 

        with torch.cuda.amp.autocast(enabled=self.mf):
            features = self.encoder(imgs)

        outputs = {}

        hypos = [None] * (len(self.ndepths))
        intervals = [None] * (len(self.ndepths))
        prob_grids = [None] * (len(self.ndepths))
        grob_grids = [None] * (len(self.ndepths))

        for level in range(len(self.ndepths)): 
            features_stage = features
            H_p, W_p = self.equi_size[0] // 2 ** (2 - level), self.equi_size[1] // 2 ** (2 - level)

            if level == 0:
                index_hypos, hypo_intervals = calculate_index_hypothesis_init(B,H_p, W_p,self.ndepths[level],imgs.device)
                hypos[level] = index_hypos
                intervals[level] = hypo_intervals

                #### Build cost volume ####
                cost_volume,position3d = proj_cost(features_stage,
                                                    self.sigmoid_param,
                                                    self.ocams,
                                                    hypos[level],
                                                    B,H_p, W_p,
                                                    disp_hypo_init,
                                                    self.phi_deg)
                #### Cost regularization ####
                occ_grid = self.cost_reg_net0(cost_volume,position3d)
                grob_grids[level] = occ_grid
                occ_grid = torch.softmax(occ_grid, dim=2)
                prob_grids[level] = occ_grid 

            ########## Depth refinement levels ##########
            else:
                #### Make depth hypothesis ####
                with torch.no_grad():
                    mask = prob_grids[level - 1] > self.threshold 
                    empty_mask = ~mask.any(dim=2, keepdim=True)
                   
                    depth_indices = torch.arange(self.ndepths[level - 1], device=mask.device).view(1,1,self.ndepths[level-1],1,1).float()
                    masked_indices = mask * depth_indices

                    min_indices = torch.where(mask, masked_indices, float('inf')).min(dim=2).values
                    max_indices = torch.where(mask, masked_indices, float('-inf')).max(dim=2).values

                    sample_count = mask.sum(dim=2).unsqueeze(1)
                    single_sample_mask = sample_count == 1 

                    min_indices = (min_indices - 1).clamp(min=0)  
                    max_indices = (max_indices + 1).clamp(max=self.ndepths[level - 1] - 1) 

                    selected_min_hypos = torch.gather(hypos[level - 1], dim=2, index=min_indices.unsqueeze(1).to(dtype=torch.int64))
                    selected_max_hypos = torch.gather(hypos[level - 1], dim=2, index=max_indices.unsqueeze(1).to(dtype=torch.int64))
                    prev_min_hypos = hypos[level - 1][:, :, :1, :, :]
                    prev_max_hypos = hypos[level - 1][:, :, -1:, :, :]
                    selected_min_hypos = torch.where(empty_mask, prev_min_hypos, selected_min_hypos)
                    selected_max_hypos = torch.where(empty_mask, prev_max_hypos, selected_max_hypos)

                    center_hyos = selected_min_hypos
                    adjustment = (self.ndepths[level] - 1) // 2

                    min_hypos_adjusted = (center_hyos - adjustment)
                    max_hypos_adjusted = (center_hyos + adjustment)
                    min_hypos_adjusted = min_hypos_adjusted.clamp(min=0)
                    max_hypos_adjusted = max_hypos_adjusted.clamp(max=191)

                    selected_min_hypos = torch.where(single_sample_mask, min_hypos_adjusted, selected_min_hypos)
                    selected_max_hypos = torch.where(single_sample_mask, max_hypos_adjusted, selected_max_hypos)

                    selected_hypos = torch.cat([selected_min_hypos, selected_max_hypos], dim=2)
                    selected_hypos = torch.repeat_interleave(selected_hypos, 2, dim=3)
                    selected_hypos = torch.repeat_interleave(selected_hypos, 2, dim=4)

                    new_interval = (selected_hypos[:, :, 1, :, :] - selected_hypos[:, :, 0, :, :]) / (self.ndepths[level] - 1) 

                    unifrom_hypos = selected_hypos[:, :, 0, :, :] + (
                            torch.arange(0, self.ndepths[level], device=selected_hypos.device,
                                         dtype=selected_hypos.dtype,
                                         requires_grad=False).reshape(1, -1, 1, 1) * new_interval)

                    hypos[level] = unifrom_hypos.unsqueeze(1)
                    intervals[level] = new_interval 

                #### Build cost volume ####
                cost_volume, position3d = proj_cost(features_stage,
                                                    self.sigmoid_param,
                                                    self.ocams,
                                                    hypos[level],
                                                    B,H_p, W_p,
                                                    disp_hypo_init,
                                                    self.phi_deg)

                #### Cost regularization ####
                if level == 1:
                    occ_grid = self.cost_reg_net1(cost_volume,position3d)
                elif level == 2:
                    occ_grid = self.cost_reg_net2(cost_volume,position3d)

                grob_grids[level] = occ_grid
                occ_grid = torch.softmax(occ_grid, dim=2)
                prob_grids[level] = occ_grid
    
        ## Return
        outputs["hypos"] = hypos
        outputs["intervals"] = intervals
        outputs["prob_grids"] = prob_grids

        return outputs
