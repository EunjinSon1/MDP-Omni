import torch
import torch.nn.functional as F


def compute_loss(outputs, gt_depth, gt_valid, ndepths):
    gt_depth = gt_depth.cuda()  
    gt_valid = gt_valid.cuda()

    hypos = outputs["hypos"]
    intervals = outputs["intervals"]
    prob_grids = outputs["prob_grids"]

    loss = []
    loss_level_weights = [1, 1]
    loss_level_reg_weights = [0.5, 0.7]

    levels = list(range(len(ndepths))) 

    ref_depth = gt_depth.unsqueeze(1) 
    ref_valid = gt_valid.unsqueeze(1)

    for level in levels:
        if level == 2: 
            final_prob = prob_grids[level]
            final_hypo = hypos[level]
            pred_idx = torch.sum(final_prob * final_hypo, dim=2).squeeze(1) 

            tmp_loss = F.smooth_l1_loss(pred_idx[gt_valid], gt_depth[gt_valid], reduction='none')
            loss.append(tmp_loss.mean())
            continue

        B, _, D, H, W = prob_grids[level].shape

        # Create gt labels
        unfold_kernel_size = int(2 ** (2 - level))
        assert unfold_kernel_size % 2 == 0 or unfold_kernel_size == 1
        unfolded_patch_depth = torch.nn.functional.unfold(ref_depth, unfold_kernel_size, dilation=1,
                                                          padding=0, stride=unfold_kernel_size)
        unfolded_patch_depth = unfolded_patch_depth.reshape(B, 1, unfold_kernel_size ** 2, H, W)

        # Approximate depth distribution from depth observations
        gt_occ_grid = torch.zeros_like(hypos[level])

        for pixel in range(unfolded_patch_depth.shape[2]):
            selected_depth = unfolded_patch_depth[:, :, pixel]
            distance_to_hypo = abs(hypos[level] - selected_depth.unsqueeze(2))

            if level == 0:
                distance_to_hypo /= intervals[level]
            else:
                diff = (hypos[level][:, :, :-1] <= selected_depth.unsqueeze(2)) & (
                        selected_depth.unsqueeze(2) < hypos[level][:, :, 1:])
                hypo_intervals = torch.where(diff, intervals[level].unsqueeze(1),
                                             torch.zeros_like(intervals[level].unsqueeze(1)))
                hypo_intervals = hypo_intervals.sum(dim=2, keepdim=True)
                distance_to_hypo /= hypo_intervals

            mask = distance_to_hypo > 1
            weights = 1 - distance_to_hypo
            level_mask = selected_depth.unsqueeze(2).expand(-1, -1, D, -1, -1)
            weights[mask] = 0
            weights[level_mask == -1] = 0
            gt_occ_grid += weights
        gt_occ_grid = gt_occ_grid / gt_occ_grid.sum(dim=2, keepdim=True)
        gt_occ_grid[torch.isnan(gt_occ_grid)] = 0 

        covered_mask = gt_occ_grid.sum(dim=2, keepdim=True) > 0 
        covered_mask_reg = covered_mask.squeeze(1).squeeze(1) 

        # SmoothL1 Loss
        level_prob = prob_grids[level]
        level_hypo = hypos[level]  

        level_idx = torch.sum(level_prob * level_hypo, dim=2).squeeze(1) 
        gt_idx = torch.sum(level_hypo * gt_occ_grid, dim=2).squeeze(1) 
        reg_loss = F.smooth_l1_loss(level_idx[covered_mask_reg], gt_idx[covered_mask_reg], reduction='none').mean()

        # KL Loss
        covered_mask = covered_mask.expand(-1, -1, D, -1, -1)
        gt_occ_grid[gt_occ_grid == 0] = 1e-8
        level_prob = torch.clamp(level_prob, min=1e-8)
        kl_loss = gt_occ_grid[covered_mask] * (
                    gt_occ_grid[covered_mask].log() - level_prob[covered_mask].log()) 
        kl_loss = kl_loss.mean()

        tmp_loss = loss_level_reg_weights[level] * reg_loss + kl_loss
        loss.append(tmp_loss)

    loss = torch.stack(loss).sum()

    return loss