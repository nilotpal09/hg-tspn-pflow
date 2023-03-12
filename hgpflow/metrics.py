import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import gc

EPS              = 1e-8
PARTICLE_PHI_STD = 1
PARTICLE_PT_STD  = 1
COSINE_LOSS_WT   = 1
HYPEREDGE_LOSS_WT = 1

INC_POS_WT       = 1

IND_LOSS_WT      = 1
INC_LOSS_WT      = 1
INC_NONZERO_WT   = 1

MIX_LOSS_WT      = 1

INC_CONTWT_X = None
INC_CONTWT_Y = None


def update_global_vars(config):
    global PARTICLE_PHI_STD, PARTICLE_PT_STD, INC_POS_WT, COSINE_LOSS_WT, HYPEREDGE_LOSS_WT
    global IND_LOSS_WT, INC_LOSS_WT, INC_NONZERO_WT
    global MIX_LOSS_WT, INC_CONTWT_X, INC_CONTWT_Y

    PARTICLE_PHI_STD = config['var transform']['particle_phi']['std']
    PARTICLE_PT_STD  = config['var transform']['particle_pt']['std']
    COSINE_LOSS_WT   = config['cosine_loss_wt']
    HYPEREDGE_LOSS_WT= config['hyperedge_loss_wt']

    MIX_LOSS_WT      = config['mix_loss_wt']

    if config['bool_inc']:
        INC_POS_WT  = config['inc_pos_wt']
    else:
        IND_LOSS_WT = config['ind_loss_wt']
        INC_LOSS_WT = config['inc_loss_wt']
        INC_NONZERO_WT = config['inc_nonzero_wt']
        
        fnp = np.load(config['path_to_inc_dist'])
        xs = 0.5 * (fnp['arr_0'][1:] + fnp['arr_0'][:-1])
        INC_CONTWT_X = torch.FloatTensor(xs[:-1])
        INC_CONTWT_Y = torch.FloatTensor(120_000 * 1/fnp['arr_1'])
        INC_CONTWT_Y[0] = 1.0

        if torch.cuda.is_available():
            INC_CONTWT_X = INC_CONTWT_X.cuda()
            INC_CONTWT_Y = INC_CONTWT_Y.cuda()

def l_split_ind(l, n):
    r = l%n
    return np.cumsum([0] + [l//n+1]*r + [l//n]*(n-r))

@ray.remote
def lsa(arr, s, e):
    return np.array([linear_sum_assignment(p) for p in arr[s:e]])

def ray_lsa(arr, n):
    l = arr.shape[0]
    ind = l_split_ind(l, n)
    arr_id = ray.put(arr)
    res = [lsa.remote(arr_id, ind[i], ind[i+1]) for i in range(n)]
    res = np.concatenate([ray.get(r) for r in res])
    return res


def LAP_loss(input, target, has_track=None, n=0, get_assignment=False, assgn_particles=1, bool_inc=True, get_individual_losses=False):

    # has_track: [1,-1,1,...]

    if bool_inc:
        return LAP_loss_bool(
            input, target, has_track=has_track, n=n, get_assignment=get_assignment, assgn_particles=assgn_particles)
    else:
        return LAP_loss_cont(
            input, target, has_track=has_track, n=n, get_assignment=get_assignment, assgn_particles=assgn_particles, 
            get_individual_losses=get_individual_losses)

def LAP_loss_bool(input, target, has_track=None, n=0, get_assignment=False, assgn_particles=1):

    pdist = F.binary_cross_entropy(
        input.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, input.size(1), -1),
        reduction='none')

    # high weight for the entire indicator
    weights = target.clone()
    weights[:,:,-1] = 2
    weights = (weights * INC_POS_WT + 1).unsqueeze(2).expand(-1, -1, input.size(1), -1)

    pdist = pdist * weights
    pdist = pdist.mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    if n > 0:
        indices = ray_lsa(pdist_, n)
    else:
        indices = np.array([linear_sum_assignment(p) for p in pdist_])

    assignment_indices = indices

    indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
    total_loss = losses.mean(1)

    # memory cleanup
    del pdist, pdist_, indices, losses #, weights
    gc.collect()

    if get_assignment == True:
        return total_loss, assignment_indices[assgn_particles:,1,:]
    return total_loss

def LAP_loss_cont(input, target, has_track=None, n=0, get_assignment=False, assgn_particles=1, get_individual_losses=False):

    # incidence only
    pdist1 = F.kl_div(
        torch.log(input+EPS).unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, input.size(1), -1),
        reduction='none') + 0.5

    # indicator only
    pdist2 = F.binary_cross_entropy(
        input.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, input.size(1), -1),
        reduction='none')

    # input, target gets 4 copies together during training
    has_track = has_track.repeat(input.size(0)//has_track.size(0),1)

    # make the mixing of charged and neutral particles VERY expensive
    has_track_pred = torch.zeros_like(has_track) - 1
    has_track_count = (has_track.size(1) + has_track.sum(dim=1)) / 2
    for i, n_track in enumerate(has_track_count):
        has_track_pred[i, :n_track.int()] = 1

    # make has_track of the shape input.shape
    mask_mix = has_track_pred.unsqueeze(-1).repeat(1, 1, input.size(2)).unsqueeze(1).expand(-1, target.size(1), -1, -1) \
               * has_track.unsqueeze(-1).repeat(1, 1, input.size(2)).unsqueeze(2).expand(-1, -1, input.size(1), -1)
               # 1 when same, -1 when different; 1-> 1, -1-> MIX_LOSS_WT 
               # (x - 1) * (-0.5) * (MIX_LOSS_WT - 1) + 1
    mask_mix = (mask_mix - 1) * (-0.5) * (MIX_LOSS_WT - 1) + 1

    # inc mask
    mask1 = torch.ones_like(input)
    mask1[:, :, -1] = 0
    mask1 = mask1.unsqueeze(1).expand(-1, target.size(1), -1, -1)

    # ind mask
    mask2 = torch.zeros_like(input)
    mask2[:, :, -1] = 1
    mask2 = mask2.unsqueeze(1).expand(-1, target.size(1), -1, -1)

    # continuos weight for inc
    cont_wt = INC_CONTWT_Y[torch.searchsorted(INC_CONTWT_X, target)]
    cont_wt = cont_wt.unsqueeze(2).expand(-1, -1, input.size(1), -1)

    pdist = INC_LOSS_WT * pdist1 * mask1 * cont_wt + IND_LOSS_WT * pdist2 * mask2
    pdist = pdist * mask_mix
    pdist = pdist.mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    if n > 0:
        indices = ray_lsa(pdist_, n)
    else:
        indices = np.array([linear_sum_assignment(p) for p in pdist_])

    assignment_indices = indices

    indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
    total_loss = losses.mean(1)

    # for book-keeping
    inc_losses = torch.gather((INC_LOSS_WT * pdist1 * mask1 * mask_mix).mean(3).flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
    inc_loss = inc_losses.mean(1)

    ind_losses = torch.gather((IND_LOSS_WT * pdist2 * mask2 * mask_mix).mean(3).flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
    ind_loss = ind_losses.mean(1)

    # memory cleanup
    del pdist, pdist_, indices, losses #, weights
    gc.collect()

    if get_individual_losses:
        total_loss = (total_loss, ind_loss, inc_loss)

    if get_assignment == True:
        return total_loss, assignment_indices[assgn_particles:,1,:]
    return total_loss


def _error_count_indicator(gt_inc, pred_inc, d):
    pred_m = pred_inc[...,-1] > 0.5
    gt_m = gt_inc[...,-1] > 0.5

    pred_inc = pred_inc[...,:-1].topk(d, dim=2, sorted=False)[1].sort()[0]
    gt_inc = gt_inc[...,:-1].topk(d, dim=2, sorted=False)[1].sort()[0]

    # batch x edge_pred x edge_gt
    eq = (pred_inc.unsqueeze(2) == gt_inc.unsqueeze(1)).all(3)
    eq = eq * pred_m.unsqueeze(2) * gt_m.unsqueeze(1)
    tp = eq.any(1).sum(1)  # count unique only
    fp = (pred_m * ~eq.any(2)).sum(1)
    fn = (gt_m * ~eq.any(1)).sum(1)
    return tp, fp, fn

def _triu_mean(x):
    if len(x.shape) < 3:
        x = x.unsqueeze(0)
    return x.triu(1).sum((1,2)) * 2. / (x.size(1) * (x.size(1)-1))
    
def _error_count_adj(gt_adj, pred_adj):
    pred_adj = pred_adj.clamp(0, 1)
    tp = _triu_mean(gt_adj * pred_adj)
    fp = _triu_mean((1 - gt_adj) * pred_adj)
    fn = _triu_mean(gt_adj * (1 - pred_adj))
    return tp, fp, fn

def error_count(type, gt, pred, **kwargs):
    assert type in ["adj", "ind"]
    if type == "adj":
        tp, fp, fn = _error_count_adj(gt, pred)
    else:
        tp, fp, fn = _error_count_indicator(gt, pred, kwargs.get("d_feats"))
    return tp, fp, fn

def precision(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    return tp / (tp + fp + EPS)

def recall(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    return tp / (tp + fn + EPS)

def f1_score(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
    return f1

def delaunay_adj_metrics(targ_adj, pred_adj, k=2):
    diag_mask = torch.eye(pred_adj.shape[2]).repeat(pred_adj.shape[0], 1, 1).bool()
    pred_adj = (pred_adj > 0.5).int()
    pred_adj[diag_mask] = 0

    tp = (targ_adj * pred_adj).sum((1,2)).float()
    tn = ((1-targ_adj) * (1-pred_adj)).sum((1,2)).float()
    fp = ((1-targ_adj) * pred_adj).sum((1,2)).float()
    fn = (targ_adj * (1-pred_adj)).sum((1,2)).float()
    
    acc = ((tp+tn) / (tp+tn+fp+fn))
    prec = (tp / (tp+fp+EPS))
    rec = (tp / (tp+fn+EPS))
    fone = 2*tp / (2*tp+fp+fn+EPS)
    return acc, fone, prec, rec

def mae_cardinality(pred, target):
    card_targ = (pred[:,:,-1]>0.5).sum(1).float()
    card_pred = (target[:,:,-1]>0.5).sum(1).float()
    return F.l1_loss(card_targ, card_pred)


# all the entries in the inputs are already rearranged according to the Hungarian matching
def hyperedge_loss(ptetaphi_truth, class_truth, regression_pred, class_pred, indicator=None, mask=None, class_weights=None):
    regression_loss_pt  = PARTICLE_PT_STD**2 * F.mse_loss(ptetaphi_truth[:,:,0], regression_pred[:,:,0], reduction='none').unsqueeze(-1)
    regression_loss_eta = F.mse_loss(ptetaphi_truth[:,:,1], regression_pred[:,:,1], reduction='none').unsqueeze(-1)
    regression_loss_phi = COSINE_LOSS_WT * (1 - torch.cos(PARTICLE_PHI_STD*(ptetaphi_truth[:,:,2] - regression_pred[:,:,2])).unsqueeze(-1))
    regression_loss = torch.cat([regression_loss_pt, regression_loss_eta, regression_loss_phi], dim=2)

    # setting the unwanted classes to zero, so that cross entropy won't break (eg. muon truth (2) for neutral prediction)
    # the loss will be made zero for these entries by the next "if mask is not None" clause
    if mask is not None:
        class_truth = class_truth * mask.squeeze(-1).long()

    classification_loss = F.cross_entropy(class_pred.reshape(-1, class_pred.shape[-1]), class_truth.reshape(-1), reduction='none')

    # garbage regression (don't regress) (n, 45, 3)
    default_value = 0
    non_garbage_mask = \
        (ptetaphi_truth[:,:,0]!=default_value) + (ptetaphi_truth[:,:,1]!=default_value) + (ptetaphi_truth[:,:,2]!=default_value)
    regression_loss = regression_loss * non_garbage_mask.unsqueeze(-1)
    classification_loss = classification_loss * non_garbage_mask.view(-1)

    if indicator is not None:
        regression_loss     = regression_loss * indicator.view(indicator.shape[0], indicator.shape[1], 1)
        classification_loss = classification_loss * indicator.view(-1)

    if class_weights is not None:
        weight_mask = torch.ones_like(class_truth)
        for idx, wt in enumerate(class_weights):
            weight_mask += (class_truth == idx) * wt
        classification_loss = classification_loss * weight_mask.view(-1)

    if mask is not None:
        regression_loss = regression_loss * mask
        classification_loss = classification_loss * mask.view(-1)

    return HYPEREDGE_LOSS_WT * (regression_loss.mean() + classification_loss.mean())



