import uproot as uproot
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


copied from v6 and not updated yet

COSINE_LOSS_WT = 4

file_path = '/storage/agrp/nilotpal/PFlow/evaluation/hypergraph/data_single_jet_fiducial_TESTskim_v6_debug_mode2.root'
branches2read = [
    'pflow_class', 'pflow_pt', 'pflow_eta', 'pflow_phi', 
    'truth_class', 'truth_pt', 'truth_eta', 'truth_phi', 
    'truth_inc', 'pred_inc'
]

entrystop = None # 1000

f = uproot.open(file_path)
tree = f['pflow_tree']

if entrystop != None:
    numentries = min(entrystop, tree.num_entries)
else:
    numentries = tree.num_entries


full_data_array = {}
for branch in branches2read:
    full_data_array[branch] = tree[branch].array(library='np', entry_stop=entrystop)


losses = {}; delta_cardinality = {}
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:

    losses[threshold] = []; delta_cardinality[threshold] = []

    for event_idx in range(numentries):

        truth_ind = np.array(full_data_array['truth_inc'][event_idx].tolist())[:,-1]
        pred_ind  = np.array(full_data_array['pred_inc'][event_idx].tolist())[:,-1]

        truth_ptetaphi = np.vstack([
            np.array(full_data_array['truth_pt'][event_idx]),
            np.array(full_data_array['truth_eta'][event_idx]),
            np.array(full_data_array['truth_phi'][event_idx])
        ]).T

        pred_ptetaphi = np.vstack([
            np.array(full_data_array['pflow_pt'][event_idx]),
            np.array(full_data_array['pflow_eta'][event_idx]),
            np.array(full_data_array['pflow_phi'][event_idx])
        ]).T

        truth_ptetaphi = torch.FloatTensor(truth_ptetaphi[truth_ind == 1])
        pred_ptetaphi  = torch.FloatTensor(pred_ptetaphi[pred_ind > threshold])

        pdist_pteta = F.mse_loss(
            pred_ptetaphi[:,:2].unsqueeze(0).expand(truth_ptetaphi.size(0), -1, -1), 
            truth_ptetaphi[:,:2].unsqueeze(1).expand(-1, pred_ptetaphi.size(0), -1),
            reduction='none')    

        pdist_phi = torch.cos(
            (pred_ptetaphi[:,2].unsqueeze(1)).unsqueeze(0).expand(truth_ptetaphi.size(0), -1, -1) -
            (truth_ptetaphi[:,2].unsqueeze(1)).unsqueeze(1).expand(-1, pred_ptetaphi.size(0), -1))

        pdist = torch.cat([pdist_pteta, COSINE_LOSS_WT * (1 - pdist_phi)], dim=-1).cpu()

        x, y, z = pdist.shape
        if x < y:
            delx, dely, delz = y-x, 0, 0
        elif x > y:
            delx, dely, delz = 0, x-y, 0
        else:
            delx, dely, delz = 0, 0, 0

        delta_cardinality[threshold].append(x - y)

        # pdist = np.pad(pdist, ((0,delx), (0,dely), (0,0)), 'constant', constant_values=0).mean(2)
        # indices = linear_sum_assignment(pdist)[1].astype(int)

        # pred_ptetaphi = pred_ptetaphi[indices]

        # loss = torch.cat([
        #     F.mse_loss(pred_ptetaphi[:,:2], truth_ptetaphi[:,:2]),
        #     torch.cos(pred_ptetaphi[:,2] - truth_ptetaphi[:,2]).unsqueeze(-1)
        # ], dim=1)
        # loss = loss.mean(dim=-1).mean()

        # losses[threshold].append(loss.item())

yval = [np.array(delta_cardinality[x]).mean() for x in thresholds]
yerr = [np.array(delta_cardinality[x]).std() for x in thresholds]
plt.errorbar(thresholds, yval, yerr = yerr)
plt.plot(thresholds, np.zeros_like(thresholds))
plt.ylabel('delta cardinality (truth - reco)')
plt.xlabel('thresolds')
plt.savefig('delta_cardinality_th_scan.png')
