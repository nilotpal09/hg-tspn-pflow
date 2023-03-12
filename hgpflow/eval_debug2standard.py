import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment

import ROOT
import uproot



THRESHOLD = 0.6
# THRESHOLD_X = np.array([    1.5*1e3,    1.75*1e3,   ])
# THRESHOLD_Y = np.array([0.1,         0.65,      0.88])

PT_H_WT_CH, DR_H_WT_CH = 0, 5
PT_H_WT_NEUT, DR_H_WT_NEUT = 1, 5


def deltaR(eta1, phi1, eta2, phi2):
    d_eta = eta1 - eta2
    phi1, phi2 = (phi1+np.pi) % (2*np.pi) - np.pi, (phi2+np.pi) % (2*np.pi) - np.pi
    d_phi = torch.min(torch.abs(phi1 - phi2), 2*np.pi - torch.abs(phi1 - phi2))
    dR = torch.sqrt(d_eta**2 + d_phi**2)
    return dR


def main():

    data_file_path  = sys.argv[1]
    debug_file_path = sys.argv[2]

    standard_file_path = ''.join(debug_file_path.split('_debug_mode'))
    print('\nstandard pflow file will saved in -\n', standard_file_path, '\n')

    truth_tree = uproot.open(data_file_path)['Low_Tree']
    debug_tree = uproot.open(debug_file_path)['pflow_tree']

    # check the number of events in both files
    n_data  = truth_tree.num_entries
    n_debug = debug_tree.num_entries

    if n_debug < n_data:
        print('\033[96m' + 'data file and debug file have differnt number of events' + '\033[0m')
        print(f'n_data: {n_data}, n_debug: {n_debug}')
        n_events = n_debug

    elif n_debug > n_data:
        raise ValueError(f'More events in debug file {n_debug} then in the data file {n_data}')

    else:
        print(f'n_data: {n_data}, n_debug: {n_debug}')
        n_events = n_debug



    #
    # read the trees
    #----------------------------

    particle_to_track = truth_tree["particle_to_track"].array(library='np', entry_stop=n_events)

    truth_pt  = debug_tree['truth_pt'].array(library='np', entry_stop=n_events)
    truth_eta = debug_tree['truth_eta'].array(library='np', entry_stop=n_events)
    truth_phi = debug_tree['truth_phi'].array(library='np', entry_stop=n_events)

    debug_pt    = debug_tree['pflow_pt'].array(library='np', entry_stop=n_events)
    debug_eta   = debug_tree['pflow_eta'].array(library='np', entry_stop=n_events)
    debug_phi   = debug_tree['pflow_phi'].array(library='np', entry_stop=n_events)
    debug_class = debug_tree['pflow_class'].array(library='np', entry_stop=n_events)

    pred_inc = debug_tree['pred_inc'].array(library='np', entry_stop=n_events)





    #
    # prep the standard tree
    #----------------------------

    outputfile = ROOT.TFile(standard_file_path,'recreate')
    outputtree = ROOT.TTree('pflow_tree','pflow_tree')

    pflow_class_vec = ROOT.vector('int')()
    outputtree.Branch('pflow_class',pflow_class_vec)

    pflow_pt_vec = ROOT.vector('float')()
    outputtree.Branch('pflow_pt',pflow_pt_vec)

    pflow_eta_vec = ROOT.vector('float')()
    outputtree.Branch('pflow_eta',pflow_eta_vec)

    pflow_phi_vec = ROOT.vector('float')()
    outputtree.Branch('pflow_phi',pflow_phi_vec)

    pflow_target_vec = ROOT.vector('int')()
    outputtree.Branch('pflow_target',pflow_target_vec)




    for i in range(n_events):

        # clear the vectors
        pflow_pt_vec.clear(); pflow_eta_vec.clear(); pflow_phi_vec.clear(); pflow_class_vec.clear();
        pflow_target_vec.clear()


        # process the event

        n_truth = particle_to_track[i].shape[0]

        truth_ch_mask   = particle_to_track[i] != -1
        truth_neut_mask = particle_to_track[i] == -1

        debug_ch_mask   = (debug_class[i] == 2) + (debug_class[i] == 3) + (debug_class[i] == 4)

        # fixed threshold
        debug_neut_mask = (debug_class[i] != 2) * (debug_class[i] != 3) * (debug_class[i] != 4) * (np.array(pred_inc[i])[:,-1] > THRESHOLD)

        # # pt dep threshold
        # debug_neut_mask = (debug_class[i] != 2) * (debug_class[i] != 3) * (debug_class[i] != 4)
        # th_neut_mask    = np.array(pred_inc[i])[:,-1] > THRESHOLD_Y[np.searchsorted(THRESHOLD_X, debug_pt[i])]
        # debug_neut_mask = debug_neut_mask * th_neut_mask

        n_truth_ch = truth_ch_mask.sum()
        n_debug_ch = debug_ch_mask.sum()
        n_truth_neut = truth_neut_mask.sum()
        n_debug_neut = debug_neut_mask.sum()





        #*********#
        # charged #
        #*********#

        truth_ch_pt  = torch.FloatTensor(truth_pt[i][:n_truth][truth_ch_mask])
        truth_ch_eta = torch.FloatTensor(truth_eta[i][:n_truth][truth_ch_mask])
        truth_ch_phi = torch.FloatTensor(truth_phi[i][:n_truth][truth_ch_mask])

        debug_ch_pt    = torch.FloatTensor(debug_pt[i][debug_ch_mask])
        debug_ch_eta   = torch.FloatTensor(debug_eta[i][debug_ch_mask])
        debug_ch_phi   = torch.FloatTensor(debug_phi[i][debug_ch_mask])
        debug_ch_class = torch.IntTensor(debug_class[i][debug_ch_mask])

        # Hungarian
        pdist_pt = F.mse_loss(
            (debug_ch_pt.unsqueeze(1)).unsqueeze(0).expand(truth_ch_pt.size(0), -1, -1),
            (truth_ch_pt.unsqueeze(1)).unsqueeze(1).expand(-1, debug_ch_pt.size(0), -1),
            reduction='none')

        pdist_deltaR = deltaR(
            (debug_ch_eta.unsqueeze(1)).unsqueeze(0).expand(truth_ch_eta.size(0), -1, -1),
            (debug_ch_phi.unsqueeze(1)).unsqueeze(0).expand(truth_ch_phi.size(0), -1, -1),
            (truth_ch_eta.unsqueeze(1)).unsqueeze(1).expand(-1, debug_ch_eta.size(0), -1),
            (truth_ch_phi.unsqueeze(1)).unsqueeze(1).expand(-1, debug_ch_phi.size(0), -1))

        pdist = torch.cat([PT_H_WT_CH*pdist_pt, DR_H_WT_CH*pdist_deltaR], dim=-1)
        pdist = pdist.mean(2)

        arange_ch, indices_ch = linear_sum_assignment(pdist)
        arange_ch, indices_ch = arange_ch.astype(int), indices_ch.astype(int)

        
        # write the trees (charged)
        absolute_ind = np.where(truth_ch_mask)[0]

        # write the trees (neutral)
        target_ch = []
        for x in range(n_debug_ch):
            if x not in indices_ch:
                target_ch.append(-1)
                # should not happen
                raise ValueError("found a charged particle w/o any match")
            else:
                ind = np.where(indices_ch==x)[0][0]
                target_ch.append(absolute_ind[arange_ch[ind]].item())

        for p_i in range(debug_ch_mask.sum()):
            pflow_pt_vec.push_back(debug_ch_pt[p_i].item())
            pflow_eta_vec.push_back(debug_ch_eta[p_i].item())
            pflow_phi_vec.push_back(debug_ch_phi[p_i].item())
            pflow_class_vec.push_back(debug_ch_class[p_i].item())

            pflow_target_vec.push_back(target_ch[p_i])
            



        #*********#
        # neutral #
        #*********#

        truth_neut_pt  = torch.FloatTensor(truth_pt[i][:n_truth][truth_neut_mask])
        truth_neut_eta = torch.FloatTensor(truth_eta[i][:n_truth][truth_neut_mask])
        truth_neut_phi = torch.FloatTensor(truth_phi[i][:n_truth][truth_neut_mask])

        debug_neut_pt    = torch.FloatTensor(debug_pt[i][debug_neut_mask])
        debug_neut_eta   = torch.FloatTensor(debug_eta[i][debug_neut_mask])
        debug_neut_phi   = torch.FloatTensor(debug_phi[i][debug_neut_mask])
        debug_neut_class = torch.IntTensor(debug_class[i][debug_neut_mask])

        # Hungarian
        pdist_delpt_sq = F.mse_loss(
            (debug_neut_pt.unsqueeze(1)).unsqueeze(0).expand(truth_neut_pt.size(0), -1, -1),
            (truth_neut_pt.unsqueeze(1)).unsqueeze(1).expand(-1, debug_neut_pt.size(0), -1),
            reduction='none')

        pdist_deltaR = deltaR(
            (debug_neut_eta.unsqueeze(1)).unsqueeze(0).expand(truth_neut_eta.size(0), -1, -1),
            (debug_neut_phi.unsqueeze(1)).unsqueeze(0).expand(truth_neut_phi.size(0), -1, -1),
            (truth_neut_eta.unsqueeze(1)).unsqueeze(1).expand(-1, debug_neut_eta.size(0), -1),
            (truth_neut_phi.unsqueeze(1)).unsqueeze(1).expand(-1, debug_neut_phi.size(0), -1))


        # sqrt((delpT/pT)^2 + dR^2)
        pdist_truth_pt = (truth_neut_pt.unsqueeze(1)).unsqueeze(1).expand(-1, debug_neut_pt.size(0), -1)
        pdist = torch.sqrt(pdist_delpt_sq / pdist_truth_pt**2 + pdist_deltaR ** 2)
        pdist = pdist.squeeze(-1)


        # old matching
        # pdist = torch.cat([PT_H_WT_NEUT*pdist_delpt_sq, DR_H_WT_NEUT*pdist_deltaR], dim=-1)
        # pdist = pdist.mean(2)


        arange_neut, indices_neut = linear_sum_assignment(pdist)
        # arange_neut, indices_neut = arange_neut.astype(int), indices_neut.astype(int)


        absolute_ind = np.where(truth_neut_mask)[0]

        # write the trees (neutral)
        target_neut = []
        for x in range(n_debug_neut):
            if x not in indices_neut:
                target_neut.append(-1)
            else:
                ind = np.where(indices_neut==x)[0][0]
                target_neut.append(absolute_ind[arange_neut[ind]].item())

        for p_i in range(debug_neut_mask.sum()):
            pflow_pt_vec.push_back(debug_neut_pt[p_i].item())
            pflow_eta_vec.push_back(debug_neut_eta[p_i].item())
            pflow_phi_vec.push_back(debug_neut_phi[p_i].item())
            pflow_class_vec.push_back(debug_neut_class[p_i].item())

            pflow_target_vec.push_back(target_neut[p_i])



        outputtree.Fill()

    outputfile.cd()
    outputtree.Write()
    outputfile.Close()



if __name__ == '__main__':
    main()

