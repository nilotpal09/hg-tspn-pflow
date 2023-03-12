import dgl
import dgl.function as fn

import sys
sys.path.append('./models/')

from lightning import PflowLightning
import sys
import os
import json
import torch
import numpy as np
from dataloader import PflowDataset, collate_graphs
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import pandas as pd
import os
from pathlib import Path
import ROOT

import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]


REDUCE_DS = -1

PT_H_WT, DR_H_WT, CL_H_WT, TR_H_WT = 1, 5, 1e2, 1e5

COSINE_LOSS_WT = 4
NUM_WORKERS = 2



PT_MEAN = 0; PT_STD = 1; ETA_MEAN = 0; ETA_STD = 1; PHI_MEAN = 0; PHI_STD = 1; E_MEAN = 0; E_STD = 1

# charged hadron : 0, electron : 1, muon : 2, neutral hadron: 3, photon: 4
# to 
# charged hadron : 2, electron : 3, muon : 4, neutral hadron: 1, photon: 0
class_swap_dict = {0:2, 1:3, 2:4, 3:1, 4:0}



def deltaR(eta1, phi1, eta2, phi2):
    d_eta = eta1 - eta2
    phi1, phi2 = (phi1+np.pi) % (2*np.pi) - np.pi, (phi2+np.pi) % (2*np.pi) - np.pi
    d_phi = torch.min(torch.abs(phi1 - phi2), 2*np.pi - torch.abs(phi1 - phi2))
    dR = torch.sqrt(d_eta**2 + d_phi**2)
    return dR


def main():

    print('\033[96m' + 'supports only batch_size = 1' + '\033[0m')
    print('\033[96m' + 'things may break with bs > 1, eg. node_energy"' + '\033[0m')

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    with open(config_path, 'r') as fp:
        config = json.load(fp)
       
    get_meanstd(config)

    outputdir = Path('/storage/agrp/nilotpal/PFlow/evaluation/hypergraph')
    outputdir.mkdir(parents=True, exist_ok=True)

    outputfilepath = os.path.join(
        outputdir, '{}_v{}_debug_mode.root'.format(
            '_'.join(config['path_to_test'].split('.')[0].split('/')[-2:]), config['version']))
    print('PFlow result will be stored in -\n', outputfilepath)

    outputfile = ROOT.TFile(outputfilepath,'recreate')
    outputtree = ROOT.TTree('pflow_tree','pflow_tree')

    pflow_class = ROOT.vector('int')()
    outputtree.Branch('pflow_class',pflow_class)

    pflow_pt = ROOT.vector('float')()
    outputtree.Branch('pflow_pt',pflow_pt)

    pflow_eta = ROOT.vector('float')()
    outputtree.Branch('pflow_eta',pflow_eta)

    pflow_phi = ROOT.vector('float')()
    outputtree.Branch('pflow_phi',pflow_phi)

    truth_pt = ROOT.vector('float')()
    outputtree.Branch('truth_pt',truth_pt)

    truth_eta = ROOT.vector('float')()
    outputtree.Branch('truth_eta',truth_eta)

    truth_phi = ROOT.vector('float')()
    outputtree.Branch('truth_phi',truth_phi)

    truth_class = ROOT.vector('float')()
    outputtree.Branch('truth_class',truth_class)

    truth_inc = ROOT.vector(ROOT.vector('float'))()
    outputtree.Branch('truth_inc',truth_inc)

    pred_inc = ROOT.vector(ROOT.vector('float'))()
    outputtree.Branch('pred_inc',pred_inc)

    truth_has_track = ROOT.vector('float')()
    outputtree.Branch('truth_has_track',truth_has_track)        



    node_energy = ROOT.vector('float')()
    outputtree.Branch('node_energy',node_energy)

    node_eta = ROOT.vector('float')()
    outputtree.Branch('node_eta',node_eta)

    node_phi = ROOT.vector('float')()
    outputtree.Branch('node_phi',node_phi)

    node_track_pt = ROOT.vector('float')()
    outputtree.Branch('node_track_pt',node_track_pt)

    node_is_track = ROOT.vector('int')()
    outputtree.Branch('node_is_track',node_is_track)



    net = PflowLightning(config)
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    dataset = PflowDataset(config['path_to_test'], config, reduce_ds=REDUCE_DS, bool_inc=config['bool_inc'], isEval=True)

    loader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS, 
        shuffle=False, collate_fn=collate_graphs)

    if torch.cuda.is_available():
        print('switching to gpu')
        net.net.cuda()
        net.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for g, inc_truth, truth_ptetaphi, truth_class_, has_track in tqdm(loader):

        # can only run with bs=1 :(
        g = g.to(device)
        if torch.cuda.is_available():
            inc_truth = inc_truth.cuda()

        inc_pred, predicted_particles = net.net.infer(g, debug=True)
        pt, eta, phi, p_class = predicted_particles.transpose(0,1)
       
        truth_ptetaphi = truth_ptetaphi.squeeze(0) # [0,:,:][mask]
        truth_ptetaphi = undo_scalling(truth_ptetaphi)
        truth_class_   = truth_class_.squeeze(0)

        has_track = has_track.squeeze(0)

        predicted_num_particles = [len(eta)]

        n_events = len(predicted_num_particles)
        particle_counter = -1

        pt  = np.exp(pt.cpu().float().data.numpy())
        eta = eta.cpu().float().data.numpy()
        phi = phi.cpu().float().data.numpy()
        p_class = p_class.cpu().int().data.numpy()

        p_class = np.array([class_swap_dict[x] for x in p_class])
        truth_class_ = torch.FloatTensor([class_swap_dict[x.item()] for x in truth_class_])

        # node info for debugging
        node_energies  = g.nodes['nodes'].data['energy'] * E_STD + E_MEAN
        node_etas      = g.nodes['nodes'].data['eta'] * ETA_STD + ETA_MEAN
        node_phis      = g.nodes['nodes'].data['phi'] * PHI_STD + PHI_MEAN
        node_track_pts = g.nodes['nodes'].data['track_pt'] * PT_STD + PT_MEAN
        node_is_tracks = g.nodes['nodes'].data['isTrack']

        for event_i in range(n_events):
            pflow_pt.clear(); pflow_eta.clear(); pflow_phi.clear(); pflow_class.clear(); 
            truth_pt.clear(); truth_eta.clear(); truth_phi.clear(); truth_class.clear(); 
            truth_inc.clear(); pred_inc.clear()
            node_energy.clear(); node_eta.clear(); node_phi.clear(); node_track_pt.clear(); 
            node_is_track.clear(); truth_has_track.clear();

            n_particles = predicted_num_particles[event_i]
    
            inc_truth = inc_truth[event_i]
            inc_truth = torch.cat([inc_truth, inc_truth.bool().any(-1, keepdim=True).float()], dim=-1)
            inc_pred  = inc_pred[event_i]

            for particle in range(int(n_particles)):
                particle_counter+=1
                
                pflow_class.push_back(int(p_class[particle_counter]))
                pflow_pt.push_back(pt[particle_counter])
                pflow_eta.push_back(eta[particle_counter])
                pflow_phi.push_back(phi[particle_counter])

                truth_class.push_back(int(truth_class_[particle_counter]))
                truth_pt.push_back(np.exp(truth_ptetaphi[particle_counter,0]))
                truth_eta.push_back(truth_ptetaphi[particle_counter,1])
                truth_phi.push_back(truth_ptetaphi[particle_counter,2])

                truth_has_track.push_back(has_track[particle_counter])

                inc_truth_particle = inc_truth[particle_counter].cpu().float().data.numpy().copy(order='C')
                v = ROOT.vector("float")(inc_truth_particle)
                truth_inc.push_back(inc_truth_particle)

                inc_pred_particle = inc_pred[particle_counter].cpu().float().data.numpy().copy(order='C')
                v = ROOT.vector("float")(inc_pred_particle)
                pred_inc.push_back(inc_pred_particle)

            for node_idx in range(len(node_energies)):
                node_energy.push_back(node_energies[node_idx])
                node_eta.push_back(node_etas[node_idx])
                node_phi.push_back(node_phis[node_idx])
                node_track_pt.push_back(node_track_pts[node_idx])
                node_is_track.push_back(int(node_is_tracks[node_idx].data))

            outputtree.Fill()

    outputfile.cd()
    outputtree.Write()
    outputfile.Close()



def undo_scalling(ptetaphi, ignore_zeros=False):
    pt  = (ptetaphi[:,0] * PT_STD  + PT_MEAN).unsqueeze(-1) 
    eta = (ptetaphi[:,1] * ETA_STD + ETA_MEAN).unsqueeze(-1)
    phi = (ptetaphi[:,2] * PHI_STD + PHI_MEAN).unsqueeze(-1)

    return torch.cat([pt, eta, phi], dim=-1)



def get_meanstd(config):
    global PT_MEAN;  global PT_STD;  global ETA_MEAN; global ETA_STD;
    global PHI_MEAN; global PHI_STD; global E_MEAN;   global E_STD
    
    PT_MEAN  = config['var transform']['particle_pt']['mean']
    PT_STD   = config['var transform']['particle_pt']['std']

    ETA_MEAN = config['var transform']['particle_eta']['mean']
    ETA_STD  = config['var transform']['particle_eta']['std']

    PHI_MEAN = config['var transform']['particle_phi']['mean']
    PHI_STD  = config['var transform']['particle_phi']['std']

    E_MEAN = config['var transform']['cell_e']['mean']
    E_STD  = config['var transform']['cell_e']['std']



if __name__ == "__main__":
    main()