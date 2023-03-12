import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn

from models.mpnn import MPNN
from models.condensation import CondNet
from models.set2set import Set2Set
from models.mlpf import MLPFNet
from models.hypergraph_refiner import IterativeRefiner
from models.hyperedge_net import HyperEdgeNet


class PflowModel(nn.Module):

    def __init__(self,config,debug):
        super().__init__()

        self.config = config

        self.track_and_cell_encoder = MPNN(config)
        self.global_encoder = MPNN(config)
        
        self.predict_set_size = (config['output model type']=='Set2Set')

        if config['output model type']=='condensation':
            self.outputnet = CondNet(config)
    
        elif config['output model type']=='Set2Set':
            self.set_size_encoder = MPNN(config['embedding model'])
            self.outputnet = Set2Set(config)
    
        elif config['output model type']=='MLPF':
            self.outputnet = MLPFNet(config)
    
        elif config['output model type']=='hypergraph':
            self.pt_mean = config['var transform']['particle_pt']['mean']
            self.pt_std  = config['var transform']['particle_pt']['std']
            self.eta_mean = config['var transform']['particle_eta']['mean']
            self.eta_std  = config['var transform']['particle_eta']['std']
            self.phi_mean = config['var transform']['particle_phi']['mean']
            self.phi_std  = config['var transform']['particle_phi']['std']

            self.outputnet = IterativeRefiner(config)
            self.hyperedgenet = HyperEdgeNet(config)

        self.debug = debug
 

    def infer(self, g, debug=False, threshold=None):
        with torch.no_grad():
            if self.config["output model type"] == "hypergraph":
                bs = g.batch_size

                self.track_and_cell_encoder(g)
                self.outputnet.init_features(g)

                inc_preds, g = self.outputnet(g, t_skip=self.config['T_TOTAL']-1, t_bp=1)
                if threshold is None:
                    indicator = (inc_preds[-1][:,:,-1] > self.config['indicator_threshold'])# .unsqueeze(-1)
                else:
                    indicator = (inc_preds[-1][:,:,-1] > threshold)# .unsqueeze(-1)

                (ptetaphi_pred, class_pred_tuple), g = self.hyperedgenet(g)

                ptetaphi_pred = self.undo_scalling(ptetaphi_pred)
                class_pred_charged, class_pred_neutral, charged_mask, neutral_mask = class_pred_tuple

                class_pred_charged = torch.argmax(class_pred_charged, dim=2).unsqueeze(-1)
                class_pred_neutral = torch.argmax(class_pred_neutral, dim=2).unsqueeze(-1)
                class_pred = class_pred_charged * charged_mask + (class_pred_neutral + 3) * neutral_mask

                particle_pred = torch.cat([ptetaphi_pred, class_pred], dim=2)

                if debug == True:
                    return inc_preds[-1], particle_pred.squeeze(0)

                particle_pred = particle_pred[indicator]
                return particle_pred

            else:
                raise ValueError("pflow_model.py:: Error! No inference code")
                

    def forward(self, g, **kwargs):
        if self.config['output model type']=='hypergraph':
            if self.debug == False:
                print('call from pflow_nodel')
                self.track_and_cell_encoder(g)
            self.outputnet.init_features(g)

            t_skip=kwargs['t_skip']; t_bp=kwargs['t_bp']

            inc_pred, g = self.outputnet(g, t_skip, t_bp)
            (ptetaphi_pred, class_pred), g = self.hyperedgenet(g)
            return inc_pred, ptetaphi_pred, class_pred, g

        else:
            self.track_and_cell_encoder(g)
            node_hidden_rep = g.nodes['nodes'].data.pop('hidden rep')
            
            self.global_encoder(g)
            global_rep = g.nodes['global node'].data.pop('global rep')
            
            if self.predict_set_size:
                self.set_size_encoder(g)
                setsize_rep = g.nodes['global node'].data.pop('global rep')
                g.nodes['global node'].data['set size rep'] = setsize_rep

            g.nodes['nodes'].data['hidden rep'] = node_hidden_rep
            g.nodes['global node'].data['global rep'] = global_rep
            self.outputnet(g)
            return g


    def undo_scalling(self, ptetaphi, ignore_zeros=False):
        if ignore_zeros == True:
            mask = (ptetaphi[:,:,0]==0) * (ptetaphi[:,:,1]==0) * (ptetaphi[:,:,2]==0)
        pt  = (ptetaphi[:,:,0] * self.pt_std  + self.pt_mean).unsqueeze(-1) 
        eta = (ptetaphi[:,:,1] * self.eta_std + self.eta_mean).unsqueeze(-1)
        phi = (ptetaphi[:,:,2] * self.phi_std + self.phi_mean).unsqueeze(-1)
     
        if ignore_zeros == True:
            pt[mask]=0; eta[mask]=0; phi[mask]=0

        return torch.cat([pt, eta, phi], dim=-1)