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

import json

class PflowModel(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config

        self.track_and_cell_encoder = MPNN(config['embedding model'],config)
        self.global_encoder = MPNN(config['embedding model'],config)
        
        self.predict_set_size = (config['output model type']=='Set2Set') or (config['output model type']=='combined')

        if config['output model type']=='condensation':
            self.outputnet = CondNet(config)
        elif config['output model type']=='Set2Set':
            self.set_size_encoder = MPNN(config['embedding model'],config)
            self.outputnet = Set2Set(config)
        elif config['output model type']=='MLPF':
            self.outputnet = MLPFNet(config)
        elif config['output model type']=='combined':
            config_condensation_path = '/srv01/agrp/dreyet/PFlow/SCD/particle_flow/experiments/configs/condensation.json'
            with open(config_condensation_path, 'r') as fp:
                config_condensation = json.load(fp)
            self.set_size_encoder = MPNN(config['embedding model'],config)
            self.outputnet = Set2Set(config)
            self.outputnext  = CondNet(config_condensation)

    def infer(self,g):
        with torch.no_grad():
            self(g)
            if self.config["output model type"] == "Set2Set": 
                predicted_particles, predicted_num_particles, pred_n_supneutral = self.outputnet.infer(g)
                return predicted_particles, predicted_num_particles, pred_n_supneutral
            elif self.config['output model type']=='combined':
                predicted_particles,predicted_num_particles = self.outputnext.infer(g)
                return predicted_particles,predicted_num_particles                
            else:
                predicted_particles,predicted_num_particles = self.outputnet.infer(g)
                return predicted_particles,predicted_num_particles
                

    def forward(self, g):
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
        if self.config['output model type']=='combined':
            self.outputnext(g)
        return g
