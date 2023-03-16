import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn
import math

from rms_norm import RMSNorm

def init_normal(m):
    
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)


class TrainableEltwiseLayer(nn.Module):

  def __init__(self, n):
    super().__init__()
    
    self.weights = nn.Parameter(torch.randn((1, n)))  # define the trainable parameter

  def forward(self, x):
    # assuming x of size (-1, n)
    return x * self.weights  # element-wise multiplication


class SlotAttention(nn.Module):
    
    def __init__(self,node_input_size,particle_input_size,config,input_class):
        super().__init__()

        self.classes = input_class 

        self.key = nn.Sequential(nn.Linear(node_input_size+4+11+6+6,100)) #,nn.ReLU(),nn.Linear(10,10)
        if self.classes == "supercharged":
            self.query = nn.Sequential(nn.Linear(26+node_input_size+3,100)) #nn.ReLU(),nn.Linear(10,10)
        else: 
            self.query = nn.Sequential(nn.Linear(particle_input_size+node_input_size,100)) #nn.ReLU(),nn.Linear(10,10)
        #self.query = nn.Sequential(nn.Linear(26+node_input_size+3,100)) #nn.ReLU(),nn.Linear(10,10)
        self.values = nn.Sequential(nn.Linear(node_input_size+4+11+6+6,particle_input_size)) #nn.ReLU(),nn.Linear(50,z_shape),)

        self.gru = nn.GRUCell(particle_input_size,particle_input_size)
                
        self.layer_norm = nn.LayerNorm(particle_input_size)
        self.norm = 1/torch.sqrt(torch.tensor([100.0]))

        self.mlp = nn.Sequential(nn.Linear(particle_input_size,64),nn.ReLU(),nn.Linear(64,particle_input_size))

        self.rmsnorm = RMSNorm(particle_input_size)
        self.lin_weights = TrainableEltwiseLayer(particle_input_size)

        self.topo_e_mean = config['var transform']['topo_e']['mean']
        self.topo_e_std  = config['var transform']['topo_e']['std']

        self.topo_eta_mean = config['var transform']['cell_eta']['mean']
        self.topo_eta_std  = config['var transform']['cell_eta']['std']


    def edge_function(self, edges):

        attention = torch.sum(edges.src['key']*edges.dst['query'],dim=1) * self.norm 
        attention = nn.ReLU()(attention) 

        values = edges.src['values']

        edges.data['attention_weights'] = attention * edges.data['edge_dR_labels']
        edges.data['attention_weights'] = edges.data['attention_weights'] * (1-edges.src['isTrack'])

        return {'attention' : attention, 'values' : values,'edge_dR_labels':edges.data['edge_dR_labels']}


    def edge_function_attention(self, edges):

        attention_weights = edges.data['edge_dR_labels']* torch.exp(edges.data['attention_weights'])/(edges.dst['exp_sum_attention'])
        return {'attention_weights': attention_weights}


    def node_update(self, nodes):

        attention_weights = nodes.mailbox['attention'].unsqueeze(2)

        weighted_sum = torch.sum(attention_weights * nodes.mailbox['values'], dim=1)
        weighted_sum = self.rmsnorm(weighted_sum) * nn.Sigmoid()(self.lin_weights(weighted_sum))

        new_hidden_rep = nodes.data['node hidden rep']+ self.mlp( self.layer_norm( self.gru(weighted_sum, nodes.data['node hidden rep']) ) )

        # exp_sum_attention = torch.sum(torch.exp(nodes.mailbox['attention']), dim=1)
        return {'node hidden rep': new_hidden_rep} #, 'exp_sum_attention': exp_sum_attention}


    def forward(self, g):
        
        self.norm = self.norm.to(g.device)

        nodes_inputs = g.nodes['nodes'].data['hidden rep']

        energy = (torch.log(g.nodes['nodes'].data['energy'].unsqueeze(1)) - self.topo_e_mean) / self.topo_e_std
        eta    = (g.nodes['nodes'].data['eta'].unsqueeze(1) - self.topo_eta_mean) / self.topo_eta_std

        eta_l1 = (g.nodes['nodes'].data['eta_l1'].unsqueeze(1) - self.topo_eta_mean) / self.topo_eta_std
        eta_l2 = (g.nodes['nodes'].data['eta_l2'].unsqueeze(1) - self.topo_eta_mean) / self.topo_eta_std
        eta_l3 = (g.nodes['nodes'].data['eta_l3'].unsqueeze(1) - self.topo_eta_mean) / self.topo_eta_std
        phi_l1 = g.nodes['nodes'].data['phi_l1'].unsqueeze(1) 
        phi_l2 = g.nodes['nodes'].data['phi_l2'].unsqueeze(1) 
        phi_l3 = g.nodes['nodes'].data['phi_l3'].unsqueeze(1)
        eta_l4 = (g.nodes['nodes'].data['eta_l4'].unsqueeze(1) - self.topo_eta_mean) / self.topo_eta_std
        eta_l5 = (g.nodes['nodes'].data['eta_l5'].unsqueeze(1) - self.topo_eta_mean) / self.topo_eta_std
        eta_l6 = (g.nodes['nodes'].data['eta_l6'].unsqueeze(1) - self.topo_eta_mean) / self.topo_eta_std
        phi_l4 = g.nodes['nodes'].data['phi_l4'].unsqueeze(1) 
        phi_l5 = g.nodes['nodes'].data['phi_l5'].unsqueeze(1) 
        phi_l6 = g.nodes['nodes'].data['phi_l6'].unsqueeze(1)


        ene_l1 = g.nodes['nodes'].data['ene_l1'].unsqueeze(1) 
        ene_l2 = g.nodes['nodes'].data['ene_l2'].unsqueeze(1) 
        ene_l3 = g.nodes['nodes'].data['ene_l3'].unsqueeze(1)
        ene_l4 = g.nodes['nodes'].data['ene_l4'].unsqueeze(1) 
        ene_l5 = g.nodes['nodes'].data['ene_l5'].unsqueeze(1) 
        ene_l6 = g.nodes['nodes'].data['ene_l6'].unsqueeze(1)

        isTrack   = g.nodes['nodes'].data['isTrack'].unsqueeze(1)
        nTracks   = g.nodes['nodes'].data['nTracks'].unsqueeze(1)
        tracks_pt = g.nodes['nodes'].data['tracks_pt'].unsqueeze(1)
        nTracks_4   = g.nodes['nodes'].data['nTracks_4'].unsqueeze(1)
        tracks_pt_4 = g.nodes['nodes'].data['tracks_pt_4'].unsqueeze(1)

        skip_info = torch.cat([
            energy,
            eta,
            g.nodes['nodes'].data['phi'].unsqueeze(1),
            g.nodes['nodes'].data['layer'].unsqueeze(1),
            eta_l1,
            eta_l2,
            eta_l3,
            eta_l4,
            eta_l5,
            eta_l6,
            phi_l1,
            phi_l2,
            phi_l3,
            phi_l4,
            phi_l5,
            phi_l6,
            ene_l1,
            ene_l2,
            ene_l3,
            ene_l4,
            ene_l5,
            ene_l6,
            isTrack,
            nTracks,
            tracks_pt,
            nTracks_4,
            tracks_pt_4
        ],dim=1)

        nodes_inputs = torch.cat([g.nodes['nodes'].data['hidden rep'],skip_info],dim=1)
        g.nodes['nodes'].data['key'] = self.key(nodes_inputs)
        g.nodes['nodes'].data['values'] = self.values(nodes_inputs)

        if self.classes == "supercharged":

            query_input = torch.cat([
                g.nodes['particles'].data['node features'],
                g.nodes['particles'].data['global rep'],
                g.nodes['particles'].data['num_nearest_tracks'].unsqueeze(1),
                g.nodes['particles'].data['pt_nearest_tracks'].unsqueeze(1),
                g.nodes['particles'].data['dR_nearest_tracks'].unsqueeze(1)
            ],dim=1)
        else:
            query_input = torch.cat([g.nodes['particles'].data['node hidden rep'],g.nodes['particles'].data['global rep']],dim=1)  

        g.nodes['particles'].data['query'] = self.query(query_input)

        g.update_all(self.edge_function, self.node_update, etype='node_to_particle')
        # g.apply_edges(self.edge_function_attention, etype='node_to_particle')

