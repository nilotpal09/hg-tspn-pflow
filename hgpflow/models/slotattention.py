import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph


import numpy as np
import torch
import torch.nn as nn
import math

# from rms_norm import RMSNorm


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
    
    def __init__(self,node_input_size,particle_input_size):
        super().__init__()

        self.key = nn.Sequential(nn.Linear(node_input_size,20)) #,nn.ReLU(),nn.Linear(10,10)
        self.query = nn.Sequential(nn.Linear(particle_input_size+node_input_size,20)) #nn.ReLU(),nn.Linear(10,10)
        self.values = nn.Sequential(nn.Linear(node_input_size, particle_input_size)) #nn.ReLU(),nn.Linear(50,z_shape),)

        self.gru = nn.GRUCell(particle_input_size,particle_input_size)
                
        self.layer_norm = nn.LayerNorm(particle_input_size)
        self.norm = 1/torch.sqrt(torch.tensor([20.0]))

        self.mlp = nn.Sequential(nn.Linear(particle_input_size,64),nn.ReLU(),nn.Linear(64,particle_input_size))

        # self.rmsnorm = RMSNorm(particle_input_size)
        self.lin_weights = TrainableEltwiseLayer(particle_input_size)


    def edge_function(self, edges):

        attention = torch.sum(edges.src['key']*edges.dst['query'],dim=1)*self.norm
        values = edges.src['values']

        edges.data['attention_weights'] = nn.ReLU()(attention)

        return {'attention' : attention, 'values' : values}


    def node_update(self, nodes):

        attention_weights = nodes.mailbox['attention'].unsqueeze(2)
        # attention_weights = nn.ReLU()(attention_weights)

        weighted_sum = torch.sum(attention_weights*nodes.mailbox['values'], dim=1)
        # weighted_sum = self.rmsnorm(weighted_sum) * nn.Sigmoid()(self.lin_weights(weighted_sum))

        new_hidden_rep = nodes.data['node hidden rep']+ self.mlp( self.layer_norm( self.gru(weighted_sum, nodes.data['node hidden rep']) ) )

        return {'node hidden rep': new_hidden_rep}


    def forward(self, g):
        
        self.norm = self.norm.to(g.device)

        nodes_inputs = g.nodes['nodes'].data['hidden rep']

        g.nodes['nodes'].data['key'] = self.key(nodes_inputs)
        g.nodes['nodes'].data['values'] = self.values(nodes_inputs)

        query_input = torch.cat([g.nodes['particles'].data['node hidden rep'],g.nodes['particles'].data['global rep']],dim=1)
        
        g.nodes['particles'].data['query'] = self.query(query_input)

        g.update_all(self.edge_function, self.node_update, etype='node_to_particle')
