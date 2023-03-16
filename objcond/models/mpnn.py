import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph


import numpy as np
import torch
import torch.nn as nn

def build_layers(inputsize,outputsize,features,add_batch_norm=False,add_activation=None):
    layers = []
    layers.append(nn.Linear(inputsize,features[0]))
    layers.append(nn.ReLU())
    for hidden_i in range(1,len(features)):
        if add_batch_norm:
            layers.append(nn.BatchNorm1d(features[hidden_i-1]))
        layers.append(nn.Linear(features[hidden_i-1],features[hidden_i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(features[-1],outputsize))
    if add_activation!=None:
        layers.append(add_activation)
    return nn.Sequential(*layers)


class NodeNetwork(nn.Module):
    def __init__(self,inputsize,outputsize,layers):
        super().__init__()
        
        self.net = build_layers(3*inputsize,outputsize,layers)

    def forward(self, x):
        
        inputs = torch.sum( x.mailbox['message'] ,dim=1)
        
        inputs = torch.cat([inputs,x.data['hidden rep'],x.data['global rep']],dim=1)
        
        output = self.net(inputs)
        output = output / torch.clamp(torch.norm(output, p='fro', dim=1, keepdim=True),1e-8)
        
        return {'hidden rep': output }

class MPNN(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config

        self.cell_init_network = build_layers(config['cell inputsize'],
                                                   config['cell hidden size'],
                                                   config['node init layers'],add_batch_norm=True)
        self.track_init_network = build_layers(config['track inputsize'],
                                                   config['track hidden size'],
                                                   config['track init layers'],add_batch_norm=True)
        
        #assumes cell and track hidden size is the same
        self.hidden_size = config['cell hidden size'] 
        
        self.node_update_networks = nn.ModuleList()
        
        self.n_blocks = config['n GN blocks']
        self.block_iterations = config['n iterations']        
        
        for block_i in range(self.n_blocks):
            
            self.node_update_networks.append(NodeNetwork(self.hidden_size, 
                                           self.hidden_size,config['node net layers']))


    def update_global_rep(self,g):
        
        
        
        global_rep = dgl.sum_nodes(g,'hidden rep',ntype='nodes')
       
    
        global_rep = global_rep / torch.clamp(torch.norm(global_rep, p='fro', dim=1, keepdim=True),1e-8)

        g.nodes['nodes'].data['global rep'] = dgl.broadcast_nodes(g,global_rep,ntype='nodes')
        g.nodes['global node'].data['global rep'] = global_rep
        
    
    def move_from_cellstracks_to_nodes(self,g,cell_info,track_info,target_name):

        g.update_all(fn.copy_src(cell_info,'m'),fn.sum('m',target_name),etype='cell_to_node')
        cell_only_data = g.nodes['nodes'].data[target_name]
        g.update_all(fn.copy_src(track_info,'m'),fn.sum('m',target_name),etype='track_to_node')
        g.nodes['nodes'].data[target_name] = g.nodes['nodes'].data[target_name]+cell_only_data

    
    def forward(self, g):
        

        #create cell, track embedding vectors
        g.nodes['cells'].data['hidden rep'] = self.cell_init_network(total_features)
        g.nodes['tracks'].data['hidden rep'] = self.track_init_network(g.nodes['tracks'].data['node features'])

        self.move_from_cellstracks_to_nodes(g,'hidden rep','hidden rep','hidden rep')

        #add engineered features to node features
        extra_features = g.nodes['cells'].data['zeta'].unsqueeze(1)
        total_features = torch.cat([g.nodes['cells'].data['node features'],extra_features],dim=1).float()

        self.update_global_rep(g)
        
        for block_i in range(self.n_blocks):
    
            for iteration_i in range(self.block_iterations[block_i]):
                
                g.update_all(fn.copy_src('hidden rep','message'), self.node_update_networks[block_i],etype= 'node_to_node' )                
                self.update_global_rep(g)
                

        return g
