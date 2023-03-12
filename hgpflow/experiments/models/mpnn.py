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
        
        self.epsilon = 1e-8
        self.net = build_layers(3*inputsize,outputsize,layers)

    def forward(self, x):
        
        inputs = torch.sum( x.mailbox['message'] ,dim=1)
        
        inputs = torch.cat([inputs,x.data['hidden rep'],x.data['global rep']],dim=1)
        
        output = self.net(inputs)
        output = output / (torch.norm(output, p='fro', dim=1, keepdim=True) + self.epsilon)
        
        return {'hidden rep': output }


class MPNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.epsilon = 1e-8

        self.transform_var = self.config['var transform']
        
        num_skip_node_features = 7
        self.hidden_size = config['embedding model']['cell hidden size'] - num_skip_node_features

        self.cell_init_network = nn.Sequential(
            nn.Linear(config['embedding model']['cell inputsize'], config['embedding model']['cell hidden size']),
            nn.ReLU(),
            nn.Linear(config['embedding model']['cell hidden size'], self.hidden_size))
        self.track_init_network = nn.Sequential(
            nn.Linear(config['embedding model']['track inputsize'], config['embedding model']['track hidden size']),
            nn.ReLU(),
            nn.Linear(config['embedding model']['track hidden size'], self.hidden_size))

        self.node_update_networks = nn.ModuleList()
        
        self.n_blocks = config['embedding model']['n GN blocks']
        self.block_iterations = config['embedding model']['n iterations']        
        
        for block_i in range(self.n_blocks):
            self.node_update_networks.append(NodeNetwork(self.hidden_size, 
                                           self.hidden_size, config['embedding model']['node net layers']))
           

    def update_global_rep(self,g):
        
        global_rep = dgl.sum_nodes(g,'hidden rep',ntype='pre_nodes')
       
        global_rep = global_rep / (torch.norm(global_rep, p='fro', dim=1, keepdim=True) + self.epsilon)

        g.nodes['pre_nodes'].data['global rep'] = dgl.broadcast_nodes(g,global_rep,ntype='pre_nodes')
        g.nodes['global node'].data['global rep'] = global_rep
        
    
    def move_from_cellstracks_to_pre_nodes(self,g,cell_info,track_info,target_name):

        g.update_all(fn.copy_src(cell_info,'m'),fn.sum('m',target_name),etype='cell_to_pre_node')
        cell_only_data = g.nodes['pre_nodes'].data[target_name]
        g.update_all(fn.copy_src(track_info,'m'),fn.sum('m',target_name),etype='track_to_pre_node')
        g.nodes['pre_nodes'].data[target_name] = g.nodes['pre_nodes'].data[target_name]+cell_only_data


    def move_from_topostracks_to_nodes(self,g,topo_info,track_info,target_name):

        g.update_all(fn.copy_src(topo_info,'m'),fn.sum('m',target_name),etype='topocluster_to_node')
        topo_only_data = g.nodes['nodes'].data[target_name]

        g.update_all(fn.copy_src(track_info,'m'),fn.sum('m',target_name),etype='track_to_node')
        g.nodes['nodes'].data[target_name] = g.nodes['nodes'].data[target_name]+topo_only_data


    def topo_vals_edge(self, edges): # cell to topo

        eta = edges.src['eta_cell'] * self.transform_var['cell_eta']['std'] + self.transform_var['cell_eta']['mean']
        phi = edges.src['phi_cell'] * self.transform_var['cell_phi']['std'] + self.transform_var['cell_phi']['mean']
        energy = torch.exp(edges.src['energy_cell'] * self.transform_var['cell_e']['std'] + self.transform_var['cell_e']['mean'])

        wtd_eta = eta * energy
        wtd_phi = phi * energy

        wtd_layer = edges.src['layer_cell'] * energy

        return { \
            'weighted_eta': wtd_eta, 'weighted_phi': wtd_phi,
            'weighted_layer': wtd_layer, 'energy': energy
        }


    def topo_vals_nodes(self, nodes):

        energy = torch.sum(nodes.mailbox['energy'], dim=1)

        eta = torch.sum(nodes.mailbox['weighted_eta'], dim=1) / energy
        eta = (eta - self.transform_var['topo_eta']['mean']) / self.transform_var['topo_eta']['std']

        phi = torch.sum(nodes.mailbox['weighted_phi'], dim=1) / energy
        phi = (phi - self.transform_var['topo_phi']['mean']) / self.transform_var['topo_phi']['std']

        layer = torch.sum(nodes.mailbox['weighted_layer'], dim=1) / energy

        energy = (torch.log(energy) - self.transform_var['topo_e']['mean']) / self.transform_var['topo_e']['std']

        return {'eta': eta, 'phi': phi, 'layer': layer, 'energy': energy}
        

    def forward(self, g):
        
        g.nodes['cells'].data['hidden rep']  = self.cell_init_network(g.nodes['cells'].data['node features'])
        g.nodes['tracks'].data['hidden rep'] = self.track_init_network(g.nodes['tracks'].data['node features'])
        
        self.move_from_cellstracks_to_pre_nodes(g,'hidden rep','hidden rep','hidden rep')
        self.update_global_rep(g)
        
        for block_i in range(self.n_blocks):    
            for iteration_i in range(self.block_iterations[block_i]):
                g.update_all(fn.copy_src('hidden rep','message'), self.node_update_networks[block_i],etype= 'pre_node_to_pre_node' )                
                self.update_global_rep(g)

        g.update_all(fn.copy_src('hidden rep','message'), fn.sum("message",'hidden rep'),etype= 'pre_node_to_topocluster')
        self.move_from_topostracks_to_nodes(g,'hidden rep','hidden rep','features_0')

        g.update_all(self.topo_vals_edge, self.topo_vals_nodes, etype='cell_to_topocluster')
        # self.move_from_topostracks_to_nodes(g,'eta','track_eta_layer_0','eta')
        # self.move_from_topostracks_to_nodes(g,'phi','track_phi_layer_0','phi')

        self.move_from_topostracks_to_nodes(g,'eta','track_eta','eta')
        self.move_from_topostracks_to_nodes(g,'phi','track_phi','phi')

        self.move_from_topostracks_to_nodes(g,'energy','energy_track','energy')
        self.move_from_topostracks_to_nodes(g,'layer','layer_track','layer')

        g.nodes['nodes'].data['features_0'] = torch.cat([
            g.nodes['nodes'].data['features_0'].clone(), 
            g.nodes['nodes'].data['eta'].view(-1,1),
            g.nodes['nodes'].data['phi'].view(-1,1),
            g.nodes['nodes'].data['energy'].view(-1,1),
            g.nodes['nodes'].data['layer'].view(-1,1), 
            g.nodes['nodes'].data['isTrack'].view(-1,1), 
            g.nodes['nodes'].data['isMuon'].view(-1,1), 
            g.nodes['nodes'].data['track_pt'].view(-1,1)
        ], dim=1)

        return g

        