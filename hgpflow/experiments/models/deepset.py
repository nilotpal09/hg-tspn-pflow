import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self, in_features,input_names):
        super().__init__()
        
        self.inputs = input_names

        small_in_features = max(math.floor(in_features/10), 1)
        self.d_k = np.sqrt(small_in_features)

        self.query = nn.Sequential(
            nn.Linear(in_features, small_in_features),
            nn.Tanh(),
        )
        self.key = nn.Linear(in_features, small_in_features)
    
    def set_edge_weight(self,edges):
        
        
        edge_weight = torch.sum( edges.src['query']*edges.dst['key'],dim=1 )/self.d_k
        

        node_data = edges.src['node input']
        
        return {'edge weight': edge_weight, 'edge message' : node_data }
    
    def node_attention(self,nodes):
    
        #these will have the shape (n_nodes, n_connected_nodes, features )
        edge_weights = nodes.mailbox['edge weight']
        edge_messages = nodes.mailbox['edge message']
        
        
        edge_weights = torch.softmax(edge_weights,dim=1)
    
        node_rep = torch.sum((edge_messages*edge_weights.unsqueeze(2)),dim=1)

        return {'node attention': node_rep}

    def forward(self, g):

        node_data = torch.cat(
            [g.ndata[inputname] for inputname in self.inputs],dim=1)

        # inp.shape should be (N,C)
        g_with_loop = dgl.add_self_loop(g)
        g_with_loop.ndata['node input'] = node_data

        g_with_loop.ndata['query'] = self.query(node_data) 
        g_with_loop.ndata['key'] = self.key(node_data)
        
        g_with_loop.update_all(self.set_edge_weight,self.node_attention)
        
        g.ndata['node attention'] = g_with_loop.ndata['node attention']
    
        return g

class DeepSetLayer(nn.Module):
    def __init__(self,inputsize,outputsize,inputnames,outputname,apply_activation=True,use_attention=True):
        super().__init__()

        self.use_attention = use_attention
        self.inputs = inputnames
        self.outputname = outputname
    
        self.layer1 = nn.Linear(inputsize,outputsize,bias=False)
        self.layer2 = nn.Linear(inputsize,outputsize,bias=True)

        if self.use_attention:
            self.attention = Attention(inputsize,inputnames)

        self.apply_activation = apply_activation
        self.activation = nn.ReLU()


    def forward(self, g):
        
        node_data = torch.cat(
            [g.nodes['particles'].data[inputname] for inputname in self.inputs],dim=1)
        

        if not self.use_attention:
            g.nodes['particles'].data['ds layer node input'] = node_data
            mean_node_inputs = dgl.broadcast_nodes(g,dgl.mean_nodes(g,'ds layer node input',ntype='particles'),ntype='particles')

            
            x = self.layer1(node_data)+self.layer2(node_data-mean_node_inputs)
        else:
            g = self.attention(g)
            
            
            x = self.layer1(node_data)+self.layer2(g.ndata['node attention'])

        x = x / torch.norm(x, p='fro', dim=1, keepdim=True)

        g.nodes['particles'].data[self.outputname] = x
        if self.apply_activation:
            g.nodes['particles'].data[self.outputname] = self.activation(g.nodes['particles'].data[self.outputname])
        
        
        return g
    
    
class DeepSet(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList()


        self.layers.append(DeepSetLayer(config['inputsize'],
        config['layers'][0],config['inputs'],'node hidden rep',apply_activation=True,use_attention=False) ) 
        
        n_layers = len(config['layers'])
        
        for i in range(n_layers-1):
            
            self.layers.append(DeepSetLayer(config['layers'][i],
                config['layers'][i+1],
                ['node hidden rep'],
                'node hidden rep',apply_activation=True,use_attention=False) )
        
        
        self.layers.append(DeepSetLayer(config['layers'][-1],
                config['outputsize'],
                ['node hidden rep'],
                'node hidden rep',apply_activation=False,use_attention=False))
        

        
    def forward(self, g):

        for i, layer in enumerate( self.layers ):
            g = layer(g)
            
        return g