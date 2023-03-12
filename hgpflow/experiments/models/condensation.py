import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph


import numpy as np
import torch
import torch.nn as nn
from condensation_inference import CondNetInference


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

class CondNet(nn.Module):
    def __init__(self,top_level_config):
        super().__init__()

        config = top_level_config['output model']
        self.inference = CondNetInference(top_level_config)



        
        self.beta_net = \
            build_layers(config['node inputsize'],outputsize=1,
                                     features=config['beta net layers'],add_activation=nn.Sigmoid()) 
        self.x_net = build_layers(config['node inputsize'],outputsize=config['x size'],
                                  features=config['x net layers'],add_batch_norm=True)


        self.nodeclass_net = build_layers(config['node inputsize'],
                                          outputsize=config['n classes'],
                                          features=config['node classifer layers']) 
        self.charge_class_net = build_layers(config['node inputsize'],
                                          outputsize=config['n charge classes'],
                                          features=config['charge classifier layers']) 

        self.particle_pt_eta_xhat_yhat_net = build_layers(config['node inputsize'],
                                          outputsize=4,
                                          features=config['ptetaxhatyhat prediction layers']) 
        

        self.particle_position_net = build_layers(config['node inputsize'],
                                          outputsize=3,
                                          features=config['prod position layers']) 
        
        
        
    def infer(self,g):
        with torch.no_grad():
            predicted_particles,predicted_num_particles = self.inference(g)
        return predicted_particles,predicted_num_particles

    def forward(self, g):

        
        ndata = torch.cat([g.nodes['nodes'].data['eta'].unsqueeze(1),g.nodes['nodes'].data['sinu_phi'].unsqueeze(1),g.nodes['nodes'].data['cosin_phi'].unsqueeze(1),
            g.nodes['nodes'].data['isMuon'].unsqueeze(1),
            g.nodes['nodes'].data['hidden rep'],g.nodes['nodes'].data['global rep']],dim=1)
        
        g.nodes['nodes'].data['beta'] = self.beta_net(ndata).view(-1)
        
        g.nodes['nodes'].data['x'] = self.x_net(ndata)
        g.nodes['nodes'].data['class pred'] = self.nodeclass_net(ndata)
        g.nodes['nodes'].data['charge pred'] = self.charge_class_net(ndata)

        g.nodes['nodes'].data['pos pred'] = self.particle_position_net(ndata)
        g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'] = self.particle_pt_eta_xhat_yhat_net(ndata)

        return g