from os import replace
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

        self.config = top_level_config
        config = top_level_config['output model']
        self.inference = CondNetInference(top_level_config)

        self.beta_net = \
            build_layers(config['node inputsize'],outputsize=1,
                                     features=config['beta net layers'],add_activation=nn.Sigmoid())

        self.x_net = build_layers(config['node inputsize'],outputsize=config['x size'],
                                  features=config['x net layers'],add_batch_norm=True)

        self.cell_class_net = build_layers(config['node inputsize'],
                                          outputsize=config['n cell classes'],
                                          features=config['cell classifer layers'])

        # Not classifying tracks
        #self.track_class_net = build_layers(config['node inputsize'],
        #                                  outputsize=config['n track classes'],
        #                                  features=config['track classifer layers'])

        self.particle_pt_net = build_layers(config['node inputsize'],
                                          outputsize=1,
                                          features=config['pt prediction layers']) 
        
        self.particle_eta_net = build_layers(config['node inputsize'],
                                          outputsize=1,
                                          features=config['eta prediction layers']) 

        self.particle_xhat_yhat_net = build_layers(config['node inputsize'],
                                          outputsize=2,
                                          features=config['xhatyhat prediction layers'])


    def zeta_term(self, g):

        layer_noise = {
            0: 13,
            1: 34.,
            2: 41.,
            3: 75.,
            4: 50.,
            5: 25.
        }

        cell_e     = torch.clone(g.nodes['nodes'].data['cell_energy'])
        cell_layer = torch.clone(g.nodes['nodes'].data['layer_cell'])

        cell_noise = cell_layer
        for layer, noise in layer_noise.items():
            cell_noise[cell_layer==layer] = noise

        #untransform
        transform = self.config['var transform']
        cell_e = cell_e*transform['cell_e']['std'] + transform['cell_e']['mean']
        cell_e = torch.exp(cell_e)

        cell_zeta = cell_e/cell_noise
        cell_zeta[g.nodes['nodes'].data['isTrack'] == 1] = 999.
        
        return cell_zeta
        
    def infer(self,g):
        with torch.no_grad():
            predicted_particles,predicted_num_particles = self.inference(g)
        return predicted_particles,predicted_num_particles

    def forward(self, g):

        g.nodes['nodes'].data['idx'] = torch.arange(g.num_nodes('nodes'),device=g.device)

        #g.nodes['nodes'].data['zeta'] = self.zeta_term(g).view(-1)

        ndata = torch.cat([g.nodes['nodes'].data['eta'].unsqueeze(1),g.nodes['nodes'].data['sinu_phi'].unsqueeze(1),g.nodes['nodes'].data['cosin_phi'].unsqueeze(1),
            g.nodes['nodes'].data['energy_cell'].unsqueeze(1),
            g.nodes['nodes'].data['layer_cell'].unsqueeze(1),
            g.nodes['nodes'].data['zeta'].unsqueeze(1),
            g.nodes['nodes'].data['winZeta'].unsqueeze(1),
            g.nodes['nodes'].data['N edges start'].unsqueeze(1),
            g.nodes['nodes'].data['nearTrack'].unsqueeze(1),
            g.nodes['nodes'].data['track_pt'].unsqueeze(1),
            g.nodes['nodes'].data['isTrack'].unsqueeze(1),
            g.nodes['nodes'].data['isMuon'].unsqueeze(1),
            g.nodes['nodes'].data['hidden rep'],g.nodes['nodes'].data['global rep']],dim=1).float()
 
        #tracks get automatic large beta 
        g.nodes['nodes'].data['beta'] = torch.where(g.nodes['nodes'].data['isTrack'] == 0, self.beta_net(ndata).view(-1), 0.98*torch.ones_like(g.nodes['nodes'].data['eta']))
        g.nodes['nodes'].data['x'] = self.x_net(ndata)
        g.nodes['nodes'].data['cell class pred'] = self.cell_class_net(ndata)
        g.nodes['nodes'].data['track class pred'] = 2*torch.ones_like( g.nodes['nodes'].data['cell class pred']).float() #self.track_class_net(ndata) Not classifying tracks

        ### Separate networks to learn offsets from the node's eta and xhat, yhat
        node_xhat_yhat = torch.cat([g.nodes['nodes'].data['cosin_phi'].unsqueeze(1),g.nodes['nodes'].data['sinu_phi'].unsqueeze(1) ], dim=1)
        particle_pt_pred        = self.particle_pt_net(ndata)
        particle_eta_pred       = g.nodes['nodes'].data['eta'].unsqueeze(1)  + self.particle_eta_net(ndata)
        particle_xhat_yhat_pred = node_xhat_yhat                             + self.particle_xhat_yhat_net(ndata)

        ### Replace eta, xhat, yhat with track quantities for nodes which are tracks. Keep pt from NN prediction for both cells and tracks!
        forcells  = torch.cat([particle_pt_pred,particle_eta_pred,                        particle_xhat_yhat_pred],dim=1)
        fortracks = torch.cat([particle_pt_pred,g.nodes['nodes'].data['eta'].unsqueeze(1),node_xhat_yhat         ],dim=1)
        g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'] = torch.where(g.nodes['nodes'].data['isTrack'].unsqueeze(1)==1,fortracks,forcells)

        return g
