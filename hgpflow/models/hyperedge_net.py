import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn


def build_net(dims, activation=None):
    layers = []
    for i in range(len(dims)-2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


class HyperEdgeNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transform_var = self.config['var transform']
        self.hyperedge_feature_size = config['output model']['hyperedge_feature_size']

        self.num_reg = 3 # pt, eta, phi
        self.num_class = 6 # ch, e, mu, gamma, neut, garbage
        self.num_class_charged = 3 # ch, e, mu # they have a track, so they can't be garbage
        self.num_class_neutral = 2 # gamma, neut (garbage)

        dim = self.hyperedge_feature_size

        if self.config['inc_assignment'] == 'hard1':
            self.ptetaphi_net = build_net(
                [dim+4] + self.config['output model']['ptetaphi_net_features'] + [self.num_reg])
            self.class_net    = build_net(
                [dim+4] + self.config['output model']['class_net_features'] + [self.num_class], 
                activation=nn.Softmax(dim=-1))

        elif self.config['inc_assignment'] == 'hard2':
            self.ptetaphi_net_charged = build_net(
                [dim+4] + self.config['output model']['ptetaphi_net_features'] + [self.num_reg])
            self.class_net_charged = build_net(
                [dim+4] + self.config['output model']['class_net_features'] + [self.num_class_charged], 
                activation=nn.Softmax(dim=-1))

            # topo_skip
            self.ptetaphi_net_neutral = build_net(
                [dim+4] + self.config['output model']['ptetaphi_net_features'] + [self.num_reg])
            self.class_net_neutral = build_net(
                [dim+4] + self.config['output model']['class_net_features'] + [self.num_class_neutral], 
                activation=nn.Softmax(dim=-1))

        else:
            self.ptetaphi_net = build_net(
                [dim] + self.config['output model']['ptetaphi_net_features'] + [self.num_reg])
            self.class_net = build_net(
                [dim] + self.config['output model']['class_net_features'] + [self.num_class], 
                activation=nn.Softmax(dim=-1))


    def forward(self,g):
        bs = g.batch_size

        if 'hard' in self.config['inc_assignment']:
            g.update_all(self.edgefn, self.nodefn, etype='node_to_pflow_particle')

            node_energy = torch.exp(g.nodes['nodes'].data['energy'] * self.config['var transform']['topo_e']['std'] \
                            + self.config['var transform']['topo_e']['std'])
            node_energy = node_energy * (g.nodes['nodes'].data['isTrack'] != 1)

            # safeguard: particle with track, w/o TC, the row will be all zero in inc_flip. so normalization will give nan (0/0)
            node_energy = node_energy + g.nodes['nodes'].data['isTrack'] * 1e-8
            node_energy = node_energy.view(bs, -1, 1)

            inc_flipped = g.edges['node_to_pflow_particle'].data['incidence_val'].view(g.batch_size, -1, self.config['max_particles'])
            inc_flipped = inc_flipped * node_energy
            inc_flipped = inc_flipped / inc_flipped.sum(dim=1, keepdim=True)
            g.edges['node_to_pflow_particle'].data['incidence_val_flipped'] = inc_flipped.view(-1)

            g.update_all(self.topo_edgefn, self.topo_nodefn, etype='node_to_pflow_particle')

        if self.config['inc_assignment'] == 'hard1':
            features = torch.cat([g.nodes['pflow_particles'].data['features'], g.nodes['pflow_particles'].data['skip_info']], dim=1)

        elif self.config['inc_assignment'] == 'hard2':
            features_charged = torch.cat([g.nodes['pflow_particles'].data['features'], g.nodes['pflow_particles'].data['skip_info']], dim=1)
            features_neutral = torch.cat([g.nodes['pflow_particles'].data['features'], g.nodes['pflow_particles'].data['skip_info_topo']], dim=1) 

        else:
            features = g.nodes['pflow_particles'].data['features']

        if self.config['inc_assignment'] == 'hard2':
            proxy_ptetaphi_ch   = g.nodes['pflow_particles'].data['skip_info'][:,:3]
            proxy_ptetaphi_neut = g.nodes['pflow_particles'].data['skip_info_topo'][:,:3]

            ptetaphi_pred_charged = self.ptetaphi_net_charged(features_charged) + proxy_ptetaphi_ch
            ptetaphi_pred_neutral = self.ptetaphi_net_neutral(features_neutral) + proxy_ptetaphi_neut

            # set the eta, phi from proxy for particles with track
            proxy_mask_1 = torch.zeros_like(ptetaphi_pred_charged)
            proxy_mask_2 = torch.zeros_like(ptetaphi_pred_charged)
            proxy_mask_1[:,0]  = 1 # pred: pT
            proxy_mask_2[:,1:] = 1 # proxy: eta, phi
            ptetaphi_pred_charged = ptetaphi_pred_charged*proxy_mask_1 + proxy_ptetaphi_ch*proxy_mask_2

            class_pred_charged = self.class_net_charged(features_charged)
            class_pred_neutral = self.class_net_neutral(features_neutral)

            charged_mask = g.nodes['pflow_particles'].data['is_charged'] # is_charged is obtained from tracks
            neutral_mask = charged_mask != 1 # also includes garbage

            ptetaphi_pred = ptetaphi_pred_charged * charged_mask + ptetaphi_pred_neutral * neutral_mask
            ptetaphi_pred = ptetaphi_pred.view(bs, -1, self.num_reg)

            class_pred_charged = class_pred_charged.view(bs, -1, self.num_class_charged)
            class_pred_neutral = class_pred_neutral.view(bs, -1, self.num_class_neutral)
            charged_mask = charged_mask.view(bs, -1, 1)
            neutral_mask = neutral_mask.view(bs, -1, 1)

            return (ptetaphi_pred, (class_pred_charged, class_pred_neutral, charged_mask, neutral_mask)), g

        else:
            ptetaphi_pred = self.ptetaphi_net(features)
            class_pred    = self.class_net(features)

            ptetaphi_pred = ptetaphi_pred.view(bs, -1, self.num_reg)
            class_pred = class_pred.view(bs, -1, self.num_class)

            return (ptetaphi_pred, class_pred), g


    def edgefn(self, edges): # node_to_pflow_particle
        mask = edges.src['isTrack'] * edges.data['incidence_val']
        mask = mask.view(-1, 1)
        skip_info = torch.cat([
            edges.src['track_pt'].view(-1,1), edges.src['eta'].view(-1,1), edges.src['phi'].view(-1,1), edges.src['isMuon'].view(-1,1)
        ], dim=1)
        skip_info = skip_info * mask
        return {'skip_info': skip_info}

    def nodefn(self, nodes):
        skip_info = nodes.mailbox['skip_info'].sum(dim=1)
        return {'skip_info': skip_info}


    def topo_edgefn(self, edges): # node_to_pflow_particle; src = [0, 0, 0,... 45 times]

        eta    = (edges.src['eta'] * self.transform_var['topo_eta']['std'] + self.transform_var['topo_eta']['mean']) \
                    * edges.data['incidence_val_flipped'] * (edges.src['isTrack'] != 1)
        
        phi    = (edges.src['phi'] * self.transform_var['topo_phi']['std'] + self.transform_var['topo_phi']['mean']) \
                    * edges.data['incidence_val_flipped'] * (edges.src['isTrack'] != 1)
        
        energy = torch.exp(edges.src['energy'] * self.transform_var['topo_e']['std'] + self.transform_var['topo_e']['mean']) \
                    * edges.data['incidence_val'] * (edges.src['isTrack'] != 1)
        
        layer  =edges.src['layer'] * edges.data['incidence_val_flipped'] * (edges.src['isTrack'] != 1)
        return {'eta': eta, 'phi': phi, 'energy': energy, 'layer': layer}

    def topo_nodefn(self, nodes):
        eta    = nodes.mailbox['eta'].sum(dim=1)
        phi    = nodes.mailbox['phi'].sum(dim=1)
        energy = nodes.mailbox['energy'].sum(dim=1)
        layer  = nodes.mailbox['layer'].sum(dim=1)
        pt     = energy / torch.cosh(eta)

        pt     = (torch.log(pt) - self.transform_var['particle_pt']['mean']) / self.transform_var['particle_pt']['std']
        eta    = (eta - self.transform_var['particle_eta']['mean']) / self.transform_var['particle_eta']['std']
        phi    = (phi - self.transform_var['particle_phi']['mean']) / self.transform_var['particle_phi']['std']

        skip_info = torch.cat([
            pt.view(-1, 1), eta.view(-1, 1), phi.view(-1, 1), layer.view(-1, 1)
        ], dim=1)
        return {'skip_info_topo': skip_info}
        