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


class IterativeRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.max_particles = config['max_particles']
        self.T = self.config['T_TOTAL']

        self.d_hid = self.config['output model']['hyperedge_feature_size']

        self.refiner = HypergraphRefiner(self.config)

        # self.proj_inputs = nn.Linear(100, self.d_hid)  ##### not using

        self.edges_mu = nn.Parameter(torch.randn(1, self.d_hid))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, self.d_hid))
        nn.init.xavier_uniform_(self.edges_logsigma)


    def init_features(self, g):
        n_hyperedges = sum(g.batch_num_nodes('pflow_particles').data)
        mu = self.edges_mu.expand(n_hyperedges, -1)
        sigma = self.edges_logsigma.exp().expand(n_hyperedges, -1)
        e_t = mu + sigma * torch.randn(mu.shape, device = g.device)
        g.nodes['pflow_particles'].data['features'] = e_t

        # v_t = self.proj_inputs(g.nodes['nodes'].data['features_00'])
        # g.nodes['nodes'].data['features_0'] = v_t
        # g.nodes['nodes'].data['features'] = v_t

        g.nodes['nodes'].data['features'] = g.nodes['nodes'].data['features_0']

    def forward(self, g, t_skip=None, t_bp=None):
        t_skip = 0 if t_skip is None else t_skip
        t_bp = self.T if t_bp is None else t_bp

        # should have done it in the dataloader # but wasn't sure how to do it
        if self.config['inc_assignment'] != "none":
            g.apply_edges(lambda edges : self.incidence_assignment(edges, g.batch_size), etype='pflow_particle_to_node')
            # g.edges['node_to_pflow_particle'].data['incidence_val'] = g.edges['pflow_particle_to_node'].data['incidence_val']

        pred_bp = []
        with torch.no_grad():
            for _ in range(t_skip):
                p = self.refiner(g)

        for _ in range(t_skip, t_skip+t_bp):
            p = self.refiner(g)
            pred_bp.append(p)

        return pred_bp, g

    def incidence_assignment(self, edges, bs):  # pflow_particle_to_node
        output = edges.data['incidence_val']

        mbs = output.shape[0] // self.max_particles
        m   = mbs // bs
        nonzero_diag = torch.arange(m) * (self.max_particles-1) + (m-1)
        nonzero_diag = nonzero_diag.repeat(bs)
        nonzero_diag_shift = (torch.arange(bs)*self.max_particles*m).repeat_interleave(m)
        nonzero_diag = nonzero_diag + nonzero_diag_shift

        diag_mask = torch.zeros_like(output)
        diag_mask[nonzero_diag.long()] = 1

        track_mask = edges.dst['isTrack'] == 1
        track_one_mask = track_mask * diag_mask.bool()

        # all track-edges = 0
        output[track_mask] = 0

        # all diag-track-edges = 1
        output[track_one_mask] = 1
        
        return{'incidence_val': output}


class HypergraphRefiner(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config
        self.max_particles = config['max_particles']

        self.hyperedge_feature_size = config['output model']['hyperedge_feature_size']
        self.node_hidden_size = config['embedding model']['node hidden size']

        self.inc_hid = self.hyperedge_feature_size # NEED TO DECIDE LATER
        self.norm_pre_n  = nn.LayerNorm(2*self.inc_hid + self.inc_hid)
        self.norm_pre_e  = nn.LayerNorm(self.inc_hid + self.inc_hid)
        self.norm_n = nn.LayerNorm(self.inc_hid)
        self.norm_e = nn.LayerNorm(self.hyperedge_feature_size)

        self.inc_proj_n_net = build_net([self.inc_hid] + self.config['output model']['inc_proj_n_net_features'] + [self.inc_hid])
        self.inc_proj_e_net = build_net([self.inc_hid] + self.config['output model']['inc_proj_e_net_features'] + [self.inc_hid])
        self.inc_proj_i_net = build_net([1] + self.config['output model']['inc_proj_i_net_features'] + [self.inc_hid])

        # self.inc_net = nn.Sequential(
        #     nn.ReLU(inplace=True), nn.Linear(self.inc_hid, 2*self.inc_hid),
        #     nn.ReLU(inplace=True), nn.Linear(2*self.inc_hid, 2*self.inc_hid), 
        #     nn.ReLU(inplace=True), nn.Linear(2*self.inc_hid, 1) , nn.Sigmoid())
        # to use this, also need comment the relu in self.incidence_update1()

        self.inc_net = build_net(
            [self.inc_hid] + self.config['output model']['inc_net_features'] + [1],
            activation=None)

        self.hyperedge_indicator = build_net(
            [self.inc_hid+1] + self.config['output model']['hyperedge_indicator_features'] + [1],
            activation=nn.Sigmoid())


        dim = self.inc_hid

        self.deepset_e = DeepSet('pflow_particles', 'pflow_to_pflow', 2*dim, self.config['output model']['deepset_e_hid_features'] + [dim])
        self.deepset_n = DeepSet('nodes', 'node_to_node', 3*dim, self.config['output model']['deepset_n_hid_features'] + [dim])

        # self.deepset_e = MessagePassingNN('pflow_particles', 'pflow_to_pflow', 2*dim, [dim, dim, dim])
        # self.deepset_n = MessagePassingNN('nodes', 'node_to_node', 3*dim, [dim, dim, dim])



    def forward(self, g):
        
        # update the incidence matrix (1)
        g.apply_edges(self.incidence_update1, etype='pflow_particle_to_node')

        # apply softmax
        pred_incidence = g.edges['pflow_particle_to_node'].data['incidence_val']
        pred_incidence = pred_incidence.reshape(g.batch_size, g.num_nodes('nodes') // g.batch_size, -1)
        pred_incidence = nn.Softmax(dim=-1)(pred_incidence)
        g.edges['pflow_particle_to_node'].data['incidence_val'] = pred_incidence.reshape(-1)

        # making the edges bidirectional (skip connection for indicator)
        g.edges['node_to_pflow_particle'].data['incidence_val_for_ind'] = g.edges['pflow_particle_to_node'].data['incidence_val']

        # get hyperedge (pflow) indicator
        # g.apply_nodes(self.hyperedge_indicator_update, ntype='pflow_particles')
        g.update_all(self.hyperedge_indicator_update_edgefn, self.hyperedge_indicator_update_nodefn, etype='node_to_pflow_particle')

        # hard assignment for tracks --
        # all edges to track are zero + set one track-edge to one for each particle + set indicators to one
        if 'hard' in self.config['inc_assignment']:
            g.apply_edges(lambda edges : self.incidence_assignment(edges, g.batch_size), etype='pflow_particle_to_node')
            track_mask = g.nodes['nodes'].data['isTrack'].reshape(g.batch_size, -1)
            n_tracks = track_mask.sum(dim=1)
            mult_zero = torch.ones_like(g.nodes['pflow_particles'].data['indicator'])
            add_one = (mult_zero * (-1)) + 1
            for i, n in enumerate(n_tracks):
                mult_zero[i*self.max_particles + torch.arange(n)] = 0
                add_one[i*self.max_particles + torch.arange(n)] = 1
            g.nodes['pflow_particles'].data['indicator'] \
                = g.nodes['pflow_particles'].data['indicator'] * mult_zero + add_one

        g.nodes['pflow_particles'].data['is_charged'] = add_one

        # making the edges bidirectional for computing puspose
        g.edges['node_to_pflow_particle'].data['incidence_val'] = g.edges['pflow_particle_to_node'].data['incidence_val']

        # update the incidence matrix according to indicator (2)
        g.apply_edges(self.incidence_update2, etype='pflow_particle_to_node')

        # making the edges bidirectional for computing puspose
        g.edges['node_to_pflow_particle'].data['incidence_val_mod'] = g.edges['pflow_particle_to_node'].data['incidence_val_mod']

        # prep and update the hyperedges
        g.update_all(self.prep_edgefn, self.hyperedge_prep_nodefn, etype='node_to_pflow_particle')
        self.deepset_e(g)
        g.nodes['pflow_particles'].data['features'] = \
            self.norm_e(g.nodes['pflow_particles'].data['features'] + g.nodes['pflow_particles'].data['deepset_feat'])

        # prep and update the nodes
        g.update_all(self.prep_edgefn, self.node_prep_nodefn, etype='pflow_particle_to_node')
        self.deepset_n(g)
        g.nodes['nodes'].data['features'] = \
            self.norm_n(g.nodes['nodes'].data['features'] + g.nodes['nodes'].data['deepset_feat'])

        # particle (src) : [0,1,2,3,0,...], nodes (dst) : [0,0,0,0,1,1,...], 
        # incidence_val : [0p0n, 1p0n, 2p0n, ...]
        # shape (n_nodes, n_particles * batch_size)
        n_nodes = g.num_nodes('nodes') // g.batch_size
        incidence = torch.zeros((n_nodes, g.num_nodes('pflow_particles')), device=g.device)
        src, dst = g.edges(etype='pflow_particle_to_node') # src: pflow_particles, dst: nodes
        dst = dst % n_nodes
        incidence[dst, src] = g.edata['incidence_val'][('pflow_particles', 'pflow_particle_to_node', 'nodes')]
        incidence = incidence.transpose(0, 1)

        pred = torch.cat([incidence, g.nodes['pflow_particles'].data['indicator']], dim=1)
        pred = pred.reshape(g.batch_size, -1, n_nodes+1)

        return pred


    #*****************************#
    # Update the Incidence matrix #
    #*****************************#

    # pre indicator incidence update
    def incidence_update1(self, edges): # pflow_particle_to_node
        inc_net_input = \
            self.inc_proj_n_net(edges.dst['features']) + \
            self.inc_proj_e_net(edges.src['features']) + \
            self.inc_proj_i_net(edges.data['incidence_val'].unsqueeze(1))
        inc_net_input = nn.ReLU(inplace=True)(inc_net_input)
        output = self.inc_net(inc_net_input).squeeze(-1)
        return {'incidence_val': output}

    def hyperedge_indicator_update_edgefn(self, edges): # node_to_pflow_particle
        inc_skip = edges.data['incidence_val_for_ind']
        return {'inc_skip': inc_skip}

    def hyperedge_indicator_update_nodefn(self, nodes): # pflow_particles
        inc_skip = torch.sum(nodes.mailbox['inc_skip'], dim=1).reshape(-1, 1)
        indicator_input = torch.cat([nodes.data['features'], inc_skip], dim=1)
        indicator_vals  = self.hyperedge_indicator(indicator_input)
        return {'indicator': indicator_vals}

    # post indicator incidence update
    def incidence_update2(self, edges): # pflow_particle_to_node
        output = edges.data['incidence_val'] * edges.src['indicator'].squeeze(1)
        return {'incidence_val_mod': output}


    #***************************************#
    # Prep the nodes/hyperedges for updates # 
    #***************************************#

    def prep_edgefn(self, edges):
        product_ie = edges.data['incidence_val_mod'].unsqueeze(1) * edges.src['features']
        return {'product_ie': product_ie}
        
    def hyperedge_prep_nodefn(self, nodes):
        product_ie_sum = torch.sum(nodes.mailbox['product_ie'], dim=1)
        hyperedge_net_input = torch.cat([nodes.data['features'], product_ie_sum], dim=1)
        hyperedge_net_input = self.norm_pre_e(hyperedge_net_input)
        return {'deepset_feat': hyperedge_net_input}

    def node_prep_nodefn(self, nodes):
        product_ie_sum = torch.sum(nodes.mailbox['product_ie'], dim=1)
        node_net_input = torch.cat([nodes.data['features_0'], nodes.data['features'], product_ie_sum], dim=1)
        node_net_input = self.norm_pre_n(node_net_input)
        return {'deepset_feat': node_net_input}
        

    #*****************#
    # Hard Assignment # 
    #*****************#

    def incidence_assignment(self, edges, bs):  # pflow_particle_to_node
        output = edges.data['incidence_val']

        mbs = output.shape[0] // self.max_particles
        m   = mbs // bs
        nonzero_diag = torch.arange(m) * (self.max_particles-1) + (m-1)
        nonzero_diag = nonzero_diag.repeat(bs)
        nonzero_diag_shift = (torch.arange(bs)*self.max_particles*m).repeat_interleave(m)
        nonzero_diag = nonzero_diag + nonzero_diag_shift

        diag_mask = torch.zeros_like(output)
        diag_mask[nonzero_diag.long()] = 1

        track_mask = edges.dst['isTrack'] == 1
        track_one_mask = track_mask * diag_mask.bool()

        # all track-edges = 0,  output[track_mask] = 0
        mask = torch.ones_like(output)
        mask[track_mask] = 0
        output = output * mask

        # all diag-track-edges = 1,  output[track_one_mask] = 1
        mask = torch.zeros_like(output)
        mask[track_one_mask] = 1

        output = output + mask
        
        return{'incidence_val': output}



class DeepSet(nn.Module):
    def __init__(self, node_name, edge_name, d_in, d_hids):
        super().__init__()
        layers = []
        layers.append(DeepSetLayer(edge_name, d_in, d_hids[0]))
        for i in range(1, len(d_hids)):
            layers.append(ReLULayerNorm(node_name, d_hids[i-1]))
            layers.append(DeepSetLayer(edge_name, d_hids[i-1], d_hids[i]))

        self.sequential = nn.Sequential(*layers)

    def forward(self, g):
        return self.sequential(g)


class DeepSetLayer(nn.Module):
    def __init__(self, edge_name, in_features, out_features):
        super().__init__()
        self.edge_name = edge_name
        self.layer1 = nn.Linear(in_features, out_features)
        self.layer2 = nn.Linear(in_features, out_features)

    def edgefn(self, edges):
        return {'msg': edges.src['deepset_feat']}

    def nodefn(self, nodes):
        msgs_mean = torch.mean(nodes.mailbox['msg'], dim=1)
        output = self.layer1(nodes.data['deepset_feat']) + self.layer2(nodes.data['deepset_feat'] - msgs_mean)
        return {'deepset_feat': output}

    def forward(self, g):
        g.update_all(self.edgefn, self.nodefn, etype=self.edge_name)
        return g


class ReLULayerNorm(nn.Module):
    def __init__(self, node_name, dim):
        super().__init__()
        self.node_name = node_name
        self.relu = nn.ReLU(inplace=True)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, g):
        g.nodes[self.node_name].data['deepset_feat'] = self.relu(g.nodes[self.node_name].data['deepset_feat'])
        g.nodes[self.node_name].data['deepset_feat'] = self.layernorm(g.nodes[self.node_name].data['deepset_feat'])
        return g




# MPNN
#---------------

class MessagePassingNN(nn.Module):
    def __init__(self, node_name, edge_name, d_in, d_hids):
        super().__init__()
        layers = []
        layers.append(MPNNLayer(edge_name, d_in, d_hids[0]))
        for i in range(1, len(d_hids)):
            layers.append(ReLULayerNorm(node_name, d_hids[i-1]))
            layers.append(MPNNLayer(edge_name, d_hids[i-1], d_hids[i]))

        self.sequential = nn.Sequential(*layers)

    def forward(self, g):
        return self.sequential(g)


class MPNNLayer(nn.Module):
    def __init__(self, edge_name, in_features, out_features):
        super().__init__()
        self.edge_name = edge_name
        self.edge_net = nn.Sequential(
            nn.Linear(2*in_features, out_features), nn.ReLU(), nn.Linear(out_features, out_features), nn.Tanh())
        self.node_net1 = nn.Sequential( # on msgs
            nn.Linear(out_features, out_features), nn.ReLU(), nn.Linear(out_features, out_features//2), nn.Tanh())
        self.node_net2 = nn.Sequential(
            nn.Linear(in_features, out_features), nn.ReLU(), nn.Linear(out_features, out_features//2), nn.Tanh())

    def edgefn(self, edges):
        edge_feat = torch.cat([edges.src['deepset_feat'], edges.dst['deepset_feat']], dim=1)
        updated_edge_feat = self.edge_net(edge_feat)
        return {'msg': updated_edge_feat}

    def nodefn(self, nodes):
        node_feat1 = self.node_net1(nodes.mailbox['msg'].sum(dim=1))
        node_feat2 = self.node_net2(nodes.data['deepset_feat'])
        output = torch.cat([node_feat1, node_feat2], dim=1)
        return {'deepset_feat': output}

    def forward(self, g):
        g.update_all(self.edgefn, self.nodefn, etype=self.edge_name)
        return g

