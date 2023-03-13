import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn

class CondNetMetrics(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.n_nearest = 5

  
    def message_dist(self, edges):

        m_belongs = torch.ones_like(edges.src['parent target'])
        if 'particle idx' in edges.dst:
            m_belongs = (edges.src['parent target']==edges.dst['particle idx'])

        x_i     = edges.src['x']
        x_c     = edges.dst['max x'] if 'max x' in edges.dst else torch.zeros_like(x_i)
        q_c     = edges.dst['max q'] if 'max q' in edges.dst else 1 # torch.ones_like()

        m_dx    = torch.norm(x_i - x_c,dim=1)
        m_dxq   = m_dx*edges.src['q']*q_c
        m_q     = edges.src['q']
        m_cond_pt = edges.src['is cond point']

        return {'m_dx': m_dx,'m_dxq': m_dxq, 'm_belongs': m_belongs, 'm_q': m_q, 'm_cond_pt': m_cond_pt}


    def compute_rms(self, nodes):

        m_dx  = nodes.mailbox['m_dx']*nodes.mailbox['m_belongs']
        N     = torch.sum(nodes.mailbox['m_belongs'],-1)
        m_dx2 = m_dx*m_dx
        rms   = torch.sqrt(torch.sum(m_dx2,dim=1)/N)

        m_dxq  = nodes.mailbox['m_dxq']
        m_dxq2 = m_dxq*m_dxq
        sum_q  = torch.sum(nodes.mailbox['m_q'])
        rmsq   = torch.sqrt(torch.sum(m_dxq2,dim=1)/(N*sum_q))

        #calculate nearest neighbor distance for each cond point
        cond_pts_dist  = nodes.mailbox['m_dx']*nodes.mailbox['m_cond_pt']
        cond_pts_dist[torch.where(cond_pts_dist<1e-8)] = 999.
        which_neighbors = torch.arange(self.n_nearest,device=cond_pts_dist.device)
        neighbors_dist  = torch.index_select(torch.sort(cond_pts_dist)[0],1,which_neighbors)

        return {'RMS': rms, 'RMSq': rmsq, 'N nodes': N, 'nearest neighbors': neighbors_dist}


    def message_rms(self, edges):

        rms = edges.src['RMS']
        x_c = edges.src['max x']
        
        return {'RMS': rms, 'max x': x_c}


    def compute_DB(self, nodes):

        Si   = nodes.mailbox['RMS']
        #DB_test = torch.sum(Si,1)
        #return {'DB': DB_test}

        Sij  = torch.unsqueeze(Si,1)
        Sij  = torch.repeat_interleave(Sij,Sij.shape[1],axis=1)
        SijT = torch.transpose(Sij,1,2)
        DSij = Sij + SijT

        xci  = nodes.mailbox['max x']
        xci  = torch.unsqueeze(xci,1)
        Xij  = torch.repeat_interleave(xci,xci.shape[2],axis=1)
        XijT = torch.transpose(Xij,1,2)
        Dij  = Xij - XijT
        Mij  = torch.linalg.norm(Dij,dim=3)
        M2ij = Mij*Mij

        Rij  = DSij/M2ij
        Rij[torch.where(torch.isnan(Rij))] = 0
        Rij[torch.where(torch.isinf(Rij))] = 0

        DB = (1/DSij.shape[2])*torch.sum(torch.max(Rij,dim=2)[0],dim=1)
    
        return {'DB': DB}


    def message_particles_to_nodes(self, edges):

        m_belongs = torch.ones_like(edges.dst['parent target'])
        if 'particle idx' in edges.src:
            m_belongs = (edges.dst['parent target']==edges.src['particle idx'])

        m_N_nodes = edges.src['N nodes']

        return {'m_belongs': m_belongs, 'm_N_nodes': m_N_nodes}

    def update_nodes(self, nodes):

        _N_nodes = nodes.mailbox['m_N_nodes']*nodes.mailbox['m_belongs']
        N_nodes = torch.sum(_N_nodes,-1)
        return {'N nodes': N_nodes}


    def update_edges(self, edges):

        #src=particles, dst=nodes
        distance = torch.norm(edges.src['max x']-edges.dst['x'],dim=1)
        belongs  = (edges.dst['parent target']==edges.src['particle idx'])
        
        return {'dist x': distance, 
                'belongs': belongs, 
                'beta': edges.dst['beta'],
                'parent class':  edges.dst['particle class'],
                'particle class': edges.src['particle class']
                }


    def init_nodes(self, g):

        g.nodes['nodes'].data['N nodes'] = torch.zeros(g.num_nodes(ntype='nodes'),device=g.device)

        g.nodes['particles'].data['RMS'] = torch.zeros(g.num_nodes(ntype='particles'),device=g.device)
        g.nodes['particles'].data['RMSq'] = torch.zeros(g.num_nodes(ntype='particles'),device=g.device)
        g.nodes['particles'].data['N nodes'] = torch.zeros(g.num_nodes(ntype='particles'),device=g.device)
        g.nodes['particles'].data['nearest neighbors'] = torch.zeros((g.num_nodes(ntype='particles'),self.n_nearest),device=g.device)

        g.nodes['global node'].data['RMS'] = torch.zeros(g.num_nodes(ntype='global node'),device=g.device)
        g.nodes['global node'].data['DB'] = torch.zeros(g.num_nodes(ntype='global node'),device=g.device)

    def init_edges(self, g):

        g.edges['particle_to_node'].data['dist x']         = torch.zeros(g.num_edges(etype='particle_to_node'),device=g.device)
        g.edges['particle_to_node'].data['belongs']        = torch.zeros(g.num_edges(etype='particle_to_node'),device=g.device)
        g.edges['particle_to_node'].data['beta']          = torch.zeros(g.num_edges(etype='particle_to_node'),device=g.device)
        #g.edges['particle_to_node'].data['node idx']       = torch.zeros(g.num_edges(etype='particle_to_node'),device=g.device)
        #g.edges['particle_to_node'].data['parent idx']     = torch.zeros(g.num_edges(etype='particle_to_node'),device=g.device)
        g.edges['particle_to_node'].data['parent class']   = torch.zeros(g.num_edges(etype='particle_to_node'),device=g.device)
        #g.edges['particle_to_node'].data['particle idx']   = torch.zeros(g.num_edges(etype='particle_to_node'),device=g.device)
        g.edges['particle_to_node'].data['particle class'] = torch.zeros(g.num_edges(etype='particle_to_node'),device=g.device)


    def forward(self, g):

        self.init_nodes(g)

        g.update_all(self.message_dist,
                      self.compute_rms,
                      etype='node_to_particle')

        g.update_all(self.message_dist,
                      self.compute_rms,
                      etype='nodes to global')

        g.update_all(self.message_rms,
                      self.compute_DB,
                      etype='particles to global')

        g.update_all(self.message_particles_to_nodes,
                      self.update_nodes,
                      etype='particle_to_node')

        g.apply_edges(self.update_edges,etype='particle_to_node')


        #WIP!
        #g.nodes['nodes'].data['dist to cond point'] = dgl.broadcast_nodes(g,g.nodes['particles'].data['max x'],ntype='nodes')

        #g.nodes['particles'].data['dist vector'] = dgl.broadcast_nodes(g,g.nodes['global node'].data['dist matrix'],ntype='particles')

        #globalrms  = g.nodes['global node'].data['RMS']#.detach().numpy()
        #DB         = g.nodes['global node'].data['DB']

        #return globalrms, DB
