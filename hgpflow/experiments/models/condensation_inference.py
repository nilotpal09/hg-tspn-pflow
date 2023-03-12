import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn


class CondNetInference(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.var_transform = self.config['var transform']

        self.x_size = config['output model']['x size']
        self.t_b = config['t_b']
        self.t_d = config['t_d']

    def pass_edge_info(sefl,edges):
    
        #assigned nodes get a beta of 0, so they can't be the max anymore.
        remove_assigned = (1-edges.src['assigned'])
        
        return {'betas': edges.src['beta']*remove_assigned,'xs' : edges.src['x'],'idx' : edges.src['idx']}

    def sort_beta(self,graphs):
        
        N = len(graphs)
    
        max_beta, where = torch.max(graphs.mailbox['betas'],dim=1)

        max_x = graphs.mailbox['xs'][torch.arange(N),where]
        max_idx = graphs.mailbox['idx'][torch.arange(N),where]
            
        beta_stack = torch.stack([graphs.data['max beta'],max_beta],dim=1)
        x_stack = torch.stack([graphs.data['max x'],max_x],dim=1)
        idx_stack = torch.stack([graphs.data['max idx'],max_idx],dim=1)
        
        max_beta, where = torch.max(beta_stack,dim=1)
        max_x = x_stack[torch.arange(N),where]
        max_idx = idx_stack[torch.arange(N),where]
        
        return {'max beta': max_beta, 'max x': max_x, 'max idx' : max_idx}

    def init_cells(self,g):
        
        
        g.nodes['nodes'].data['idx'] = torch.arange(g.num_nodes('nodes'),device=g.device)
        
        g.nodes['nodes'].data['assigned'] = torch.zeros(g.num_nodes('nodes'),device=g.device)
        
        betas = g.nodes['nodes'].data['beta']

        g.nodes['nodes'].data['cluster idx'] = -1*torch.ones(g.num_nodes('nodes'),device=g.device)
        g.nodes['nodes'].data['is cond point'] = torch.zeros(g.num_nodes('nodes'),device=g.device)
    
    def find_condensation_points(self, g):
        
        self.init_cells(g)
            
        index = -1
        while True:
            
            index+=1
            ## init the global nodes
            g.nodes['global node'].data['max beta'] = torch.zeros(g.num_nodes('global node'),device=g.device)
            g.nodes['global node'].data['max x'] = torch.zeros(g.num_nodes('global node'),self.x_size,device=g.device)
            g.nodes['global node'].data['max idx'] = torch.zeros(g.num_nodes('global node'),device=g.device)

            #find the max beta and associated x for each graph.
            g.update_all(self.pass_edge_info,self.sort_beta,etype='nodes to global')
            
            
            #assign cells and tracks to the current "object" with max beta in each graph in the batch
            
            count_unassigned = 0 #when all nodes have been assigned (or labeled as bkg), stop.
            
                
            g.nodes['nodes'].data['max x'] = dgl.broadcast_nodes(g, 
                                                                g.nodes['global node'].data['max x'], 
                                                                ntype='nodes' )
            
            max_idx = dgl.broadcast_nodes(g,g.nodes['global node'].data['max idx'], 
                                                                ntype='nodes' )
            
            max_betas = dgl.broadcast_nodes(g,g.nodes['global node'].data['max beta'], 
                                                                ntype='nodes' )
            
            
            cond_points = (max_idx==g.nodes['nodes'].data['idx']) & (max_betas >= self.t_b)
        
            
            g.nodes['nodes'].data['is cond point'][cond_points] = 1
            g.nodes['nodes'].data['distance'] = torch.norm(g.nodes['nodes'].data['max x']-
                                                                g.nodes['nodes'].data['x'],dim=1)
            distance = g.nodes['nodes'].data['distance']
            selected = (distance <= self.t_d) & (max_betas >= self.t_b)
            
            
            g.nodes['nodes'].data['cluster idx'][selected] = index
            g.nodes['nodes'].data['assigned'][selected] = 1
            
            unassigned_nodes = (g.nodes['nodes'].data['assigned'] < 1) & (g.nodes['nodes'].data['beta'] >= self.t_b)
            
            n_unassigned = len(torch.where(unassigned_nodes)[0])
            count_unassigned+=n_unassigned
            
            #check if all nodes with beta > t_b are assigned - if yes, stop.
            #break
            if count_unassigned==0:
                break

    def label_with_batch_index(self,g):
        
        batch_indx = torch.repeat_interleave(torch.arange(g.batch_size,device=g.device),g.batch_num_nodes('nodes'))
        g.nodes['nodes'].data['batch idx'] = batch_indx


    def create_particles(self,g):

        #self.label_with_batch_index(g)
        predicted_num_particles = dgl.sum_nodes(g,'is cond point',ntype='nodes')

        #cond_points = g.nodes['nodes'].data['is cond point'] > 3
        cond_points = g.nodes['nodes'].data['is cond point'] > 0
        
        particle_pt_eta_xhat_yhat = g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'][cond_points]
        particle_pos = g.nodes['nodes'].data['pos pred'][cond_points]
       
        particle_class  = torch.argmax( g.nodes['nodes'].data['class pred'][cond_points], dim=1)
        particle_charge = torch.argmax( g.nodes['nodes'].data['charge pred'][cond_points], dim=1)


        return (particle_pt_eta_xhat_yhat, particle_pos, particle_class, particle_charge),predicted_num_particles
    

    def undo_scaling(self,particles):

        particle_pt_eta_xhat_yhat, particle_pos, particle_class, particle_charge = particles

        pt = particle_pt_eta_xhat_yhat[:,0]
        pt = self.var_transform['particle_pt']['std']*pt+self.var_transform['particle_pt']['mean']
        pt = torch.exp(pt)

        eta = particle_pt_eta_xhat_yhat[:,1]
        eta = self.var_transform['particle_eta']['std']*eta+self.var_transform['particle_eta']['mean']
        xhat = particle_pt_eta_xhat_yhat[:,2]
        xhat = self.var_transform['particle_xhat']['std']*xhat+self.var_transform['particle_xhat']['mean']
        yhat = particle_pt_eta_xhat_yhat[:,3]
        yhat = self.var_transform['particle_yhat']['std']*yhat+self.var_transform['particle_yhat']['mean']
      
        phi = torch.atan2(yhat,xhat)
        px = pt*torch.cos(phi)
        py = pt*torch.sin(phi)

        theta = 2.0*torch.arctan( -torch.exp( eta ) )
        pz = pt/torch.tan(theta)

        #fdibello
        prod_x = particle_pos[:,0]*self.var_transform['particle_prod_x']['std']+self.var_transform['particle_prod_x']['mean']
        prod_y = particle_pos[:,1]*self.var_transform['particle_prod_y']['std']+self.var_transform['particle_prod_y']['mean']
        prod_z = particle_pos[:,2]*self.var_transform['particle_prod_z']['std']+self.var_transform['particle_prod_z']['mean']

        #prod_x = particle_pos[:,0]*self.var_transform['prod x']['std']+self.var_transform['prod x']['mean']
        #prod_y = particle_pos[:,1]*self.var_transform['prod y']['std']+self.var_transform['prod y']['mean']
        #prod_z = particle_pos[:,2]*self.var_transform['prod z']['std']+self.var_transform['prod z']['mean']
        
        particle_pos = torch.stack([prod_x,prod_y,prod_z],dim=1)

        particle_pxpypz = torch.stack([px,py,pz],dim=1)

        return eta, phi, pt, particle_pxpypz, particle_pos, particle_class, particle_charge

    def forward(self, g):

        self.find_condensation_points(g)

        predicted_particles,predicted_num_particles = self.create_particles(g)
        predicted_particles = self.undo_scaling(predicted_particles) # <-- NOTE: THE TUPLE ARRANGEMENT IS CHANGED W.R.T. PREV LINE!!!

        return predicted_particles,predicted_num_particles