import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn


class CondNetInference(nn.Module):
    def __init__(self,config,use_truth=False):
        super().__init__()

        self.config = config
        self.var_transform = self.config['var transform']
        self.t_b_opt = self.config['t_b_opt']
        self.t_d_opt = self.config['t_d_opt']
        self.pt_bins = self.config['bin_opt']

        self.optimize = False #Use pt-binned values for the tb and td thresholds

        self.use_truth = use_truth
        if 'truth inference' in self.config:
            self.use_truth = self.use_truth or bool(self.config['truth inference'])

        #print('DEBUG: self.use_truth = ', self.use_truth)

        self.x_size = config['output model']['x size']
        self.t_b = config['t_b']
        self.t_d = config['t_d']

    def decorate_nodes_with_thresholds(self,g):

        if self.optimize:
            for bin_i, pt_lo in enumerate(self.pt_bins):

                if bin_i == len(self.pt_bins)-1:
                    continue
                pt_hi = self.pt_bins[bin_i+1]
                if bin_i+1 == len(self.pt_bins)-1:
                    pt_hi = 9999.

                t_b = self.t_b_opt[bin_i]
                t_d = self.t_d_opt[bin_i]

                true_pt = g.nodes['nodes'].data['parent_pt'].clone()
                true_pt = true_pt*self.var_transform['particle_pt']['std'] + self.var_transform['particle_pt']['mean']
                true_pt = torch.exp(true_pt)
                true_pt = true_pt/1000. #MeV --> GeV

                pred_pt = g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'][:,0].clone()
                pred_pt = self.var_transform['particle_pt']['std']*pred_pt+self.var_transform['particle_pt']['mean']
                pred_pt = torch.exp(pred_pt)/1000.  #MeV --> GeV
                print(pred_pt[-1],pt_lo,pt_hi,)

                node_in_bin = torch.logical_and(pred_pt >= pt_lo, pred_pt < pt_hi)
                g.nodes['nodes'].data['t_b'] = torch.where(node_in_bin,t_b*torch.ones_like(g.nodes['nodes'].data['t_b']),g.nodes['nodes'].data['t_b'])
                g.nodes['nodes'].data['t_d'] = torch.where(node_in_bin,t_d*torch.ones_like(g.nodes['nodes'].data['t_d']),g.nodes['nodes'].data['t_d'])
        else:
                g.nodes['nodes'].data['t_b'] = self.t_b*torch.ones_like(g.nodes['nodes'].data['t_b'])
                g.nodes['nodes'].data['t_d'] = self.t_d*torch.ones_like(g.nodes['nodes'].data['t_d'])


    def pass_edge_info(sefl,edges):
    
        #assigned nodes get a beta of 0, so they can't be the max anymore.
        remove_assigned = (1-edges.src['assigned'])
        
        return {'betas': edges.src['beta']*remove_assigned,'xs' : edges.src['x'],'idx' : edges.src['idx'],  'isTrack': edges.src['isTrack']*remove_assigned}

    def sort_beta(self,graphs):
        
        N = len(graphs)

        #give crazy (but equal) boost to beta of nodes which are unassigned tracks
        max_beta, where = torch.max(graphs.mailbox['betas']*(1 + 999*graphs.mailbox['isTrack']),dim=1)

        max_beta = graphs.mailbox['betas'][torch.arange(N),where]
        max_x    = graphs.mailbox['xs'][torch.arange(N),where]
        max_idx  = graphs.mailbox['idx'][torch.arange(N),where]
        max_isTrack = graphs.mailbox['isTrack'][torch.arange(N),where]
            
        #N.B.: don't know what these stacks and the subsequent max is for... gives same result except where = 1
        #beta_stack = torch.stack([graphs.data['max beta'],max_beta],dim=1)
        #x_stack = torch.stack([graphs.data['max x'],max_x],dim=1)
        #idx_stack = torch.stack([graphs.data['max idx'],max_idx],dim=1)
        
        #max_beta, where = torch.max(beta_stack,dim=1)
        #max_x = x_stack[torch.arange(N),where]
        #max_idx = idx_stack[torch.arange(N),where]

        return {'max beta': max_beta, 'max x': max_x, 'max idx' : max_idx, 'max isTrack': max_isTrack}

    def init_cells(self,g):
        
        #g.nodes['nodes'].data['idx'] = torch.arange(g.num_nodes('nodes'),device=g.device) #moved to dataloader
        
        g.nodes['nodes'].data['assigned'] = torch.zeros(g.num_nodes('nodes'),device=g.device)
        g.nodes['nodes'].data['cluster idx'] = -1*torch.ones(g.num_nodes('nodes'),device=g.device)
        g.nodes['nodes'].data['is cond point'] = torch.zeros(g.num_nodes('nodes'),device=g.device)
        g.nodes['nodes'].data['t_b'] = -1.*torch.ones(g.num_nodes('nodes'),device=g.device)
        g.nodes['nodes'].data['t_d'] = -1.*torch.ones(g.num_nodes('nodes'),device=g.device)
    
    def find_condensation_points(self, g):
                    
        index = -1
        n_unassigned = 999
        while True:
        
            #print('starting round ',index) 
            index+=1
            ## init the global nodes
            g.nodes['global node'].data['max beta'] = torch.zeros(g.num_nodes('global node'),device=g.device)
            g.nodes['global node'].data['max x'] = torch.zeros(g.num_nodes('global node'),self.x_size,device=g.device)
            g.nodes['global node'].data['max idx'] = torch.zeros(g.num_nodes('global node'),device=g.device)
            g.nodes['global node'].data['max isTrack'] = torch.zeros(g.num_nodes('global node'),device=g.device)

            #find the max beta and associated x for each graph.
            g.update_all(self.pass_edge_info,self.sort_beta,etype='nodes to global')
            
            
            #assign cells and tracks to the current "object" with max beta in each graph in the batch
                        
                
            g.nodes['nodes'].data['max x'] = dgl.broadcast_nodes(g, 
                                                                g.nodes['global node'].data['max x'], 
                                                                ntype='nodes' )
            
            max_idx = dgl.broadcast_nodes(g,g.nodes['global node'].data['max idx'], 
                                                                ntype='nodes' )
            
            max_betas = dgl.broadcast_nodes(g,g.nodes['global node'].data['max beta'], 
                                                                ntype='nodes' )
       
            max_isTrack = dgl.broadcast_nodes(g,g.nodes['global node'].data['max isTrack'], 
                                                                ntype='nodes' )

            cond_points = (max_idx==g.nodes['nodes'].data['idx']) & ((max_betas >= g.nodes['nodes'].data['t_b']) | (g.nodes['nodes'].data['isTrack']==1))
            newTrackCPs = (max_idx==g.nodes['nodes'].data['idx']) & (g.nodes['nodes'].data['isTrack']==1)
        
            
            g.nodes['nodes'].data['is cond point'][cond_points] = 1
            g.nodes['nodes'].data['distance'] = torch.norm(g.nodes['nodes'].data['max x']-
                                                                g.nodes['nodes'].data['x'],dim=1)
            distance = g.nodes['nodes'].data['distance']


            #if the new CP is a track, no elimination based on t_d (artifically huge distance thresh)
            #newselected = (g.nodes['nodes'].data['assigned']==0) & (distance <= self.t_d*(1 - max_isTrack)) & (g.nodes['nodes'].data['beta'] >= self.t_b) & (g.nodes['nodes'].data['isTrack']==0) #these selected will be disqualified from next round
            newselected = (g.nodes['nodes'].data['assigned']==0) & (distance <= g.nodes['nodes'].data['t_d']) & (g.nodes['nodes'].data['beta'] >= g.nodes['nodes'].data['t_b']) & (g.nodes['nodes'].data['isTrack']==0) #these selected will be disqualified from next round
            newselected = newselected | newTrackCPs
            
            g.nodes['nodes'].data['cluster idx'][newselected] = index
            g.nodes['nodes'].data['assigned'][newselected] = 1
            
            unassigned_candidates = (g.nodes['nodes'].data['assigned'] < 1) & ((g.nodes['nodes'].data['beta'] >= g.nodes['nodes'].data['t_b']) | (g.nodes['nodes'].data['isTrack']==1))
            
            n_unassigned_pre = n_unassigned
            n_unassigned = len(torch.where(unassigned_candidates)[0])

            if n_unassigned==0 or (n_unassigned_pre == n_unassigned):
                break

    def find_condensation_points_truth(self, g):
        
        true_cond_wheres = g.nodes['particles'].data['max idx'].to(torch.long)
        g.nodes['nodes'].data['is cond point'][true_cond_wheres] = 1


    def label_with_batch_index(self,g):
        
        batch_indx = torch.repeat_interleave(torch.arange(g.batch_size,device=g.device),g.batch_num_nodes('nodes'))
        g.nodes['nodes'].data['batch idx'] = batch_indx


    def create_particles(self,g):

        cell_class_pred  = torch.argmax( g.nodes['nodes'].data['cell class pred'],  dim=1)
        track_class_pred = torch.argmax( g.nodes['nodes'].data['track class pred'], dim=1) + 2*torch.ones_like(g.nodes['nodes'].data['isTrack'])
        particle_class   = torch.where(g.nodes['nodes'].data['isTrack']==1,track_class_pred,cell_class_pred)
        mislabeled_cell_asCharged = ((g.nodes['nodes'].data['isTrack']==0) & (particle_class > 1 )).long() #for cells which are identified as charged classes, ignore

        #self.label_with_batch_index(g)
        g.nodes['nodes'].data['is cond point'] = g.nodes['nodes'].data['is cond point']*(1-g.nodes['nodes'].data['n'])*(1-mislabeled_cell_asCharged)
        predicted_num_particles = dgl.sum_nodes(g,'is cond point',ntype='nodes')

        #cond_points = g.nodes['nodes'].data['is cond point'] > 3
        cond_points = g.nodes['nodes'].data['is cond point'] > 0

        target_pt_eta_xhat_yhat = torch.cat([g.nodes['nodes'].data['parent_pt'].unsqueeze(1)
            ,g.nodes['nodes'].data['parent_eta'].unsqueeze(1),
            g.nodes['nodes'].data['parent_xhat'].unsqueeze(1),g.nodes['nodes'].data['parent_yhat'].unsqueeze(1) ],dim=1)

        node_xhat_yhat = torch.cat([g.nodes['nodes'].data['cosin_phi'].unsqueeze(1),g.nodes['nodes'].data['sinu_phi'].unsqueeze(1) ], dim=1)
        node_pt_eta_xhat_yhat  = torch.cat([g.nodes['nodes'].data['track_pt'].unsqueeze(1),g.nodes['nodes'].data['eta'].unsqueeze(1), node_xhat_yhat],dim=1).float()

        track_pt = g.nodes['nodes'].data['track_pt'].clone()
        track_pt = torch.where(track_pt!=0,track_pt*self.var_transform['track_pt']['std'] + self.var_transform['track_pt']['mean'],track_pt)
        track_pt = torch.where(track_pt!=0,torch.exp(track_pt),track_pt)
        track_pt = track_pt/1000. #MeV --> GeV

        #print("DEBUG: track_pt")
        #print(track_pt)
        #print("DEBUG: target_pt_eta_xhat_yhat")
        #print(target_pt_eta_xhat_yhat)
        #print("DEBUG: pt_eta_xhat_yhat_pred")
        #print(g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'])

        low_pt_tracks = torch.logical_and(g.nodes['nodes'].data['isTrack']==1 , track_pt<15).unsqueeze(-1)
        g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'] = torch.where(low_pt_tracks,node_pt_eta_xhat_yhat,g.nodes['nodes'].data['pt_eta_xhat_yhat_pred']) #NASTY HACK!        
        particle_pt_eta_xhat_yhat = g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'][cond_points]
       
        relabel_as_0        = 0*torch.ones_like(g.nodes['nodes'].data['particle class'])
        relabel_as_1        = 1*torch.ones_like(g.nodes['nodes'].data['particle class'])
        relabel_as_2        = 2*torch.ones_like(g.nodes['nodes'].data['particle class'])
        relabel_as_4        = 4*torch.ones_like(g.nodes['nodes'].data['particle class'])
        not_cond_point      = torch.where(cond_points != cond_points,True,False)

        mislabeled_track = (g.nodes['nodes'].data['isTrack']==1) & ((particle_class==0) | (particle_class==1))
        particle_class   = torch.where(mislabeled_track,relabel_as_2,particle_class) #for tracks which are identified as neutral, relabel by hand as charged

        neutral_cell_layer0 = ((g.nodes['nodes'].data['isTrack']==0) & (particle_class==1) & (g.nodes['nodes'].data['layer_cell']==0))
        particle_class      = torch.where(neutral_cell_layer0,relabel_as_0,particle_class) #for neutral had. predictions in layer 0, they are probably photons


        particle_class   = particle_class[cond_points] #take only the values of CPs
        
        #return also the parent particle index of this node for truth links for performance metrics
        particle_parent  = g.nodes['nodes'].data['parent target'][cond_points]

        return (particle_pt_eta_xhat_yhat, particle_class, particle_parent),predicted_num_particles
    

    def undo_scaling(self,particles):

        particle_pt_eta_xhat_yhat, particle_class, particle_parent = particles

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
        
        particle_pxpypz = torch.stack([px,py,pz],dim=1)

        return eta, phi, pt, particle_pxpypz, particle_class, particle_parent

    def forward(self, g):

        self.init_cells(g)

        self.decorate_nodes_with_thresholds(g)

        if self.use_truth:
            self.find_condensation_points_truth(g)
        else:
            self.find_condensation_points(g)

        predicted_particles,predicted_num_particles = self.create_particles(g)
        predicted_particles = self.undo_scaling(predicted_particles) # <-- NOTE: THE TUPLE ARRANGEMENT IS CHANGED W.R.T. PREV LINE!!!

        return predicted_particles,predicted_num_particles
