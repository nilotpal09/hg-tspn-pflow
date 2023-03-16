import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class HungarianLoss(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config

        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.regression_loss = nn.MSELoss(reduction='none')

    def deltaR(self,phi0,eta0,phi1,eta1):

        deta = eta0-eta1
        dphi = phi0-phi1
        dphi[dphi > np.pi] = dphi[dphi > np.pi]-2*np.pi
        dphi[dphi < - np.pi] = dphi[dphi < - np.pi]+2*np.pi

        dR = torch.sqrt( deta**2+dphi**2 )

        return dR


    def compute_pair_loss(self,edges):

        class_loss = self.CrossEntropyLoss(edges.dst['pred particle class'],edges.src['particle class'])
        charge_loss = self.CrossEntropyLoss(edges.dst['pred charge class'],edges.src['charge class'])
        pos_loss = 0.1*torch.norm(edges.dst['pred pos']-edges.src['prod pos'],dim=1) 
        
        momentum_loss= 0.1*self.regression_loss(edges.dst['pred momentum'],edges.src['particle_pt']) 
        
        #dRloss = self.deltaR( edges.dst['pred phi'], edges.dst['pred eta'],  edges.src['particle_phi'], edges.src['particle_eta'])
        # pred_phieta = torch.stack( [edges.dst['pred phi'], edges.dst['pred eta']],dim=1 )
        # target_phieta = torch.stack( [edges.src['particle_phi'], edges.src['particle_eta']],dim=1 )
        phi_loss = self.regression_loss(edges.dst['pred phi'],  edges.src['particle_phi'])
        eta_loss = self.regression_loss(edges.dst['pred eta'],  edges.src['particle_eta'])
        #dRloss = torch.norm(pred_phieta-target_phieta,dim=1) 
        dRloss = phi_loss+eta_loss #torch.sqrt( phi_loss+eta_loss )

        loss = dRloss #+class_loss+momentum_loss #this is the loss that will determine the assignments
        
        return {'loss': loss, 'class l': class_loss, 'charge l': charge_loss, 'pos l' :pos_loss,
               'mom l': momentum_loss, 'dRloss': dRloss}


    def forward(self, g):
        
        # g.apply_edges(self.compute_pair_loss,etype='to_pflow')

        
        # data = g.edges['to_pflow'].data['loss'].cpu().data.numpy()#+0.1
        # u = g.all_edges(etype='to_pflow')[0].cpu().data.numpy().astype(int)
        # v = g.all_edges(etype='to_pflow')[1].cpu().data.numpy().astype(int)
        # m = csr_matrix((data,(u,v)))
        
        # selected_columns = min_weight_full_bipartite_matching(m)[1]

        
        # #converting the selected columns from the sparse matrix to their index in the edge list of the graph
        #n_particles_per_event = [n.item() for n in g.batch_num_nodes('particles')]
        # col_offest = np.repeat( np.cumsum([0]+n_particles_per_event[:-1]), n_particles_per_event)
        # row_offset = np.concatenate([[0]]+[[n]*n for n in n_particles_per_event])[:-1]
        # row_offset = np.cumsum(row_offset)
        
        # g.nodes['pflow particles'].data['class l' ] = g.edges['to_pflow'].data['class l' ][selected_columns-col_offest+row_offset]
        # g.nodes['pflow particles'].data['charge l'] = g.edges['to_pflow'].data['charge l'][selected_columns-col_offest+row_offset]
        # g.nodes['pflow particles'].data['pos l'   ] = g.edges['to_pflow'].data['pos l'   ][selected_columns-col_offest+row_offset]
        # g.nodes['pflow particles'].data['mom l'   ] = g.edges['to_pflow'].data['mom l'   ][selected_columns-col_offest+row_offset]
        # g.nodes['pflow particles'].data['dRloss'  ] = g.edges['to_pflow'].data['dRloss'  ][selected_columns-col_offest+row_offset]
        
        n_particles_per_event = g.batch_num_nodes('particles')
        selected_columns = g.nodes['pflow particles'].data['matched particles']
        
        class_loss = self.CrossEntropyLoss(g.nodes['pflow particles'].data['pred particle class'],g.nodes['particles'].data['particle class'][selected_columns])
        charge_loss = self.CrossEntropyLoss(g.nodes['pflow particles'].data['pred charge class'],g.nodes['particles'].data['charge class'][selected_columns])
        pos_loss = 0.1*torch.norm(g.nodes['pflow particles'].data['pred pos']-g.nodes['particles'].data['prod pos'][selected_columns],dim=1) 
        
        momentum_loss= 0.1*self.regression_loss(g.nodes['pflow particles'].data['pred momentum'],g.nodes['particles'].data['particle_pt'][selected_columns]) 
        
       
        phi_loss = self.regression_loss(g.nodes['pflow particles'].data['pred phi'],  g.nodes['particles'].data['particle_phi'][selected_columns])
        eta_loss = self.regression_loss(g.nodes['pflow particles'].data['pred eta'],  g.nodes['particles'].data['particle_eta'][selected_columns])
        
        dRloss = phi_loss+eta_loss #self.deltaR( g.nodes['pflow particles'].data['pred phi'],g.nodes['pflow particles'].data['pred eta'],
                        #                         g.nodes['particles'].data['particle_phi'][selected_columns], 
                         #                        g.nodes['particles'].data['particle_eta'][selected_columns]) #torch.sqrt( phi_loss+eta_loss )


        g.nodes['pflow particles'].data['class l' ] = class_loss
        g.nodes['pflow particles'].data['charge l'] = charge_loss
        g.nodes['pflow particles'].data['pos l'   ] = pos_loss
        g.nodes['pflow particles'].data['mom l'   ] = momentum_loss
        g.nodes['pflow particles'].data['dRloss'  ] = dRloss


        classloss = dgl.mean_nodes(g,'class l',ntype='pflow particles').mean()
        chargeloss = dgl.mean_nodes(g,'charge l',ntype='pflow particles').mean()
        posloss   = dgl.mean_nodes(g,'pos l',ntype='pflow particles').mean()
        momloss   = dgl.mean_nodes(g,'mom l',ntype='pflow particles').mean()
        dRloss     = dgl.mean_nodes(g,'dRloss',ntype='pflow particles').mean()
        
       
        
        
        #number of particles prediction
        predicted_setsizes = g.nodes['global node'].data['predicted_setsizes']
        
        setsize_loss = self.CrossEntropyLoss(predicted_setsizes,n_particles_per_event)
        setsize_loss = setsize_loss.mean()
        
        loss = dRloss #+classloss+chargeloss+posloss+momloss+setsize_loss

        return {'loss':loss ,
                'setsize loss' : setsize_loss.item(),
               'class l': classloss.item(), 'charge l': classloss.item(), 'pos l' :posloss.item()
               ,'mom l': momloss.item(), 'dRloss': dRloss.item()}

class ChamferLoss(nn.Module):
    def __init__(self,config):
        super(ChamferLoss, self).__init__()

        self.config = config

        self.cel = nn.CrossEntropyLoss(reduction='none')
        self.regression_loss = nn.MSELoss(reduction='none')


    def compute_pair_loss(self,edges):

        class_loss = self.cel(edges.dst['particle class'],edges.src['particle class'])
        charge_loss = self.cel(edges.dst['charge class'],edges.src['charge class'])
        pos_loss = torch.norm(edges.src['prod pos']-edges.dst['prod pos'],dim=1)
        momentum_loss = torch.norm(edges.src['3 momentum']-edges.dst['3 momentum'],dim=1)
        energy_loss= self.regression_loss(edges.dst['energy'],edges.src['energy'])
        
        loss = class_loss+charge_loss+pos_loss+momentum_loss+energy_loss
        
        return {'loss': loss, 'class l': class_loss, 'charge l': charge_loss, 'pos l' :pos_loss,
               '3mom l': momentum_loss, 'energy l': energy_loss}
    
    
    
    def get_min_loss(self,nodes):
        
        min_loss,_ = torch.min(nodes.mailbox['m'],dim=1)
        
        return {'min loss' : min_loss}
        
    def forward(self, g):
        
        g.apply_edges(self.compute_pair_loss,etype='to_pflow')

        u = g.all_edges(etype='to_pflow')[0]
        v = g.all_edges(etype='to_pflow')[1]
        
        
        g.edges['from_pflow'].data['loss'] = g.edges['to_pflow'].data['loss']
        
        g.update_all(fn.copy_edge('loss','m'),self.get_min_loss,etype='to_pflow')
        g.update_all(fn.copy_edge('loss','m'),self.get_min_loss,etype='from_pflow')
        
        particle_loss = dgl.sum_nodes(g,'min loss',ntype='particles').mean()
        pflow_loss = dgl.sum_nodes(g,'min loss',ntype='pflow particles').mean()
        
        

        #number of particles prediction
        n_particles_per_event = g.batch_num_nodes('particles')
        predicted_setsizes = dgl.mean_nodes(g,'predicted_setsizes',ntype='pflow particles')
        
        setsize_loss = self.cel(predicted_setsizes,n_particles_per_event.long())
        setsize_loss = setsize_loss.mean()
        
        loss = particle_loss+pflow_loss+setsize_loss
        return {'loss':loss , 'particle loss': particle_loss.item(),
                'setsize loss' : setsize_loss.item(),'pflow loss' : pflow_loss.item()}