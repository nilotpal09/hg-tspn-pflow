import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph


import numpy as np
import torch
import torch.nn as nn
from deepset import DeepSet
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from slotattention import SlotAttention


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


#Transformer Set Prediction Network
class TSPN(nn.Module):

    def __init__(self,top_level_config,class_name):
        super().__init__()

        self.class_name = class_name
        config = top_level_config['output model']
        self.var_transform = top_level_config['var transform']


        self.output_setsize_predictor = build_layers(config['set size predictor input size'],outputsize=config['set size max'],
                                         features=config['set size predictor layers']) 


        self.particle_pt_eta_phi_net = build_layers(config['inputsize'],
                                          outputsize=3,
                                          features=config['ptetaphi prediction layers'],add_batch_norm=False) #Ask Jonathan about this

        z_shape = config['z size']
        self.z_shape = config['z size']
        self.z_emb = torch.nn.Embedding(num_embeddings=config['set size max'],embedding_dim=z_shape)
        config_emb = top_level_config['embedding model']
        self.slotattn = nn.ModuleList()
        for i in range(3):
            self.slotattn.append(SlotAttention(config_emb['cell hidden size'],self.z_shape))


    def deltaR(self,phi0,eta0,phi1,eta1):

        deta = eta0-eta1
        dphi = phi0-phi1
        dphi[dphi > np.pi] = dphi[dphi > np.pi]-2*np.pi
        dphi[dphi < - np.pi] = dphi[dphi < - np.pi]+2*np.pi

        dR = torch.sqrt( deta**2+dphi**2 )

        return dR


    def compute_pair_distance(self,edges):
               
        phi_loss = self.regression_loss(edges.dst['pred phi'],  edges.src['particle_phi'])
        eta_loss = self.regression_loss(edges.dst['pred eta'],  edges.src['particle_eta'])
        #dRloss = torch.sqrt( phi_loss+eta_loss )
        #dRloss = torch.norm(pred_phieta-target_phieta,dim=1) 
        dRloss = phi_loss+eta_loss#self.deltaR( edges.dst['init_phi'], edges.dst['init_eta'],  edges.src['particle_phi'], edges.src['particle_eta'])+0.001 #torch.sqrt( phi_loss+eta_loss )

        #dRloss = phi_loss
        
        return {'dRloss': dRloss, 'phi' :   edges.src['particle_phi'], 'eta' : edges.src['particle_eta']}


    def find_matched_phi_eta(self,g):

        g.apply_edges(self.compute_pair_distance,etype='to_pflow')
        
        data = g.edges['to_pflow'].data['dRloss'].detach().cpu().data.numpy()#+0.1
        u = g.all_edges(etype='to_pflow')[0].cpu().data.numpy().astype(int)
        v = g.all_edges(etype='to_pflow')[1].cpu().data.numpy().astype(int)
        m = csr_matrix((data,(u,v)))
        
        selected_columns = min_weight_full_bipartite_matching(m)[1]

        # n_particles_per_event = [n.item() for n in g.batch_num_nodes('particles')]
        # col_offest = np.repeat( np.cumsum([0]+n_particles_per_event[:-1]), n_particles_per_event)
        # row_offset = np.concatenate([[0]]+[[n]*n for n in n_particles_per_event])[:-1]
        # row_offset = np.cumsum(row_offset)

        matched_phi = g.nodes['particles'].data['particle_phi'][selected_columns] #g.edges['to_pflow'].data['phi' ][selected_columns-col_offest+row_offset]
        matched_eta = g.nodes['particles'].data['particle_eta'][selected_columns] #g.edges['to_pflow'].data['eta'][selected_columns-col_offest+row_offset]

        g.nodes['pflow particles'].data['matched particles'] = torch.LongTensor(selected_columns)

        return matched_phi, matched_eta


    def create_outputgraphs(self, g, n_nodes, nparticles):

        outputgraphs = []

        for n_node,N in zip(n_nodes, nparticles):

            n_node = n_node.cpu().data;  N = N.cpu().data
            num_nodes_dict = {
                'particles' : N,
                'nodes' : n_node
            }

            estart = torch.repeat_interleave( torch.arange(n_node),N)
            eend   = torch.arange(N).repeat(n_node)

            data_dict = {
                ('nodes','node_to_particle','particles') : (estart,eend)
            }

            outputgraphs.append( dgl.heterograph(data_dict, num_nodes_dict, device=g.device) )

        outputgraphs = dgl.batch(outputgraphs)
        outputgraphs.nodes['nodes'].data['hidden rep'] = g.nodes['nodes'].data['hidden rep']

        indexses = torch.cat([torch.linspace(0,N-1,N,device=g.device).view(N).long() for N in nparticles],dim=0) 
        Z = self.z_emb(indexses)

        #change name
        outputgraphs.nodes['particles'].data['node hidden rep'] = Z

        # create hidden rep for the output objects based on the global rep of the input set and the init for the new objects        
        inputset_global = g.nodes['global node'].data['global rep']
       
        outputgraphs.nodes['particles'].data['global rep'] = dgl.broadcast_nodes(outputgraphs,inputset_global,ntype='particles')
        
        for i, slotatt in enumerate(self.slotattn):
            slotatt(outputgraphs)

        ndata = torch.cat( [outputgraphs.nodes['particles'].data['node hidden rep'],outputgraphs.nodes['particles'].data['global rep']],dim=1)

        outputgraphs.nodes['particles'].data['pt_eta_phi_pred'] = self.particle_pt_eta_phi_net(ndata)
   
        return outputgraphs


    def forward(self, g):

        outputgraphs = self.create_outputgraphs(g,g.batch_num_nodes('nodes'), g.batch_num_nodes('pflow '+self.class_name))
        nprediction = outputgraphs.nodes['particles'].data['pt_eta_phi_pred']
        g.nodes['pflow '+self.class_name].data["pt_eta_phi_pred"] = nprediction
        g.edges['node_to_pflow_'+self.class_name].data["pred_attention"] = outputgraphs.edges['node_to_particle'].data['attention_weights']

        return g
    
    
    def create_particles(self,g):
        
        # predicted_setsizes coming soon!
        # cells_global = dgl.mean_nodes(g,'set size rep',ntype='cells') 
        # tracks_global = dgl.mean_nodes(g,'set size rep',ntype='tracks')
        # all_global = torch.cat([cells_global,tracks_global],dim=1)

        # predicted_setsizes = self.output_setsize_predictor(all_global)

        # predicted_n = torch.torch.multinomial(torch.softmax(predicted_setsizes,dim=1),1).view(-1)

        predicted_n = g.number_of_nodes(self.class_name)
        outputgraphs = self.create_outputgraphs(g, g.batch_num_nodes('nodes'), g.batch_num_nodes('pflow '+self.class_name))
        ndata = outputgraphs.ndata['node hidden rep']

        # particle_charge = torch.argmax(self.charge_class_net(ndata),dim=1)
        # particle_e = self.particle_energy_net(ndata).view(-1)
        # particle_pos = self.particle_position_net(ndata)
        # particle_3mom = self.particle_3momentum(ndata)

        # return (particle_e,particle_3mom, particle_pos, particle_class,particle_charge),predicted_setsizes

        g.nodes['pflow ' + self.class_name].data['pt_eta_phi_pred'] = outputgraphs.nodes['particles'].data['pt_eta_xhat_yhat_pred']

    
    def undo_scaling(self,particles):

        particle_e, particle_pxpypz, particle_pos, particle_class, particle_charge = particles

        p_e = self.var_transform['particle e']['std']*particle_e+self.var_transform['particle e']['mean']
        p_e = torch.exp(p_e)

        px = particle_pxpypz[:,0]*self.var_transform['p_x']['std']+self.var_transform['p_x']['mean']
        py = particle_pxpypz[:,1]*self.var_transform['p_y']['std']+self.var_transform['p_y']['mean']
        pz = particle_pxpypz[:,2]*self.var_transform['p_z']['std']+self.var_transform['p_z']['mean']

        particle_pxpypz = torch.stack([px,py,pz],dim=1)

        prod_x = particle_pos[:,0]*self.var_transform['prod x']['std']+self.var_transform['prod x']['mean']
        prod_y = particle_pos[:,1]*self.var_transform['prod y']['std']+self.var_transform['prod y']['mean']
        prod_z = particle_pos[:,2]*self.var_transform['prod z']['std']+self.var_transform['prod z']['mean']
        
        particle_pos = torch.stack([prod_x,prod_y,prod_z],dim=1)

        return p_e, particle_pxpypz, particle_pos, particle_class, particle_charge

    
    def infer(self,g):
        
        self.create_particles(g)

