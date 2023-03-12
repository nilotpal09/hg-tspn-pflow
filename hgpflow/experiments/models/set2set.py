import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

from tspn import TSPN
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
class Set2Set(nn.Module):

    def __init__(self,top_level_config):
    
        super().__init__()

        config = top_level_config
        self.var_transform = top_level_config['var transform']

        self.tspns = nn.ModuleDict()
        self.classes = config['classes'] # ["neutral"] # ["photon","charged","neutral","electron","muon"]
        for cl in self.classes:
            self.tspns[cl] = TSPN(top_level_config, cl)


    def undo_scaling(self, particles):

        pt, eta, xhat, yhat, particle_class, particle_charge, prod_x, prod_y, prod_z = particles.transpose(0,1)

        pt = self.var_transform['particle_pt']['std']*pt+self.var_transform['particle_pt']['mean']
        pt = torch.exp(pt)

        eta = self.var_transform['particle_eta']['std']*eta+self.var_transform['particle_eta']['mean']
        xhat = self.var_transform['particle_xhat']['std']*xhat+self.var_transform['particle_xhat']['mean']
        yhat = self.var_transform['particle_yhat']['std']*yhat+self.var_transform['particle_yhat']['mean']
      
        phi = np.arctan2(yhat,xhat)
        px = pt*torch.cos(phi)
        py = pt*torch.sin(phi)

        theta = 2.0*torch.arctan( -torch.exp( eta ) )
        pz = pt/torch.tan(theta)

        prod_x = prod_x * self.var_transform['particle_prod_x']['std'] + self.var_transform['particle_prod_x']['mean']
        prod_y = prod_y * self.var_transform['particle_prod_y']['std'] + self.var_transform['particle_prod_y']['mean']
        prod_z = prod_z * self.var_transform['particle_prod_z']['std'] + self.var_transform['particle_prod_z']['mean']
        
        particle_pos = torch.stack([prod_x,prod_y,prod_z],dim=1)

        particle_pxpypz = torch.stack([px,py,pz],dim=1)

        return eta, phi, pt, particle_pxpypz, particle_pos, particle_class, particle_charge


    def forward(self, g):

        for cl in self.classes:
            if g.number_of_nodes(cl) != 0:
               self.tspns[cl](g)

        return g


    def infer(self,g):

        predicted_num_particles = torch.zeros([g.batch_size])
        predicted_particles_tmp = []
        pred_attention_wts_tmp = []

        for cl_inx, cl in enumerate(self.classes):
            if g.number_of_nodes(cl) != 0:

                self.tspns[cl].infer(g)

                predicted_num_particles_cl = g.batch_num_nodes('pflow '+ cl)
                predicted_num_particles += predicted_num_particles_cl

                particle_properties = torch.cat([
                    g.nodes['pflow '+cl].data['pt_eta_xhat_yhat_pred'],
                    (torch.zeros(g.number_of_nodes(cl)) + cl_inx).unsqueeze(1),
                    torch.zeros((g.number_of_nodes(cl), 4)) # charge, pos(3)
                ], dim=1)

                predicted_particles_tmp.append(torch.split(particle_properties, predicted_num_particles_cl.tolist()))

                # attention
                num_nodes = g.batch_num_nodes('nodes') * predicted_num_particles_cl
                pred_attention_wts_tmp.append(torch.split(g.edges['node_to_pflow_'+cl].data['pred_attention'], num_nodes.tolist()))

        predicted_particles = []
        predicted_attention_weights = []

        for batch_i in range(g.batch_size):
            for cl_inx, cl in enumerate(self.classes):

                num_particles = predicted_particles_tmp[cl_inx][batch_i].shape[0] 

                if num_particles != 0:              
                    predicted_particles.append(predicted_particles_tmp[cl_inx][batch_i])

                    predicted_attention_weights.extend(
                        pred_attention_wts_tmp[cl_inx][batch_i].reshape(-1, num_particles).transpose(0,1)
                    )

        predicted_particles = torch.cat(predicted_particles)        
        predicted_particles = self.undo_scaling(predicted_particles)

        return predicted_particles, predicted_num_particles, predicted_attention_weights

