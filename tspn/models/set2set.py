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

# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# import copy


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
        self.classes = ["supercharged","superneutral"]
        # self.classes = ["supercharged","neutral","photon"]
        for cl in self.classes:
            self.tspns[cl] = TSPN(top_level_config,cl)


    def undo_scaling(self, particles):

        #print(particles.transpose(0,1).shape)

        pt, eta, phi, particle_class,target,particle_charge, prod_x, prod_y, prod_z = particles.transpose(0,1)
        #pt, eta, phi, particle_class, particle_charge,target, prod_x, prod_y, prod_z = particles.transpose(0,1)

        #print("undo",target)

        pt = self.var_transform['particle_pt']['std']*pt+self.var_transform['particle_pt']['mean']
        pt = torch.exp(pt)

        eta_tra = self.var_transform['particle_eta']['std']*eta+self.var_transform['particle_eta']['mean']
        phi_tra = self.var_transform['particle_phi']['std']*phi+self.var_transform['particle_phi']['mean']
        
        eta[particle_class < 2] = eta_tra[particle_class < 2]
        phi[particle_class < 2] = phi_tra[particle_class < 2]

        #phi = np.arctan2(yhat,xhat)
        px = pt*torch.cos(phi)
        py = pt*torch.sin(phi)

        theta = 2.0*torch.arctan( -torch.exp( eta ) )
        pz = pt/torch.tan(theta)

        prod_x = prod_x * self.var_transform['particle_prod_x']['std'] + self.var_transform['particle_prod_x']['mean']
        prod_y = prod_y * self.var_transform['particle_prod_y']['std'] + self.var_transform['particle_prod_y']['mean']
        prod_z = prod_z * self.var_transform['particle_prod_z']['std'] + self.var_transform['particle_prod_z']['mean']
        
        particle_pos = torch.stack([prod_x,prod_y,prod_z],dim=1)

        particle_pxpypz = torch.stack([px,py,pz],dim=1)

        return eta, phi, pt, particle_pxpypz, particle_pos, particle_class, particle_charge, target


    # def mp_fun(self, args):
    #     cl, g = args
    #     self.tspns[cl](g)
    #     g = g.to(torch.device('cpu'))

    #     return g


    def forward(self, g):

        for cl in self.classes:
            if g.number_of_nodes(cl) != 0:
               self.tspns[cl](g)

        return g

        # pool = mp.Pool(processes=5)
        # args = zip(self.classes, [g,g,g,g,g])
        # results = pool.map(self.mp_fun, args)
        # pool.close()
        # pool.join()

        # return results.get()[1]


    def infer(self,g):
        predicted_num_particles = torch.zeros([g.batch_size])
        predicted_particles_tmp = []
        pred_attention_wts_tmp = []

        n_dict = {
        "supercharged": 0,
        "superneutral": 0
#        "photon": 0,
#        "neutral": 0
        }

        for cl_inx, cl in enumerate(self.classes):

            print(g.number_of_nodes(cl),cl,"class")
            if g.number_of_nodes(cl) > -1:

                self.tspns[cl].infer(g,0)

        

                predicted_num = 0
                #predicted_num_particles_cl is in the batch while pred_n_gamma is not for the batch
                if cl=="photon": 
                    pred_n_gamma_low      = g.nodes['global node'].data["n_"+cl+"_low"]
                    pred_n_gamma_midlow   = g.nodes['global node'].data["n_"+cl+"_midlow"] 
                    pred_n_gamma_midhigh  = g.nodes['global node'].data["n_"+cl+"_midhigh"]
                    pred_n_gamma_high     = g.nodes['global node'].data["n_"+cl+"_high"] 

                    pred_n_gamma = torch.stack((pred_n_gamma_low,pred_n_gamma_midlow,pred_n_gamma_midhigh,pred_n_gamma_high), dim=0)
                    #predicted_num_particles_cl = torch.sum(pred_n_gamma, dim=0)
                    g.nodes['global node'].data["n_"+cl+"_tot"] =  pred_n_gamma_low + pred_n_gamma_midlow +  pred_n_gamma_midhigh + pred_n_gamma_high
                    #g.nodes['global node'].data["n_"+cl+"_tot"]  = g.batch_num_nodes(cl) 
                    predicted_num = g.nodes['global node'].data["n_"+cl+"_tot"] 
                    predicted_num_particles_cl = predicted_num
                    #print(predicted_num_particles_cl)

                elif cl=="neutral": 
                    pred_n_neutral_low      = g.nodes['global node'].data["n_"+cl+"_low"]
                    pred_n_neutral_midlow   = g.nodes['global node'].data["n_"+cl+"_midlow"] 
                    pred_n_neutral_midhigh  = g.nodes['global node'].data["n_"+cl+"_midhigh"]
                    pred_n_neutral_high     = g.nodes['global node'].data["n_"+cl+"_high"] 

                    g.nodes['global node'].data["n_"+cl+"_tot"] =  pred_n_neutral_low + pred_n_neutral_midlow +  pred_n_neutral_midhigh + pred_n_neutral_high
                    predicted_num = g.nodes['global node'].data["n_"+cl+"_tot"] 
                    #print("---->",pred_n_neutral_low)

                elif cl=="superneutral": 
                    pred_n_supneutral_low      = g.nodes['global node'].data["n_"+cl+"_low"]
                    pred_n_supneutral_midlow   = g.nodes['global node'].data["n_"+cl+"_midlow"] 
                    pred_n_supneutral_midhigh  = g.nodes['global node'].data["n_"+cl+"_midhigh"]
                    pred_n_supneutral_high     = g.nodes['global node'].data["n_"+cl+"_high"] 

                    g.nodes['global node'].data["n_"+cl+"_tot"] =  pred_n_supneutral_low + pred_n_supneutral_midlow +  pred_n_supneutral_midhigh + pred_n_supneutral_high
                    predicted_num = g.nodes['global node'].data["n_"+cl+"_tot"] 

                
                    pred_n_supneutral=torch.stack((pred_n_supneutral_low,pred_n_supneutral_midlow,pred_n_supneutral_midhigh,pred_n_supneutral_high),dim=0)
                    #predicted_num_particles_cl = torch.sum(pred_n_neutral, dim=0)
                    predicted_num_particles_cl = predicted_num



                #supercharged does not need any number prediction

                if cl == "supercharged":
                    particle_class = torch.argmax(g.nodes['pflow '+cl].data['class_pred'],dim=1)
                    particle_class[particle_class==3] = 4
                    particle_class[particle_class==1] = 3
                    particle_class[particle_class==2] = 3
                    particle_class[particle_class==0] = 2
                    particle_prediction = g.nodes['pflow '+cl].data['pt_eta_phi_pred']
                    particle_target = g.nodes['pflow '+cl].data['target idx']
                    ncl = g.number_of_nodes(cl)
                    predicted_num_particles_cl = g.batch_num_nodes('pflow '+ cl)    

                else:
                    pred_graph = self.tspns[cl].infer(g,1)
                    #particle_class = pred_graph.nodes['particles_'+cl].data['class_pred'] 
                    particle_class = torch.argmax(pred_graph.nodes['particles_'+cl].data['class_pred'],dim=1)
                    #particle_prediction = g.nodes['pflow '+cl].data['pt_eta_phi_pred']
                    particle_prediction = pred_graph.nodes['particles_'+cl].data['pt_eta_phi_pred'] 
                    particle_target = pred_graph.nodes['particles_'+cl].data['target idx']  
                    ncl = pred_graph.number_of_nodes('particles_'+cl)  
                    predicted_num_particles_cl = pred_graph.batch_num_nodes('particles_'+cl)    
         
                
                #elif cl == "supercharged":
                predicted_num_particles+=predicted_num_particles_cl
                #print(predicted_num_particles_cl)
            


                if particle_prediction.shape[0] > 0:
                    print("cl",cl,particle_class,ncl)
                    particle_properties = torch.cat([
                        particle_prediction,
                        particle_class.unsqueeze(1),
                        particle_target.unsqueeze(1),
                        #(torch.zeros(g.number_of_nodes(cl)) + cl_inx).unsqueeze(1),
                        torch.zeros((ncl, 4)) # charge, pos(3)
                        ], dim=1)
                #if there are no reco particles
                else:  
                    particle_properties = torch.cat([torch.zeros(0),torch.zeros(0)],dim=1)
  
                #print (predicted_num_particles_cl)
                if cl == "supercharged": number = g.number_of_nodes(cl) 
                else:  number = pred_graph.number_of_nodes("particles_"+cl)
                n_dict[cl] = number

                if number != 0:
                    predicted_particles_tmp.append(torch.split(particle_properties, predicted_num_particles_cl.tolist()))

                    # attention
                    num_nodes = g.batch_num_nodes('nodes') * predicted_num_particles_cl
                #pred_attention_wts_tmp.append(torch.split(g.edges['node_to_pflow_'+cl].data['pred_attention'], num_nodes.tolist()))

        predicted_particles = []
        predicted_attention_weights = []

        for batch_i in range(g.batch_size):
            counter = -1
            for cl_inx, cl in enumerate(self.classes):
                if cl == "supercharged":
                 if g.number_of_nodes(cl) != 0: 
                  counter = counter + 1
                  num_particles = predicted_particles_tmp[counter][batch_i].shape[0] 
                 else:
                  num_particles = 0

                 if num_particles != 0:              
                    predicted_particles.append(predicted_particles_tmp[counter][batch_i])
                else:
                 if n_dict[cl] != 0: 
                  counter = counter + 1

                  num_particles = predicted_particles_tmp[counter][batch_i].shape[0] 
                 else:
                  num_particles = 0

                 if num_particles != 0:              
                    predicted_particles.append(predicted_particles_tmp[counter][batch_i])



                    #predicted_attention_weights.extend(
                    #    pred_attention_wts_tmp[cl_inx][batch_i].reshape(-1, num_particles).transpose(0,1)
                    #)

        predicted_particles = torch.cat(predicted_particles)        
        predicted_particles = self.undo_scaling(predicted_particles)

#        return predicted_particles, predicted_num_particles, predicted_attention_weights
        return predicted_particles, predicted_num_particles, pred_n_supneutral

