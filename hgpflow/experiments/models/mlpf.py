import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph


import numpy as np
import torch
import torch.nn as nn

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





class MLPFNet(nn.Module):
    def __init__(self,config):
        super(MLPFNet, self).__init__()

        self.config = config

        self.beta_net = nn.ModuleDict()
        self.x_net= nn.ModuleDict()
        self.nodeclass_net= nn.ModuleDict()
        self.charge_class_net = nn.ModuleDict()
        self.particle_position_net = nn.ModuleDict()
        self.particle_energy_net = nn.ModuleDict()

        for node_type in ['cells','tracks']:
#            self.beta_net[node_type] = \
#                build_layers(config[node_type+' node inputsize'],outputsize=1,
#                                         features=config['beta net layers'],add_activation=nn.Sigmoid())
#            self.x_net[node_type] = build_layers(config[node_type+' node inputsize'],outputsize=config['x size'],
#                                      features=config['x net layers'],add_batch_norm=True)


            self.nodeclass_net[node_type] = build_layers(config['output model'][node_type+' node inputsize'],
                                              outputsize=config['output model']['n classes']+1,
                                              features=config['output model']['node classifer layers'])
            self.charge_class_net[node_type] = build_layers(config['output model'][node_type+' node inputsize'],
                                              outputsize=config['output model']['n charge classes'],
                                              features=config['output model']['charge classifier layers'])

            self.particle_energy_net[node_type] = build_layers(config['output model'][node_type+' node inputsize'],
                                              outputsize=1,
                                              features=config['output model']['energy prediction layers'])

            self.particle_position_net[node_type] = build_layers(config['output model'][node_type+' node inputsize'],
                                              outputsize=3,
                                              features=config['output model']['prod position layers'])



    def forward(self, g):


        for ntype in ['cells']:
        #for ntype in ['cells','tracks']:
            data2 = torch.cat([g.nodes[ntype].data['node features'], g.nodes[ntype].data['hidden rep 2'],g.nodes[ntype].data['global rep 2']],dim=1)

            g.nodes[ntype].data['class pred'] = self.nodeclass_net[ntype](data2)
            g.nodes[ntype].data['charge pred'] = self.charge_class_net[ntype](data2)

            g.nodes[ntype].data['pos pred'] = self.particle_position_net[ntype](data2)
            g.nodes[ntype].data['e pred'] = self.particle_energy_net[ntype](data2).view(-1)

        return g




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





#Need to change here the definition of the what is a particle
    def create_particles(self,g):

        for ntype in ['cells']:
        #for ntype in ['cells','tracks']:
          data2 = torch.cat([g.nodes[ntype].data['node features'], g.nodes[ntype].data['hidden rep 2'],g.nodes[ntype].data['global rep 2']],dim=1)

          g.nodes[ntype].data['class pred'] = self.nodeclass_net[ntype](data2)
          g.nodes[ntype].data['charge pred'] = self.charge_class_net[ntype](data2)

          g.nodes[ntype].data['pos pred'] = self.particle_position_net[ntype](data2)
          g.nodes[ntype].data['e pred'] = self.particle_energy_net[ntype](data2).view(-1)


          truth_class = g.nodes[ntype].data['particle class'] 



        particle_class = g.nodes[ntype].data['class pred'] 
        #count how many particles have class > 0
        #particle_class = particle_class.argmax()
        m = nn.Softmax(dim=1)
        output = m(particle_class)
        print("******************",output) 



        return output, truth_class






    def infer(self,g):
        
        predicted_particles, truth_particles = self.create_particles(g)
        #predicted_num_particles = output



        #predicted_particles = self.create_particles(g)
        #predicted_particles,predicted_num_particles = self.create_particles(g)
        #predicted_particles = self.undo_scaling(predicted_particles)
        #return p_e, particle_pxpypz, particle_pos, particle_class, particle_charge


        predicted_num_particles = len(predicted_particles) 
        return predicted_particles,predicted_num_particles,truth_particles