import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
import math

import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

import gc


def collate_graphs(samples):

    batched_g = dgl.batch(samples)

    return batched_g


class PflowDataset(Dataset):
    def __init__(self, filename,config=None,reduce_ds=-1,entry_start=0):
        # photon : 0
        # neutral hadron: n, pion0, K0, Xi0, lambda: 1
        # charged hadron: p+-, K+-, pion+-, Xi+, Omega, Sigma : 2
        # electron : 3
        # muon : 4
        self.class_labels = {-3112 : 2,
                                3112 : 2,
                                3222 : 2,
                            -3222 : 2,
                            -3334 : 2,
                            3334 : 2,
                            -3122: 1,
                            3122 : 1,
                                310 : 1,
                                3312 : 2,
                            -3312: 2,
                            3322 : 1,
                            -3322 : 1,
                            2112: 1,
                            321: 2,
                            130: 1,
                            -2112: 1,
                            2212: 2,
                            11: 3,
                            -211: 2,
                            13: 4,
                            211: 2,
                            -13: 4,
                            -11: 3,
                            22: 0,
                            -2212: 2,
                            -321: 2}


        self.charge_labels = {130: 0,
                            -3322: 0,
                            3334: 2,
                            11: 2,
                            13: 2,
                            -3312: 1,
                            22: 0,
                            3222: 1,
                            2212: 1,
                            3112: 2,
                            -211: 2,
                            3122: 0,
                            310: 0,
                            -321: 2,
                            -2112: 0,
                            321: 1,
                            2112: 0,
                            -3122: 0,
                            211: 1,
                            -3112: 1,
                            -2212: 2,
                            -3334: 1,
                            -3222: 2,
                            3312: 2,
                            -13: 1,
                            -11: 1,
                            3322: 0}

        #class - particle_to_track
        self.class_particle_to_track = {
                              0: 0,
                              1 : 1,
                              -1: 5,
                              2 : 2,
                              -2: 2,
                              3 : 3,
                              -3: 5,
                              4 : 4,
                              -4: 5}
        self.classIsMuon = {
                              0: 0,
                              1 : 0,
                              2 : 0,
                              3 : 0,
                              4 : 1}

        f = uproot.open(filename)
        if 'EventTree' in f:
            self.tree = f['EventTree']
        else:
            self.tree = f['Low_Tree']
        
        self.config=config
        self.var_transform = self.config['var transform']
        self.entry_start = entry_start

        self.nevents = self.tree.num_entries
        if reduce_ds < 1.0 and reduce_ds > 0:
            self.nevents = int(self.nevents*reduce_ds)
        if reduce_ds >= 1.0:
            self.nevents = reduce_ds
        print('we have ',self.nevents, ' events')
        
        self.track_variables = ['track_parent_idx', 'track_d0',
                  'track_z0',
                  'cosin_track_phi',
                  'sinu_track_phi',
                  'track_theta',
                  'track_phi',
                  'track_eta_layer_0',
                  'track_eta_layer_1',
                  'track_eta_layer_2',
                  'track_eta_layer_3',
                  'track_eta_layer_4',
                  'track_eta_layer_5',
                  'sinu_track_phi_layer_0',
                  'sinu_track_phi_layer_1',
                  'sinu_track_phi_layer_2',
                  'sinu_track_phi_layer_3',
                  'sinu_track_phi_layer_4',
                  'sinu_track_phi_layer_5',
                  'cosin_track_phi_layer_0',
                  'cosin_track_phi_layer_1',
                  'cosin_track_phi_layer_2',
                  'cosin_track_phi_layer_3',
                  'cosin_track_phi_layer_4',
                  'cosin_track_phi_layer_5',
                  'track_qoverp']
        
        self.track_inputs = [
            'track_d0',
                  'track_z0',
                  'cosin_track_phi',
                  'sinu_track_phi',
                  'track_qoverp',
                  'track_pt',
                  'track_eta',
                  'track_eta_layer_0',
                  'track_eta_layer_1',
                  'track_eta_layer_2',
                  'track_eta_layer_3',
                  'track_eta_layer_4',
                  'track_eta_layer_5',
                  'sinu_track_phi_layer_0',
                  'sinu_track_phi_layer_1',
                  'sinu_track_phi_layer_2',
                  'sinu_track_phi_layer_3',
                  'sinu_track_phi_layer_4',
                  'sinu_track_phi_layer_5',
                  'cosin_track_phi_layer_0',
                  'cosin_track_phi_layer_1',
                  'cosin_track_phi_layer_2',
                  'cosin_track_phi_layer_3',
                  'cosin_track_phi_layer_4',
                  'cosin_track_phi_layer_5',
                  'track_isMuon',
                  'track_isIso'
        ]

        self.cell_variables = ['cell_x','cell_y','cell_z','cell_e','cell_eta','sinu_cell_phi','cosin_cell_phi','cell_layer','cell_parent_idx']
        
        self.cell_inputs = ['cell_x','cell_y','cell_z','cell_eta','sinu_cell_phi','cosin_cell_phi','cell_layer','cell_e']
        #self.cell_inputs = ['cell_eta','cell_phi','cell_e','cell_layer']
        
        self.particle_variables = ['particle_pdgid','particle_px','particle_py','particle_pz','particle_e',
                                    'particle_prod_x','particle_prod_y','particle_prod_z']

        self.all_graphs = {} 
        self.full_data_array = {}

        print('loading data:')

        # # loading the part needed for the truth attention weights !! need to flatten it in order to make it a tensor
        self.full_data_array["particle_to_node_weight"] = self.tree["particle_to_node_weight"].array(library='np', entry_stop=self.nevents)
        self.full_data_array["particle_to_node_idx"]    = self.tree["particle_to_node_idx"].array(library='np', entry_stop=self.nevents)

        # needed for track selection
        self.full_data_array["track_not_reg"] = self.tree["track_not_reg"].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents)
        self.full_data_array["particle_isIso"] = self.tree["particle_isIso"].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents)

        # self.full_data_array["particle_to_track"] = self.tree["particle_to_track"].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents) #Etienne: needed for condensation?
        self.full_data_array["particle_pdgid_noC"] =  np.copy(self.tree["particle_pdgid"].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents))
        # self.full_data_array["particle_to_track"] = np.concatenate( self.full_data_array["particle_to_track"] ) #Etienne: needed for condensation?

        # transform in -1 and 1
        # self.full_data_array["particle_to_track"] = np.where(self.full_data_array["particle_to_track"]==-1, self.full_data_array["particle_to_track"],1)
        self.n_photons   =  [] 
        self.n_muons     =  [] 
        self.n_electrons =  [] 
        self.n_charged   =  [] 
        self.n_neutral   =  [] 

        for var in tqdm( self.cell_variables+self.particle_variables+self.track_variables):
            newvar = ""
            if "cosin_" in var or "sinu_" in var:
                replace = ""
                if "cosin_" in var: replace = "cosin_"
                if "sinu_" in var:  replace = "sinu_"
                newvar = var.replace(replace, '')
                self.full_data_array[var] = np.copy(self.tree[newvar].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents))
                

            else: 
                self.full_data_array[var] = self.tree[var].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents)
            #fdibello need to change the number of tracks and add the dedicated class
            if "track" in var or "particle" in var:
                if var == "track_parent_idx":
                    self.full_data_array["track_isMuon"] = np.copy(self.full_data_array["track_parent_idx"])
                    self.full_data_array["track_isIso"] = np.copy(self.full_data_array["track_parent_idx"])
                   # self.full_data_array["track_isMuon"][:][:] = 0

                for nev in range(self.nevents):

                  if var == "particle_pdgid": 
                     photons    =  [x for x in self.full_data_array[var][nev] if self.class_labels[x] == 0 ] 
                     neutral    =  [x for x in self.full_data_array[var][nev] if self.class_labels[x] == 1 ] 
                     charged    =  [x for x in self.full_data_array[var][nev] if self.class_labels[x] == 2 ]
                     electrons  =  [x for x in self.full_data_array[var][nev] if self.class_labels[x] == 3 ] 
                     muons      =  [x for x in self.full_data_array[var][nev] if self.class_labels[x] == 4 ] 
                     self.n_photons  .append(len(photons)) 
                     self.n_muons    .append(len(muons)) 
                     self.n_electrons.append(len(electrons)) 
                     self.n_charged  .append(len(charged)) 
                     self.n_neutral  .append(len(neutral)) 

                  if(len(self.full_data_array[var][nev]) ==  len(self.full_data_array["track_not_reg"][nev])):
                   
                   self.full_data_array[var][nev] = [ self.full_data_array[var][nev][i] for i in range(0,len(self.full_data_array["track_not_reg"][nev])) if self.full_data_array["track_not_reg"][nev][i] == 0]
                  if var == "track_parent_idx":
                    self.full_data_array["track_isMuon"][nev] = [ self.full_data_array["track_isMuon"][nev][i] for i in range(0,len(self.full_data_array["track_not_reg"][nev])) if self.full_data_array["track_not_reg"][nev][i] == 0]
                    self.full_data_array["track_isMuon"][nev] = [self.classIsMuon[self.class_labels[self.full_data_array["particle_pdgid_noC"][nev][x]]] for x in self.full_data_array["track_parent_idx"][nev]]

                    self.full_data_array["track_isIso"][nev] = [ self.full_data_array["particle_isIso"][nev][x] for x in self.full_data_array["track_parent_idx"][nev]]



            if var == 'cell_x':
                self.n_cells = [len(x) for x in self.full_data_array[var]]
            elif var=='track_d0':
                self.n_tracks = [len(x) for x in self.full_data_array[var]]
            elif var=='particle_pdgid':
                self.n_particles = [len(x) for x in self.full_data_array[var]]
               

            #flatten the arrays
            self.full_data_array[var] = np.concatenate( self.full_data_array[var] )

            if newvar in ['cell_phi']:
                self.full_data_array[var][self.full_data_array[var] > np.pi] = self.full_data_array[var][self.full_data_array[var] > np.pi]-2*np.pi

            if "cosin_" in var: self.full_data_array[var] = np.cos(self.full_data_array[var]) 
            if "sinu_" in var: self.full_data_array[var] = np.sin(self.full_data_array[var]) 

            if var=='particle_pdgid':
                #add a new class for charged with no tracks - check if this is needed or not
                self.particle_class = torch.tensor([self.class_labels[x] for x in  self.full_data_array[var]])
                self.particle_charge =  torch.tensor([self.charge_labels[x] for x in  self.full_data_array[var]])

            if var in ['track_d0','track_z0']:
                self.full_data_array[var] = np.sign(self.full_data_array[var])*np.log(1+50.0*abs(self.full_data_array[var]))

            if var in ['cell_e','particle_e']:
                self.full_data_array[var] = np.log(self.full_data_array[var])

            if var in ['track_theta']:
                self.full_data_array['track_eta'] =  -np.log( np.tan( self.full_data_array[var]/2 )) 

            if var in ['track_qoverp']:
                self.full_data_array['track_pt'] =  np.log(np.abs(1./self.full_data_array["track_qoverp"]) * np.sin(self.full_data_array["track_theta"]))
        

        #add decorater to the track
        self.full_data_array["track_isMuon"] = np.concatenate( self.full_data_array["track_isMuon"] )  
        self.full_data_array["track_isIso"] = np.concatenate( self.full_data_array["track_isIso"] )

        #particle properties
        particle_phi   = np.arctan2(self.full_data_array['particle_py'],self.full_data_array['particle_px'])
        particle_p     = np.linalg.norm(np.column_stack([self.full_data_array['particle_px'],self.full_data_array['particle_py'],self.full_data_array['particle_pz']]),axis=1)
        particle_theta = np.arccos( self.full_data_array['particle_pz']/particle_p)
        particle_eta   =  -np.log( np.tan( particle_theta/2 )) 
        particle_xhat = np.cos(particle_phi )
        particle_yhat = np.sin(particle_phi )

        particle_pt = particle_p*np.sin(particle_theta)
        particle_pt = np.log(particle_pt)

        self.full_data_array['particle_phi'] = particle_phi
        self.full_data_array['particle_pt'] = particle_pt
        self.full_data_array['particle_theta'] = particle_theta
        self.full_data_array['particle_eta'] = particle_eta
        self.full_data_array['particle_xhat'] = particle_xhat
        self.full_data_array['particle_yhat'] = particle_yhat
        
        #transform variables and transform to tensors
        for var in tqdm( self.track_variables+self.cell_variables+self.particle_variables+['track_eta','track_pt','particle_phi','particle_pt','particle_theta','particle_eta','particle_xhat','particle_yhat']): 
            if var in self.var_transform:
                self.full_data_array[var] = (self.full_data_array[var]-self.var_transform[var]['mean'])/self.var_transform[var]['std']
            if var in ['cell_layer','cell_parent_idx']:
                self.full_data_array[var] = torch.LongTensor(self.full_data_array[var])
            else:
                self.full_data_array[var] = torch.FloatTensor(self.full_data_array[var])
            # self.full_data_array["track_isMuon"] = torch.tensor(self.full_data_array["track_isMuon"].clone().detach()) 

        self.full_data_array["track_isMuon"] = torch.tensor(self.full_data_array["track_isMuon"])
        self.full_data_array["track_isIso"] = torch.tensor(self.full_data_array["track_isIso"])

        self.cell_cumsum = np.cumsum([0]+self.n_cells)
        self.track_cumsum = np.cumsum([0]+self.n_tracks)
        self.particle_cumsum = np.cumsum([0]+self.n_particles)

        self.edge_c_to_c_start =  [torch.tensor(x) for x in self.tree['cell_to_cell_edge_start'].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents)]
        self.edge_c_to_c_end =   [torch.tensor(x) for x in self.tree['cell_to_cell_edge_end'].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents) ]
        
        self.edge_t_to_c_start =  [torch.tensor(x) for x in  self.tree['track_to_cell_edge_start'].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents)]
        self.edge_t_to_c_end =   [torch.tensor(x) for x in self.tree['track_to_cell_edge_end'].array(library='np',entry_start=self.entry_start,entry_stop=self.entry_start+self.nevents)]


        # bad practice, apparently
        for key in self.full_data_array.keys():
            if key in ['particle_pdgid_noC', 'track_not_reg']:
                continue
            exec("self.{} = self.full_data_array['{}']".format(key, key))            
            # exec("self.{}.share_memory_()".format(key))
 
        # del self.full_data_array 

        # clean up to save memory (experimental): Nilotpal
        del self.tree
        del particle_phi, particle_p, particle_theta, particle_eta, particle_xhat, particle_yhat, particle_pt
        #del particle_to_node_idx, particle_to_node_weight

        gc.collect()

        # for idx, var in enumerate(self.full_data_array.keys()):
        #     if torch.is_tensor(self.full_data_array[var]):
        #         self.full_data_array[var].share_memory_()
        # self.attention_idx_cumsum.share_memory_()
        # self.attention_idx_cumsum_eventwise.share_memory_()


    def get_single_item(self,idx):
        
        n_cells = self.n_cells[idx]
        n_tracks = self.n_tracks[idx]
        n_particles = self.n_particles[idx]
        n_nodes = n_cells+n_tracks

        if (n_particles==0 or n_cells==0) and idx>0: #or n_tracks==0) and idx>0:
            print('HACK: replacing empty event {} with the preceding event {}'.format(idx,idx-1))
            print('HACK: n_particles=={} or n_cells=={} or n_tracks=={}'.format(n_particles,n_cells,n_tracks))
            return self.get_single_item(idx-1)

        cell_start, cell_end = self.cell_cumsum[idx],self.cell_cumsum[idx+1]
        track_start, track_end = self.track_cumsum[idx],self.track_cumsum[idx+1]
        particle_start, particle_end = self.particle_cumsum[idx],self.particle_cumsum[idx+1]

        particle_pdg = self.particle_pdgid[particle_start:particle_end]
        particle_class  =  self.particle_class[particle_start:particle_end]
        particle_charge = self.particle_charge[particle_start:particle_end]

        particle_px =self.particle_px[particle_start:particle_end]
        particle_py =self.particle_py[particle_start:particle_end]
        particle_pz =self.particle_pz[particle_start:particle_end]
        particle_e = self.particle_e[particle_start:particle_end]
        particle_phi   = self.particle_phi[particle_start:particle_end]
        particle_pt     = self.particle_pt[particle_start:particle_end]
        particle_theta = self.particle_theta[particle_start:particle_end]
        particle_eta   = self.particle_eta[particle_start:particle_end]
        particle_xhat   = self.particle_xhat[particle_start:particle_end]
        particle_yhat   = self.particle_yhat[particle_start:particle_end]
        particle_prod_x = self.particle_prod_x[particle_start:particle_end]
        particle_prod_y = self.particle_prod_y[particle_start:particle_end]
        particle_prod_z = self.particle_prod_z[particle_start:particle_end]

        particle_to_node_weight = self.particle_to_node_weight[idx]
        particle_to_node_idx = self.particle_to_node_idx[idx]

        truth_attention_weights = np.zeros((n_particles,n_tracks+n_cells))
        for idx_x, p_idx in enumerate(particle_to_node_idx):
             row_idx = np.repeat(idx_x,len(p_idx))
             column_idx = np.array(p_idx).astype(int)
             weights = np.array(particle_to_node_weight[idx_x])
             truth_attention_weights[row_idx,column_idx] = weights


        track_features = []
        for var in self.track_inputs:
            if var != "track_isIso":
                # arr = self.full_data_array[var][track_start:track_end]
                exec('track_features.append(self.{}[track_start:track_end])'.format(var))

        track_features = torch.stack( track_features ,dim=1).float()

        cell_features = []
        for var in self.cell_inputs:
            # arr =  self.full_data_array[var][cell_start:cell_end]

            exec('cell_features.append(self.{}[cell_start:cell_end])'.format(var))

        cell_features = torch.stack( cell_features  ,dim=1).float()
        
        num_nodes_dict = {
            'cells' : n_cells,
            'tracks' : n_tracks,
            'nodes' : n_cells+n_tracks,
            'particles' : n_particles,
            'global node' : 1,
        }

        #all children are connected to all potential parents:
        edge_list1 = torch.repeat_interleave( torch.arange(n_nodes),n_particles)
        edge_list2 = torch.arange(n_particles).repeat(n_nodes)

        #connect cells and tracks to their nodes
        cell_to_node = torch.arange(n_cells)
        track_to_node = torch.arange(n_tracks)

        edge_c_to_c_start = self.edge_c_to_c_start[idx]
        edge_c_to_c_end =  self.edge_c_to_c_end[idx]
        
        edge_t_to_c_start = self.edge_t_to_c_start[idx]
        edge_t_to_c_end =  self.edge_t_to_c_end[idx]

        #connect each cell to every track
        edge_list1_track_to_cell = torch.repeat_interleave( torch.arange(n_tracks),n_cells)
        edge_list2_track_to_cell = torch.arange(n_cells).repeat(n_tracks)

        #convert track edges to proper indx for nodes by adding n_cells to track start
        node_to_node_start = torch.cat([edge_c_to_c_start, n_cells+edge_t_to_c_start, edge_t_to_c_end],dim=0)
        node_to_node_end = torch.cat([edge_c_to_c_end, edge_t_to_c_end, n_cells+edge_t_to_c_start],dim=0)

        data_dict = {
                    ('cells','cell_to_node','nodes') : (cell_to_node,cell_to_node),
                    ('tracks','track_to_node','nodes') : (track_to_node, n_cells+track_to_node),
                    ('tracks','track_to_cell','cells') : (edge_list1_track_to_cell, edge_list2_track_to_cell),

                    ('nodes','node_to_node','nodes') : (node_to_node_end, node_to_node_start), #[TODO]-->DONE! reverse order of list (however, not consistent with latest SCD code)

                    ('nodes', 'node_to_particle', 'particles') : (edge_list1,edge_list2),                  
                    ('particles', 'particle_to_node', 'nodes'): (edge_list2, edge_list1),

                    #all nodes connected to the global node
                    ('nodes','nodes to global','global node'): (torch.arange(n_nodes).int(),torch.zeros(n_nodes).int()),

                    #all particles connected to the global node
                    ('particles','particles to global','global node'): (torch.arange(n_particles).int(),torch.zeros(n_particles).int()),
                    }
        
        g = dgl.heterograph(data_dict,num_nodes_dict)
        g.nodes['particles'].data['particle idx'] = torch.arange(n_particles) 
        g.nodes['particles'].data['particle class'] = torch.LongTensor(particle_class)
        g.nodes['particles'].data['charge_class'] = torch.LongTensor(particle_charge)
        g.nodes['particles'].data['prod pos'] = torch.stack( [particle_prod_x,particle_prod_y,particle_prod_z],dim=1) 
        g.nodes['particles'].data['particle_phi'] = particle_phi
        g.nodes['particles'].data['particle_eta'] = particle_eta
        g.nodes['particles'].data['particle_pt'] = particle_pt
        g.nodes['particles'].data['particle_xhat'] = particle_xhat
        g.nodes['particles'].data['particle_yhat'] = particle_yhat

        g.nodes['cells'].data['N edges start'] = (torch.bincount(torch.cat([edge_c_to_c_start,torch.arange(n_cells)])) - 1).int()
        g.nodes['cells'].data['N edges end'] = (torch.bincount(torch.cat([edge_c_to_c_end,torch.arange(n_cells)])) - 1).int()
        g.nodes['cells'].data['parent target'] = self.cell_parent_idx[cell_start:cell_end].float()
        g.nodes['tracks'].data['parent target'] =  self.track_parent_idx[track_start:track_end].float()
        g.nodes['cells'].data['node features'] = cell_features
        g.nodes['tracks'].data['node features'] = track_features
        g.nodes['cells'].data['isTrack'] = torch.LongTensor(np.zeros(n_cells))
        g.nodes['tracks'].data['isTrack'] = torch.LongTensor(np.ones(n_tracks))
        g.nodes['cells'].data['layer_cell'] = self.cell_layer[cell_start:cell_end].long()

        g.nodes['tracks'].data['track_eta'] = self.var_transform["track_eta"]['std']*self.track_eta[track_start:track_end].float()+self.var_transform["track_eta"]['mean']
        track_phi  = torch.atan2(self.sinu_track_phi,self.cosin_track_phi)
#        g.nodes['tracks'].data['track_phi'] = self.track_phi[track_start:track_end].float()
        g.nodes['tracks'].data['track_phi'] = track_phi[track_start:track_end].float()
        g.nodes['tracks'].data['track_isIso'] = self.track_isIso[track_start:track_end].float()
        g.nodes['tracks'].data['track_eta_layer_0']       = self.track_eta_layer_0[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_0'] = self.cosin_track_phi_layer_0[track_start:track_end].float()

        self.dummy_vector = torch.LongTensor(np.zeros(cell_end))
        self.dummy_vector_track = torch.LongTensor(np.zeros(track_end))
        self.dummy_negones_track = torch.LongTensor(-1*np.ones(track_end))

        g.nodes['nodes'].data['N edges start'] = torch.cat([g.nodes['cells'].data['N edges start'], self.dummy_vector_track[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['N edges end'] = torch.cat([g.nodes['cells'].data['N edges end'], self.dummy_vector_track[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['isTrack'] = torch.cat([g.nodes['cells'].data['isTrack'], g.nodes['tracks'].data['isTrack']],dim=0)
        g.nodes['nodes'].data['isMuon'] = torch.cat([self.dummy_vector[cell_start:cell_end], self.track_isMuon[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['energy_cell'] = torch.cat([self.cell_e[cell_start:cell_end],self.dummy_vector_track[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['phi_cell'] = torch.cat([torch.asin(self.sinu_cell_phi[cell_start:cell_end]),self.dummy_vector_track[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['eta_cell'] = torch.cat([self.cell_eta[cell_start:cell_end],self.dummy_vector_track[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['layer_cell'] = torch.cat([self.cell_layer[cell_start:cell_end],self.dummy_negones_track[track_start:track_end]],dim=0)



        g.nodes['nodes'].data['cosin_phi'] = torch.cat([self.cosin_cell_phi[cell_start:cell_end],self.cosin_track_phi[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['sinu_phi'] = torch.cat([self.sinu_cell_phi[cell_start:cell_end],self.sinu_track_phi[track_start:track_end]],dim=0)
        #g.nodes['nodes'].data['phi'] = torch.cat([self.cell_phi[cell_start:cell_end], self.track_phi[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['eta'] = torch.cat([ self.var_transform["cell_eta"]['std']*self.cell_eta[cell_start:cell_end]+self.var_transform["cell_eta"]['mean'], g.nodes['tracks'].data['track_eta']],dim=0)

        g.nodes['nodes'].data['cell_energy'] = torch.cat([self.cell_e[cell_start:cell_end], torch.tensor(np.zeros(track_end-track_start))],dim=0)
        g.nodes['nodes'].data['track_p'] = torch.cat([torch.tensor(np.zeros(cell_end-cell_start)), torch.abs(self.track_qoverp[track_start:track_end])],dim=0)
        g.nodes['nodes'].data['track_pt'] = torch.cat([torch.tensor(np.zeros(cell_end-cell_start)), torch.abs(self.track_pt[track_start:track_end])],dim=0)

        ### Copy ALL the track projections!
        g.nodes['nodes'].data['track_eta_layer_0']       = torch.cat([self.dummy_vector[cell_start:cell_end], self.track_eta_layer_0[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['cosin_track_phi_layer_0'] = torch.cat([self.dummy_vector[cell_start:cell_end], self.cosin_track_phi_layer_0[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['sinu_track_phi_layer_0']  = torch.cat([self.dummy_vector[cell_start:cell_end], self.sinu_track_phi_layer_0[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['track_eta_layer_1']       = torch.cat([self.dummy_vector[cell_start:cell_end], self.track_eta_layer_1[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['cosin_track_phi_layer_1'] = torch.cat([self.dummy_vector[cell_start:cell_end], self.cosin_track_phi_layer_1[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['sinu_track_phi_layer_1']  = torch.cat([self.dummy_vector[cell_start:cell_end], self.sinu_track_phi_layer_1[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['track_eta_layer_2']       = torch.cat([self.dummy_vector[cell_start:cell_end], self.track_eta_layer_2[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['cosin_track_phi_layer_2'] = torch.cat([self.dummy_vector[cell_start:cell_end], self.cosin_track_phi_layer_2[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['sinu_track_phi_layer_2']  = torch.cat([self.dummy_vector[cell_start:cell_end], self.sinu_track_phi_layer_2[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['track_eta_layer_3']       = torch.cat([self.dummy_vector[cell_start:cell_end], self.track_eta_layer_3[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['cosin_track_phi_layer_3'] = torch.cat([self.dummy_vector[cell_start:cell_end], self.cosin_track_phi_layer_3[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['sinu_track_phi_layer_3']  = torch.cat([self.dummy_vector[cell_start:cell_end], self.sinu_track_phi_layer_3[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['track_eta_layer_4']       = torch.cat([self.dummy_vector[cell_start:cell_end], self.track_eta_layer_4[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['cosin_track_phi_layer_4'] = torch.cat([self.dummy_vector[cell_start:cell_end], self.cosin_track_phi_layer_4[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['sinu_track_phi_layer_4']  = torch.cat([self.dummy_vector[cell_start:cell_end], self.sinu_track_phi_layer_4[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['track_eta_layer_5']       = torch.cat([self.dummy_vector[cell_start:cell_end], self.track_eta_layer_5[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['cosin_track_phi_layer_5'] = torch.cat([self.dummy_vector[cell_start:cell_end], self.cosin_track_phi_layer_5[track_start:track_end].float()],dim=0)
        g.nodes['nodes'].data['sinu_track_phi_layer_5']  = torch.cat([self.dummy_vector[cell_start:cell_end], self.sinu_track_phi_layer_5[track_start:track_end].float()],dim=0)

        g.nodes['tracks'].data['track_eta_layer_0']       = self.track_eta_layer_0[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_0'] = self.cosin_track_phi_layer_0[track_start:track_end].float()
        g.nodes['tracks'].data['sinu_track_phi_layer_0']  = self.sinu_track_phi_layer_0[track_start:track_end].float()
        g.nodes['tracks'].data['track_eta_layer_1']       = self.track_eta_layer_1[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_1'] = self.cosin_track_phi_layer_1[track_start:track_end].float()
        g.nodes['tracks'].data['sinu_track_phi_layer_1']  = self.sinu_track_phi_layer_1[track_start:track_end].float()
        g.nodes['tracks'].data['track_eta_layer_2']       = self.track_eta_layer_2[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_2'] = self.cosin_track_phi_layer_2[track_start:track_end].float()
        g.nodes['tracks'].data['sinu_track_phi_layer_2']  = self.sinu_track_phi_layer_2[track_start:track_end].float()
        g.nodes['tracks'].data['track_eta_layer_3']       = self.track_eta_layer_3[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_3'] = self.cosin_track_phi_layer_3[track_start:track_end].float()
        g.nodes['tracks'].data['sinu_track_phi_layer_3']  = self.sinu_track_phi_layer_3[track_start:track_end].float()
        g.nodes['tracks'].data['track_eta_layer_4']       = self.track_eta_layer_4[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_4'] = self.cosin_track_phi_layer_4[track_start:track_end].float()
        g.nodes['tracks'].data['sinu_track_phi_layer_4']  = self.sinu_track_phi_layer_4[track_start:track_end].float()
        g.nodes['tracks'].data['track_eta_layer_5']       = self.track_eta_layer_5[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_5'] = self.cosin_track_phi_layer_5[track_start:track_end].float()
        g.nodes['tracks'].data['sinu_track_phi_layer_5']  = self.sinu_track_phi_layer_5[track_start:track_end].float()

        g.nodes['cells'].data['eta'] = self.cell_eta[cell_start:cell_end]
        g.nodes['cells'].data['cosin_phi'] = self.cosin_cell_phi[cell_start:cell_end]
        g.nodes['cells'].data['sinu_phi'] = self.sinu_cell_phi[cell_start:cell_end]
        g.nodes['tracks'].data['eta'] = self.track_eta[track_start:track_end]
        g.nodes['tracks'].data['cosin_phi'] = self.cosin_track_phi[track_start:track_end]
        g.nodes['tracks'].data['sinu_phi'] = self.sinu_track_phi[track_start:track_end]

        ### Add cell energy / noise as cell and node feature ###
        layer_noise = {
            0: 13,
            1: 34.,
            2: 41.,
            3: 75.,
            4: 50.,
            5: 25.
        }

        cell_e     = torch.clone(g.nodes['nodes'].data['cell_energy'])
        cell_layer = torch.clone(g.nodes['nodes'].data['layer_cell'])

        cell_noise = cell_layer
        for layer, noise in layer_noise.items():
            cell_noise[cell_layer==layer] = noise

        cell_e = cell_e*self.var_transform['cell_e']['std'] + self.var_transform['cell_e']['mean']
        cell_e = torch.exp(cell_e)

        cell_z    = cell_e/cell_noise
        where_cells  = torch.logical_and( (g.nodes['nodes'].data['isTrack'] == 0), (cell_z>0))
        where_tracks = (g.nodes['nodes'].data['isTrack'] == 1)
        cell_zeta = cell_z
        cell_zeta[where_tracks] = 9999.

        g.nodes['cells'].data['zeta'] = cell_zeta[where_cells]       
        g.nodes['nodes'].data['zeta'] = cell_zeta

        # #build attention edges
        g.edges['node_to_particle'].data["truth_attention"] = torch.tensor(truth_attention_weights[g.nodes["particles"].data['particle idx'],:].T.flatten()).float()

        return g


    def __len__(self):

        return self.nevents 


    def __getitem__(self, idx):
        
        return  self.get_single_item(idx)

