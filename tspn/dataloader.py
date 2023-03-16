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
        gs, idxs = zip(*samples)
        batched_g = dgl.batch(gs)

        return batched_g, idxs



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
        #self.class_particle_to_track = {
        #                      0: 0,
        #                      1 : 1,
        #                      -1: 5,
        #                      2 : 2,
        #                      -2: 2,
        #                      3 : 3,
        #                      -3: 5,
        #                      4 : 4,
        #                      -4: 5}
        self.classIsMuon = {
                              0: 0,
                              1 : 0,
                              2 : 0,
                              3 : 0,
                              4 : 1}


        f = uproot.open(filename)
        self.tree = f['Low_Tree']
        
        self.config=config
        self.var_transform = self.config['var transform']
        self.cell_e_mean, self.cell_e_std = self.var_transform['cell_e']['mean'], self.var_transform['cell_e']['std']

        self.entry_start = entry_start

        self.nevents = self.tree.num_entries
        if reduce_ds < 1.0 and reduce_ds > 0:
            self.nevents = int(self.nevents*reduce_ds)
        if reduce_ds >= 1.0:
            self.nevents = reduce_ds
        print('We have ',self.nevents, ' events')

        self.track_variables = ['track_parent_idx', 'track_d0',
                  'track_z0',
                  'cosin_track_phi',
                  'sinu_track_phi',
                  'track_theta',
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

        self.cell_variables = ['cell_x','cell_y','cell_z','cell_e','cell_eta','sinu_cell_phi','cosin_cell_phi','cell_layer','cell_parent_idx', 'cell_topo_idx']
        
        self.cell_inputs = ['cell_x','cell_y','cell_z','cell_eta','sinu_cell_phi','cosin_cell_phi','cell_layer','cell_e']
        #self.cell_inputs = ['cell_eta','cell_phi','cell_e','cell_layer']


        
        self.particle_variables = ['particle_pdgid','particle_px','particle_py','particle_pz','particle_e',
                                    'particle_prod_x','particle_prod_y','particle_prod_z', 'particle_dep_energy']

        self.all_graphs = {} 
        self.full_data_array = {}

        print('loading data:')

        # # loading the part needed for the truth attention weights !! need to flatten it in order to make it a tensor
        self.full_data_array["particle_to_node_weight"] = self.tree["particle_to_node_weight"].array(library='np', entry_stop=self.nevents)
        self.full_data_array["particle_to_node_idx"]    = self.tree["particle_to_node_idx"].array(library='np', entry_stop=self.nevents)

        # experimental : Nilotpal #
        # shape (num_events, num_particles, num_connected_nodes)
        # attention_idx_cumsum : total number of edges per event (cumulated)
        # attention_idx_cumsum_eventwise : total number of edges per particle (cumulated) per event (flatten)

        # particle_to_node_idx    = self.tree["particle_to_node_idx"].array(library='np', entry_stop=self.nevents)
        # particle_to_node_weight = self.tree["particle_to_node_weight"].array(library='np', entry_stop=self.nevents)

        # self.full_data_array["particle_to_node_idx"]    = []
        # self.full_data_array["particle_to_node_weight"] = []

        # self.attention_idx_cumsum = [0]; self.attention_idx_cumsum_eventwise = []
        
        # for event_i, event in enumerate(particle_to_node_idx):

        #     node_idxs    = np.hstack(event).astype(int)
        #     node_weights = np.hstack(particle_to_node_weight[event_i])

        #     self.attention_idx_cumsum.append(node_idxs.shape[0])

        #     tmp_tensor = torch.cumsum(torch.tensor([len(x)for x in event]), dim=0)
        #     tmp_tensor.share_memory_()
        #     self.attention_idx_cumsum_eventwise.extend(tmp_tensor.tolist())            

        #     # flattened features
        #     self.full_data_array["particle_to_node_idx"].extend(node_idxs)
        #     self.full_data_array["particle_to_node_weight"].extend(node_weights)

        #self.full_data_array["particle_to_node_idx"] = torch.tensor(self.full_data_array["particle_to_node_idx"])
        #self.full_data_array["particle_to_node_weight"] = torch.tensor(self.full_data_array["particle_to_node_weight"])

        #self.attention_idx_cumsum = torch.cumsum(torch.tensor(self.attention_idx_cumsum), dim=0)
        #self.attention_idx_cumsum_eventwise = torch.tensor(self.attention_idx_cumsum_eventwise)




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
        self.n_superneutral   =  []
    
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
                     self.n_superneutral  .append(len(neutral)+len(photons))

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
            elif var == 'cell_topo_idx':
                self.n_topos = [max(x) for x in self.full_data_array[var]] 
                self.full_data_array[var]=self.full_data_array[var]-1
            #flatten the arrays
            self.full_data_array[var] = np.concatenate( self.full_data_array[var] )

            #if newvar in ['cell_phi']:
            #    self.full_data_array[var][self.full_data_array[var] > np.pi] = self.full_data_array[var][self.full_data_array[var] > np.pi]-2*np.pi

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
        particle_phi   = (np.arctan2(self.full_data_array['particle_py'],self.full_data_array['particle_px'])+np.pi)%(2.*np.pi)-np.pi
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
            self.full_data_array[var] = torch.tensor(self.full_data_array[var])
            # self.full_data_array["track_isMuon"] = torch.tensor(self.full_data_array["track_isMuon"].clone().detach()) 

        self.full_data_array["track_isMuon"] = torch.tensor(self.full_data_array["track_isMuon"])
        self.full_data_array["track_isIso"] = torch.tensor(self.full_data_array["track_isIso"])


        self.cell_cumsum = np.cumsum([0]+self.n_cells)
        self.track_cumsum = np.cumsum([0]+self.n_tracks)
        self.particle_cumsum = np.cumsum([0]+self.n_particles)
        self.photon_cumsum = np.cumsum([0]+self.n_photons)
        self.charged_cumsum = np.cumsum([0]+self.n_charged)
        self.neutral_cumsum = np.cumsum([0]+self.n_neutral)
        self.electron_cumsum = np.cumsum([0]+self.n_electrons)
        self.muon_cumsum = np.cumsum([0]+self.n_muons)
        self.superneutral_cumsum = np.cumsum([0]+self.n_superneutral)

        #define exclusive classes
        classes = ["photon","neutral","charged","electron","muon","superneutral"]
        for idx, cl in enumerate(classes):
            self.index = np.where(self.particle_class == idx)[0] # why self??
            if cl == "superneutral": self.index = np.where((self.particle_class == 0) | (self.particle_class == 1) ) [0] 
            self.full_data_array[cl+'_idx']             = torch.tensor(self.index)
            self.full_data_array[cl+'_phi']             = self.full_data_array['particle_phi'][self.index]
            self.full_data_array[cl+'_pt']              = self.full_data_array['particle_pt'][self.index]
            self.full_data_array[cl+'_theta']           = self.full_data_array['particle_theta'][self.index]
            self.full_data_array[cl+'_eta']             = self.full_data_array['particle_eta'][self.index]
            self.full_data_array[cl+'_xhat']            = self.full_data_array['particle_xhat'][self.index]
            self.full_data_array[cl+'_yhat']            = self.full_data_array['particle_yhat'][self.index]
            self.full_data_array[cl+'_charge_class']    = self.particle_charge[self.index]
            self.full_data_array[cl+'_particle_class']    = self.particle_class[self.index]
            self.full_data_array[cl+'_particle_prod_x'] = self.full_data_array["particle_prod_x"][self.index] 
            self.full_data_array[cl+'_particle_prod_y'] = self.full_data_array["particle_prod_y"][self.index] 
            self.full_data_array[cl+'_particle_prod_z'] = self.full_data_array["particle_prod_z"][self.index] 
            self.full_data_array[cl+'_dep_energy']      = self.full_data_array["particle_dep_energy"][self.index] 

        self.full_data_array['supercharged_idx']        = self.full_data_array['track_parent_idx']

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


    def get_single_item(self,idx):

        n_topos = self.n_topos[idx]
        n_cells = self.n_cells[idx]
        n_tracks = self.n_tracks[idx]
        n_particles = self.n_particles[idx]
        n_photon    = self.n_photons[idx]
        n_charged   = self.n_charged[idx]
        n_neutral   = self.n_neutral[idx]
        n_electron  = self.n_electrons[idx]
        n_muon      = self.n_muons[idx]
        n_nodes = n_topos+n_tracks
        n_prenodes = n_cells+n_tracks
        n_supercharged = n_tracks
        n_superneutral = n_photon+n_neutral


        cell_start, cell_end = self.cell_cumsum[idx],self.cell_cumsum[idx+1]
        track_start, track_end = self.track_cumsum[idx],self.track_cumsum[idx+1]
        particle_start, particle_end = self.particle_cumsum[idx],self.particle_cumsum[idx+1]
        photon_start, photon_end = self.photon_cumsum[idx],self.photon_cumsum[idx+1]
        charged_start,   charged_end = self.charged_cumsum[idx],self.charged_cumsum[idx+1]
        neutral_start,   neutral_end = self.neutral_cumsum[idx],self.neutral_cumsum[idx+1]
        electron_start,  electron_end = self.electron_cumsum[idx],self.electron_cumsum[idx+1]
        muon_start,      muon_end = self.muon_cumsum[idx],self.muon_cumsum[idx+1]
        superneutral_start, superneutral_end = self.superneutral_cumsum[idx],self.superneutral_cumsum[idx+1]
        class_start = [photon_start,neutral_start,charged_start,electron_start,muon_start,superneutral_start]
        class_end = [photon_end,neutral_end,charged_end,electron_end,muon_end,superneutral_end]

        cell_topo_idx = self.cell_topo_idx[cell_start:cell_end]

        cell_e = self.cell_e[cell_start:cell_end]

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
        particle_dep_energy = self.particle_dep_energy[particle_start:particle_end]
        
        particle_to_node_weight = self.particle_to_node_weight[idx]
        particle_to_node_idx = self.particle_to_node_idx[idx]
        
        # ############################
        # # sum for one particle = 1 # (A)
        # ############################

        # truth_incidence = np.zeros((n_nodes, n_particles))

        # for p_idx, (n_idx, n_weights) in enumerate(zip(particle_to_node_idx, particle_to_node_weight)):

        #     n_idx     = np.array(n_idx, dtype=int)
        #     n_weights = np.array(n_weights)

        #     # ghost particles
        #     if len(n_idx) == 0:
        #         continue

        #     # track attention
        #     if n_weights[-1] == 0.5:
        #         if len(n_idx) == 1:
        #             truth_incidence[n_idx[-1] - n_cells + n_topos, p_idx] = 1
        #         else:
        #             truth_incidence[n_idx[-1] -  n_cells + n_topos, p_idx] = 0.5

        #     # topocluster attention
        #     n_weights = n_weights[n_idx<n_cells]
        #     n_idx = n_idx[n_idx<n_cells]

        #     if len(n_idx) > 0:
        #         bc = np.bincount(cell_topo_idx[n_idx], weights=n_weights)
        #         truth_incidence[:len(bc), p_idx] =  bc


        ########################
        # sum for one topo = 1 # (B)
        ########################

        truth_topoattention_weights = np.zeros((n_particles,n_topos+n_tracks))
        topo_e=np.array([np.sum(np.exp( np.array(cell_e * self.cell_e_std + self.cell_e_mean).astype(float)[np.where(idx==cell_topo_idx)] ) )    for idx in range(n_topos)])
        for idx_p, cell_particle_idx in enumerate(particle_to_node_idx):
            ptn_weight=np.array(particle_to_node_weight[idx_p]).astype(float)
            if len(ptn_weight)==0: continue
            if len(ptn_weight)==1 and ptn_weight[0]==0.5: continue        
            cells_list_idx=np.array(cell_particle_idx).astype(int)
            p_en_dep=np.array(particle_dep_energy[idx_p]).astype(float)
            if(ptn_weight[-1]==0.5): 
                topo_fraction=2*ptn_weight*p_en_dep
                cells_list_idx=cells_list_idx[:-1]
            else: 
                topo_fraction=ptn_weight*p_en_dep
            list_topo_idx=cell_topo_idx[cells_list_idx]#-1
            for i, topo_idx in enumerate(list_topo_idx):
                truth_topoattention_weights[idx_p,topo_idx]+=topo_fraction[i]/topo_e[topo_idx]
        truth_topoattention_weights[np.where(truth_topoattention_weights<0.01)] = 0

        ### topo "parents"
        topo_parents = np.zeros(n_topos)
        for topo_idx in range(n_topos):
            parents  = self.cell_parent_idx[cell_start:cell_end][topo_idx==cell_topo_idx]
            energies = cell_e[topo_idx==cell_topo_idx]
            topo_parents[topo_idx] = torch.argmax(torch.bincount(parents+1,weights=energies)) - 1


        track_features = []
        for var in self.track_inputs:
            if var != "track_isIso":
                # arr = self.full_data_array[var][track_start:track_end]
                exec('track_features.append(self.{}[track_start:track_end])'.format(var))

        track_features = torch.stack( track_features ,dim=1)

        cell_features = []
        for var in self.cell_inputs:
            # arr =  self.full_data_array[var][cell_start:cell_end]

            exec('cell_features.append(self.{}[cell_start:cell_end])'.format(var))

        cell_features = torch.stack( cell_features  ,dim=1)
        
        num_nodes_dict = {
            'topos' : n_topos,
            'cells' : n_cells,
            'tracks' : n_tracks,
            'pre_nodes' : n_cells+n_tracks,
            'nodes' : n_topos+n_tracks,
            'photon' :   n_photon,
            'charged' :  n_charged,
            'neutral' :  n_neutral,
            'electron' : n_electron,
            'muon' :     n_muon,
            'particles' : n_particles,
            'supercharged' : n_supercharged,
            'superneutral' : n_superneutral,
            'global node' : 1,
            'pflow particles' : n_particles,
            'pflow photon' :   n_photon,
            'pflow charged' :  n_charged,
            'pflow neutral' :  n_neutral,
            'pflow electron' : n_electron,
            'pflow muon' :     n_muon,
            'pflow supercharged' : n_supercharged,
            'pflow superneutral' : n_superneutral,
        }


        #all children are connected to all potential parents:

        edge_list1 = torch.repeat_interleave( torch.arange(n_nodes),n_particles)
        edge_list2 = torch.arange(n_particles).repeat(n_nodes)

        #@cell_to_t = torch.arange(n_cells)
        #cell_to_topo = torch.tensor(topo_idx, dtype=int)
        cell_to_topo = cell_topo_idx.clone().detach().type(dtype=torch.int64)

        edge_list1_topo = torch.repeat_interleave(torch.arange(n_topos), n_topos)
        edge_list2_topo = torch.arange(n_topos).repeat(n_topos)

        edge_list1_topo_particle = torch.repeat_interleave( torch.arange(n_topos),n_particles)
        edge_list2_topo_particle = torch.arange(n_particles).repeat(n_topos)


        edge_list1_photon = torch.repeat_interleave( torch.arange(n_nodes),n_photon)
        edge_list2_photon = torch.arange(n_photon).repeat(n_nodes)

        edge_list1_supercharged_truth = torch.repeat_interleave( torch.arange(n_particles),n_supercharged)
        edge_list2_supercharged_truth = torch.arange(n_supercharged).repeat(n_particles)

        edge_list1_superneutral_truth = torch.repeat_interleave( torch.arange(n_superneutral),n_superneutral)
        edge_list2_superneutral_truth = torch.arange(n_superneutral).repeat(n_superneutral)

        edge_list1_charged = torch.repeat_interleave( torch.arange(n_nodes),n_charged)
        edge_list2_charged = torch.arange(n_charged).repeat(n_nodes)

        edge_list1_neutral = torch.repeat_interleave( torch.arange(n_nodes),n_neutral)
        edge_list2_neutral = torch.arange(n_neutral).repeat(n_nodes)

        edge_list1_electron = torch.repeat_interleave( torch.arange(n_nodes),n_electron)
        edge_list2_electron = torch.arange(n_electron).repeat(n_nodes)

        edge_list1_muon = torch.repeat_interleave( torch.arange(n_nodes),n_muon)
        edge_list2_muon = torch.arange(n_muon).repeat(n_nodes)

        edge_list1_supercharged = torch.repeat_interleave( torch.arange(n_nodes),n_supercharged)
        edge_list2_supercharged = torch.arange(n_supercharged).repeat(n_nodes)

        edge_list1_superneutral = torch.repeat_interleave( torch.arange(n_nodes),n_superneutral)
        edge_list2_superneutral = torch.arange(n_superneutral).repeat(n_nodes)

        #connect cells and tracks to their nodes
        cell_to_node = torch.arange(n_cells)
        track_to_node = torch.arange(n_tracks)
        topo_to_node = torch.arange(n_topos)

        edge_list_1_track_cell = torch.repeat_interleave( torch.arange(n_tracks),n_cells)
        edge_list_2_track_cell   = torch.arange(n_cells).repeat(n_tracks)

        edge_c_to_c_start = self.edge_c_to_c_start[idx]
        edge_c_to_c_end =  self.edge_c_to_c_end[idx]
        
        edge_t_to_c_start = self.edge_t_to_c_start[idx]
        edge_t_to_c_end =  self.edge_t_to_c_end[idx]
        
        #convert track edges to proper indx for nodes by adding n_cells to track start
        pre_node_to_pre_node_start = torch.cat([edge_c_to_c_start, n_cells+edge_t_to_c_start, edge_t_to_c_end],dim=0)
        pre_node_to_pre_node_end   = torch.cat([edge_c_to_c_end, edge_t_to_c_end, n_cells+edge_t_to_c_start],dim=0)

        node_to_node_start = torch.cat([edge_c_to_c_start, n_cells+edge_t_to_c_start, edge_t_to_c_end],dim=0)
        node_to_node_end   = torch.cat([edge_c_to_c_end, edge_t_to_c_end, n_cells+edge_t_to_c_start],dim=0)

        particle_to_particle_edge_start = torch.arange(n_particles).repeat(n_particles)
        particle_to_particle_edge_end = torch.repeat_interleave( torch.arange(n_particles),n_particles) 
        
        photon_to_photon_edge_start = torch.arange(n_photon).repeat(n_photon)
        photon_to_photon_edge_end = torch.repeat_interleave( torch.arange(n_photon),n_photon) 

        charged_to_charged_edge_start = torch.arange(n_charged).repeat(n_charged)
        charged_to_charged_edge_end = torch.repeat_interleave( torch.arange(n_charged),n_charged) 

        neutral_to_neutral_edge_start = torch.arange(n_neutral).repeat(n_neutral)
        neutral_to_neutral_edge_end = torch.repeat_interleave( torch.arange(n_neutral),n_neutral) 

        electron_to_electron_edge_start = torch.arange(n_electron).repeat(n_electron)
        electron_to_electron_edge_end = torch.repeat_interleave( torch.arange(n_electron),n_electron) 

        muon_to_muon_edge_start = torch.arange(n_muon).repeat(n_muon)
        muon_to_muon_edge_end = torch.repeat_interleave( torch.arange(n_muon),n_muon) 

        data_dict = {

            ('cells','cell_to_pre_node','pre_nodes') : (cell_to_node,cell_to_node),
            ('tracks','track_to_pre_node','pre_nodes') : (track_to_node, n_cells+track_to_node),
            #all nodes connected to the global node
            ('cells','cells to global','global node'): (torch.arange(n_cells).int(),torch.zeros(n_cells).int()),
            ('tracks','tracks to global','global node'): (torch.arange(n_tracks).int(),torch.zeros(n_tracks).int()),
            ('topos','topos to global','global node') : (torch.arange(n_topos).int(),torch.zeros(n_topos).int()), 

            ('pre_nodes','pre_node_to_pre_node','pre_nodes') : (pre_node_to_pre_node_start, pre_node_to_pre_node_end),
       
            ('topos','topo_to_node','nodes') : (topo_to_node, topo_to_node), # list needed
            ('tracks','track_to_node','nodes') : (track_to_node, n_topos+track_to_node),
            ('tracks','track_to_cell','cells') : (edge_list_1_track_cell, edge_list_2_track_cell),


            # # ('nodes','node_to_node','nodes') : (node_to_node_start, node_to_node_end),

            ('nodes', 'node_to_photon' , 'photon')  :  (edge_list1_photon  ,edge_list2_photon),
            ('nodes', 'node_to_charged' , 'charged')  : (edge_list1_charged ,edge_list2_charged),
            ('nodes', 'node_to_neutral' , 'neutral')  : (edge_list1_neutral ,edge_list2_neutral),
            ('nodes', 'node_to_electron', 'electron') : (edge_list1_electron,edge_list2_electron),
            ('nodes', 'node_to_muon'    , 'muon')     : (edge_list1_muon    ,edge_list2_muon),

            ('pre_nodes', 'pre_node_to_topo', 'topos') : (cell_to_node, cell_to_topo),
            ('cells', 'cell_to_topo', 'topos') : (cell_to_node, cell_to_topo),
            ('topos', 'topo_to_cell', 'cells') : (cell_to_topo, cell_to_node),                    
            ('topos', 'topo_to_topo', 'topos') : (edge_list1_topo, edge_list2_topo),
            ('topos', 'topo_to_particle', 'particles') : (edge_list1_topo_particle, edge_list2_topo_particle),
            #for now we assume same number of pflow and truth objects
            #will be moved to the TSPN part
            ('nodes', 'node_to_pflow_photon' , 'pflow photon')  :   (edge_list1_photon  ,edge_list2_photon),
            ('nodes', 'node_to_pflow_charged' , 'pflow charged')  : (edge_list1_charged ,edge_list2_charged),
            ('nodes', 'node_to_pflow_neutral' , 'pflow neutral')  : (edge_list1_neutral ,edge_list2_neutral),
            ('nodes', 'node_to_pflow_electron', 'pflow electron') : (edge_list1_electron,edge_list2_electron),
            ('nodes', 'node_to_pflow_muon'    , 'pflow muon')     : (edge_list1_muon    ,edge_list2_muon),

            # for the new TSPN
            ('nodes', 'node_to_supercharged', 'supercharged')     : (edge_list1_supercharged    ,edge_list2_supercharged),
            ('nodes', 'node_to_pflow_supercharged', 'pflow supercharged')     : (edge_list1_supercharged    ,edge_list2_supercharged),

            ('nodes', 'node_to_superneutral', 'superneutral')     : (edge_list1_superneutral    ,edge_list2_superneutral),
            ('nodes', 'node_to_pflow_superneutral', 'pflow superneutral')     : (edge_list1_superneutral    ,edge_list2_superneutral),


            ('nodes', 'node_to_particle', 'particles') : (edge_list1,edge_list2),                  
            ('particles', 'particle_to_node', 'nodes'): (edge_list2, edge_list1),
         

            #all nodes connected to the global node
            ('nodes','nodes to global','global node'): (torch.arange(n_nodes).int(),torch.zeros(n_nodes).int()),

            #all particles connected to the global node
            #this is needed to get the prediction for one event
            ('particles','particles to global','global node'): (torch.arange(n_particles).int(),torch.zeros(n_particles).int()),
            ('photon','photon to global','global node'): (torch.arange(n_photon).int(),torch.zeros(n_photon).int()),
            ('neutral','neutral to global','global node'): (torch.arange(n_neutral).int(),torch.zeros(n_neutral).int()),
            ('superneutral','superneutral to global','global node'): (torch.arange(n_superneutral).int(),torch.zeros(n_superneutral).int()),

            ('particles','to_pflow','pflow particles') : (particle_to_particle_edge_start,particle_to_particle_edge_end),
            ('pflow particles','from_pflow','particles') : (particle_to_particle_edge_end,particle_to_particle_edge_start),
            

            #new TSPN             
            ('particles','to_pflow_supercharged','pflow supercharged') : (edge_list1_supercharged_truth,edge_list2_supercharged_truth),
            ('pflow supercharged','from_pflow_supercharged','particles') : (edge_list2_supercharged_truth,edge_list1_supercharged_truth),

            ('superneutral','to_pflow_superneutral','pflow superneutral') : (edge_list1_superneutral_truth,edge_list2_superneutral_truth),
            ('pflow superneutral','from_pflow_superneutral','superneutral') : (edge_list2_superneutral_truth,edge_list1_superneutral_truth),

            ('photon','to_pflow_photon','pflow photon')   : (photon_to_photon_edge_start,photon_to_photon_edge_end),
            ('pflow photon','from_pflow_photon','photon') : (photon_to_photon_edge_end,  photon_to_photon_edge_start),

            ('charged','to_pflow_charged','pflow charged')   : (charged_to_charged_edge_start,charged_to_charged_edge_end),
            ('pflow charged','from_pflow_charged','charged') : (charged_to_charged_edge_end,  charged_to_charged_edge_start),

            ('neutral','to_pflow_neutral','pflow neutral')   : (neutral_to_neutral_edge_start,neutral_to_neutral_edge_end),
            ('pflow neutral','from_pflow_neutral','neutral') : (neutral_to_neutral_edge_end,  neutral_to_neutral_edge_start),

            ('electron','to_pflow_electron','pflow electron')   : (electron_to_electron_edge_start,electron_to_electron_edge_end),
            ('pflow electron','from_pflow_electron','electron') : (electron_to_electron_edge_end,  electron_to_electron_edge_start),

            ('muon','to_pflow_muon','pflow muon')   : (muon_to_muon_edge_start,muon_to_muon_edge_end),
            ('pflow muon','from_pflow_muon','muon') : (muon_to_muon_edge_end,  muon_to_muon_edge_start)

        }


        #print(data_dict)
        #print(num_nodes_dict)
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
        g.nodes['particles'].data['particle_dep_energy'] = particle_dep_energy


        #need to change the dataloader - add a new class called charged and fix the number to the number of tracks


        classes = ["photon","neutral","charged","electron","muon","superneutral"]
        for idx_c, cl in enumerate(classes):
            p_start = class_start[idx_c]
            p_end   = class_end[idx_c]

            # g.nodes[cl].data['particle idx']      = self.full_data_array[cl+'_idx'][p_start:p_end] - self.particle_cumsum[idx]
            # g.nodes[cl].data['particle_phi']      = self.full_data_array[cl+'_phi'][p_start:p_end] 
            # g.nodes[cl].data['particle_pt']       = self.full_data_array[cl+'_pt'][p_start:p_end]
            # g.nodes[cl].data['particle_theta']    = self.full_data_array[cl+'_theta'][p_start:p_end]
            # g.nodes[cl].data['particle_eta']      = self.full_data_array[cl+'_eta'][p_start:p_end]
            # g.nodes[cl].data['particle_xhat']     = self.full_data_array[cl+'_xhat'][p_start:p_end]
            # g.nodes[cl].data['particle_yhat']     = self.full_data_array[cl+'_yhat'][p_start:p_end]
            # g.nodes[cl].data['charge class']      = self.full_data_array[cl+'_charge class'][p_start:p_end] 
            # g.nodes[cl].data['prod pos']          = torch.stack([ self.full_data_array[cl+'_particle_prod_x'][p_start:p_end],
            #                                                       self.full_data_array[cl+'_particle_prod_y'][p_start:p_end],
            #                                                       self.full_data_array[cl+'_particle_prod_z'][p_start:p_end]
            #                                                     ], dim=1) 

            exec("g.nodes[cl].data['particle idx']      = self.{}_idx[p_start:p_end] - self.particle_cumsum[idx]".format(cl))
            exec("g.nodes[cl].data['particle_phi']      = self.{}_phi[p_start:p_end] ".format(cl))
            exec("g.nodes[cl].data['particle_pt']       = self.{}_pt[p_start:p_end]".format(cl))
            exec("g.nodes[cl].data['particle_theta']    = self.{}_theta[p_start:p_end]".format(cl))
            exec("g.nodes[cl].data['particle_eta']      = self.{}_eta[p_start:p_end]".format(cl))
            exec("g.nodes[cl].data['particle_xhat']     = self.{}_xhat[p_start:p_end]".format(cl))
            exec("g.nodes[cl].data['particle_yhat']     = self.{}_yhat[p_start:p_end]".format(cl))
            exec("g.nodes[cl].data['charge_class']      = self.{}_charge_class[p_start:p_end] ".format(cl))
            exec("g.nodes[cl].data['particle_class']      = self.{}_particle_class[p_start:p_end] ".format(cl))
            exec("g.nodes[cl].data['prod pos']          = torch.stack([ self.{}_particle_prod_x[p_start:p_end], self.{}_particle_prod_y[p_start:p_end], self.{}_particle_prod_z[p_start:p_end] ], dim=1)".format(cl, cl, cl))
            exec("g.nodes[cl].data['particle_dep_energy']      = self.{}_dep_energy[p_start:p_end] ".format(cl))

        g.nodes['supercharged'].data['particle idx'] = self.supercharged_idx[track_start:track_end]
        g.nodes['superneutral'].data['particle idx'] = self.superneutral_idx[superneutral_start:superneutral_end]

        g.nodes['cells'].data['parent target'] = self.cell_parent_idx[cell_start:cell_end].float()
        g.nodes['tracks'].data['parent target'] =  self.track_parent_idx[track_start:track_end].float()
        g.nodes['cells'].data['node features'] = cell_features
        g.nodes['tracks'].data['node features'] =  track_features 

        g.nodes['topos'].data['isTrack'] = torch.LongTensor(np.zeros(n_topos))
        g.nodes['topos'].data['parent target'] = torch.Tensor(topo_parents)

        # g.nodes['cells'].data['isTrack'] = torch.LongTensor(np.zeros(n_cells))

        g.nodes['tracks'].data['isTrack'] = torch.LongTensor(np.ones(n_tracks))

        g.nodes['topos'].data['track_pt'] = torch.LongTensor(np.zeros(n_topos))
        g.nodes['tracks'].data['track_pt'] = self.track_pt[track_start:track_end].float()

        g.nodes['tracks'].data['track_eta_layer_0'] = self.var_transform["track_eta_layer_0"]['std']*self.track_eta_layer_0[track_start:track_end].float()+self.var_transform["track_eta_layer_0"]['mean']
        g.nodes['tracks'].data['track_eta_layer_4'] = self.var_transform["track_eta_layer_4"]['std']*self.track_eta_layer_4[track_start:track_end].float()+self.var_transform["track_eta_layer_4"]['mean']


        g.nodes['tracks'].data['cosin_track_phi_layer_0'] = self.cosin_track_phi_layer_0[track_start:track_end].float()
        g.nodes['tracks'].data['sinu_track_phi_layer_0'] = self.sinu_track_phi_layer_0[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_0'] = self.cosin_track_phi_layer_0[track_start:track_end].float()
        g.nodes['tracks'].data['track_phi_layer_0'] = torch.atan2(g.nodes['tracks'].data['sinu_track_phi_layer_0'], g.nodes['tracks'].data['cosin_track_phi_layer_0'])

        g.nodes['tracks'].data['cosin_track_phi_layer_4'] = self.cosin_track_phi_layer_4[track_start:track_end].float()
        g.nodes['tracks'].data['sinu_track_phi_layer_4'] = self.sinu_track_phi_layer_4[track_start:track_end].float()
        g.nodes['tracks'].data['cosin_track_phi_layer_4'] = self.cosin_track_phi_layer_4[track_start:track_end].float()
        g.nodes['tracks'].data['track_phi_layer_4'] = torch.atan2(g.nodes['tracks'].data['sinu_track_phi_layer_4'], g.nodes['tracks'].data['cosin_track_phi_layer_4'])


        g.nodes['tracks'].data['track_isIso'] = self.track_isIso[track_start:track_end].float()


        self.dummy_vector = torch.FloatTensor(np.zeros(cell_end))
        self.dummy_vector_track = torch.FloatTensor(np.zeros(track_end)) + 0.0000001

        g.nodes['nodes'].data['isTrack'] = torch.cat([g.nodes['topos'].data['isTrack'], g.nodes['tracks'].data['isTrack']],dim=0)
        g.nodes['nodes'].data['isMuon'] = torch.cat([g.nodes['topos'].data['isTrack'], self.track_isMuon[track_start:track_end]],dim=0)
        g.nodes['nodes'].data['track_pt'] = torch.cat([g.nodes['topos'].data['track_pt'], g.nodes['tracks'].data['track_pt']],dim=0)

        g.nodes['cells'].data['sin_phi_cell'] = self.sinu_cell_phi[cell_start:cell_end]
        g.nodes['cells'].data['cosin_phi_cell'] = self.cosin_cell_phi[cell_start:cell_end]

        g.nodes['cells'].data['phi_cell'] = torch.atan2(self.sinu_cell_phi[cell_start:cell_end],self.cosin_cell_phi[cell_start:cell_end])
        g.nodes['cells'].data['eta_cell'] = self.cell_eta[cell_start:cell_end]

        g.nodes['cells'].data['layer_cell']  = self.cell_layer[cell_start:cell_end]
        g.nodes['cells'].data['energy_cell'] = self.cell_e[cell_start:cell_end]

        g.nodes['tracks'].data['layer_track'] = self.dummy_vector_track[track_start:track_end]
        g.nodes['tracks'].data['energy_track'] = self.dummy_vector_track[track_start:track_end]
        g.nodes['tracks'].data['track_eta'] = self.var_transform["track_eta"]['std']*self.track_eta[track_start:track_end].float()+self.var_transform["track_eta"]['mean']
        track_phi  = torch.atan2(self.sinu_track_phi,self.cosin_track_phi)
        g.nodes['tracks'].data['track_phi'] = track_phi[track_start:track_end].float()
        # # (A) - defined above
        # g.edges['node_to_supercharged'].data["truth_attention"] = torch.tensor(truth_incidence[:,g.nodes['supercharged'].data['particle idx']].flatten()).float()

        # (B) - defined above
        g.edges['node_to_supercharged'].data["truth_attention"] = torch.tensor(truth_topoattention_weights[g.nodes['supercharged'].data['particle idx'],:].T.flatten()).float()
        #needs to implement the attention for superneutral# TODO

        return g


    def __len__(self):

        return self.nevents 


    def __getitem__(self, idx):
        
        return  self.get_single_item(idx), idx

