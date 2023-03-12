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

import gc


def collate_graphs(samples):
	graphs  = [x[0] for x in samples]
	incidence_truth = [x[1].unsqueeze(0) for x in samples]
	ptetaphi_truth  = [x[2].unsqueeze(0) for x in samples]
	class_truth     = [x[3].unsqueeze(0) for x in samples]
	has_track       = [x[4].unsqueeze(0) for x in samples]

	batched_g = dgl.batch(graphs)
	incidence_truth = torch.cat(incidence_truth)
	ptetaphi_truth  = torch.cat(ptetaphi_truth)
	class_truth     = torch.cat(class_truth)
	has_track       = torch.cat(has_track)

	return batched_g, incidence_truth, ptetaphi_truth, class_truth, has_track


class PflowSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the jets
        :param batch_size: batch size
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.drop_last = False

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            if n_nodes_i <= 1:
                continue
            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = max(n_of_size / self.batch_size, 1)

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]


class PflowDataset(Dataset):

	def __init__(self, filename, config=None, reduce_ds=1.0, bool_inc=False, isEval=False):
		# charged hadron: p+-, K+-, pion+-, Xi+, Omega, Sigma : 0 (0)
		# electron : 1 (1)
		# muon : 2 (2)
		# neutral hadron: n, pion0, K0, Xi0, lambda: 3 (0)
		# photon : 4 (1)
		# reordered classess inside the brackets
	

		self.config = config
		self.bool_inc = bool_inc
		self.isEval = isEval

		self.max_particles = self.config['max_particles']
		self.init_label_dicts()
		if (self.config['inc_assignment'] == 'hard2') and (isEval==False):
			self.remap_classes()

		f = uproot.open(filename)
		self.tree = f['Low_Tree']
		
		self.config=config
		self.var_transform = self.config['var transform']

		self.nevents = self.tree.num_entries
		if reduce_ds < 1.0 and reduce_ds > 0:
			self.nevents = int(self.nevents*reduce_ds)
		if reduce_ds >= 1.0:
			self.nevents = reduce_ds
		print(' we have ',self.nevents, ' events')
		
		self.init_variables_list()
		self.full_data_array = {}

		print('loading data:')

		# shape for each event (n_particle, something)
		self.full_data_array["particle_to_node_idx"]    = self.tree["particle_to_node_idx"].array(library='np', entry_stop=self.nevents)
		self.full_data_array["particle_to_node_weight"] = self.tree["particle_to_node_weight"].array(library='np', entry_stop=self.nevents)

		self.full_data_array["track_not_reg"] = self.tree["track_not_reg"].array(library='np',entry_stop=self.nevents)
		self.full_data_array["particle_pdgid_noC"] =  np.copy(self.tree["particle_pdgid"].array(library='np',entry_stop=self.nevents))

		self.full_data_array["particle_to_track"] = self.tree["particle_to_track"].array(library='np',entry_stop=self.nevents)
		self.full_data_array["particle_to_track"] = np.concatenate( self.full_data_array["particle_to_track"] )

		# transform in -1 and 1
		self.full_data_array["particle_to_track"] = np.where(self.full_data_array["particle_to_track"]==-1, self.full_data_array["particle_to_track"],1)

		for var in tqdm( self.cell_variables+self.particle_variables+self.track_variables):
			newvar = ""
			if "cosin_" in var or "sinu_" in var:
				replace = ""
				if "cosin_" in var: replace = "cosin_"
				if "sinu_" in var:  replace = "sinu_"
				newvar = var.replace(replace, '')
				self.full_data_array[var] = np.copy(self.tree[newvar].array(library='np',entry_stop=self.nevents))
			else: 
				self.full_data_array[var] = self.tree[var].array(library='np',entry_stop=self.nevents)

			#fdibello need to change the number of tracks and add the dedicated class
			if "track" in var or "particle" in var:
				if var == "track_parent_idx":
					self.full_data_array["track_isMuon"] = np.copy(self.full_data_array["track_parent_idx"])

				for nev in range(self.nevents):
					if(len(self.full_data_array[var][nev]) ==  len(self.full_data_array["track_not_reg"][nev])):
						self.full_data_array[var][nev] = [ self.full_data_array[var][nev][i] for i in range(0,len(self.full_data_array["track_not_reg"][nev])) if self.full_data_array["track_not_reg"][nev][i] == 0]
					
					if var == "track_parent_idx":
						self.full_data_array["track_isMuon"][nev] = [ self.full_data_array["track_isMuon"][nev][i] for i in range(0,len(self.full_data_array["track_not_reg"][nev])) if self.full_data_array["track_not_reg"][nev][i] == 0]
						self.full_data_array["track_isMuon"][nev] = [self.classIsMuon[self.class_labels[self.full_data_array["particle_pdgid_noC"][nev][x]]] for x in self.full_data_array["track_parent_idx"][nev]]

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
		
		# adding the raw phis as well. Computing it from cosin and sin, coz it guarentees that phi will be in (-pi, +pi)
		for var in tqdm(self.cell_inputs + self.track_inputs):
			if 'phi' in var and ('sinu' not in var and 'cosin' not in var):
				self.full_data_array[var] = np.arctan2(self.full_data_array['sinu_'+var], self.full_data_array['cosin_'+var])

		#add decorater to the track
		self.full_data_array["track_isMuon"] = np.concatenate( self.full_data_array["track_isMuon"] )

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

		for var in tqdm(self.track_inputs + self.cell_inputs + ['particle_pt','particle_eta','particle_phi'] + ['cell_topo_idx']):
			if var in self.var_transform:
				self.full_data_array[var] = (self.full_data_array[var] - self.var_transform[var]['mean']) / self.var_transform[var]['std']
			self.full_data_array[var] = torch.tensor(self.full_data_array[var])

		self.cell_cumsum = np.cumsum([0]+self.n_cells)
		self.track_cumsum = np.cumsum([0]+self.n_tracks)
		self.particle_cumsum = np.cumsum([0]+self.n_particles)

		self.edge_c_to_c_start =  [torch.tensor(x) for x in self.tree['cell_to_cell_edge_start'].array(library='np',entry_stop=self.nevents)]
		self.edge_c_to_c_end =   [torch.tensor(x) for x in self.tree['cell_to_cell_edge_end'].array(library='np',entry_stop=self.nevents) ]
		
		self.edge_t_to_c_start =  [torch.tensor(x) for x in  self.tree['track_to_cell_edge_start'].array(library='np',entry_stop=self.nevents)]
		self.edge_t_to_c_end =   [torch.tensor(x) for x in self.tree['track_to_cell_edge_end'].array(library='np',entry_stop=self.nevents)]

		del self.tree
		del particle_phi, particle_p, particle_theta, particle_eta, particle_xhat, particle_yhat, particle_pt

		# needed for batch sampling
		self.n_nodes = []; cell_ind_stop = 0
		for i in range(len(self.n_cells)):
			cell_ind_start, cell_ind_stop = cell_ind_stop, cell_ind_stop + self.n_cells[i]
			if cell_ind_start != cell_ind_stop:
				n_topo = max(self.full_data_array['cell_topo_idx'][cell_ind_start:cell_ind_stop]).item()
			else:
				n_topo = 0
			self.n_nodes.append(n_topo + self.n_tracks[i])
		self.n_nodes = np.array(self.n_nodes)

		# charge particles with tracks that don't deposit any energy are defined to be electrons
		no_energy_dep_ch_mask = self.full_data_array['particle_to_track'] \
			* (self.full_data_array['particle_dep_energy'] == 0)\
			* (self.particle_class != 2).detach().numpy()
		self.particle_class[no_energy_dep_ch_mask] = 1 ##!!!!!!!!! Hard-coded, we don't care about it during evaluation

		# # print('n_nodes:', self.n_nodes)
		# # print('n_particles:', self.n_particles)
		# print('max/min n_nodes:', max(self.n_nodes), min(self.n_nodes))
		# print('max/min n_particles:', max(self.n_particles), min(self.n_particles))
		# raise ValueError("stop!")

		print('done loading data\n')
		gc.collect()

		self.neg_contribs = []
		self.fake_TC_count = []


	def get_single_item(self, idx):
		
		# particles       : true hyperedges
		# pflow_particles : predicted hyperedges
		# we compare their incidence matrices (node-particle vs node-pflow_particles)

		n_cells  = self.n_cells[idx]
		n_tracks = self.n_tracks[idx]
		n_particles = self.n_particles[idx]

		cell_start, cell_end = self.cell_cumsum[idx],self.cell_cumsum[idx+1]
		track_start, track_end = self.track_cumsum[idx],self.track_cumsum[idx+1]
		particle_start, particle_end = self.particle_cumsum[idx],self.particle_cumsum[idx+1]

		particle_pdg    = self.full_data_array['particle_pdgid'][particle_start:particle_end]
		particle_class  = self.particle_class[particle_start:particle_end]

		particle_pt     = self.full_data_array['particle_pt'][particle_start:particle_end]
		particle_eta    = self.full_data_array['particle_eta'][particle_start:particle_end]
		particle_phi    = self.full_data_array['particle_phi'][particle_start:particle_end]
		particle_e      = self.full_data_array['particle_e'][particle_start:particle_end]
		particle_dep_energy = self.full_data_array['particle_dep_energy'][particle_start:particle_end]
		
		# not a tensor
		particle_to_track =  self.full_data_array['particle_to_track'][particle_start:particle_end]

		track_features = []
		for var in self.track_inputs:
			arr = self.full_data_array[var][track_start:track_end]
			track_features.append(arr)
		track_features = torch.stack( track_features ,dim=1)

		cell_features = []
		for var in self.cell_inputs:
			arr =  self.full_data_array[var][cell_start:cell_end]
			cell_features.append(arr)
		cell_features = torch.stack( cell_features  ,dim=1)
		
		# cell_topo_idx starts with 1 | all the other indices are float64
		cell_topo_idx = self.full_data_array['cell_topo_idx'][cell_start:cell_end].type(torch.LongTensor) - 1
		if len(cell_topo_idx ) != 0:
			n_topoclusters = int(max(cell_topo_idx)) + 1
		else:
			n_topoclusters = 0
		n_nodes = n_topoclusters+n_tracks

		num_nodes_dict = {
			'cells' : n_cells,
			'tracks' : n_tracks,
			'pre_nodes' : n_cells+n_tracks, 
			'nodes' : n_topoclusters+n_tracks,
			'particles' : self.max_particles,
			'global node' : 1,
			'pflow_particles' : self.max_particles,
			'topoclusters' : n_topoclusters,
			'global_node' : 1,
			'status' : 1
		}

		#all children are connected to all potential parents:

		edge_list1 = torch.repeat_interleave( torch.arange(n_nodes),self.max_particles)
		edge_list2 = torch.arange(self.max_particles).repeat(n_nodes)

		cell_to_topocluster_start = torch.arange(n_cells)

		cell_to_pre_node = torch.arange(n_cells)
		track_to_pre_node = torch.arange(n_tracks)

		#connect topoclusters and tracks to their nodes
		topocluster_to_node = torch.arange(n_topoclusters)
		track_to_node = torch.arange(n_tracks)

		edge_c_to_c_start = self.edge_c_to_c_start[idx]
		edge_c_to_c_end =  self.edge_c_to_c_end[idx]
		
		edge_t_to_c_start = self.edge_t_to_c_start[idx]
		edge_t_to_c_end =  self.edge_t_to_c_end[idx]
		
		pre_node_to_pre_node_start = torch.cat([edge_c_to_c_start, n_cells+edge_t_to_c_start, edge_t_to_c_end],dim=0)
		pre_node_to_pre_node_end   = torch.cat([edge_c_to_c_end, edge_t_to_c_end, n_cells+edge_t_to_c_start],dim=0)

		node_to_node_start, node_to_node_end = list(zip(*itertools.permutations(np.arange(n_nodes), 2)))
		node_to_node_start, node_to_node_end = torch.tensor(node_to_node_start), torch.tensor(node_to_node_end)

		particle_to_particle_edge_start = torch.arange(self.max_particles).repeat(self.max_particles)
		particle_to_particle_edge_end = torch.repeat_interleave( torch.arange(self.max_particles),self.max_particles) 
		
		data_dict = {
			('cells','cell_to_pre_node','pre_nodes') : (cell_to_pre_node, cell_to_pre_node),
			('tracks','track_to_pre_node','pre_nodes') : (track_to_pre_node, track_to_pre_node),
			('pre_nodes','pre_node_to_pre_node','pre_nodes') : (pre_node_to_pre_node_end, pre_node_to_pre_node_start), # Flipped to Fix

			('pre_nodes','pre_node_to_topocluster','topoclusters') : (cell_to_topocluster_start, cell_topo_idx),
			('cells','cell_to_topocluster','topoclusters') : (cell_to_topocluster_start, cell_topo_idx), # indices are same as above line

			('topoclusters','topocluster_to_node','nodes') : (topocluster_to_node,topocluster_to_node),
			('tracks','track_to_node','nodes') : (track_to_node, n_topoclusters+track_to_node),

			('nodes','node_to_node','nodes') : (node_to_node_start, node_to_node_end),
	   
			('nodes', 'node_to_particle', 'particles') : (edge_list1,edge_list2),                  
			('particles', 'particle_to_node', 'nodes'): (edge_list2, edge_list1),
		 
			('nodes', 'node_to_pflow_particle', 'pflow_particles') : (edge_list1,edge_list2),                  
			('pflow_particles', 'pflow_particle_to_node', 'nodes'): (edge_list2, edge_list1),

			('particles','to_pflow','pflow_particles') : (particle_to_particle_edge_start, particle_to_particle_edge_end),
			('pflow_particles','from_pflow','particles') : (particle_to_particle_edge_end, particle_to_particle_edge_start),

			('pflow_particles','pflow_to_pflow','pflow_particles') : (particle_to_particle_edge_end, particle_to_particle_edge_start),

            # all nodes connected to the global node (NOT SURE IF WE NEED THIS)
			('nodes','nodes_to_global','global_node'): (torch.arange(n_nodes).int(),torch.zeros(n_nodes).int()),
		}

		g = dgl.heterograph(data_dict, num_nodes_dict)

		g.nodes['cells'].data['node features']  = cell_features
		g.nodes['tracks'].data['node features'] = track_features

		dummy_topo_pt = torch.FloatTensor(np.zeros(n_topoclusters)) # + 1e-8 # log(pT)
		dummy_track_layer  = torch.FloatTensor(np.zeros(n_tracks)) - 1
		dummy_track_energy = torch.FloatTensor(np.zeros(n_tracks)) # + 1e-8 # log(energy)

		g.nodes['topoclusters'].data['isTrack'] = torch.LongTensor(np.zeros(n_topoclusters))
		g.nodes['tracks'].data['isTrack'] = torch.LongTensor(np.ones(n_tracks))

		g.nodes['topoclusters'].data['track_pt'] = dummy_topo_pt
		g.nodes['tracks'].data['track_pt'] = self.full_data_array['track_pt'][track_start:track_end].float()

		g.nodes['tracks'].data['track_eta'] = self.full_data_array['track_eta'][track_start:track_end].float()
		g.nodes['tracks'].data['track_phi'] = self.full_data_array['track_phi'][track_start:track_end].float()

		g.nodes['tracks'].data['track_eta_layer_0'] = self.full_data_array['track_eta_layer_0'][track_start:track_end].float()
		g.nodes['tracks'].data['track_phi_layer_0'] = self.full_data_array['track_phi_layer_0'][track_start:track_end].float()

		g.nodes['nodes'].data['isTrack'] = torch.cat([g.nodes['topoclusters'].data['isTrack'], g.nodes['tracks'].data['isTrack']],dim=0)
		g.nodes['nodes'].data['isMuon'] = torch.cat([g.nodes['topoclusters'].data['isTrack'], self.full_data_array['track_isMuon'][track_start:track_end]],dim=0)
		g.nodes['nodes'].data['track_pt'] = torch.cat([g.nodes['topoclusters'].data['track_pt'], g.nodes['tracks'].data['track_pt']],dim=0)

		g.nodes['cells'].data['phi_cell'] = self.full_data_array['cell_phi'][cell_start:cell_end]
		g.nodes['cells'].data['eta_cell'] = self.full_data_array['cell_eta'][cell_start:cell_end]

		g.nodes['cells'].data['layer_cell']  = self.full_data_array['cell_layer'][cell_start:cell_end]
		g.nodes['cells'].data['energy_cell'] = self.full_data_array['cell_e'][cell_start:cell_end]

		g.nodes['tracks'].data['layer_track']  = dummy_track_layer
		g.nodes['tracks'].data['energy_track'] = dummy_track_energy

		# shape (n_particles, something) after transpose # using old node convention
		particle_to_node_idx    = self.full_data_array['particle_to_node_idx'][idx]
		particle_to_node_weight = self.full_data_array['particle_to_node_weight'][idx]

		truth_incidence = np.zeros((n_nodes, self.max_particles))

		# print("n_tracks", n_tracks)
		# print("n_topoclusters", n_topoclusters)
		# print("n_particles", n_particles)


		#***************************************#
		# incidence 1: sum for one particle = 1 # (A)
		#***************************************#

		for p_idx, (n_idx, n_weights) in enumerate(zip(particle_to_node_idx, particle_to_node_weight)):

			n_idx     = np.array(n_idx, dtype=int)
			n_weights = np.array(n_weights)

			# ghost particles
			if len(n_idx) == 0:
				continue

			# track attention
			if n_weights[-1] == 0.5:
				truth_incidence[n_idx[-1] - n_cells + n_topoclusters, p_idx] = 1

			# topocluster attention
			n_weights = n_weights[n_idx<n_cells]
			n_idx = n_idx[n_idx<n_cells]

			if len(n_idx) > 0:
				bc = np.bincount(cell_topo_idx[n_idx], weights=n_weights)

				if particle_to_track[p_idx] == -1: # already sums to 1
					truth_incidence[:len(bc), p_idx] = bc
				else:
					truth_incidence[:len(bc), p_idx] = 2*bc

		truth_incidence = truth_incidence.T



		#*********************************#
		# incidence 2: sum for one TC = 1 # (B) (uses A)
		#*********************************#

		# ch particles with tracks but no deposited energy
		particle_dep_energy[(particle_dep_energy == 0) * (particle_to_track == 1)] = 1

		particle_dep_energy_padded = np.zeros(self.max_particles)
		particle_dep_energy_padded[:particle_dep_energy.shape[0]] = particle_dep_energy

		truth_incidence_pre = truth_incidence
		truth_incidence = truth_incidence * particle_dep_energy_padded.reshape(-1,1)


		# negative energy contribution
		# for a given col

		# 	   	       Now       opt1     opt2
		# -----------------------------------------
		# |  A ->  8  |   8/10  |  8/10  |  8/11  |
		# |  B ->  3  |   3/10  |  3/10  |  3/11  |
		# |  C -> -1  |  -1/10  |  0/10  |  0/11  |
		# -----------------------------------------
		# |  TC -> 10 |              X   |

		# TC = sum of the col = 8 + 3 - 1 = 10 (in dataloader.py)
		# during NN, TC = 10




		if (truth_incidence < 0).sum() != 0:
			self.neg_contribs.extend( truth_incidence[ np.where(truth_incidence < 0)[0], np.where(truth_incidence < 0)[1]] )
			truth_incidence[ np.where(truth_incidence < 0)[0], np.where(truth_incidence < 0)[1]] = 0




		# creating fake hyperedges in the incidence matrix for the fake TCs
		# won't regress thier pt,eta etc. (masking for this is done with trut_ptetaphi in metrics.py)
		n_fake_TCs = (truth_incidence.sum(axis=0) == 0).sum()
		if n_fake_TCs != 0:
			row_pos = np.arange(n_fake_TCs) + n_particles
			col_pos = np.where(truth_incidence.sum(axis=0) == 0)[0]
			truth_incidence[row_pos, col_pos] = 1

		self.fake_TC_count.append(n_fake_TCs)

		truth_incidence = truth_incidence / truth_incidence.sum(axis=0, keepdims=True)

		if self.bool_inc==True:
			truth_incidence = truth_incidence > 0.01

		g.edges['pflow_particle_to_node'].data['incidence_val'] = torch.zeros(g.num_edges('pflow_particle_to_node'))
		g.edges['node_to_pflow_particle'].data['incidence_val'] = torch.zeros(g.num_edges('node_to_pflow_particle'))

		ptetaphi = torch.zeros(self.max_particles, 3)
		ptetaphi[:n_particles,0] = particle_pt
		ptetaphi[:n_particles,1] = particle_eta
		ptetaphi[:n_particles,2] = particle_phi

		# del particle_pdg, particle_e, particle_pt, particle_eta, particle_phi, 
		# track_features, cell_features, cell_topo_idx

		# assigning class 1 (same as neutrals) to garbage particles
		# this is to keep class prediction for neutral restricted to 2 classes
		# these won't be used in the loss computation anyway
		class_truth = torch.zeros(self.max_particles) + 1 # garbage are class 1
		class_truth[:n_particles] = particle_class

		gc.collect()

		# has track is not zero padded as we use it only for masking etc (no loss computation)
		has_track = torch.zeros(self.max_particles) - 1
		has_track[:n_particles] = torch.FloatTensor(particle_to_track)
		return g, torch.FloatTensor(truth_incidence), ptetaphi, class_truth.long(), has_track



	def __len__(self):
		return self.nevents 


	def __getitem__(self, idx):
		return  self.get_single_item(idx)


	def remap_classes(self):
		# TL,DR; neutral partilcles have     class_lable -= 3
		# charged hadron: 0, electron: 1, muon: 2, neutral hadron: 3, photon: 4
		# to 											   *				  *
		# charged hadron: 0, electron: 1, muon: 2, neutral hadron: 0, photon: 1
		#											      
		swaps = {0:0, 1:1, 2:2, 3:0, 4:1}
		for key, val in self.class_labels.items():
			self.class_labels[key] = swaps[val]

	def init_label_dicts(self):

		# photon: 0, charged hadron: 1, neutral hadron: 2, electron: 3, muon: 4
		self.class_labels_old = {-3112 : 1,
							  3112 : 1,
							  3222 : 1,
							 -3222 : 1,
							 -3334 : 1,
							  3334 : 1,
							 -3122 : 2,
							  3122 : 2,
							   310 : 2,
							  3312 : 1,
							 -3312 : 1,
							  3322 : 2,
							 -3322 : 2,
							  2112 : 2,
							   321 : 1,
							   130 : 2,
							 -2112 : 2,
							  2212 : 1,
								11 : 3,
							  -211 : 1,
								13 : 4,
							   211 : 1,
							   -13 : 4,
							   -11 : 3,
								22 : 0,
							 -2212 : 1,
							  -321 : 1}

		self.class_labels = {-3112 : 0,
							  3112 : 0,
							  3222 : 0,
							 -3222 : 0,
							 -3334 : 0,
							  3334 : 0,
							 -3122 : 3,
							  3122 : 3,
							   310 : 3,
							  3312 : 0,
							 -3312 : 0,
							  3322 : 3,
							 -3322 : 3,
							  2112 : 3,
							   321 : 0,
							   130 : 3,
							 -2112 : 3,
							  2212 : 0,
								11 : 1,
							  -211 : 0,
								13 : 2,
							   211 : 0,
							   -13 : 2,
							   -11 : 1,
								22 : 4,
							 -2212 : 0,
							  -321 : 0}


		self.charge_labels = {130 : 0,
							-3322 : 0,
							 3334 : 2,
							   11 : 2,
							   13 : 2,
							-3312 : 1,
							   22 : 0,
							 3222 : 1,
							 2212 : 1,
							 3112 : 2,
							 -211 : 2,
							 3122 : 0,
							  310 : 0,
							 -321 : 2,
							-2112 : 0,
							  321 : 1,
							 2112 : 0,
							-3122 : 0,
							  211 : 1,
							-3112 : 1,
							-2212 : 2,
							-3334 : 1,
							-3222 : 2,
							 3312 : 2,
							  -13 : 1,
							  -11 : 1,
							 3322 : 0}

		#class - particle_to_track
		self.class_particle_to_track = {
							   0 : 0,
							   1 : 1,
							  -1 : 5,
							   2 : 2,
							  -2 : 2,
							   3 : 3,
							  -3 : 5,
							   4 : 4,
							  -4 : 5}
		self.classIsMuon = {
							  0 : 0,
							  1 : 0,
							  2 : 0,
							  3 : 0,
							  4 : 1}


	def init_variables_list(self):

		self.track_variables = [
			'track_parent_idx',
			'track_d0',
			'track_z0',
			'sinu_track_phi',
			'cosin_track_phi',
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
			'track_qoverp'
		]
		
		self.track_inputs = [
			'track_d0',
			'track_z0',
			'track_pt',
			'track_eta',
			'track_phi',
			'sinu_track_phi',
			'cosin_track_phi',
			'track_eta_layer_0',
			'track_eta_layer_1',
			'track_eta_layer_2',
			'track_eta_layer_3',
			'track_eta_layer_4',
			'track_eta_layer_5',
			'track_phi_layer_0',
			'track_phi_layer_1',
			'track_phi_layer_2',
			'track_phi_layer_3',
			'track_phi_layer_4',
			'track_phi_layer_5',
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
			'track_isMuon'
		]

		self.cell_variables = ['cell_x','cell_y','cell_z','cell_e','cell_eta','cosin_cell_phi','sinu_cell_phi','cell_layer','cell_particle_target','cell_parent_idx','cell_topo_idx']
		
		self.cell_inputs =    ['cell_x','cell_y','cell_z','cell_e','cell_eta','cell_phi','cosin_cell_phi','sinu_cell_phi','cell_layer']
		
		self.particle_variables = ['particle_pdgid','particle_px','particle_py','particle_pz','particle_e',
									'particle_prod_x','particle_prod_y','particle_prod_z', 'particle_dep_energy']




