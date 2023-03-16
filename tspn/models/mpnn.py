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


class NodeNetwork(nn.Module):
    def __init__(self,inputsize,outputsize,layers):
        super().__init__()
        
        self.net = build_layers(3*inputsize,outputsize,layers)

    def forward(self, x):
        
        inputs = torch.sum( x.mailbox['message'] ,dim=1)
        
        inputs = torch.cat([inputs,x.data['hidden rep'],x.data['global rep']],dim=1)
        
        output = self.net(inputs)
        output = output / torch.norm(output, p='fro', dim=1, keepdim=True)
        
        return {'hidden rep': output }

class MPNN(nn.Module):
    def __init__(self,config,high_level_config=None):
        super().__init__()

        self.config = config

        self.cell_init_network = build_layers(config['cell inputsize'],
                                                   config['cell hidden size'],
                                                   config['node init layers'],add_batch_norm=True)
        self.track_init_network = build_layers(config['track inputsize'],
                                                   config['track hidden size'],
                                                   config['track init layers'],add_batch_norm=True)
        
        
        # right now assuming cell and track hidden size
        # is the same
        self.hidden_size = config['cell hidden size'] 
        
        self.node_update_networks = nn.ModuleList()
        
        self.n_blocks = config['n GN blocks']
        self.block_iterations = config['n iterations']        
        
        self.transform_var = high_level_config['var transform']

        for block_i in range(self.n_blocks):    
            self.node_update_networks.append(NodeNetwork(self.hidden_size, 
                                           self.hidden_size,config['node net layers']))


    def topo_message(self,edges):
        topo_number = 1-edges.src['isTrack']

        topo_number_track_1   = topo_number*(edges.src['nTracks'] > 0)*(edges.src['layer'] < 1)
        topo_number_notrack_1 = topo_number*(edges.src['nTracks'] == 0)*(edges.src['layer'] < 1)
        topo_number_track_2   = topo_number*(edges.src['nTracks'] > 0)*((edges.src['layer'] > 1) & (edges.src['layer'] < 2))
        topo_number_notrack_2 = topo_number*(edges.src['nTracks'] == 0)*((edges.src['layer'] > 1) & (edges.src['layer'] < 2))
        topo_number_track_3   = topo_number*(edges.src['nTracks'] > 0)*((edges.src['layer'] > 2) & (edges.src['layer'] < 3))
        topo_number_notrack_3 = topo_number*(edges.src['nTracks'] == 0)*((edges.src['layer'] > 2) & (edges.src['layer'] < 3))

        topo_number_track_4   = topo_number*(edges.src['nTracks'] > 0)*((edges.src['layer'] > 3) & (edges.src['layer'] < 4))
        topo_number_notrack_4 = topo_number*(edges.src['nTracks'] == 0)*((edges.src['layer'] > 3) & (edges.src['layer'] < 4))
        topo_number_track_5   = topo_number*(edges.src['nTracks'] > 0)*((edges.src['layer'] > 4) & (edges.src['layer'] < 5))
        topo_number_notrack_5 = topo_number*(edges.src['nTracks'] == 0)*((edges.src['layer'] > 4) & (edges.src['layer'] < 5))
        topo_number_track_6   = topo_number*(edges.src['nTracks'] > 0)*(edges.src['layer'] > 5 )
        topo_number_notrack_6 = topo_number*(edges.src['nTracks'] == 0)*(edges.src['layer'] > 5)

        return{'m_topo_number':topo_number,'m_topo_number_track_1':topo_number_track_1,'m_topo_number_notrack_1':topo_number_notrack_1,
                'm_topo_number_track_2':topo_number_track_2,'m_topo_number_notrack_2':topo_number_notrack_2,
                'm_topo_number_track_3':topo_number_track_3,'m_topo_number_notrack_3':topo_number_notrack_3,
                'm_topo_number_track_4':topo_number_track_4,'m_topo_number_notrack_4':topo_number_notrack_4,
                'm_topo_number_track_5':topo_number_track_5,'m_topo_number_notrack_5':topo_number_notrack_5,
                'm_topo_number_track_6':topo_number_track_6,'m_topo_number_notrack_6':topo_number_notrack_6}

    def global_update_topos(self,nodes):
        topo_number         = torch.sum(nodes.mailbox['m_topo_number'],dim=1)
        topo_number_track_1   = torch.sum(nodes.mailbox['m_topo_number_track_1'],dim=1)
        topo_number_notrack_1 = torch.sum(nodes.mailbox['m_topo_number_notrack_1'],dim=1)
        topo_number_track_2   = torch.sum(nodes.mailbox['m_topo_number_track_2'],dim=1)
        topo_number_notrack_2 = torch.sum(nodes.mailbox['m_topo_number_notrack_2'],dim=1)
        topo_number_track_3   = torch.sum(nodes.mailbox['m_topo_number_track_3'],dim=1)
        topo_number_notrack_3 = torch.sum(nodes.mailbox['m_topo_number_notrack_3'],dim=1)
        topo_number_track_4   = torch.sum(nodes.mailbox['m_topo_number_track_4'],dim=1)
        topo_number_notrack_4 = torch.sum(nodes.mailbox['m_topo_number_notrack_4'],dim=1)
        topo_number_track_5   = torch.sum(nodes.mailbox['m_topo_number_track_5'],dim=1)
        topo_number_notrack_5 = torch.sum(nodes.mailbox['m_topo_number_notrack_5'],dim=1)
        topo_number_track_6   = torch.sum(nodes.mailbox['m_topo_number_track_6'],dim=1)
        topo_number_notrack_6 = torch.sum(nodes.mailbox['m_topo_number_notrack_6'],dim=1)
        return{'number_topos':topo_number,'number_topos_track_1':topo_number_track_1,'number_topos_notrack_1':topo_number_notrack_1,
                'number_topos_track_2':topo_number_track_2,'number_topos_notrack_2':topo_number_notrack_2,
                'number_topos_track_3':topo_number_track_3,'number_topos_notrack_3':topo_number_notrack_3,
                'number_topos_track_4':topo_number_track_4,'number_topos_notrack_4':topo_number_notrack_4,
                'number_topos_track_5':topo_number_track_5,'number_topos_notrack_5':topo_number_notrack_5,
                'number_topos_track_6':topo_number_track_6,'number_topos_notrack_6':topo_number_notrack_6}


    def track_message(self,edges):
        track_pt     = edges.src['track_pt']
        track_number = edges.src['isTrack']
        return{'m_track_pt':track_pt,'m_track_number':track_number}

    def global_update_tracks(self,nodes):
        track_pt = torch.sum(nodes.mailbox['m_track_pt'],dim=1)
        track_number = torch.sum(nodes.mailbox['m_track_number'],dim=1)
        return{'tracks_pt':track_pt,'number_tracks':track_number}


    def cell_message(self,edges):
        energy_layer_all_layer = edges.src['energy_cell']
        cell_layer = edges.src['layer_cell']

        energy_layer_0 = energy_layer_all_layer * (cell_layer==0)
        energy_layer_1 = energy_layer_all_layer * (cell_layer==1)
        energy_layer_2 = energy_layer_all_layer * (cell_layer==2)
        energy_layer_3 = energy_layer_all_layer * (cell_layer==3)
        energy_layer_4 = energy_layer_all_layer * (cell_layer==4)
        energy_layer_5 = energy_layer_all_layer * (cell_layer==5)

        return{'m_energy_layer_0':energy_layer_0,'m_energy_layer_1':energy_layer_1,'m_energy_layer_2':energy_layer_2,
        'm_energy_layer_3':energy_layer_3,'m_energy_layer_4':energy_layer_4,'m_energy_layer_5':energy_layer_5}

    def global_update_energy(self,nodes):

        energy_l_0 = torch.sum(nodes.mailbox['m_energy_layer_0'],dim=1)
        energy_l_1 = torch.sum(nodes.mailbox['m_energy_layer_1'],dim=1)
        energy_l_2 = torch.sum(nodes.mailbox['m_energy_layer_2'],dim=1)
        energy_l_3 = torch.sum(nodes.mailbox['m_energy_layer_3'],dim=1)
        energy_l_4 = torch.sum(nodes.mailbox['m_energy_layer_4'],dim=1)
        energy_l_5 = torch.sum(nodes.mailbox['m_energy_layer_5'],dim=1)

        return{'energy_l_0':energy_l_0,'energy_l_1':energy_l_1,'energy_l_2':energy_l_2,
        'energy_l_3':energy_l_3,'energy_l_4':energy_l_4,'energy_l_5':energy_l_5}


    def update_global_rep(self,g):
        
        global_rep = dgl.sum_nodes(g,'hidden rep', ntype='pre_nodes')  
        # global_rep = global_rep / torch.norm(global_rep, p='fro', dim=1, keepdim=True)

        g.nodes['pre_nodes'].data['global rep'] = dgl.broadcast_nodes(g, global_rep, ntype='pre_nodes')
        g.nodes['global node'].data['global rep'] = global_rep
        
    
    def move_from_cellstracks_to_pre_nodes(self,g,cell_info,track_info,target_name):

        g.update_all(fn.copy_src(cell_info,'m'),fn.sum('m',target_name),etype='cell_to_pre_node')
        cell_only_data = g.nodes['pre_nodes'].data[target_name]
        g.update_all(fn.copy_src(track_info,'m'),fn.sum('m',target_name),etype='track_to_pre_node')
        g.nodes['pre_nodes'].data[target_name] = g.nodes['pre_nodes'].data[target_name]+cell_only_data


    def move_from_topostracks_to_nodes(self,g,topo_info,track_info,target_name):


        g.update_all(fn.copy_src(topo_info,'m'),fn.sum('m',target_name),etype='topo_to_node')
        topo_only_data = g.nodes['nodes'].data[target_name]

        g.update_all(fn.copy_src(track_info,'m'),fn.sum('m',target_name),etype='track_to_node')
        g.nodes['nodes'].data[target_name] = g.nodes['nodes'].data[target_name]+topo_only_data

    def deltaR(self,phi0,eta0,phi1,eta1):

        deta = eta0-eta1
        dphi = phi0-phi1
        dphi[dphi > np.pi] = dphi[dphi > np.pi]-2*np.pi
        dphi[dphi < - np.pi] = dphi[dphi < - np.pi]+2*np.pi

        dR = torch.sqrt( deta**2+dphi**2 )

        return dR

    def track_vals_edge(self, edges):
        eta_track = edges.src['track_eta_layer_0']
        eta_cell =  edges.dst['eta_cell'] * self.transform_var['cell_eta']['std'] + self.transform_var['cell_eta']['mean']


        phi_track = edges.src['track_phi_layer_0']
        phi_cell  = edges.dst['phi_cell']

        cell_layer = (edges.dst['layer_cell'] == 0) | (edges.dst['layer_cell'] == 1) | (edges.dst['layer_cell'] == 2)
        cell_layer_4 =  (edges.dst['layer_cell'] == 3) | (edges.dst['layer_cell'] == 4) | (edges.dst['layer_cell'] == 5)

        eta_track_4 = edges.src['track_eta_layer_4']
        phi_track_4 = edges.src['track_phi_layer_4']

        dR = self.deltaR(phi_cell,eta_cell,phi_track,eta_track)
        dR_4 = self.deltaR(phi_cell,eta_cell,phi_track_4,eta_track_4)
        th_R = 0.035
        th_R4 = 0.15
        edge_dR_labels_4 = (dR_4<th_R4) * cell_layer_4 
        edge_dR_labels = (dR<th_R) * cell_layer
        tr_pt = edges.src['track_pt'] * edge_dR_labels
        tr_pt_4 = edges.src['track_pt'] * edge_dR_labels_4


        return{'edge_dR_labels':edge_dR_labels,'tr_pt':tr_pt, 'edge_dR_labels_4':edge_dR_labels_4,'tr_pt_4':tr_pt_4
        }


    def cell_vals_nodes(self, nodes):
        nTracks        = torch.sum(nodes.mailbox['edge_dR_labels'], dim=1)
        sumTrack_pt    = torch.sum(nodes.mailbox['tr_pt'], dim=1)

        nTracks_4        = torch.sum(nodes.mailbox['edge_dR_labels_4'], dim=1)
        sumTrack_pt_4    = torch.sum(nodes.mailbox['tr_pt_4'], dim=1)

        return{'cell_nTracks':nTracks,'cell_sumTrack_pt':sumTrack_pt,'cell_nTracks_4':nTracks_4,'cell_sumTrack_pt_4':sumTrack_pt_4
        }



    def topo_vals_edge(self, edges):

        nTracks = edges.src['cell_nTracks']
        tracks_pt = edges.src['cell_sumTrack_pt']

        nTracks_4 = edges.src['cell_nTracks_4']
        tracks_pt_4 = edges.src['cell_sumTrack_pt_4']

        eta = edges.src['eta_cell'] * self.transform_var['cell_eta']['std'] + self.transform_var['cell_eta']['mean']
        energy = torch.exp(edges.src['energy_cell'] * self.transform_var['cell_e']['std'] + self.transform_var['cell_e']['mean'])

        wtd_eta = eta * energy
        wtd_phi = edges.src['phi_cell'] * energy

        wtd_layer = edges.src['layer_cell'] * energy

        wtd_eta_l1 = eta * energy * (edges.src['layer_cell']==0)
        wtd_eta_l2 = eta * energy * (edges.src['layer_cell']==1)
        wtd_eta_l3 = eta * energy * (edges.src['layer_cell']==2)

        wtd_eta_l4 = eta * energy * (edges.src['layer_cell']==3)
        wtd_eta_l5 = eta * energy * (edges.src['layer_cell']==4)
        wtd_eta_l6 = eta * energy * (edges.src['layer_cell']==5)

        wtd_phi_l1 = edges.src['phi_cell'] * energy * (edges.src['layer_cell']==0)
        wtd_phi_l2 = edges.src['phi_cell'] * energy * (edges.src['layer_cell']==1)
        wtd_phi_l3 = edges.src['phi_cell'] * energy * (edges.src['layer_cell']==2)

        wtd_phi_l4 = edges.src['phi_cell'] * energy * (edges.src['layer_cell']==3)
        wtd_phi_l5 = edges.src['phi_cell'] * energy * (edges.src['layer_cell']==4)
        wtd_phi_l6 = edges.src['phi_cell'] * energy * (edges.src['layer_cell']==5)

        wtd_ene_l1 = energy * (edges.src['layer_cell']==0)
        wtd_ene_l2 = energy * (edges.src['layer_cell']==1)
        wtd_ene_l3 = energy * (edges.src['layer_cell']==2)
        wtd_ene_l4 = energy * (edges.src['layer_cell']==0)
        wtd_ene_l5 = energy * (edges.src['layer_cell']==1)
        wtd_ene_l6 = energy * (edges.src['layer_cell']==2)


        return { \
            'weighted_eta': wtd_eta, 'weighted_phi': wtd_phi,
            'weighted_layer': wtd_layer, 'energy': energy, 'weighted_eta_l1': wtd_eta_l1, 'weighted_eta_l2': wtd_eta_l2, 
            'weighted_eta_l3': wtd_eta_l3, 'weighted_phi_l1': wtd_phi_l1, 'weighted_phi_l2': wtd_phi_l2, 'weighted_phi_l3': wtd_phi_l3,
            'weighted_eta_l4': wtd_eta_l4,'weighted_eta_l5': wtd_eta_l5,'weighted_eta_l6': wtd_eta_l6,
            'weighted_phi_l4': wtd_phi_l4, 'weighted_phi_l5': wtd_phi_l5, 'weighted_phi_l6': wtd_phi_l6,
            'w_nTracks':nTracks,'tracks_pt':tracks_pt, 'w_nTracks_4':nTracks_4,'tracks_pt_4':tracks_pt_4,
            'wtd_ene_l1':wtd_ene_l1, 'wtd_ene_l2':wtd_ene_l2,'wtd_ene_l3':wtd_ene_l3,'wtd_ene_l4':wtd_ene_l4,
            'wtd_ene_l5':wtd_ene_l5,'wtd_ene_l6':wtd_ene_l6

        }


    def topo_vals_nodes(self, nodes):

        nTracks   = torch.sum(nodes.mailbox['w_nTracks'], dim=1)
        tracks_pt = torch.sum(nodes.mailbox['tracks_pt'], dim=1)



        nTracks_4   = torch.sum(nodes.mailbox['w_nTracks_4'], dim=1)
        tracks_pt_4 = torch.sum(nodes.mailbox['tracks_pt_4'], dim=1)

        nTracks = nTracks.float()
        nTracks_4 = nTracks_4.float()


        energy    = torch.sum(nodes.mailbox['energy'], dim=1)

        eta = torch.sum(nodes.mailbox['weighted_eta'], dim=1) / energy
        phi = torch.sum(nodes.mailbox['weighted_phi'], dim=1) /  energy
        layer     = torch.sum(nodes.mailbox['weighted_layer'], dim=1) / energy

        eta_l1 = torch.sum(nodes.mailbox['weighted_eta_l1'], dim=1) / energy
        eta_l2 = torch.sum(nodes.mailbox['weighted_eta_l2'], dim=1) / energy
        eta_l3 = torch.sum(nodes.mailbox['weighted_eta_l3'], dim=1) / energy
        phi_l1 = torch.sum(nodes.mailbox['weighted_phi_l1'], dim=1) / energy
        phi_l2 = torch.sum(nodes.mailbox['weighted_phi_l2'], dim=1) / energy
        phi_l3 = torch.sum(nodes.mailbox['weighted_phi_l3'], dim=1) / energy

        eta_l4 = torch.sum(nodes.mailbox['weighted_eta_l4'], dim=1) / energy
        eta_l5 = torch.sum(nodes.mailbox['weighted_eta_l5'], dim=1) / energy
        eta_l6 = torch.sum(nodes.mailbox['weighted_eta_l6'], dim=1) / energy
        phi_l4 = torch.sum(nodes.mailbox['weighted_phi_l4'], dim=1) / energy
        phi_l5 = torch.sum(nodes.mailbox['weighted_phi_l5'], dim=1) / energy
        phi_l6 = torch.sum(nodes.mailbox['weighted_phi_l6'], dim=1) / energy


        ene_l1 = torch.sum(nodes.mailbox['wtd_ene_l1'], dim=1) / energy
        ene_l2 = torch.sum(nodes.mailbox['wtd_ene_l2'], dim=1) / energy
        ene_l3 = torch.sum(nodes.mailbox['wtd_ene_l3'], dim=1) / energy
        ene_l4 = torch.sum(nodes.mailbox['wtd_ene_l4'], dim=1) / energy
        ene_l5 = torch.sum(nodes.mailbox['wtd_ene_l5'], dim=1) / energy
        ene_l6 = torch.sum(nodes.mailbox['wtd_ene_l6'], dim=1) / energy


        return {'eta': eta, 'phi': phi, 'layer': layer, 'energy': energy, 'eta_l1': eta_l1,
        'eta_l2': eta_l2,'eta_l3': eta_l3, 'phi_l1': phi_l1, 'phi_l2': phi_l2, 'phi_l3': phi_l3, 'eta_l4': eta_l4,
        'eta_l5': eta_l5,'eta_l6': eta_l6, 'phi_l4': phi_l4, 'phi_l5': phi_l5, 'phi_l6': phi_l6, 
        'nTracks':nTracks, 'tracks_pt':tracks_pt,
        'nTracks_4':nTracks_4, 'tracks_pt_4':tracks_pt_4, 'ene_l1': ene_l1, 'ene_l2': ene_l2,
        'ene_l3': ene_l3,'ene_l4': ene_l4,'ene_l5': ene_l5,'ene_l6': ene_l6
        }
        

    def forward(self, g):
        
        g.nodes['cells'].data['hidden rep']  = self.cell_init_network(g.nodes['cells'].data['node features'])
        g.nodes['tracks'].data['hidden rep'] = self.track_init_network(g.nodes['tracks'].data['node features'])


        #store the summary statistics for cells
        g.update_all(self.cell_message,
                     self.global_update_energy,
                     etype='cells to global')




        self.move_from_cellstracks_to_pre_nodes(g,'hidden rep','hidden rep','hidden rep')
        self.update_global_rep(g)
        
        for block_i in range(self.n_blocks):
    
            for iteration_i in range(self.block_iterations[block_i]):
                g.update_all(fn.copy_src('hidden rep','message'), self.node_update_networks[block_i],etype= 'pre_node_to_pre_node' ) 
                self.update_global_rep(g)

        g.update_all(fn.copy_src('hidden rep','message'), fn.sum("message",'hidden rep'),etype= 'pre_node_to_topo')
        self.move_from_topostracks_to_nodes(g,'hidden rep','hidden rep','hidden rep')
        self.move_from_topostracks_to_nodes(g,'parent target','parent target','parent target')

        g.update_all(self.track_vals_edge, self.cell_vals_nodes, etype='track_to_cell')

        g.update_all(self.topo_vals_edge, self.topo_vals_nodes, etype='cell_to_topo')

        #update topo information with deltaR of closest track, and its pT
        #Ideally we shall store the info if a cell is matched or not to a track


        self.move_from_topostracks_to_nodes(g,'eta','track_eta_layer_0','eta')
        self.move_from_topostracks_to_nodes(g,'phi','track_phi_layer_0','phi')
        self.move_from_topostracks_to_nodes(g,'layer','energy_track','layer')

        
        #energy_track is a dummy vector
        self.move_from_topostracks_to_nodes(g,'phi_l1','energy_track','phi_l1')
        self.move_from_topostracks_to_nodes(g,'phi_l2','energy_track','phi_l2')
        self.move_from_topostracks_to_nodes(g,'phi_l3','energy_track','phi_l3')
        self.move_from_topostracks_to_nodes(g,'eta_l1','energy_track','eta_l1')
        self.move_from_topostracks_to_nodes(g,'eta_l2','energy_track','eta_l2')
        self.move_from_topostracks_to_nodes(g,'eta_l3','energy_track','eta_l3')

        self.move_from_topostracks_to_nodes(g,'phi_l4','energy_track','phi_l4')
        self.move_from_topostracks_to_nodes(g,'phi_l5','energy_track','phi_l5')
        self.move_from_topostracks_to_nodes(g,'phi_l6','energy_track','phi_l6')
        self.move_from_topostracks_to_nodes(g,'eta_l4','energy_track','eta_l4')
        self.move_from_topostracks_to_nodes(g,'eta_l5','energy_track','eta_l5')
        self.move_from_topostracks_to_nodes(g,'eta_l6','energy_track','eta_l6')


        self.move_from_topostracks_to_nodes(g,'ene_l1','energy_track','ene_l1')
        self.move_from_topostracks_to_nodes(g,'ene_l2','energy_track','ene_l2')
        self.move_from_topostracks_to_nodes(g,'ene_l3','energy_track','ene_l3')
        self.move_from_topostracks_to_nodes(g,'ene_l4','energy_track','ene_l4')
        self.move_from_topostracks_to_nodes(g,'ene_l5','energy_track','ene_l5')
        self.move_from_topostracks_to_nodes(g,'ene_l6','energy_track','ene_l6')

        self.move_from_topostracks_to_nodes(g,'energy','energy_track','energy')
        self.move_from_topostracks_to_nodes(g,'nTracks','energy_track','nTracks')
        self.move_from_topostracks_to_nodes(g,'tracks_pt','energy_track','tracks_pt')

        self.move_from_topostracks_to_nodes(g,'nTracks_4','energy_track','nTracks_4')
        self.move_from_topostracks_to_nodes(g,'tracks_pt_4','energy_track','tracks_pt_4')


        #store the summary statistics for tracks
        g.update_all(self.track_message,
                     self.global_update_tracks,
                     etype='tracks to global')

        #store the summary statistics for tracks
        g.update_all(self.topo_message,
                     self.global_update_topos,
                     etype='topos to global')


        return g
