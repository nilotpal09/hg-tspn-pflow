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


def build_layers(inputsize,outputsize,features,add_batch_norm=False,add_activation=None,dropout=None):
    layers = []
    layers.append(nn.Linear(inputsize,features[0]))
    if dropout is not None:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.ReLU())
    for hidden_i in range(1,len(features)):
        if add_batch_norm:
            layers.append(nn.BatchNorm1d(features[hidden_i-1]))
        layers.append(nn.Linear(features[hidden_i-1],features[hidden_i]))
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(features[-1],outputsize))
    # if dropout is not None: # do we need this??
    #     layers.append(nn.Dropout(p=dropout))
    if add_activation!=None:
        layers.append(add_activation)
    return nn.Sequential(*layers)


#Transformer Set Prediction Network
class TSPN(nn.Module):

    def __init__(self,top_level_config,class_name):
        super().__init__()

        self.regression_loss = nn.MSELoss(reduction='none')

        self.class_name = class_name
        config = top_level_config['output model']
        self.var_transform = top_level_config['var transform']

        if self.class_name == "supercharged": add_dim = 26+12
        else: add_dim = 0

        self.nodeclass_net = build_layers(config['inputsize']+add_dim-100,
                                          outputsize=4, #charged, non-iso e-, isolated e- and muons
                                          features=config['node classifer layers'])
        self.nodeclass_net_neutral = build_layers(config['inputsize']+add_dim-100,
                                          outputsize=2, #charged, non-iso e-, isolated e- and muons
                                          features=config['node classifer layers'])

        self.output_setsize_predictor = build_layers(config['set size predictor input size']+9+12, 25,
                                         features=config['set size predictor layers']) 
        
        self.output_setsize_predictor_low     = build_layers(config['set size predictor input size']+9+12, 25,features=config['set size predictor layers'],dropout=0.5) 
        self.output_setsize_predictor_midlow  = build_layers(config['set size predictor input size']+9+12, 25,features=config['set size predictor layers'],dropout=0.5) 
        self.output_setsize_predictor_midhigh = build_layers(config['set size predictor input size']+9+12, 25,features=config['set size predictor layers'],dropout=0.5) 
        self.output_setsize_predictor_high    = build_layers(config['set size predictor input size']+9+12, 25,features=config['set size predictor layers'],dropout=0.5) 


        if self.class_name == "supercharged": add_dim = 26
        else: add_dim = 0
        self.particle_pt_eta_phi_net = build_layers(config['inputsize']+add_dim-100,
                                          outputsize=3,
                                          features=config['ptetaphi prediction layers'],add_batch_norm=False) #Ask Jonathan about this

        z_shape = config['z size']
        self.z_shape = config['z size']
        self.z_emb = torch.nn.Embedding(num_embeddings=config['set size max'],embedding_dim=z_shape)
        config_emb = top_level_config['embedding model']
        self.slotattn = nn.ModuleList()
        for i in range(3):
            self.slotattn.append(SlotAttention(config_emb['cell hidden size'],self.z_shape,top_level_config,self.class_name))

    def compute_pair_loss(self,edges):

        target_pt_eta_phi    = torch.stack([edges.src['particle_pt'], edges.src['particle_eta'], edges.src['particle_phi']],dim=1)
        predicted_pt_eta_phi = edges.dst['pt_eta_phi_pred']

        loss = torch.sum( self.regression_loss(target_pt_eta_phi,predicted_pt_eta_phi), dim=1)
   
        return {'loss': loss, 'target idx':edges.src['particle idx']}

    def deltaR(self,phi0,eta0,phi1,eta1):

        deta = eta0-eta1
        dphi = phi0-phi1
        dphi[dphi > np.pi] = dphi[dphi > np.pi]-2*np.pi
        dphi[dphi < - np.pi] = dphi[dphi < - np.pi]+2*np.pi

        dR = torch.sqrt( deta**2+dphi**2 )

        return dR

    def particle_message(self,edges):
        energy_layer_all_layer = edges.src['energy']*edges.data['edge_dR_labels']
        cell_layer = edges.src['layer']

        energy_layer_0 = energy_layer_all_layer * (cell_layer==0)
        energy_layer_1 = energy_layer_all_layer * (cell_layer==1)
        energy_layer_2 = energy_layer_all_layer * (cell_layer==2)
        energy_layer_3 = energy_layer_all_layer * (cell_layer==3)
        energy_layer_4 = energy_layer_all_layer * (cell_layer==4)
        energy_layer_5 = energy_layer_all_layer * (cell_layer==5)

        return{'m_energy_layer_0':energy_layer_0,'m_energy_layer_1':energy_layer_1,'m_energy_layer_2':energy_layer_2,
        'm_energy_layer_3':energy_layer_3,'m_energy_layer_4':energy_layer_4,'m_energy_layer_5':energy_layer_5}


    def particle_update(self,nodes):

        energy_l_0 = torch.sum(nodes.mailbox['m_energy_layer_0'],dim=1)
        energy_l_1 = torch.sum(nodes.mailbox['m_energy_layer_1'],dim=1)
        energy_l_2 = torch.sum(nodes.mailbox['m_energy_layer_2'],dim=1)
        energy_l_3 = torch.sum(nodes.mailbox['m_energy_layer_3'],dim=1)
        energy_l_4 = torch.sum(nodes.mailbox['m_energy_layer_4'],dim=1)
        energy_l_5 = torch.sum(nodes.mailbox['m_energy_layer_5'],dim=1)

        senergy_l_0 = torch.std(nodes.mailbox['m_energy_layer_0'],dim=1)
        senergy_l_1 = torch.std(nodes.mailbox['m_energy_layer_1'],dim=1)
        senergy_l_2 = torch.std(nodes.mailbox['m_energy_layer_2'],dim=1)
        senergy_l_3 = torch.std(nodes.mailbox['m_energy_layer_3'],dim=1)
        senergy_l_4 = torch.std(nodes.mailbox['m_energy_layer_4'],dim=1)
        senergy_l_5 = torch.std(nodes.mailbox['m_energy_layer_5'],dim=1)

        return{'energy_l_0':energy_l_0,'energy_l_1':energy_l_1,'energy_l_2':energy_l_2,
        'energy_l_3':energy_l_3,'energy_l_4':energy_l_4,'energy_l_5':energy_l_5,
        'senergy_l_0':senergy_l_0,'senergy_l_1':senergy_l_1,'senergy_l_2':senergy_l_2,
        'senergy_l_3':senergy_l_3,'senergy_l_4':senergy_l_4,'senergy_l_5':senergy_l_5
        }


    def edge_function_dR_reco(self, edges):

        eta_src = edges.src['eta']
        phi_src = edges.src['phi']

        if self.class_name != "supercharged": 
          #hack to se all dR labels == 1 to make work the attention block
          phi_dst_t = edges.dst['node hidden rep'].shape
          phi_dst_t = phi_dst_t[0]
          dR =  torch.ones(phi_dst_t, device=self.device)
          edge_dR_labels = (dR>0)


        elif self.class_name == "supercharged": 

          eta_dst = edges.dst['track_eta_layer_0']
          phi_dst = edges.dst['phi']
          
          dR = self.deltaR(phi_src,eta_src,phi_dst,eta_dst)
          th_R = 0.4
          if(self.class_name == "neutral" or self.class_name == "photon"): th_R = 1.0
          edge_dR_labels = (dR<th_R) #havng this function of pT?
        return  { \
            'dR': dR , 'edge_dR_labels':edge_dR_labels,'eta':eta_src,
            'cell_energy':edges.src['energy'],'cell_layer':edges.src['layer']
        }

    def track_proxm_edgefn(self, edges):

        eta_src = edges.src['eta']
        phi_src = edges.src['phi']

        pt_src  = edges.src['track_pt']

        eta_dst = edges.dst['track_eta_layer_0']
        phi_dst = edges.dst['phi']

        dR = self.deltaR(phi_src,eta_src,phi_dst,eta_dst)

        th_R = 0.1
        edge_dR_labels = (dR<th_R) * (dR!=0)

        track_edges = edges.src['isTrack'] == 1
        track_index_nearest = track_edges*edge_dR_labels
        track_pt_nearest    = pt_src * track_index_nearest
        track_dR_nearest    =  dR * track_index_nearest
        return  { \
            'track_index_nearest': track_index_nearest, 'track_pt_nearest': track_pt_nearest,
            'track_dR_nearest': track_dR_nearest
        }

    def track_proxm_nodefn(self, nodes):

        num_nearest_tracks = torch.sum(nodes.mailbox['track_index_nearest'],dim=1)
        pt_nearest_tracks  = torch.sum(nodes.mailbox['track_pt_nearest'],dim=1)
        dR_nearest_tracks  = torch.min(nodes.mailbox['track_dR_nearest'],dim=1)[0]

        return {
            'num_nearest_tracks': num_nearest_tracks, 'pt_nearest_tracks': pt_nearest_tracks,
            'dR_nearest_tracks': dR_nearest_tracks
        }



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
        #fdibello - nparticles is actually the number of tracks
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
        #decorate the output graph with the track feature, track 0 -> particle 0 and so on

        outputgraphs.nodes['nodes'].data['phi'] = g.nodes['nodes'].data['phi']
        outputgraphs.nodes['nodes'].data['eta'] = g.nodes['nodes'].data['eta']
        outputgraphs.nodes['nodes'].data['energy'] = g.nodes['nodes'].data['energy']
        outputgraphs.nodes['nodes'].data['layer'] = g.nodes['nodes'].data['layer']
        outputgraphs.nodes['nodes'].data['isTrack'] = g.nodes['nodes'].data['isTrack']
        outputgraphs.nodes['nodes'].data['track_pt'] = g.nodes['nodes'].data['track_pt']
        outputgraphs.nodes['nodes'].data['eta_l1'] = g.nodes['nodes'].data['eta_l1']
        outputgraphs.nodes['nodes'].data['eta_l2'] = g.nodes['nodes'].data['eta_l2']
        outputgraphs.nodes['nodes'].data['eta_l3'] = g.nodes['nodes'].data['eta_l3']
        outputgraphs.nodes['nodes'].data['phi_l1'] = g.nodes['nodes'].data['phi_l1']
        outputgraphs.nodes['nodes'].data['phi_l2'] = g.nodes['nodes'].data['phi_l2']
        outputgraphs.nodes['nodes'].data['phi_l3'] = g.nodes['nodes'].data['phi_l3']


        outputgraphs.nodes['nodes'].data['eta_l4'] = g.nodes['nodes'].data['eta_l4']
        outputgraphs.nodes['nodes'].data['eta_l5'] = g.nodes['nodes'].data['eta_l5']
        outputgraphs.nodes['nodes'].data['eta_l6'] = g.nodes['nodes'].data['eta_l6']
        outputgraphs.nodes['nodes'].data['phi_l4'] = g.nodes['nodes'].data['phi_l4']
        outputgraphs.nodes['nodes'].data['phi_l5'] = g.nodes['nodes'].data['phi_l5']
        outputgraphs.nodes['nodes'].data['phi_l6'] = g.nodes['nodes'].data['phi_l6']


        outputgraphs.nodes['nodes'].data['ene_l1'] = g.nodes['nodes'].data['ene_l1']
        outputgraphs.nodes['nodes'].data['ene_l2'] = g.nodes['nodes'].data['ene_l2']
        outputgraphs.nodes['nodes'].data['ene_l3'] = g.nodes['nodes'].data['ene_l3']
        outputgraphs.nodes['nodes'].data['ene_l4'] = g.nodes['nodes'].data['ene_l4']
        outputgraphs.nodes['nodes'].data['ene_l5'] = g.nodes['nodes'].data['ene_l5']
        outputgraphs.nodes['nodes'].data['ene_l6'] = g.nodes['nodes'].data['ene_l6']

        outputgraphs.nodes['nodes'].data['nTracks'] = g.nodes['nodes'].data['nTracks']
        outputgraphs.nodes['nodes'].data['nTracks_4'] = g.nodes['nodes'].data['nTracks_4']
        outputgraphs.nodes['nodes'].data['tracks_pt'] = g.nodes['nodes'].data['tracks_pt']
        outputgraphs.nodes['nodes'].data['tracks_pt_4'] = g.nodes['nodes'].data['tracks_pt_4']

        #outputgraphs.nodes['nodes'].data['phi_cell'] = g.nodes['nodes'].data['phi_cell']
        #outputgraphs.nodes['nodes'].data['eta_cell'] = g.nodes['nodes'].data['eta_cell']
        #outputgraphs.nodes['nodes'].data['energy_cell'] = g.nodes['nodes'].data['energy_cell']
        #outputgraphs.nodes['nodes'].data['layer_cell'] = g.nodes['nodes'].data['layer_cell']
        #outputgraphs.nodes['nodes'].data['eta'] = g.nodes['nodes'].data['eta']
        #outputgraphs.nodes['nodes'].data['cosin_phi'] = g.nodes['nodes'].data['cosin_phi']
        #outputgraphs.nodes['nodes'].data['sinu_phi'] = g.nodes['nodes'].data['sinu_phi']


        if self.class_name == "supercharged": 
          outputgraphs.nodes['particles'].data['node features'] = g.nodes['tracks'].data['node features']
          outputgraphs.nodes['particles'].data['track_eta_layer_0'] = g.nodes['tracks'].data['track_eta_layer_0']
          outputgraphs.nodes['particles'].data['phi'] = g.nodes['tracks'].data['track_phi_layer_0']
          outputgraphs.nodes['particles'].data['isIso'] = g.nodes['tracks'].data['track_isIso']


        indexses = torch.cat([torch.linspace(0,N-1,N,device=g.device).view(N).long() for N in nparticles],dim=0) 
        Z = self.z_emb(indexses)

        #change name
        outputgraphs.nodes['particles'].data['node hidden rep'] = Z

        # create hidden rep for the output objects based on the global rep of the input set and the init for the new objects        
        inputset_global = g.nodes['global node'].data['global rep']
        #here I need to make some sort of association
       
        outputgraphs.nodes['particles'].data['global rep'] = dgl.broadcast_nodes(outputgraphs,inputset_global,ntype='particles')
        
        outputgraphs.apply_edges(self.edge_function_dR_reco, etype='node_to_particle')

        if self.class_name == "supercharged": 
         outputgraphs.update_all(self.track_proxm_edgefn, self.track_proxm_nodefn, etype='node_to_particle')

        #store the summary statistics for each class
        outputgraphs.update_all(self.particle_message,
                     self.particle_update,
                     etype='node_to_particle')



        for i, slotatt in enumerate(self.slotattn):
            slotatt(outputgraphs)
        

        #fdibello - get now the track properties
        #g.apply_edges(ApplyToChildEdgeLabel,etype='particle_to_node')

        if self.class_name == "supercharged": 
            ndata = torch.cat( [outputgraphs.nodes['particles'].data['node hidden rep'],outputgraphs.nodes['particles'].data['node features']],dim=1)
            ndata_class_1 = torch.stack([
            outputgraphs.nodes['particles'].data['energy_l_0'],outputgraphs.nodes['particles'].data['energy_l_1'],outputgraphs.nodes['particles'].data['energy_l_2'],
            outputgraphs.nodes['particles'].data['energy_l_3'],outputgraphs.nodes['particles'].data['energy_l_4'],outputgraphs.nodes['particles'].data['energy_l_5'],
            outputgraphs.nodes['particles'].data['senergy_l_0'],outputgraphs.nodes['particles'].data['senergy_l_1'],outputgraphs.nodes['particles'].data['senergy_l_2'],
            outputgraphs.nodes['particles'].data['senergy_l_3'],outputgraphs.nodes['particles'].data['senergy_l_4'],outputgraphs.nodes['particles'].data['senergy_l_5'],
            ],dim=1)
            ndata_class = torch.cat( [ndata,ndata_class_1],dim=1)
        else: 
            ndata = outputgraphs.nodes['particles'].data['node hidden rep']
            ndata_class = ndata
        outputgraphs.nodes['particles'].data['pt_eta_phi_pred'] = self.particle_pt_eta_phi_net(ndata)
        if self.class_name == "supercharged": outputgraphs.nodes['particles'].data['class_pred'] = self.nodeclass_net(ndata_class)
        else: outputgraphs.nodes['particles'].data['class_pred'] = self.nodeclass_net_neutral(ndata_class)


        if self.class_name == "supercharged": 
            pt, eta, phi = outputgraphs.nodes['particles'].data['pt_eta_phi_pred'].transpose(0,1)
            eta = g.nodes['tracks'].data['track_eta']
            phi = g.nodes['tracks'].data['track_phi']
            outputgraphs.nodes['particles'].data['pt_eta_phi_pred'] =  torch.stack([pt,eta,phi],dim=1) 
        
        #Add the eta, phi and the prodcution position
   
        return outputgraphs


    def forward(self, g):

        self.device = g.device

        #fdibello
        if self.class_name == "supercharged": npart = g.batch_num_nodes('tracks')
        else: npart = g.batch_num_nodes('pflow '+self.class_name)
        
        outputgraphs = self.create_outputgraphs(g,g.batch_num_nodes('nodes'), npart)
        #outputgraphs = self.create_outputgraphs(g,g.batch_num_nodes('nodes'), g.batch_num_nodes('tracks'))
        nprediction = outputgraphs.nodes['particles'].data['pt_eta_phi_pred']
        g.nodes['pflow '+self.class_name].data["pt_eta_phi_pred"] = nprediction
        g.nodes['pflow '+self.class_name].data["class_pred"] = outputgraphs.nodes['particles'].data['class_pred']
        #add the index 
        if self.class_name == "supercharged":
          g.nodes['pflow ' + self.class_name].data['parent target'] = g.nodes['tracks'].data['parent target']
          #g.edges['node_to_pflow_'+self.class_name].data["pred_attention"] = outputgraphs.edges['node_to_particle'].data['attention_weights']
          #g.edges['node_to_pflow_'+self.class_name].data["edge_dR_labels"] = outputgraphs.edges['node_to_particle'].data['edge_dR_labels']


          g.nodes['pflow ' + self.class_name].data['energy_l_0'] = outputgraphs.nodes['particles'].data['energy_l_0']
          g.nodes['pflow ' + self.class_name].data['energy_l_1'] = outputgraphs.nodes['particles'].data['energy_l_1']
          g.nodes['pflow ' + self.class_name].data['energy_l_2'] = outputgraphs.nodes['particles'].data['energy_l_2']
          g.nodes['pflow ' + self.class_name].data['energy_l_3'] = outputgraphs.nodes['particles'].data['energy_l_3']
          g.nodes['pflow ' + self.class_name].data['energy_l_4'] = outputgraphs.nodes['particles'].data['energy_l_4']
          g.nodes['pflow ' + self.class_name].data['energy_l_5'] = outputgraphs.nodes['particles'].data['energy_l_5']


          g.nodes['pflow ' + self.class_name].data['senergy_l_0'] = outputgraphs.nodes['particles'].data['senergy_l_0']
          g.nodes['pflow ' + self.class_name].data['senergy_l_1'] = outputgraphs.nodes['particles'].data['senergy_l_1']
          g.nodes['pflow ' + self.class_name].data['senergy_l_2'] = outputgraphs.nodes['particles'].data['senergy_l_2']
          g.nodes['pflow ' + self.class_name].data['senergy_l_3'] = outputgraphs.nodes['particles'].data['senergy_l_3']
          g.nodes['pflow ' + self.class_name].data['senergy_l_4'] = outputgraphs.nodes['particles'].data['senergy_l_4']
          g.nodes['pflow ' + self.class_name].data['senergy_l_5'] = outputgraphs.nodes['particles'].data['senergy_l_5']
          g.nodes['pflow ' + self.class_name].data['isIso'] = outputgraphs.nodes['particles'].data['isIso']

        # predicted_setsizes coming soon!
        cells_global = dgl.mean_nodes(g,'hidden rep',ntype='topos') 
        tracks_global = dgl.mean_nodes(g,'hidden rep',ntype='tracks')
        #skip connections
        size = g.nodes['global node'].data['number_tracks'].shape[0]
        number_track  = g.nodes['global node'].data['number_tracks'].view(size,1)
        sum_pt        = g.nodes['global node'].data['tracks_pt'].view(size,1)
        calo_l0       = g.nodes['global node'].data['energy_l_0'].view(size,1)
        calo_l1       = g.nodes['global node'].data['energy_l_1'].view(size,1)
        calo_l2       = g.nodes['global node'].data['energy_l_2'].view(size,1)
        calo_l3       = g.nodes['global node'].data['energy_l_3'].view(size,1)
        calo_l4       = g.nodes['global node'].data['energy_l_4'].view(size,1)
        calo_l5       = g.nodes['global node'].data['energy_l_5'].view(size,1)
        number_topo   = g.nodes['global node'].data['number_topos'].view(size,1)
        number_topo_tracks_1   = g.nodes['global node'].data['number_topos_track_1'].view(size,1)
        number_topo_notracks_1   = g.nodes['global node'].data['number_topos_notrack_1'].view(size,1)
        number_topo_tracks_2   = g.nodes['global node'].data['number_topos_track_2'].view(size,1)
        number_topo_notracks_2   = g.nodes['global node'].data['number_topos_notrack_2'].view(size,1)
        number_topo_tracks_3   = g.nodes['global node'].data['number_topos_track_3'].view(size,1)
        number_topo_notracks_3   = g.nodes['global node'].data['number_topos_notrack_3'].view(size,1)
        number_topo_tracks_4   = g.nodes['global node'].data['number_topos_track_4'].view(size,1)
        number_topo_notracks_4   = g.nodes['global node'].data['number_topos_notrack_4'].view(size,1)
        number_topo_tracks_5   = g.nodes['global node'].data['number_topos_track_5'].view(size,1)
        number_topo_notracks_5   = g.nodes['global node'].data['number_topos_notrack_5'].view(size,1)
        number_topo_tracks_6   = g.nodes['global node'].data['number_topos_track_6'].view(size,1)
        number_topo_notracks_6   = g.nodes['global node'].data['number_topos_notrack_6'].view(size,1)
#        print("FDIBELLO ECAL 1 ---->",number_topo,"withTracks",number_topo_tracks_1,"noTracks",number_topo_notracks_1)
#        print("FDIBELLO ECAL 2 ---->",number_topo,"withTracks",number_topo_tracks_2,"noTracks",number_topo_notracks_2)
#        print("FDIBELLO ECAL 3 ---->",number_topo,"withTracks",number_topo_tracks_3,"noTracks",number_topo_notracks_3)
#        print("FDIBELLO HCAL 1---->",number_topo,"withTracks",number_topo_tracks_4,"noTracks",number_topo_notracks_4)
#        print("FDIBELLO HCAL 2---->",number_topo,"withTracks",number_topo_tracks_5,"noTracks",number_topo_notracks_5)
#        print("FDIBELLO HCAL 3---->",number_topo,"withTracks",number_topo_tracks_6,"noTracks",number_topo_notracks_6)
#        print("number_track---->",number_track)
#        if(self.class_name=="superneutral"):print("npart ",npart,self.class_name,g.nodes[self.class_name].data['particle_class'])
#        else: print("npart ",npart,self.class_name)
        all_global = torch.cat([cells_global,tracks_global,number_track,sum_pt,
                                number_topo,number_topo_tracks_1,number_topo_notracks_1,
                                number_topo_tracks_2,number_topo_notracks_2,
                                number_topo_tracks_3,number_topo_notracks_3,
                                number_topo_tracks_4,number_topo_notracks_4,
                                number_topo_tracks_5,number_topo_notracks_5,
                                number_topo_tracks_6,number_topo_notracks_6,
                                calo_l0,calo_l1,calo_l2,calo_l3,calo_l4,calo_l5],dim=1)

        #predicted_setsizes = self.output_setsize_predictor(all_global)

        if self.class_name != "supercharged":
            predicted_setsizes_low     = self.output_setsize_predictor_low(all_global)
            predicted_setsizes_midlow  = self.output_setsize_predictor_midlow(all_global)
            predicted_setsizes_midhigh = self.output_setsize_predictor_midhigh(all_global)
            predicted_setsizes_high    = self.output_setsize_predictor_high(all_global)
    
            #g.nodes['global node'].data['predicted_setsize'] = predicted_setsizes
            g.nodes['global node'].data['predicted_setsize_'+self.class_name+'_low'] = predicted_setsizes_low
            g.nodes['global node'].data['predicted_setsize_'+self.class_name+'_midlow'] = predicted_setsizes_midlow
            g.nodes['global node'].data['predicted_setsize_'+self.class_name+'_midhigh'] = predicted_setsizes_midhigh
            g.nodes['global node'].data['predicted_setsize_'+self.class_name+'_high'] = predicted_setsizes_high

        return g
    
    def create_particles(self,g,do_particle_pred):
        print("Creating particles "+self.class_name)

        # predicted_setsizes coming soon!
        cells_global = dgl.mean_nodes(g,'hidden rep',ntype='topos') 
        tracks_global = dgl.mean_nodes(g,'hidden rep',ntype='tracks')
        #skip connections
        size = g.nodes['global node'].data['number_tracks'].shape[0]
        number_track  = g.nodes['global node'].data['number_tracks'].view(size,1)
        sum_pt        = g.nodes['global node'].data['tracks_pt'].view(size,1)
        calo_l0       = g.nodes['global node'].data['energy_l_0'].view(size,1)
        calo_l1       = g.nodes['global node'].data['energy_l_1'].view(size,1)
        calo_l2       = g.nodes['global node'].data['energy_l_2'].view(size,1)
        calo_l3       = g.nodes['global node'].data['energy_l_3'].view(size,1)
        calo_l4       = g.nodes['global node'].data['energy_l_4'].view(size,1)
        calo_l5       = g.nodes['global node'].data['energy_l_5'].view(size,1)
        number_topo   = g.nodes['global node'].data['number_topos'].view(size,1)
        number_topo_tracks_1   = g.nodes['global node'].data['number_topos_track_1'].view(size,1)
        number_topo_notracks_1   = g.nodes['global node'].data['number_topos_notrack_1'].view(size,1)
        number_topo_tracks_2   = g.nodes['global node'].data['number_topos_track_2'].view(size,1)
        number_topo_notracks_2   = g.nodes['global node'].data['number_topos_notrack_2'].view(size,1)
        number_topo_tracks_3   = g.nodes['global node'].data['number_topos_track_3'].view(size,1)
        number_topo_notracks_3   = g.nodes['global node'].data['number_topos_notrack_3'].view(size,1)
        number_topo_tracks_4   = g.nodes['global node'].data['number_topos_track_4'].view(size,1)
        number_topo_notracks_4   = g.nodes['global node'].data['number_topos_notrack_4'].view(size,1)
        number_topo_tracks_5   = g.nodes['global node'].data['number_topos_track_5'].view(size,1)
        number_topo_notracks_5   = g.nodes['global node'].data['number_topos_notrack_5'].view(size,1)
        number_topo_tracks_6   = g.nodes['global node'].data['number_topos_track_6'].view(size,1)
        number_topo_notracks_6   = g.nodes['global node'].data['number_topos_notrack_6'].view(size,1)

        all_global = torch.cat([cells_global,tracks_global,number_track,sum_pt,
                                number_topo,number_topo_tracks_1,number_topo_notracks_1,
                                number_topo_tracks_2,number_topo_notracks_2,
                                number_topo_tracks_3,number_topo_notracks_3,
                                number_topo_tracks_4,number_topo_notracks_4,
                                number_topo_tracks_5,number_topo_notracks_5,
                                number_topo_tracks_6,number_topo_notracks_6,
                                calo_l0,calo_l1,calo_l2,calo_l3,calo_l4,calo_l5],dim=1)


        #predicted_setsizes = self.output_setsize_predictor(all_global)

        #predicted_n = torch.torch.multinomial(torch.softmax(predicted_setsizes,dim=1),1).view(-1)

        predicted_setsizes_low     = self.output_setsize_predictor_low(all_global)
        predicted_setsizes_midlow  = self.output_setsize_predictor_midlow(all_global)
        predicted_setsizes_midhigh = self.output_setsize_predictor_midhigh(all_global)
        predicted_setsizes_high    = self.output_setsize_predictor_high(all_global)

        predicted_low     = torch.torch.multinomial(torch.softmax(predicted_setsizes_low    ,dim=1),1).view(-1)
        predicted_midlow  = torch.torch.multinomial(torch.softmax(predicted_setsizes_midlow ,dim=1),1).view(-1)
        predicted_midhigh = torch.torch.multinomial(torch.softmax(predicted_setsizes_midhigh,dim=1),1).view(-1)
        predicted_high    = torch.torch.multinomial(torch.softmax(predicted_setsizes_high   ,dim=1),1).view(-1)

        predicted_n = predicted_low+predicted_midlow+predicted_midhigh+predicted_high
        #print("low", np.round(torch.softmax(predicted_setsizes_low    ,dim=1).numpy(),3))
        #print("low", np.round(torch.torch.multinomial(torch.softmax(predicted_setsizes_low    ,dim=1), 1).numpy(),3))

        
        

        ntot = g.num_nodes('pflow '+self.class_name)

        #decorate output with the best guess we have on the hungarian matching
        if self.class_name == "supercharged":
            g.nodes['pflow '+self.class_name].data["target idx"] = torch.tensor([0]).repeat(ntot)
                
        

        elif self.class_name == "neutral" or self.class_name == "photon" or self.class_name == "superneutral"  :

            g.nodes["global node"].data["n_"+self.class_name+"_low"] = predicted_low    
            g.nodes["global node"].data["n_"+self.class_name+"_midlow"] = predicted_midlow 
            g.nodes["global node"].data["n_"+self.class_name+"_midhigh"] = predicted_midhigh
            g.nodes["global node"].data["n_"+self.class_name+"_high"] = predicted_high   


#            g.apply_edges(self.compute_pair_loss,etype='to_pflow_'+self.class_name)
#            data = g.edges['to_pflow_'+self.class_name].data['loss'].cpu().data.numpy()+0.00000001
#            u = g.all_edges(etype='to_pflow_'+self.class_name)[0].cpu().data.numpy().astype(int)
#            v = g.all_edges(etype='to_pflow_'+self.class_name)[1].cpu().data.numpy().astype(int)
#            m = csr_matrix((data,(u,v)))
            
#            n_objects_per_event = [n.item() for n in g.batch_num_nodes('pflow '+self.class_name)]
#            reco_columns, truth_columns = min_weight_full_bipartite_matching(m)
#            col_offest = np.repeat( np.cumsum([0]+n_objects_per_event[:-1]), n_objects_per_event)
#            row_offset = np.concatenate([[0]]+[[n]*n for n in n_objects_per_event])[:-1]
#            row_offset = np.cumsum(row_offset)
#            edge_indices = truth_columns-col_offest+row_offset
#            g.nodes['pflow '+self.class_name].data['loss'] = g.edges['to_pflow_'+self.class_name].data['loss' ][edge_indices] 
#            g.nodes['pflow '+self.class_name].data['target idx'] = g.edges['to_pflow_'+self.class_name].data['target idx'][edge_indices] 


        #fdibello
        if self.class_name == "supercharged": npart = g.batch_num_nodes('tracks')
        #else: npart = g.batch_num_nodes('pflow '+self.class_name) 
        #fdibello else: npart = g.batch_num_nodes('pflow '+self.class_name) 
        elif do_particle_pred == 0: npart = g.batch_num_nodes('pflow '+self.class_name) 
        elif do_particle_pred == 1: npart = g.nodes['global node'].data["n_"+self.class_name+"_tot"] 


        predicted_n = g.number_of_nodes(self.class_name)
        outputgraphs = self.create_outputgraphs(g, g.batch_num_nodes('nodes'), npart)
        #ndata = outputgraphs.ndata['node hidden rep']

        # particle_charge = torch.argmax(self.charge_class_net(ndata),dim=1)
        # particle_e = self.particle_energy_net(ndata).view(-1)
        # particle_pos = self.particle_position_net(ndata)
        # particle_3mom = self.particle_3momentum(ndata)

        # return (particle_e,particle_3mom, particle_pos, particle_class,particle_charge),predicted_setsizes

        #need to define new node features 

        if do_particle_pred == 1:
            if self.class_name == "supercharged": g.nodes['pflow ' + self.class_name].data['pt_eta_phi_pred'] = outputgraphs.nodes['particles'].data['pt_eta_phi_pred']
        
        if self.class_name == "supercharged": 
            g.nodes['pflow '+self.class_name].data["class_pred"] = outputgraphs.nodes['particles'].data['class_pred']

            #print(g.nodes['pflow '+self.class_name].data["class_pred"])
            #print(torch.tensor([1]).repeat(ntot))


    def create_particles_pred(self,g,do_particle_pred):


        pred_graph = []
        nparticles = g.nodes['global node'].data["n_"+self.class_name+"_tot"]
        #nparticles = g.batch_num_nodes(self.class_name)
        ntruth_particles = g.batch_num_nodes(self.class_name)
        #fdibello - nparticles is actually the number of tracks
        for N,Ntruth in zip(nparticles,ntruth_particles):

            #need to generalize to truth particles for the hungarian

            Ntruth = Ntruth.cpu().data;  N = N.cpu().data
            num_nodes_dict = {
                'particles_'+self.class_name : N,
                'truth' : Ntruth
            }

            estart = torch.repeat_interleave( torch.arange(Ntruth),N)
            eend   = torch.arange(N).repeat(Ntruth)

            data_dict = {
                ('truth','truth_to_particle','particles_'+self.class_name) : (estart,eend)
            }

            pred_graph.append( dgl.heterograph(data_dict, num_nodes_dict, device=g.device) )


        pred_graph = dgl.batch(pred_graph)


        outputgraphs = self.create_outputgraphs(g, g.batch_num_nodes('nodes'), nparticles)


        pred_graph.nodes['particles_'+self.class_name].data['pt_eta_phi_pred'] = outputgraphs.nodes['particles'].data['pt_eta_phi_pred']
        if self.class_name == "photon":
            pred_graph.nodes['particles_'+self.class_name].data["class_pred"] = torch.tensor([0]).repeat(torch.sum(nparticles))
        elif self.class_name == "neutral":
            pred_graph.nodes['particles_'+self.class_name].data["class_pred"] = torch.tensor([1]).repeat(torch.sum(nparticles))
        elif self.class_name == "superneutral":
            pred_graph.nodes['particles_'+self.class_name].data["class_pred"] = outputgraphs.nodes['particles'].data['class_pred'] 
            print("=======>",pred_graph.nodes['particles_'+self.class_name].data["class_pred"])



        #print("====> fdibello ",nparticles,ntruth_particles)
        #if torch.sum(nparticles) != 0 and torch.sum(ntruth_particles)   != 0:
        #            #hungarian matching
        #    pred_graph.nodes['truth'].data['particle_pt']  = g.nodes[self.class_name].data["particle_pt"]
        #    pred_graph.nodes['truth'].data['particle_eta']  = g.nodes[self.class_name].data["particle_eta"]
        #    pred_graph.nodes['truth'].data['particle_phi']  = g.nodes[self.class_name].data["particle_phi"]
        #    pred_graph.nodes['truth'].data['particle idx']  = g.nodes[self.class_name].data["particle idx"]

        #    pred_graph.apply_edges(self.compute_pair_loss,etype='truth_to_particle')
        #    data = pred_graph.edges['truth_to_particle'].data['loss'].cpu().data.numpy()+0.00000001
        #    u = pred_graph.all_edges(etype='truth_to_particle')[0].cpu().data.numpy().astype(int)
        #    v = pred_graph.all_edges(etype='truth_to_particle')[1].cpu().data.numpy().astype(int)
        #    m = csr_matrix((data,(u,v)))
        #   
        #    n_objects_per_event = [n.item() for n in pred_graph.batch_num_nodes('particles_'+self.class_name)]
        #    reco_columns, truth_columns = min_weight_full_bipartite_matching(m)
        #    col_offest = np.repeat( np.cumsum([0]+n_objects_per_event[:-1]), n_objects_per_event)
        #    row_offset = np.concatenate([[0]]+[[n]*n for n in n_objects_per_event])[:-1]
        #    row_offset = np.cumsum(row_offset)
        #    edge_indices = truth_columns-col_offest+row_offset
        #    pred_graph.nodes['particles_'+self.class_name].data['loss'] = pred_graph.edges['truth_to_particle'].data['loss' ][edge_indices] 
        #    pred_graph.nodes['particles_'+self.class_name].data['target idx'] = pred_graph.edges['truth_to_particle'].data['target idx'][edge_indices]

        #else: 
        pred_graph.nodes['particles_'+self.class_name].data['loss']  = torch.tensor([0]).repeat(torch.sum(nparticles))
        pred_graph.nodes['particles_'+self.class_name].data['target idx'] = torch.tensor([-1]).repeat(torch.sum(nparticles))


        return pred_graph
    
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

    
    def infer(self,g,do_particle_pred):
        
        self.create_particles(g,do_particle_pred)
        if do_particle_pred == 1 and (self.class_name == "superneutral" or self.class_name == "neutral" or self.class_name == "photon"):
            pred_graph = self.create_particles_pred(g,do_particle_pred)
            return pred_graph

