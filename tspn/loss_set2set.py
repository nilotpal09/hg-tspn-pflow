import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

import gc
from copy import deepcopy



class Set2SetLoss(nn.Module):

    def __init__(self, config=None):
    
        super().__init__()
        self.config = config
        self.var_transform = self.config["var transform"]

        self.pT_ranges = self.config["pt ranges"]

        self.class_node_loss = nn.CrossEntropyLoss(reduction='none',weight=torch.tensor([0.5,2.0, 5.0, 2.5]))
        self.class_node_loss_neutral = nn.CrossEntropyLoss(reduction='none',weight=torch.tensor([1.0,1.0]))
        self.regression_loss = nn.MSELoss(reduction='none')
        self.regression_loss_attention = nn.MSELoss(reduction='sum')
        self.class_n_particle = nn.CrossEntropyLoss(reduction='mean')


    def compute_pair_loss(self,edges):

        target_pt_eta_phi    = torch.stack([edges.src['particle_pt'], edges.src['particle_eta'], edges.src['particle_phi']],dim=1)
        predicted_pt_eta_phi = edges.dst['pt_eta_phi_pred']
        pt_mean,  pt_std  = self.var_transform['particle_pt']['mean'], self.var_transform['particle_pt']['std']

        released_energy = edges.src['particle_dep_energy']
        released_energy = (released_energy > 0)
#        pt_weight = (pt_std*edges.src['particle_pt']+pt_mean).view(-1, 1)
        pt_weight =torch.exp(edges.src['particle_pt']*pt_std+pt_mean).view(-1,1)/10000.
        #classification loss for superneutral
        #compute node loss
        pt_weight_flat = torch.flatten(pt_weight)
        loss_class = pt_weight_flat*self.class_node_loss_neutral(edges.dst['class_pred'],
                                                           edges.src['particle_class'])
        loss = torch.sum( pt_weight * self.regression_loss(target_pt_eta_phi,predicted_pt_eta_phi), dim=1)
        #loss = torch.sum( released_energy.unsqueeze(-1) * pt_weight * self.regression_loss(target_pt_eta_phi,predicted_pt_eta_phi), dim=1)
        return {'loss': loss,'loss_class':loss_class}


    def particle_properties_message_func(self,edges):
        return {'m_class': edges.src['particle class']*edges.data['edge label'],
            # 'm_charge': edges.src['charge_class']*edges.data['edge label'],
            # 'm_pos' : edges.src['prod pos']*edges.data['edge label'].unsqueeze(1).repeat(1,3),
            'm_pt' :  edges.src['particle_pt']*edges.data['edge label'],
            'm_eta' : edges.src['particle_eta']*edges.data['edge label'],
            'm_phi' : edges.src['particle_phi']*edges.data['edge label'],
            'm_count' : edges.src['particle idx']*edges.data['edge label'],
            'isolation':edges.dst['isIso']*edges.data['edge label'],
            'm_particle_dep_energy':edges.src['particle_dep_energy']*edges.data['edge label'],
           }


    def particle_properties_node_update(self,nodes):
        p_class = torch.sum(nodes.mailbox['m_class'],dim=1)
        # c_class = torch.sum(nodes.mailbox['m_charge'],dim=1)
        # parent_pos = torch.sum(nodes.mailbox['m_pos'],dim=1)
        parent_pt = torch.sum(nodes.mailbox['m_pt'],dim=1)
        parent_eta = torch.sum(nodes.mailbox['m_eta'],dim=1)
        parent_phi = torch.sum(nodes.mailbox['m_phi'],dim=1)
        # parent_dep_energy = torch.sum(nodes.mailbox['m_parent_dep_energy'],dim=1)
        parent_dep_energy = torch.sum(nodes.mailbox['m_particle_dep_energy'],dim=1)
        isMatch = torch.sum(nodes.mailbox['m_count'],dim=1)
        p_isIso = torch.sum(nodes.mailbox["isolation"],dim=1)
        p_class[p_class==2] = 0
        p_class[(p_class==3) * (p_isIso==1) ] = 1
        p_class[(p_class==3) * (p_isIso==0) ] = 2

        p_class[p_class==4] = 3

        #electrons with no em energies are charged
        em_energy = nodes.data['energy_l_0']+nodes.data['energy_l_1']
        p_class[(em_energy==0) * (p_class==1)] = 0
        p_class[(em_energy==0) * (p_class==2)] = 0

        return {'particle class': p_class.long(),
                'parent_pt' : parent_pt, 'parent_phi' : parent_phi, 'parent_eta' : parent_eta,'parent_idx' : isMatch,'parent_dep_energy':parent_dep_energy}
                

    def ApplyToChildEdgeLabel(self,edges):
        edge_labels = (edges.dst['parent target']==edges.src['particle idx'])

        return {'edge label' : (edge_labels).float() }


    def ApplyToParentEdgeLabel(self,edges):
        edge_labels = (edges.dst['particle idx']==edges.src['parent target'])

        return {'edge label' : (edge_labels).float() }

    def my_loss(self,output, target, edge_label):
        target = target * 2
        loss = 10000*(output - target)**2
        loss = loss[(edge_label!=0.)*(target!=1.0)]
        loss_f = loss
        return loss_f

    def undo_scalling(self, inp):
        pt_mean,  pt_std  = self.var_transform['particle_pt']['mean'], self.var_transform['particle_pt']['std']
        eta_mean, eta_std = self.var_transform['particle_eta']['mean'], self.var_transform['particle_eta']['std']
        phi_mean, phi_std = self.var_transform['particle_phi']['mean'], self.var_transform['particle_phi']['std']

        inp[:,0] = inp[:,0]*pt_std + pt_mean
        inp[:,1] = inp[:,1]*eta_std + eta_mean
        inp[:,2] = inp[:,2]*phi_std + phi_mean

        inp[:,2][inp[:,2]>np.pi]  = inp[:,2][inp[:,2]>np.pi] - 2*np.pi
        inp[:,2][inp[:,2]<-np.pi] = inp[:,2][inp[:,2]<-np.pi] + 2*np.pi

        return inp

    def pt_edge(self, edges):
        pt_mean,  pt_std  = self.var_transform['particle_pt']['mean'], self.var_transform['particle_pt']['std']
        pT =torch.exp(edges.src['particle_pt']*pt_std+pt_mean)
        has_energy = (edges.src['particle_dep_energy'] > 0)
        isLow    =  pT<self.pT_ranges["low_pT"]["max"]
        isMidLow =  (pT>self.pT_ranges["midlow_pT"]["min"]) * (pT<self.pT_ranges["midlow_pT"]["max"])
        isMidHigh=  (pT>self.pT_ranges["midhigh_pT"]["min"]) * (pT<self.pT_ranges["midhigh_pT"]["max"])
        isHigh   =  pT>self.pT_ranges["high_pT"]["min"]
        return{'isLow':isLow, 'isMidLow':isMidLow, 'isMidHigh':isMidHigh, 'isHigh':isHigh}


    def n_for_global_pt(self,nodes):
        n_Low     = torch.sum(1*nodes.mailbox['isLow'], dim=1)
        n_MidLow  = torch.sum(1*nodes.mailbox['isMidLow'], dim=1)
        n_MidHigh = torch.sum(1*nodes.mailbox['isMidHigh'], dim=1)
        n_High    = torch.sum(1*nodes.mailbox['isHigh'], dim=1)

        #n_tot =n_Low + n_MidLow + n_MidHigh + n_High
        return {'n_Low': n_Low, 'n_MidLow' : n_MidLow, 'n_MidHigh' : n_MidHigh, 'n_High' : n_High}#, 'n_tot':n_tot}


    def forward(self, g,hungarian_info, scatter=True, num_particles_in_first_event=None, var_trans=None):
        
        loss = 0; loss_a = 0; loss_p = 0; loss_c = 0

        scatter_dict = {}
        set_size_dict = {}

        # cache
        run_hungarian, hungarian_matches = hungarian_info


        fill_num_particles = False
        if num_particles_in_first_event is not None:
            if bool(num_particles_in_first_event) == False: # is empty
                fill_num_particles = True

        #fdibello
        classes = ["supercharged", "superneutral"]
        #classes = ["supercharged", "neutral", "photon"]
        n_node = g.number_of_nodes("nodes")
        n_particle = g.number_of_nodes("particles")
        


        n_tracks = g.number_of_nodes("tracks")
        loss_class = 0
        for cl in classes:

            if g.number_of_nodes(cl) != 0: 
    
                n_objects_per_event = [n.item() for n in g.batch_num_nodes('pflow '+cl)]

                if cl == "supercharged":
                    #fdibello - get now the track properties
                    g.apply_edges(self.ApplyToChildEdgeLabel,etype='to_pflow_'+cl)
                    g.apply_edges(self.ApplyToParentEdgeLabel,etype='from_pflow_'+cl)
                    g.update_all(self.particle_properties_message_func,
                             self.particle_properties_node_update,
                             etype='to_pflow_'+cl)
        
    
                    # n_objects_per_event = [n.item() for n in g.batch_num_nodes('pflow '+cl)]
        
                    target_pt_eta_phi = torch.cat([
                        g.nodes['pflow '+cl].data['parent_pt'].unsqueeze(1),
                        g.nodes['pflow '+cl].data['parent_eta'].unsqueeze(1),
                        g.nodes['pflow '+cl].data['parent_phi'].unsqueeze(1)
                    ],dim=1)
        
                    
                    phi_std = self.var_transform['particle_phi']['std']
                    phi_mean = self.var_transform['particle_phi']['mean']
                    has_energy = (g.nodes['pflow '+cl].data['parent_dep_energy'].unsqueeze(1) > 0) 

                    pt_loss = self.regression_loss(g.nodes['pflow '+cl].data['pt_eta_phi_pred'][:,0], target_pt_eta_phi[:,0])
                    #pt_loss = has_energy*self.regression_loss(g.nodes['pflow '+cl].data['pt_eta_phi_pred'][:,0], target_pt_eta_phi[:,0])

                    pt_eta_phi_loss = pt_loss #+ eta_loss + phi_loss
        
                    #compute node loss
                    loss_class = self.class_node_loss(g.nodes['pflow '+cl].data['class_pred'],
                                                           g.nodes['pflow '+cl].data['particle class']).mean()
                    loss_class = loss_class*5
        
                    loss_particle = pt_eta_phi_loss.mean()
                    loss += loss_class  + loss_particle 
                    loss_p += loss_particle; loss_c += loss_class;  loss_a += 0
                
                # now hungarian matching
                elif cl == "neutral" or cl == "photon" or cl == "superneutral":
    
                    g.apply_edges(self.compute_pair_loss,etype='to_pflow_'+cl)

                    if run_hungarian:
                        data = g.edges['to_pflow_'+cl].data['loss'].cpu().data.numpy()+0.00000001
                        u = g.all_edges(etype='to_pflow_'+cl)[0].cpu().data.numpy().astype(int)
                        v = g.all_edges(etype='to_pflow_'+cl)[1].cpu().data.numpy().astype(int)
                        m = csr_matrix((data,(u,v)))
                   

                        reco_columns, truth_columns = min_weight_full_bipartite_matching(m)
   
                        # caching the Hungarian matching
                        n_obj_cumsum = [0] + np.cumsum(n_objects_per_event).tolist()
                        for idx, H in enumerate(hungarian_matches):
                            start, stop = n_obj_cumsum[idx], n_obj_cumsum[idx+1]
                            H[cl] = truth_columns[start : stop] - n_obj_cumsum[idx]
                    else:
    
                        truth_columns = []
    
                        offset = 0
                        for idx, H in enumerate(hungarian_matches):
                            truth_columns.extend(H[cl] + offset)
                            offset += len(H[cl])
                        reco_columns, truth_columns = np.arange(len(truth_columns)), np.array(truth_columns)
    
                    # n_objects_per_event = [n.item() for n in g.batch_num_nodes('pflow '+cl)]
    
                    col_offest = np.repeat( np.cumsum([0]+n_objects_per_event[:-1]), n_objects_per_event)
                    row_offset = np.concatenate([[0]]+[[n]*n for n in n_objects_per_event])[:-1]
                    row_offset = np.cumsum(row_offset)
    
                    edge_indices = truth_columns-col_offest+row_offset
                    g.nodes['pflow '+cl].data['loss'] = g.edges['to_pflow_'+cl].data['loss' ][edge_indices]+5*g.edges['to_pflow_'+cl].data['loss_class' ][edge_indices] 

                    loss_particle = dgl.sum_nodes(g,'loss', ntype='pflow '+cl).mean() * g.batch_size / sum(n_objects_per_event)

                    loss += 5*loss_particle
                    loss_p += loss_particle; loss_c += loss_class;  loss_a += 0
    
                    #pred_n_class = g.nodes['global node'].data['predicted_setsize']
    
            if cl != "supercharged":

                # if cl == "neutral":
                #     continue

                g.update_all(self.pt_edge, self.n_for_global_pt, etype=cl+' to global')

                pred_n_class_low     = g.nodes['global node'].data['predicted_setsize_'+cl+'_low']
                pred_n_class_midlow  = g.nodes['global node'].data['predicted_setsize_'+cl+'_midlow']
                pred_n_class_midhigh = g.nodes['global node'].data['predicted_setsize_'+cl+'_midhigh']
                pred_n_class_high    = g.nodes['global node'].data['predicted_setsize_'+cl+'_high']

                
                loss_low     = self.class_n_particle(pred_n_class_low    , g.nodes["global node"].data["n_Low"])
                loss_midlow  = self.class_n_particle(pred_n_class_midlow , g.nodes["global node"].data["n_MidLow"])   
                loss_midhigh = self.class_n_particle(pred_n_class_midhigh, g.nodes["global node"].data["n_MidHigh"])    
                loss_high    = self.class_n_particle(pred_n_class_high   , g.nodes["global node"].data["n_High"]) 
    
                loss += 10 * (loss_low + loss_midlow + loss_midhigh + loss_high)
 

            if scatter == True:
                if cl == 'supercharged':
                    pred = torch.cat([g.nodes['pflow '+cl].data['pt_eta_phi_pred'], torch.argmax(g.nodes['pflow '+cl].data['class_pred'], dim=1).reshape(-1,1)], dim=1)
                    target = torch.stack([
                               g.nodes['pflow '+cl].data['parent_pt'], 
                               g.nodes['pflow '+cl].data['parent_eta'], 
                               g.nodes['pflow '+cl].data['parent_phi'],
                               g.nodes['pflow '+cl].data['particle class']
                           ],dim=1)

                    target_copy, pred_copy = deepcopy(target.cpu().data), deepcopy(pred.cpu().data)
                    target_copy, pred_copy = self.undo_scalling(target_copy), self.undo_scalling(pred_copy)

                    scatter_dict[cl] = [target_copy, pred_copy]
                    del target, pred

                elif cl == 'photon' or cl == 'neutral' or cl == "superneutral":
#                    pred = torch.cat([g.nodes['pflow '+cl].data['pt_eta_phi_pred'], torch.argmax(g.nodes['pflow '+cl].data['class_pred'], dim=1).reshape(-1,1)], dim=1)
                    pred_kine = g.nodes['pflow '+cl].data['pt_eta_phi_pred']
                    pred_class = torch.argmax(g.nodes['pflow '+cl].data['class_pred'],dim=1).reshape(-1,1)
                    pred_kine[truth_columns] = pred_kine[reco_columns]
                    pred_class[truth_columns] = pred_class[reco_columns]
                    pred = torch.cat([pred_kine, pred_class],dim=1)

                    target = torch.stack([
                                g.nodes[cl].data['particle_pt'], 
                                g.nodes[cl].data['particle_eta'], 
                                g.nodes[cl].data['particle_phi'], 
                                g.nodes[cl].data['particle_class']
                            ], dim=1)

                    scatter_dict[cl] = [target, pred]

                    #set_size_dict[cl] = [g.nodes["global node"].data["n_MidHigh"].detach().data, torch.argmax(pred_n_class_midhigh, dim=1).detach().data]

            if fill_num_particles:
                n_obj_copy = deepcopy(n_objects_per_event[0])
                num_particles_in_first_event[cl] = n_obj_copy
                del n_objects_per_event
       
            gc.collect()
        
        if scatter == True:
            return  {
                'loss':loss, 
                'particle_loss': loss_p.detach(),
                'class_loss': loss_c.detach(), 

                'num_loss_low': loss_low.detach(),
                'num_loss_midlow': loss_midlow.detach(),
                'num_loss_midhigh': loss_midhigh.detach(),
                'num_loss_high': loss_high.detach(),

                'scatter_dict': scatter_dict, 
                'set_size_dict': set_size_dict,
                'num_particles_in_first_event': num_particles_in_first_event
            }

        return  {'loss':loss, 'particle loss': loss_p.detach(),'class loss': loss_c.detach()}#,'loss att': loss_a.detach()} #, 'attention loss': loss_a.detach()}


