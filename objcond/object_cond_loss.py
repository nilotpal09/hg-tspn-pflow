import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy  as np
from copy import deepcopy


def particle_properties_message_func(edges):
    return {'m_class': edges.src['particle class']*edges.data['edge label'],
            'm_charge': edges.src['charge_class']*edges.data['edge label'],
            'm_pos' : edges.src['prod pos']*edges.data['edge label'].unsqueeze(1).repeat(1,3),
            'm_pt' :  edges.src['particle_pt']*edges.data['edge label'],
#            'm_phi' : edges.src['particle_phi']*edges.data['edge label'],
            'm_eta' : edges.src['particle_eta']*edges.data['edge label'],
            'm_xhat' : edges.src['particle_xhat']*edges.data['edge label'],
            'm_yhat' : edges.src['particle_yhat']*edges.data['edge label'],
            'm_N_nodes': edges.src['N nodes']*edges.data['edge label'],
            'm_has_Track': edges.src['has_track']*edges.data['edge label']
           }

def particle_properties_node_update(nodes):
    p_class = torch.sum(nodes.mailbox['m_class'],dim=1)
    c_class = torch.sum(nodes.mailbox['m_charge'],dim=1)
    parent_pos = torch.sum(nodes.mailbox['m_pos'],dim=1)
    parent_pt = torch.sum(nodes.mailbox['m_pt'],dim=1)
#    parent_phi = torch.sum(nodes.mailbox['m_phi'],dim=1)
    parent_eta = torch.sum(nodes.mailbox['m_eta'],dim=1)
    parent_xhat = torch.sum(nodes.mailbox['m_xhat'],dim=1)
    parent_yhat = torch.sum(nodes.mailbox['m_yhat'],dim=1)
    parent_N_nodes = torch.sum(nodes.mailbox['m_N_nodes'],dim=1)
    parent_has_track = torch.sum(nodes.mailbox['m_has_Track'],dim=1)

    return {'particle class': p_class.long(), 'charge_class' : c_class.long(),
                'parent_pos' : parent_pos, 'parent_pt' : parent_pt, 'parent_xhat' : parent_xhat,'parent_yhat' : parent_yhat, 'parent_eta' : parent_eta, 'N nodes': parent_N_nodes, 'has_track': parent_has_track}


def ApplyToChildEdgeLabel(edges):
    edge_labels = (edges.dst['parent target']==edges.src['particle idx'])
    
    return {'edge label' : (edge_labels).float() }

def ApplyToParentEdgeLabel(edges):
    edge_labels = (edges.dst['particle idx']==edges.src['parent target'])
    
    return {'edge label' : (edge_labels).float() }


def ChildToParentEF(edges):
    
    qs = edges.src['q']*edges.data['edge label']
    beta = edges.src['beta']*edges.data['edge label']
    node_x = edges.src['x']*edges.data['edge label'].unsqueeze(1)
    node_idx = edges.src['idx']*edges.data['edge label']
    parent_class = edges.dst['particle class']*edges.data['edge label']
    zeta = edges.src['zeta']*edges.data['edge label']
    isTrack = edges.src['isTrack']*edges.data['edge label']

    return {'qs' : qs, 'betas':beta, 'child_x' : node_x, 'child_idx': node_idx, 'parent_class': parent_class, 'isTrack': isTrack, 'zetas':zeta}

def ChildToParentNF_maxbeta(nodes):

    N = len(nodes)

    beta_mail = torch.where(nodes.mailbox['parent_class']<2, nodes.mailbox['betas'], nodes.mailbox['isTrack']*nodes.mailbox['betas'])
    all_betas = torch.cat([ nodes.data['max beta'].unsqueeze(1),beta_mail],dim=1 )

    max_beta,where = torch.max(all_betas,dim=1) #largest predicted beta per particle

    all_qs = torch.cat([ nodes.data['max q'].unsqueeze(1),nodes.mailbox['qs']],dim=1 )
    max_q = all_qs[torch.arange(N),where]
    
    all_xs = torch.cat([ nodes.data['max x'].unsqueeze(1),nodes.mailbox['child_x']],dim=1 )
    max_x = all_xs[torch.arange(N),where]
    
    all_idxs = torch.cat([ nodes.data['max idx'].unsqueeze(1),nodes.mailbox['child_idx']],dim=1 )
    max_idx = all_idxs[torch.arange(N),where]
    
    return {'max q' : max_q, 'max x' : max_x, 'max beta': max_beta, 'max idx': max_idx}

def ChildToParentNF_maxzeta(nodes):

    N = len(nodes)

    all_zetas = torch.cat([ nodes.data['max zeta'].unsqueeze(1),nodes.mailbox['zetas']],dim=1 )

    max_zeta,where = torch.max(all_zetas,dim=1) #largest signal-to-noise ratio (zeta) per particle

    beta_mail = torch.where(nodes.mailbox['parent_class']<2, nodes.mailbox['betas'], nodes.mailbox['isTrack']*nodes.mailbox['betas'])
    all_betas = torch.cat([ nodes.data['max beta'].unsqueeze(1),beta_mail],dim=1 )
    max_beta = all_betas[torch.arange(N),where]

    all_qs = torch.cat([ nodes.data['max q'].unsqueeze(1),nodes.mailbox['qs']],dim=1 )
    max_q = all_qs[torch.arange(N),where]
    
    all_xs = torch.cat([ nodes.data['max x'].unsqueeze(1),nodes.mailbox['child_x']],dim=1 )
    max_x = all_xs[torch.arange(N),where]
    
    all_idxs = torch.cat([ nodes.data['max idx'].unsqueeze(1),nodes.mailbox['child_idx']],dim=1 )
    max_idx = all_idxs[torch.arange(N),where]
    
    return {'max q' : max_q, 'max x' : max_x, 'max beta': max_beta, 'max idx': max_idx, 'max zeta': max_zeta}

def ParentToChildEF(edges):
        
    distance = torch.norm(edges.src['max x']-edges.dst['x'],dim=1)
    distancesquard = distance*distance

    max_q = edges.src['max q']
    
    one_minus_dist = (2.0-distance)
    one_minus_dist = torch.clip(one_minus_dist,min=0.0)
    attractive_potential = 3.0*edges.data['edge label']*(max_q)*distancesquard
    repulsive_potential = 1.0*(1-edges.data['edge label'])*one_minus_dist*max_q
       
    return {'V_attract' : attractive_potential, 'V_repulse' : repulsive_potential}

def ParentToChildNF(nodes):
    
    total_attract = torch.sum( nodes.mailbox['V_attract'],dim=1)
    total_repulse = torch.sum( nodes.mailbox['V_repulse'],dim=1)
    
    return {'node Lv' : nodes.data['q']*(total_attract+total_repulse)} 


def move_from_cellstracks_to_nodes(g,cell_info,track_info,target_name):


    g.update_all(fn.copy_src(cell_info,'m'),fn.sum('m',target_name),etype='cell_to_node')
    cell_only_data = g.nodes['nodes'].data[target_name]
    g.update_all(fn.copy_src(track_info,'m'),fn.sum('m',target_name),etype='track_to_node')
    g.nodes['nodes'].data[target_name] = g.nodes['nodes'].data[target_name]+cell_only_data



class ObjectCondenstationLoss(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.var_transform = self.config["var transform"]

        self.cell_class_loss = nn.CrossEntropyLoss(reduction='none',weight=torch.tensor(self.config["cell class weights"]))
        self.track_class_loss = nn.CrossEntropyLoss(reduction='none',weight=torch.tensor(self.config["track class weights"]))
        self.regression_loss = nn.MSELoss(reduction='none')
        self.class_CP_loss   = nn.CrossEntropyLoss(reduction='none',weight=torch.tensor([0.1,10.0]))

        self.qmin = config['qmin']
        self.s_c = config['s_c']
        self.s_b = config['s_b']

        self.use_BCE_beta_loss = config['use_BCE_beta_loss']
        self.use_zeta_CPs      = config['use_zeta_CPs']

    def undo_scaling(self, inp):
        pt_mean,  pt_std  = self.var_transform['particle_pt']['mean'], self.var_transform['particle_pt']['std']
        eta_mean, eta_std = self.var_transform['particle_eta']['mean'], self.var_transform['particle_eta']['std']
        phi_mean, phi_std = self.var_transform['particle_phi']['mean'], self.var_transform['particle_phi']['std']

        inp[:,0] = inp[:,0]*pt_std + pt_mean
        inp[:,1] = inp[:,1]*eta_std + eta_mean
        inp[:,2] = inp[:,2]*phi_std + phi_mean

        inp[:,2][inp[:,2]>np.pi]  = inp[:,2][inp[:,2]>np.pi] - 2*np.pi
        inp[:,2][inp[:,2]<-np.pi] = inp[:,2][inp[:,2]<-np.pi] + 2*np.pi


        return inp

    def forward(self, g, scatter=True, num_particles_in_first_event=None):

        g.nodes['particles'].data['N nodes'] = torch.zeros(g.num_nodes('particles'),device=g.device)
        g.nodes['nodes'].data['N nodes'] = torch.zeros(g.num_nodes('nodes'),device=g.device)

        g.nodes['particles'].data['has_track'] = torch.zeros(g.num_nodes('particles'),device=g.device)
        g.nodes['nodes'].data['has_track'] = torch.zeros(g.num_nodes('nodes'),device=g.device)

        self.bkg_classes = self.config['bkg classes'] if 'bkg classes' in self.config else [2,3,4]
        scatter_dict = {}

        fill_num_particles = False
        if num_particles_in_first_event is not None:
            if bool(num_particles_in_first_event) == False: # is empty
                fill_num_particles = True

        n_objects_per_event = [n.item() for n in g.batch_num_nodes('particles')]

        move_from_cellstracks_to_nodes(g,'parent target','parent target','parent target')

        g.apply_edges(ApplyToChildEdgeLabel,etype='particle_to_node')
        g.apply_edges(ApplyToParentEdgeLabel,etype='node_to_particle')

        g.update_all(fn.copy_edge('edge label','belongs'),fn.sum('belongs','N nodes'),etype='node_to_particle')
        g.nodes['nodes'].data['isTrack'] = g.nodes['nodes'].data['isTrack'].float()
        g.update_all(fn.u_mul_e('isTrack','edge label','itsatrack'),fn.sum('itsatrack','has_track'),etype='node_to_particle')
        g.nodes['nodes'].data['isTrack'] = g.nodes['nodes'].data['isTrack'].int()
        
        g.update_all(particle_properties_message_func,
                     particle_properties_node_update,
                     etype='particle_to_node')
        
      
        
        # label pileup/fake nodes as background nodes
        n = (g.nodes['nodes'].data['parent target'] < 0).float()
        g.nodes['nodes'].data['n'] = n

        # also ignore cells which have less than 2 incoming edges
        where_island = torch.where(torch.logical_and(g.nodes['nodes'].data['N edges start'] < 2,g.nodes['nodes'].data['isTrack']==0),True,False)
        g.nodes['nodes'].data['n'][where_island] = 1

        #protect against 0 beta
        eps = 0.0000001
        clamped_beta = torch.clamp( g.nodes['nodes'].data['beta'], min=eps,max=1.0-eps)
    
        #replace trackless charged particles with their neutral counterparts (clean the target)
        tracklessCharged = torch.logical_and(g.nodes['nodes'].data['particle class']==2,g.nodes['nodes'].data['has_track']==0)
        g.nodes['nodes'].data['particle class'][tracklessCharged] = 1

        tracklessElectron = torch.logical_and(g.nodes['nodes'].data['particle class']==3,g.nodes['nodes'].data['has_track']==0)
        g.nodes['nodes'].data['particle class'][tracklessElectron] = 0

        #node-level class predictions
        cells_class_upto_2   = torch.where(g.nodes['nodes'].data['particle class']>2,2*torch.ones_like(g.nodes['nodes'].data['particle class']),g.nodes['nodes'].data['particle class'])
        tracks_class_minus_2 = torch.where(g.nodes['nodes'].data['particle class']<2,g.nodes['nodes'].data['particle class'],g.nodes['nodes'].data['particle class']- 2*torch.ones_like(g.nodes['nodes'].data['particle class']))

        #class loss
        cell_class_loss      = self.cell_class_loss(g.nodes['nodes'].data['cell class pred'],cells_class_upto_2)
        track_class_loss     = self.track_class_loss(g.nodes['nodes'].data['track class pred'],tracks_class_minus_2)
        nodes_class_loss     = torch.where(g.nodes['nodes'].data['isTrack']==1,track_class_loss,cell_class_loss)

        #target kinematics
        target_pt_eta_xhat_yhat = torch.cat([g.nodes['nodes'].data['parent_pt'].unsqueeze(1)
            ,g.nodes['nodes'].data['parent_eta'].unsqueeze(1),
            g.nodes['nodes'].data['parent_xhat'].unsqueeze(1),g.nodes['nodes'].data['parent_yhat'].unsqueeze(1) ],dim=1)

        #kinematics loss
        pt_eta_xhat_yhat_loss = torch.sum( self.regression_loss(g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'].float(),
                                                    target_pt_eta_xhat_yhat.float()), dim=1)
        pt_eta_xhat_yhat_loss = torch.sqrt(pt_eta_xhat_yhat_loss+eps)

        #compute total node properties loss
        class_boost = 0.2
        g.nodes['nodes'].data['node loss'] = (1.0+class_boost)*nodes_class_loss + (1.0-class_boost)*pt_eta_xhat_yhat_loss

        #compute true pt
        pt_true = target_pt_eta_xhat_yhat[:,0]
        pt_true = torch.exp(pt_true)
        pt_true = pt_true*self.var_transform['particle_pt']['std'] + self.var_transform['particle_pt']['mean']
        pt_true = pt_true/1000. #MeV --> GeV

        #compute track pt
        pt_track = torch.exp(g.nodes['nodes'].data['track_pt'])
        pt_track = pt_track*self.var_transform['particle_pt']['std'] + self.var_transform['particle_pt']['mean']
        pt_track = pt_track/1000. #MeV --> GeV

        #zero-weight nodes where track_pt is way way overestimated (20x larger)
        bad_track_mask = torch.where(g.nodes['nodes'].data['isTrack']*pt_track/(pt_true + 1e-6) > 20,0,1)

        #weight node properties loss by beta
        archtan_beta2 = torch.arctanh( clamped_beta )**2
        g.nodes['nodes'].data['q'] = (archtan_beta2+self.qmin).view(-1)
        g.nodes['nodes'].data['xi'] = (1-g.nodes['nodes'].data['n'])*archtan_beta2*bad_track_mask
        g.nodes['nodes'].data['xitimesL'] = g.nodes['nodes'].data['node loss']*g.nodes['nodes'].data['xi']
        g.nodes['nodes'].data['ntimesb'] = g.nodes['nodes'].data['n']*g.nodes['nodes'].data['beta']

        #init the particles beta/x/q arrays
        g.nodes['particles'].data['max beta'] = 0.0*torch.ones(g.num_nodes('particles')).float().to(g.device)
        g.nodes['particles'].data['max q'] = torch.zeros(g.num_nodes('particles')).to(g.device)
        g.nodes['particles'].data['max x'] = torch.zeros(g.num_nodes('particles'),self.config['output model']['x size']).to(g.device)
        g.nodes['particles'].data['max idx'] = torch.zeros(g.num_nodes('particles')).to(g.device)
        g.nodes['particles'].data['max zeta'] = 0.0*torch.ones(g.num_nodes('particles')).float().to(g.device)
      
        #compute the attractive and repulsive potentials
        if self.use_zeta_CPs:
            g.update_all(ChildToParentEF,ChildToParentNF_maxzeta,etype='node_to_particle')
        else:
            g.update_all(ChildToParentEF,ChildToParentNF_maxbeta,etype='node_to_particle')
        g.update_all(ParentToChildEF,ParentToChildNF,etype='particle_to_node')

        #label the (truth) condensation points
        g.nodes['nodes'].data['is cond point truth'] = torch.zeros(g.num_nodes('nodes')).long().to(g.device)
        true_cond_wheres = g.nodes['particles'].data['max idx'].to(torch.long)
        g.nodes['nodes'].data['is cond point truth'][true_cond_wheres] = 1

        #compute the potential loss 
        x_loss = dgl.mean_nodes(g, 'node Lv' ,ntype='nodes')        
        x_loss = x_loss.mean()
            
        # first sum over nodes n*beta (background nodes term)
        sum_nxb = dgl.sum_nodes(g, 'ntimesb' ,ntype='nodes') 
        
        total_nBackground = dgl.sum_nodes(g,'n',ntype='nodes')
        total_nBackground[total_nBackground == 0] = 1.0 # protect against division by 0, if there are no background nodes, the numerator will also be 0
        background_term = sum_nxb/total_nBackground

        #compute the beta loss
        if self.use_BCE_beta_loss:
            #use BCE loss instead
            pred_beta_class = torch.cat([(1 - g.nodes['nodes'].data['beta']).unsqueeze(-1),g.nodes['nodes'].data['beta'].unsqueeze(1)],dim=-1)
            #object_sum_beta = torch.sum(self.class_CP_loss(pred_beta_class,g.nodes['nodes'].data['is cond point truth'])/g.nodes['nodes'].data['N nodes'])
            object_sum_beta = torch.sum(self.class_CP_loss(pred_beta_class,g.nodes['nodes'].data['is cond point truth'])*(1-g.nodes['nodes'].data['isTrack'])*(1.+10.*(g.nodes['nodes'].data['particle class']==1))/(g.nodes['nodes'].data['N nodes']+g.nodes['nodes'].data['n']))
            beta_loss = 0.05*(object_sum_beta+self.s_b*background_term).mean()
        else:
            #next sum over objects max beta (to encourage condensation)
            g.nodes['particles'].data['1 min max beta'] = 1-g.nodes['particles'].data['max beta']
            object_sum_beta = dgl.sum_nodes(g,'1 min max beta',ntype='particles')
            beta_loss = (object_sum_beta+self.s_b*background_term).mean()

        #the Lp' term from the paper 
        sum_xi = dgl.sum_nodes(g, 'xi' ,ntype='nodes')
        sum_xiL = dgl.sum_nodes(g, 'xitimesL' ,ntype='nodes')

        #summed losses
        node_loss = (sum_xiL/(sum_xi+1e-6)).mean()
        beta_loss = self.s_c * beta_loss
        x_loss = self.s_c * x_loss

        #total loss
        loss = node_loss + beta_loss + x_loss

        #debugging NAN loss
        if torch.isnan(loss):
            print('FOUND NAN!')
            print('node loss: ', node_loss)
            print('beta loss: ', beta_loss)
            print('x loss: ',    x_loss)
            print('FEATURES:')
            print('clamped_beta: ',clamped_beta.detach().cpu().numpy().tolist())
            print('archtan_beta2: ',archtan_beta2.detach().cpu().numpy().tolist())


        #plotting scatter plots of target and prediction during training
        if scatter == True:
            pred_x_y = torch.cat([g.nodes['nodes'].data['pt_eta_xhat_yhat_pred'], g.nodes['nodes'].data['particle class'].unsqueeze(1)], dim=1)
            pred     = torch.stack([
                            pred_x_y[:,0],
                            pred_x_y[:,1],
                            torch.atan2(pred_x_y[:,3],pred_x_y[:,2]),
                            pred_x_y[:,4],
            ], dim=1
            )

            target = torch.stack([
                        g.nodes['nodes'].data['parent_pt'], 
                        g.nodes['nodes'].data['parent_eta'], 
                        torch.atan2(g.nodes['nodes'].data['parent_yhat'],g.nodes['nodes'].data['parent_xhat']),
                        g.nodes['nodes'].data['particle class']
                    ],dim=1)

            target_copy, pred_copy = deepcopy(target.cpu().data), deepcopy(pred.cpu().data)
            target_copy, pred_copy = self.undo_scaling(target_copy), self.undo_scaling(pred_copy)

            scatter_dict['nodes'] = [target_copy, pred_copy]
            del target, pred

        if fill_num_particles:
            n_obj_copy = deepcopy(n_objects_per_event[0])
            num_particles_in_first_event[cl] = n_obj_copy
            del n_objects_per_event

        if scatter == True:
            return  {
                'loss':loss, 
                'node loss':node_loss.item(),
                'beta loss' : beta_loss.item(),
                'x loss': x_loss.item(),
                'scatter_dict': scatter_dict,
                'num_particles_in_first_event': num_particles_in_first_event
            }

        return {'loss':loss,'node loss':node_loss.item(),'beta loss' : beta_loss.item(), 'x loss': x_loss.item() }
