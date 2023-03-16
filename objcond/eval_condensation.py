import argparse
import dgl
import dgl.function as fn
from lightning import PflowLightning
import sys
import os
import json
from time import sleep
import torch
import numpy as np
from dataloader import PflowDataset, collate_graphs
from condensation_metrics import CondNetMetrics
from object_cond_loss import ObjectCondenstationLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import pandas as pd
import os
import ROOT

def get_nodes_dict(g, epoch=-1, config=None):
    nodes_dict = {}

    node_feats = ['x','beta','q','n','is cond point','isTrack','parent target','particle class']
    if args.truth_inference:
        node_feats = node_feats + ['N nodes']
    for feta in node_feats:
        arr = g.nodes['nodes'].data[feta].detach().cpu().numpy()
        if len(arr.shape) > 1:
            for comp in range(arr.shape[-1]):
                nodes_dict[feta+str(comp)] = arr[:,comp]
        else: 
            nodes_dict[feta] = arr

    #special treatment for parent class, sup cond pts
    #nodes_dict['parent class']  = g.nodes['particles'].data['particle class'].detach().cpu().numpy()[nodes_dict['parent target'].astype(int)]

    idxs              = g.nodes['nodes'].data['idx'].detach().cpu().numpy().astype(int)
    super_cond_wheres = g.nodes['particles'].data['max idx'].detach().cpu().numpy().astype(int)
    super_cond_points = np.zeros(len(idxs)).astype(bool)
    super_cond_points[super_cond_wheres] = 1
    nodes_dict['is cond point sup'] = super_cond_points

    cell_class_pred     = torch.argmax( g.nodes['nodes'].data['cell class pred'],  dim=1)
    track_class_pred    = torch.argmax( g.nodes['nodes'].data['track class pred'], dim=1) + 2*torch.ones_like(g.nodes['nodes'].data['isTrack'])
    particle_class_pred = torch.where(g.nodes['nodes'].data['isTrack']==1,track_class_pred,cell_class_pred)
    nodes_dict['class pred'] = particle_class_pred.detach().cpu().numpy().astype(int)

    true_pt = g.nodes['nodes'].data['parent_pt'].clone()
    if config is not None:
        true_pt = true_pt*config['var transform']['particle_pt']['std'] + config['var transform']['particle_pt']['mean']
    true_pt = torch.exp(true_pt)
    true_pt = true_pt/1000. #MeV --> GeV

    nodes_dict['parent_pt'] = true_pt.detach().cpu().numpy()

    #also add the epoch
    if epoch >= 0:
        nodes_dict['epoch'] = np.repeat(epoch,len(nodes_dict['beta']))

    return nodes_dict

def get_targets_dict(g, epoch=-1):
    targets_dict = {}

    node_feats = ['particle class','particle_pt','particle_eta','particle_phi','max x','max beta','max q']
    if args.truth_inference and doMetrics: 
        node_feats = node_feats + ['N nodes','nearest neighbors']
    for feta in node_feats:
        arr = g.nodes['particles'].data[feta].detach().cpu().numpy()
        if len(arr.shape) > 1:
            for comp in range(arr.shape[-1]):
                targets_dict[feta+str(comp)] = arr[:,comp]
        else:
            targets_dict[feta] = arr

    #also add the epoch
    if epoch >= 0:
        targets_dict['epoch'] = np.repeat(epoch,len(targets_dict['max beta']))

    return targets_dict

def get_predictions_dict():

    #to be filled later
    predictions_dict = {}
    predictions_dict['epoch'] = []
    predictions_dict['event'] = []
    predictions_dict['t_b'] = []
    predictions_dict['t_d'] = []
    predictions_dict['pred pt'] = []
    predictions_dict['pred eta'] = []
    predictions_dict['pred phi'] = []
    predictions_dict['pred class'] = []

    return predictions_dict


def get_edges_dict(g, epoch=-1):
    edges_dict = {}
    edge_feats = ['dist x', 'belongs', 'beta', 'parent class', 'particle class']

    for feta in edge_feats:
        edges_dict[feta] = g.edges['particle_to_node'].data[feta].detach().cpu().numpy()

    #also add the epoch
    #if epoch >= 0:
    #    edges_dict['epoch'] = np.repeat(epoch,len(edges_dict['dist x']))

    return edges_dict


parser = argparse.ArgumentParser(description='Evaluate predictions of given pflow checkpoint')
parser.add_argument('--gpu', dest='gpu', type=int)
parser.add_argument('--config', dest='config', required=True)
parser.add_argument('--checkpoint', dest='checkpoint', required=False, default='epoch=0')
parser.add_argument('--output', dest='output',default='my_evaluation.root')
parser.add_argument('--max_events', dest='max_events', type=int, default=-1)
parser.add_argument('--truth_inference', dest='truth_inference', type=int, default=0)
parser.add_argument('--t_b', dest='t_b', type=float)
parser.add_argument('--t_d', dest='t_d', type=float)
parser.add_argument('--path_to_test', dest='path_to_test')
parser.add_argument('--performance_test', dest='performance_test', type=int, default=0)

args = parser.parse_args()

doMetrics = True

#py3 eval_condensation.py configs/condensation.json path/to/checkpoint.ckpt 0
if args.gpu: os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


if __name__ == "__main__":

    with open(args.config, 'r') as fp:
        config = json.load(fp)

    if args.t_b:
        config['t_b'] = args.t_b

    if args.t_d:
        config['t_d'] = args.t_d

    if args.path_to_test:
        config['path_to_test'] = args.path_to_test

    if not os.path.exists('evaluations'):
                os.mkdir('evaluations')
   
    epoch = int(args.checkpoint.split(".")[0].split("epoch=")[1])
    newoutput = args.output
    newoutput = newoutput.split(".root")[0] + "_" + str(epoch) + ".root"

    outputfile = ROOT.TFile(newoutput,'recreate')
    outputtree = ROOT.TTree('pflow_tree','pflow_tree')
    nodesdict  = {}

    event_vec = ROOT.vector('int')()
    outputtree.Branch('event',event_vec)
    pflow_class = ROOT.vector('int')()
    outputtree.Branch('pflow_class',pflow_class)
    pflow_charge = ROOT.vector('int')()
    outputtree.Branch('pflow_charge',pflow_charge)
    pflow_prod_x = ROOT.vector('float')()
    outputtree.Branch('pflow_prod_x',pflow_prod_x)
    pflow_prod_y = ROOT.vector('float')()
    outputtree.Branch('pflow_prod_y',pflow_prod_y)
    pflow_prod_z = ROOT.vector('float')()
    outputtree.Branch('pflow_prod_z',pflow_prod_z)
    
    pflow_px = ROOT.vector('float')()
    outputtree.Branch('pflow_px',pflow_px)
    pflow_py = ROOT.vector('float')()
    outputtree.Branch('pflow_py',pflow_py)
    pflow_pz = ROOT.vector('float')()
    outputtree.Branch('pflow_pz',pflow_pz)

    pflow_pt = ROOT.vector('float')()
    outputtree.Branch('pflow_pt',pflow_pt)

    pflow_eta = ROOT.vector('float')()
    outputtree.Branch('pflow_eta',pflow_eta)

    pflow_phi = ROOT.vector('float')()
    outputtree.Branch('pflow_phi',pflow_phi)

    #Branches to store target (parent) properties for condensation points
    #truth_class = ROOT.vector('int')()
    #outputtree.Branch('truth_class',truth_class)

    #truth_pt = ROOT.vector('float')()
    #outputtree.Branch('truth_pt',truth_pt)

    #truth_eta = ROOT.vector('float')()
    #outputtree.Branch('truth_eta',truth_eta)

    #truth_phi = ROOT.vector('float')()
    #outputtree.Branch('truth_phi',truth_phi)

    truth_parent = ROOT.vector('int')()
    outputtree.Branch('truth_parent',truth_parent)

    epoch_vec = ROOT.vector('int')()
    outputtree.Branch('epoch',epoch_vec)

    ##Condensation sanity infos
    if args.truth_inference and doMetrics:
        neighbor_dist_vec = ROOT.vector('float')()
        outputtree.Branch('neighbor_dist',neighbor_dist_vec)

        N_metric_vec = ROOT.vector('int')()
        outputtree.Branch('N_metric',N_metric_vec)

        DB_vec = ROOT.vector('float')()
        RMS_vec = ROOT.vector('float')()
        outputtree.Branch('DB',DB_vec)
        outputtree.Branch('RMS',RMS_vec)

    if config['output model type'] == "Set2Set":
        attention = ROOT.vector(ROOT.vector('float'))()
        outputtree.Branch('attention',attention)

    config['truth inference'] = int(args.truth_inference)
    net = PflowLightning(config)
    if args.checkpoint != "epoch=0":
        checkpoint = torch.load(args.checkpoint,map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    dataset = PflowDataset(config['path_to_test'],config,reduce_ds=args.max_events)

    if not config['output model type'] == "Set2Set":
        metrics   = CondNetMetrics(config)
        loss_func = ObjectCondenstationLoss(config)

    loader = DataLoader(dataset, batch_size=config['batchsize'], num_workers=config['num_workers'], 
        shuffle=False,collate_fn=collate_graphs)

    if torch.cuda.is_available():
        print('switching to gpu')
        net.net.cuda()
        net.cuda()
        loss_func.cuda()
        metrics.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    nodes_df   = pd.DataFrame()
    targets_df = pd.DataFrame()
    predictions_df = pd.DataFrame()
    edges_df   = pd.DataFrame()

    global_event_idx = 0

    ### PERFORMANCE BENCHMARKING ###    
    if args.performance_test==1:

        times = []

        ### GPU warm up
        for g in loader:
            g = g.to(device)
            net(g)
            break

        for g_idx, g in tqdm( enumerate(loader) ):

                g = g.to(device)

                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                # start
                starter.record()

                # network prediction
                net(g)
                loss_func(g)
                predicted_particles, predicted_num_particles = net.net.infer(g)

                # finish
                ender.record()

                # wait to synchronize...
                torch.cuda.synchronize()

                # measure difference
                curr_time = starter.elapsed_time(ender)
                times.append(curr_time)

        print(f'\nList of times:')
        print(times)

        print(f'\nAverage time over {len(loader)} events:')
        print(np.mean(times))

        # for monitoring memory usage
        print('\nsleeping for 10 seconds...')
        sleep(10)
        print('finished!')
        
        exit()

    ###################################

    for g_idx, g in tqdm( enumerate(loader) ):

        g = g.to(device)

        if config['output model type'] == "Set2Set":    
            predicted_particles, predicted_num_particles, predicted_attention_weights = net.net.infer(g)
        else:
            net(g)
            loss_func(g)
            predicted_particles, predicted_num_particles = net.net.infer(g)
            if args.truth_inference and doMetrics:
                metrics(g)

        eta, phi, pt, p_pxpypz, p_pos, p_class, p_charge, p_parent_idx = predicted_particles
        
        n_events = len(predicted_num_particles)
        particle_counter = -1
        n_nodes          = [gb.num_nodes(ntype='nodes') for gb in dgl.unbatch(g)]
        n_particle_nodes = [gb.num_nodes(ntype='particles') for gb in dgl.unbatch(g)]
        n_edges          = [gb.num_edges(etype='particle_to_node') for gb in dgl.unbatch(g)]

        predicted_num_particles = predicted_num_particles.cpu().int().data.numpy()
        pt   = pt.cpu().float().data.numpy()
        eta = eta.cpu().float().data.numpy()
        phi = phi.cpu().float().data.numpy()
        p_pxpypz = p_pxpypz.cpu().float().data.numpy()
        p_pos = p_pos.cpu().float().data.numpy()
        p_class = p_class.cpu().int().data.numpy()
        p_charge = p_charge.cpu().int().data.numpy()
        p_parent_idx = p_parent_idx.long() #.cpu().int().data.numpy()

        #for gb in dgl.unbatch(g):
        #parent_class = g.nodes['particles'].data['particle class'][p_parent_idx].detach().cpu().numpy()
        #parent_pt    = g.nodes['particles'].data['particle_pt'][p_parent_idx].detach().cpu().numpy()
        #parent_eta   = g.nodes['particles'].data['particle_eta'][p_parent_idx].detach().cpu().numpy()
        #parent_phi   = g.nodes['particles'].data['particle_phi'][p_parent_idx].detach().cpu().numpy()
        parent_idx   = p_parent_idx.detach().cpu().numpy()

        if args.truth_inference and doMetrics:
            neighbor_dist = g.nodes['particles'].data['nearest neighbors'].detach().cpu().numpy()  if device==torch.device('cuda') else g.nodes['particles'].data['nearest neighbors'].detach().numpy()
            N_metric      = g.nodes['particles'].data['N nodes'].detach().cpu().numpy().astype(int)  if device==torch.device('cuda') else g.nodes['particles'].data['N nodes'].detach().numpy().astype(int)  
            DB  = g.nodes['global node'].data['DB'].detach().cpu().numpy()  if device==torch.device('cuda') else g.nodes['global node'].data['DB'].detach().numpy()
            RMS = g.nodes['global node'].data['RMS'].detach().cpu().numpy() if device==torch.device('cuda') else g.nodes['global node'].data['RMS'].detach().numpy()

        nodes_dict = get_nodes_dict(g,epoch,config)
        targets_dict = get_targets_dict(g,epoch)
        predictions_dict = get_predictions_dict()
        if args.truth_inference and doMetrics:
            edges_dict = get_edges_dict(g,epoch)

        for event_i in range(n_events):

            event_nodes_array = np.repeat(global_event_idx, n_nodes[event_i])
            if 'event' not in nodes_dict:
                nodes_dict['event'] = event_nodes_array
            else:
                nodes_dict['event'] = np.append(nodes_dict['event'],event_nodes_array)

            event_particles_array = np.repeat(global_event_idx, n_particle_nodes[event_i])
            if 'event' not in targets_dict:
                targets_dict['event'] = event_particles_array
            else:
                targets_dict['event'] = np.append(targets_dict['event'],event_particles_array)

            if args.truth_inference and doMetrics:
                event_edges_array = np.repeat(global_event_idx, n_edges[event_i])
                if 'event' not in edges_dict:
                    edges_dict['event'] = event_edges_array
                else:
                    edges_dict['event'] = np.append(edges_dict['event'],event_edges_array)

            event_vec.clear()
            pflow_class.clear()
            pflow_charge.clear()
            pflow_prod_x.clear()
            pflow_prod_y.clear()
            pflow_prod_z.clear()
            pflow_px.clear()
            pflow_py.clear()
            pflow_pz.clear()
            pflow_pt.clear()
            pflow_eta.clear()
            pflow_phi.clear()
            #truth_class.clear()
            #truth_pt.clear()
            #truth_eta.clear()
            #truth_phi.clear()
            truth_parent.clear()
            if args.truth_inference and doMetrics:
                neighbor_dist_vec.clear()
                N_metric_vec.clear()
                RMS_vec.clear()
                DB_vec.clear()
            epoch_vec.clear()

            if config['output model type'] == "Set2Set":
                attention.clear()

            n_particles = predicted_num_particles[event_i]
    
            for particle in range(int(n_particles)):
                particle_counter+=1
                event_vec.push_back(event_i)
                pflow_class.push_back(        int(p_class[particle_counter])        )
                pflow_charge.push_back(       int(p_charge[particle_counter])         )
                pflow_prod_x.push_back(      p_pos[particle_counter][0]          )
                pflow_prod_y.push_back(      p_pos[particle_counter][1]          )
                pflow_prod_z.push_back(      p_pos[particle_counter][2]          )
                pflow_px.push_back(  p_pxpypz[particle_counter][0]              )
                pflow_py.push_back(  p_pxpypz[particle_counter][1]              )
                pflow_pz.push_back(  p_pxpypz[particle_counter][2]              )
                pflow_pt.push_back(         pt[particle_counter]       )
                pflow_eta.push_back(        eta[particle_counter]       )
                pflow_phi.push_back(        phi[particle_counter]       )

                #truth_class.push_back( int(parent_class[particle_counter]) )
                #truth_pt.push_back(    parent_pt[particle_counter]  )
                #truth_eta.push_back(   parent_eta[particle_counter] )
                #truth_phi.push_back(   parent_phi[particle_counter] )
                truth_parent.push_back( int(parent_idx[particle_counter]) )

                epoch_vec.push_back(epoch)
                if args.truth_inference and doMetrics:
                    neighbor_dist_vec.push_back(neighbor_dist[particle_counter][0])
                    N_metric_vec.push_back(int(N_metric[particle_counter]))
                    DB_vec.push_back(DB[event_i])
                    RMS_vec.push_back(RMS[event_i])

                predictions_dict['epoch'].append(epoch)
                predictions_dict['event'].append(global_event_idx)
                predictions_dict['t_b'].append(config['t_b'])
                predictions_dict['t_d'].append(config['t_d'])
                predictions_dict['pred pt'].append(pt[particle_counter])
                predictions_dict['pred eta'].append(eta[particle_counter])
                predictions_dict['pred phi'].append(phi[particle_counter])
                predictions_dict['pred class'].append(int(p_class[particle_counter]))

            global_event_idx += 1

            outputtree.Fill()

        nodes_df_i = pd.DataFrame(nodes_dict)
        if len(nodes_df) == 0:
            nodes_df = nodes_df_i
        else:
            nodes_df = nodes_df.append(nodes_df_i)

        targets_df_i = pd.DataFrame(targets_dict)
        if len(targets_df) == 0:
            targets_df = targets_df_i
        else:
            targets_df = targets_df.append(targets_df_i)

        predictions_df_i = pd.DataFrame(predictions_dict)
        if len(predictions_df) == 0:
            predictions_df = predictions_df_i
        else:
            predictions_df = predictions_df.append(predictions_df_i)

        if args.truth_inference and doMetrics:
            edges_df_i = pd.DataFrame(edges_dict)
            if len(edges_df) == 0:
                edges_df = edges_df_i
            else:
                edges_df = edges_df.append(edges_df_i)

    pkloutput = newoutput
    pkloutput = pkloutput.replace('.root','')
    nodes_df.to_pickle(pkloutput + "_nodes_pickle.pkl")
    targets_df.to_pickle(pkloutput + "_targets_pickle.pkl")
    predictions_df.to_pickle(pkloutput + "_predictions_pickle.pkl")
    if args.truth_inference and doMetrics:
        edges_df.to_pickle(pkloutput + "_edges_pickle.pkl")

    outputfile.cd()
    outputtree.Write()
    outputfile.Close()

    print('results written to ',newoutput)
