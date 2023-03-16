import dgl
import dgl.function as fn
from lightning import PflowLightning
import sys
import os
import json
import torch
import numpy as np
from dataloader import PflowDataset, collate_graphs
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import pandas as pd
import os
import ROOT
from array import array 

print(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]


if __name__ == "__main__":

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    with open(config_path, 'r') as fp:
        config = json.load(fp)
    
    if not os.path.exists('evaluations'):
                os.mkdir('evaluations')
   
   
    outputfile = ROOT.TFile(config['name']+'_unsup.root','recreate')
    outputtree = ROOT.TTree('pflow_tree','pflow_tree')

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

    pflow_target = ROOT.vector('int')()
    outputtree.Branch('pflow_target',pflow_target)

    n_neutral_low     = array("i",[0])
    n_neutral_midlow  = array("i",[0])
    n_neutral_midhigh = array("i",[0])
    n_neutral_high    = array("i",[0])
    n_supneutral_low     = array("i",[0])
    n_supneutral_midlow  = array("i",[0])
    n_supneutral_midhigh = array("i",[0])
    n_supneutral_high    = array("i",[0])
    outputtree.Branch('n_supneutral_low',n_supneutral_low    ,'n_supneutral_low/I')
    outputtree.Branch('n_supneutral_midlow',n_supneutral_midlow ,'n_supneutral_midlow /I')
    outputtree.Branch('n_supneutral_midhigh',n_supneutral_midhigh,'n_supneutral_midhigh/I')
    outputtree.Branch('n_supneutral_high',n_supneutral_high   ,'n_supneutral_high/I')

    outputtree.Branch('n_neutral_low',n_neutral_low    ,'n_neutral_low/I')
    outputtree.Branch('n_neutral_midlow',n_neutral_midlow ,'n_neutral_midlow /I')
    outputtree.Branch('n_neutral_midhigh',n_neutral_midhigh,'n_neutral_midhigh/I')
    outputtree.Branch('n_neutral_high',n_neutral_high   ,'n_neutral_high/I')

    n_gamma_low     = array("i",[0])
    n_gamma_midlow  = array("i",[0])
    n_gamma_midhigh = array("i",[0])
    n_gamma_high    = array("i",[0])
    outputtree.Branch('n_gamma_low',n_gamma_low    , "n_gamma_low/I")
    outputtree.Branch('n_gamma_midlow',n_gamma_midlow , "n_gamma_midlow /I")
    outputtree.Branch('n_gamma_midhigh',n_gamma_midhigh, "n_gamma_midhigh/I")
    outputtree.Branch('n_gamma_high',n_gamma_high   , "n_gamma_high/I")


    if config['output model type'] == "Set2Set":
        attention = ROOT.vector(ROOT.vector('float'))()
        outputtree.Branch('attention',attention)


    net = PflowLightning(config)
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print("Config Path to Test", config['path_to_test'])

    dataset = PflowDataset(config['path_to_test'],config,reduce_ds=config["reduce_ds"]) 

    loader = DataLoader(dataset, batch_size=32, num_workers=config['num_workers'], 
        shuffle=False,collate_fn=collate_graphs)

    if torch.cuda.is_available():
        print('switching to gpu')
        net.net.cuda()
        net.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    device = torch.device('cpu')
    for g1 in tqdm( loader ):
        g = g1[0]
        g = g.to(device)

        if config['output model type'] == "Set2Set":    
            predicted_particles, predicted_num_particles, pred_n_supnu = net.net.infer(g)
        else:
            predicted_particles, predicted_num_particles = net.net.infer(g)

        eta, phi, pt, p_pxpypz, p_pos, p_class, p_charge, p_target = predicted_particles
        
        n_events = len(predicted_num_particles)
        particle_counter = -1

        predicted_num_particles = predicted_num_particles.cpu().int().data.numpy()
        pt   = pt.cpu().float().data.numpy()
        eta = eta.cpu().float().data.numpy()
        phi = phi.cpu().float().data.numpy()
        p_pxpypz = p_pxpypz.cpu().float().data.numpy()
        p_pos = p_pos.cpu().float().data.numpy()
        p_class = p_class.cpu().int().data.numpy()
        p_charge = p_charge.cpu().int().data.numpy()
        p_target = p_target.cpu().int().data.numpy()

        for event_i in range(n_events):
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
            pflow_target.clear()
            n_gamma_low    [0]=-1#.clear()#=-1
            n_gamma_midlow [0]=-1#.clear()#=-1
            n_gamma_midhigh[0]=-1#.clear()#=-1
            n_gamma_high   [0]=-1#.clear()#=-1
            n_neutral_low    [0]=-1#.clear()#=-1
            n_neutral_midlow [0]=-1#.clear()#=-1
            n_neutral_midhigh[0]=-1#.clear()#=-1
            n_neutral_high   [0]=-1#.clear()#=-1

            if config['output model type'] == "Set2Set":
                attention.clear()

            n_particles = predicted_num_particles[event_i]
            

            n_supneutral_low[0]=int(pred_n_supnu[0][event_i])
            n_supneutral_midlow[0]=int(pred_n_supnu[1][event_i])
            n_supneutral_midhigh[0]=int(pred_n_supnu[2][event_i])
            n_supneutral_high[0]=int(pred_n_supnu[3][event_i])
#            n_gamma_low    [0]=int(pred_n_gamma[0][event_i])
#            n_gamma_midlow [0]=int(pred_n_gamma[1][event_i])
#            n_gamma_midhigh[0]=int(pred_n_gamma[2][event_i])
#            n_gamma_high   [0]=int(pred_n_gamma[3][event_i])

            for particle in range(int(n_particles)):
                particle_counter+=1
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
                pflow_target.push_back(        int(p_target[particle_counter])       )

                # if config['output model type'] == "Set2Set":
                #     attention_weights = predicted_attention_weights[particle_counter].cpu().float().data.numpy()
                #     attention_weights = attention_weights.copy(order='C')

                #     v = ROOT.vector("float")(attention_weights)
                #     attention.push_back(v)

            outputtree.Fill()

    outputfile.cd()
    outputtree.Write()
    outputfile.Close()

