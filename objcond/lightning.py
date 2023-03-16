#import comet_ml
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
sys.path.append('./models/')

from pflow_model import PflowModel

from dataloader import PflowDataset, collate_graphs
import numpy as np
import dgl
from object_cond_loss import ObjectCondenstationLoss
from hungrian_loss import HungarianLoss
from mlpf_loss import MLPFLoss
from loss_set2set import Set2SetLoss

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import seaborn as sn
import pandas as pd
from sklearn import metrics

import json

class PflowLightning(LightningModule):

    def __init__(self, config, comet_exp=None):
        super().__init__()
        
        self.config = config
        
        self.net = PflowModel(self.config)

        if config['output model type']=='condensation':
            self.loss = ObjectCondenstationLoss(config)
        elif config['output model type']=='Set2Set':
            self.loss = Set2SetLoss(config)
        elif config['output model type']=='MLPF':
            self.loss = MLPFLoss(config)
        elif config['output model type']=='combined':
            print('We are running in combined mode!')
            config_condensation_path = '/srv01/agrp/dreyet/PFlow/SCD/particle_flow/experiments/configs/condensation.json'
            with open(config_condensation_path, 'r') as fp:
                config_condensation = json.load(fp)
                self.config_condensation = config_condensation
            self.loss2 = ObjectCondenstationLoss(config_condensation)
            self.loss = Set2SetLoss(config)

        self.comet_exp = comet_exp
        self.num_particles_in_first_event = {}


    def set_comet_exp(self, comet_exp):
        self.comet_exp = comet_exp


    def forward(self, g):

        return self.net(g)

    
    def training_step(self, batch, batch_idx):
        
        g = batch
        
        self(g)
        
        loss = self.loss(g)
        if self.config['output model type']=='combined':
            loss2 = self.loss2(g)

        return_dict = {
            'loss' : loss['loss'] + loss2['loss'] if self.config['output model type']=='combined' else loss['loss']
        }

        return_dict['log'] = {}
        #if not self.config['output model type']=='combined':
        for loss_type in self.config['loss types']:
            return_dict['log'][loss_type] = loss[loss_type]

        if self.config['output model type']=='combined':
            cond_loss_types = ["node loss","beta loss","x loss"]
 
            for cond_loss_type in cond_loss_types:
                return_dict['log'][cond_loss_type] = loss2[cond_loss_type]

        return return_dict
        

    def validation_step(self, batch, batch_idx):
        
        g = batch
        self(g)

        if self.config['output model type']=='condensation': loss = self.loss(g)
        else: loss = self.loss(g, True, self.num_particles_in_first_event)
        #print('loss:', loss['loss'], end='\t')
        
        if self.config['output model type']=='combined':
            loss2 = self.loss2(g)
            #print('loss2:', loss2, end='\t')
            #return {'val_loss'     : loss['loss'] + loss2['loss'], 'tspn_loss'     : loss['loss'], 'cond_loss': loss2['loss']}

        if self.config['output model type']=='condensation' or self.config['output model type']=='combined':

            g0 = g # dgl.unbatch(g)[0] #HACK! Trying to fix memory access error
            target_class   = g0.nodes['particles'].data['particle class'].detach()
            target_pt      = g0.nodes['particles'].data['particle_pt'].detach()
            target_eta     = g0.nodes['particles'].data['particle_eta'].detach()
            target_phi     = g0.nodes['particles'].data['particle_phi'].detach()

            predicted_particles, predicted_num_particles = self.net.infer(g0)
            predicted_eta, predicted_phi, predicted_pt, predicted_particle_pxpypz, predicted_particle_pos, predicted_particle_class, predicted_particle_charge, _ = predicted_particles

            scatter_dict = {}

            classes = ["photon","neutral","charged","electron","muon"]

            for idx, cl in enumerate(classes):

                predicted_where = torch.where(predicted_particle_class==idx)
                predicted = torch.stack([
                            predicted_pt[predicted_where],
                            predicted_eta[predicted_where],
                            predicted_phi[predicted_where],
                            predicted_particle_class[predicted_where], #silly??
                ], dim=1)

                target_where = torch.where(target_class==idx)
                target    = torch.stack([
                            target_pt[target_where], 
                            target_eta[target_where], 
                            target_phi[target_where],
                            target_class[target_where]
                ],dim=1)

                #target_copy, pred_copy = deepcopy(target.cpu().data), deepcopy(pred.cpu().data)
                #target_copy, pred_copy = self.undo_scaling(target_copy), self.undo_scaling(pred_copy) ##WARNING!! tuple shape change!

                scatter_dict[cl] = [target, predicted]
                del target, predicted
            

        return_dict = {
            'val_loss'     : loss['loss'] + loss2['loss'] if self.config['output model type']=='combined' else loss['loss'],
            'scatter_dict' : loss['scatter_dict'] if not (self.config['output model type']=='condensation' or self.config['output model type']=='combined') else scatter_dict,
            'num_particles_in_first_event': loss['num_particles_in_first_event']
        }

        for loss_type in self.config['loss types']:
            self.log(loss_type, loss[loss_type], batch_size=self.config['batchsize'])
            return_dict[loss_type] = loss[loss_type]
            print(loss_type + ":", loss[loss_type], end='\t')

        if self.config['output model type']=='combined':
            return_dict['tspn_loss'] = loss['loss']
            return_dict['cond_loss'] = loss2['loss']

            cond_loss_types = ["node loss","beta loss","x loss"]
 
            for cond_loss_type in cond_loss_types:
                self.log(cond_loss_type, loss2[cond_loss_type])
                return_dict[cond_loss_type] = loss2[cond_loss_type]


        return return_dict

    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['learningrate'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12,64], gamma=0.1, verbose=True)

        #if not self.config['output model type']=='condensation':
        return {"optimizer": optimizer} #, "lr_scheduler": scheduler}
        #else:
        #    return torch.optim.AdamW(self.parameters(), lr=self.config['learningrate'])
    
    def train_dataloader(self):
        
        if 'reduce_ds' in self.config:
            reduce_ds = self.config['reduce_ds']
        else:
            reduce_ds = 1

        dataset = PflowDataset(self.config['path_to_train'], self.config, reduce_ds=reduce_ds,entry_start=(self.config['entry_start'] if 'entry_start' in self.config else 0))

        loader = DataLoader(dataset, batch_size=self.config['batchsize'], 
                            num_workers=self.config['num_workers'], shuffle=True, collate_fn=collate_graphs, pin_memory=False)
        return loader

    
    def val_dataloader(self):

        if 'reduce_ds' in self.config:
            reduce_ds = self.config['reduce_ds']
        else:
            reduce_ds = 1

        reduce_ds = 1000 #HACK!

        dataset = PflowDataset(self.config['path_to_valid'], self.config, reduce_ds=reduce_ds)

        loader = DataLoader(dataset, batch_size=self.config['batchsize'], num_workers=self.config['num_workers'], 
                            shuffle=False, collate_fn=collate_graphs, pin_memory=False)
        return loader


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.config['output model type']=='combined':
            avg_loss_tspn = torch.stack([x['tspn_loss'] for x in outputs]).mean()
            avg_loss_cond = torch.stack([x['cond_loss'] for x in outputs]).mean()
            print('avg_loss={},\tavg_loss_tspn={},\tavg_loss_cond={}'.format(avg_loss,avg_loss_tspn,avg_loss_cond))

        return_dict = {'val_loss': avg_loss, 'log': {'val_loss': avg_loss}}
        for loss_type in self.config['loss types']:
            if self.config['output model type']=='condensation' or self.config['output model type']=='combined':
                return_dict['log']['val_'+loss_type] = np.mean([x[loss_type] for x in outputs])
            elif not self.config['output model type']=='combined':
                return_dict['log']['val_'+loss_type] = np.mean([x[loss_type].cpu().data for x in outputs])

        if not self.config['output model type']=='combined' or True:

            # scatter plot
            classes = ["supercharged"] # ["photon","charged","neutral","electron","muon"]
            if self.config['output model type']=='condensation' or self.config['output model type']=='combined': classes =  ["photon","neutral","charged","electron","muon"]

            if self.config['output model type']=='condensation' or self.config['output model type']=='combined':
                #plt.clf()
                fig, axes = plt.subplots(2, 3, sharey=True,sharex=True,dpi=400,tight_layout=True)
            elif plt.get_fignums():
                plt.clf()
                fig = plt.gcf()
            else:
                fig = plt.figure(figsize=(len(classes)*5+2, 6*5), dpi=100, tight_layout=True)

            canvas = FigureCanvas(fig) 

            for cl_idx, cl in enumerate(classes):

                if self.config['output model type']=='condensation' or self.config['output model type']=='combined':

                    target_first = outputs[0]['scatter_dict'][cl][0].cpu()
                    pred_first   = outputs[0]['scatter_dict'][cl][1].cpu()

                    cls_dict = {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2)}
                    ax = axes[cls_dict[cl_idx]]

                    ax.title.set_text('$\eta$-$\phi$ (class {})'.format(cl))
                    ax.scatter(target_first[:,1], target_first[:,2],s=20,color='green',marker='x')
                    ax.scatter(pred_first[:,1], pred_first[:,2],s=5,edgecolors='red',facecolors='none',marker='.')
                    ax.set_xlim([-np.pi, np.pi])
                    ax.set_ylim([-4, 4])

                else:
                    target = np.vstack([x['scatter_dict'][cl][0] for x in outputs if cl in x['scatter_dict'].keys()])
                    pred   = np.vstack([x['scatter_dict'][cl][1] for x in outputs if cl in x['scatter_dict'].keys()])

                    ax1 = fig.add_subplot(6, len(classes), cl_idx + 1)
                    ax1.scatter(target[:,0], pred[:,0], s=3)
                    ax1.set_title(cl + ' - log(pT)')
                    ax1.set_xlabel('Truth'); ax1.set_ylabel('Reco')

                    ax2 = fig.add_subplot(6, len(classes), cl_idx + len(classes) + 1)
                    ax2.scatter(target[:,1], pred[:,1], s=3)
                    ax2.set_title(cl + ' - eta')
                    ax2.set_xlabel('Truth'); ax2.set_ylabel('Reco')

                    ax3 = fig.add_subplot(6, len(classes), cl_idx + 2*len(classes) + 1)
                    ax3.scatter(target[:,2], pred[:,2], s=3)
                    ax3.set_title(cl + ' - phi')
                    ax3.set_xlabel('Truth'); ax3.set_ylabel('Reco')

                    ax4 = fig.add_subplot(6, len(classes), cl_idx + 3*len(classes) + 1)
                    del_phi = target[:,2] - pred[:,2]
                    del_phi[del_phi<-np.pi] = del_phi[del_phi<-np.pi] + 2*np.pi
                    del_phi[del_phi>np.pi]  = del_phi[del_phi>np.pi] - 2*np.pi
                    ax4.scatter(target[:,2], del_phi, s=3)
                    ax4.set_title(cl + ' - residual phi')
                    ax4.set_xlabel('truth phi'); ax4.set_ylabel('phi truth - phi reco')

                    if cl in self.num_particles_in_first_event.keys():
                        target_first = outputs[0]['scatter_dict'][cl][0][:self.num_particles_in_first_event[cl]]
                        pred_first   = outputs[0]['scatter_dict'][cl][1][:self.num_particles_in_first_event[cl]]

                        pT_truth, eta_truth, phi_truth = target_first[:,0], target_first[:,1], target_first[:,2]
                        pT_pred,  eta_pred,  phi_pred  = pred_first[:,0],   pred_first[:,1],   pred_first[:,2]

                        ax5 = fig.add_subplot(6, len(classes), cl_idx + 4*len(classes) + 1)
                        ax5.scatter(eta_truth, phi_truth, c='red', s=3, label='Truth')
                        ax5.scatter(eta_pred, phi_pred, c='blue', s=3, label='Reco')
                        ax5.set_title(cl + ' - event display')
                        ax5.set_xlabel('eta'); ax5.set_ylabel('phi')
                        ax5.legend()

                    ax6 = fig.add_subplot(6, len(classes), cl_idx + 5*len(classes) + 1)
                    cm = metrics.confusion_matrix(target[:,3], pred[:,3], labels=[0,1,2,3])
                    df_cm = pd.DataFrame(cm, index=['charged', 'iso e-', 'non-iso e-', 'muon'], columns=['charged', 'iso e-', 'non-iso e-', 'muon'])
                    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap=sn.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
                    ax6.set_title(cl + ' - classes')
                    ax6.set_xlabel('Truth'); ax6.set_ylabel('Reco')

            canvas.draw()
            w, h = fig.get_size_inches() * fig.get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)

            if self.comet_exp is not None:
                self.comet_exp.log_image(
                    image_data=image,
                    name='truth vs reco scatter',
                    overwrite=False, 
                    image_format="png",
                )

            if self.config['output model type'] == 'condensation':
                plt.savefig(self.config['scatter dir']+'/scatter_{}.png'.format(str(self.current_epoch)))
                plt.close(fig)
            elif self.config['output model type']=='combined':
                plt.savefig(self.config_condensation['scatter dir']+'/scatter_{}.png'.format(str(self.current_epoch)))
                plt.close(fig)
            else: plt.savefig('/storage/agrp/dreyet/PFlow/SCD/particle_flow/experiments/TSPN/scatter.png')
            #else: plt.savefig('/storage/agrp/nilotpal/PFlow/experiments/Set2Set/scatter.png')

        self.num_particles_in_first_event = {}

        return return_dict
        
        
