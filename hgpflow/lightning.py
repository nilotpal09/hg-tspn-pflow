import comet_ml

import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torch.nn.functional as F
import sys

from pflow_model import PflowModel

import numpy as np

import metrics
import misc
from numpy.random import default_rng

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix

import gc

from dataloader import PflowDataset, collate_graphs, PflowSampler

# Miscellaneous
SEED = 123456
RNG = default_rng(SEED)
# pl.seed_everything(SEED)

N_RAY = 0
if N_RAY > 0:
    ray.init(num_cpus=N_RAY,include_dashboard=False)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)


class PflowLightning(LightningModule):

    def __init__(self, config, comet_exp=None):
        super().__init__()
        
        self.config = config
        self.frozen = False
        if 'frozen' in self.config:
            self.frozen = self.config['frozen']

        self.N_BPTT  = self.config['N_BPTT']
        self.T_BPTT  = self.config['T_BPTT']
        self.T_TOTAL = self.config['T_TOTAL']

        self.indicator_threshold = self.config['indicator_threshold']
        self.inc_only_epochs = self.config['inc_only_epochs']

        self.acc_grad_iter = self.config['effective_batchsize'] // self.config['batchsize']

        self.net = PflowModel(self.config, self.debug)
        # self.net.apply(init_weights)

        if config['output model type']=='hypergraph':
            self.sampler = misc.IntegerPartitionSampler(self.T_TOTAL-self.T_BPTT*self.N_BPTT, self.N_BPTT, RNG)
        self.automatic_optimization = False
        self.comet_exp = comet_exp

        metrics.update_global_vars(config)
        self.pt_mean = config['var transform']['particle_pt']['mean']
        self.pt_std  = config['var transform']['particle_pt']['std']
        self.eta_mean = config['var transform']['particle_eta']['mean']
        self.eta_std  = config['var transform']['particle_eta']['std']
        self.phi_mean = config['var transform']['particle_phi']['mean']
        self.phi_std  = config['var transform']['particle_phi']['std']


    def set_comet_exp(self, comet_exp):
        self.comet_exp = comet_exp


    def forward(self, g):
        if self.debug == False:
            self.net.track_and_cell_encoder(g)        
        self.net.outputnet.init_features(g)
        inc_preds, g = self.net.outputnet(g, t_skip=self.T_TOTAL-1, t_bp=1)
        hyperedge_pred, g = self.net.hyperedgenet(g)
        return inc_preds, hyperedge_pred, g

    
    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)

        g, target, ptetaphi_truth, class_truth, has_track = batch
        target = torch.cat([target, target.bool().any(-1, keepdim=True).float()], dim=-1)

        bs = g.batch_size
        opt = self.optimizers()
        opt.zero_grad()

        loss_per_upd = []

        if self.debug == False:
            self.net.track_and_cell_encoder(g)        
        self.net.outputnet.init_features(g)

        # if frozen hypergraph
        if self.frozen:
            preds, g = self.net.outputnet(g, t_skip=self.T_TOTAL-1, t_bp=1)
            g.nodes['pflow_particles'].data['features'] = g.nodes['pflow_particles'].data['features'].detach()
            g.nodes['nodes'].data['features'] = g.nodes['nodes'].data['features'].detach()
            g.nodes['nodes'].data['features_0'] = g.nodes['nodes'].data['features_0'].detach()
            g.edges['pflow_particle_to_node'].data['incidence_val'] = g.edges['pflow_particle_to_node'].data['incidence_val'].detach()
            g.edges['node_to_pflow_particle'].data['incidence_val'] = g.edges['node_to_pflow_particle'].data['incidence_val'].detach()

            hyperedge_pred, g = self.net.hyperedgenet(g)

            preds_ = preds[-1].detach()
            indicator_ = preds_[:,:,-1].detach()
            targets = target.repeat(self.T_BPTT, 1, 1)    

            with torch.no_grad():
                _, indices = metrics.LAP_loss(
                    preds_, target.detach(), has_track, n=min(N_RAY, target.size(0)), get_assignment=True, assgn_particles=-bs, bool_inc=self.config['bool_inc'])

            indicator  = torch.zeros_like(indicator_)

            regression_pred_, class_pred_ = hyperedge_pred
            regression_pred = torch.zeros_like(regression_pred_)

            class_pred_charged_, class_pred_neutral_, charged_mask_, neutral_mask_ = class_pred_
            class_pred_charged, class_pred_neutral = torch.zeros_like(class_pred_charged_), torch.zeros_like(class_pred_neutral_)
            charged_mask, neutral_mask = torch.zeros_like(charged_mask_), torch.zeros_like(neutral_mask_)
            for i in range(indices.shape[0]):
                regression_pred[i] = regression_pred_[i][indices[i]]
                class_pred_charged[i] = class_pred_charged_[i][indices[i]]
                class_pred_neutral[i] = class_pred_neutral_[i][indices[i]]
                charged_mask[i] = charged_mask_[i][indices[i]]
                neutral_mask[i] = neutral_mask_[i][indices[i]]
                indicator[i]    = indicator_[i][indices[i]]

            hyperedge_loss_charged = metrics.hyperedge_loss(
                ptetaphi_truth, class_truth, regression_pred, class_pred_charged, indicator, mask=charged_mask, class_weights=[1,40,500])
            hyperedge_loss_neutral = metrics.hyperedge_loss(
                ptetaphi_truth, class_truth, regression_pred, class_pred_neutral, indicator, mask=neutral_mask, class_weights=[3,1])
            hyperedge_loss = hyperedge_loss_charged + hyperedge_loss_neutral
            loss = hyperedge_loss

            self.manual_backward(loss)
            if (batch_idx+1)%self.acc_grad_iter == 0:
                opt.step()
                opt.zero_grad()

            with torch.no_grad():
                logs = {
                    "loss": hyperedge_loss.item(),
                    "hyperedge_loss": hyperedge_loss.item()}
                self.log_dict({f"train/{k}":v for k,v in logs.items()})


        # if not frozen hypergraph
        else:
            t_pre = self.sampler()

            bptt_i = 1
            for t in t_pre:
                preds, g = self.net.outputnet(g, t_skip=t, t_bp=self.T_BPTT)

                # save the last one
                if bptt_i == self.N_BPTT:
                    preds_ = preds[-1].detach()

                preds = torch.cat(preds, dim=0)
                targets = target.repeat(self.T_BPTT, 1, 1)

                loss_per_t = metrics.LAP_loss(
                    preds, 
                    targets,
                    has_track, 
                    n=min(N_RAY, bs),
                    bool_inc=self.config['bool_inc'])
                loss_inc = loss_per_t.mean(0)

                preds = preds.clamp(0,1)
                indicator_ = preds[-bs:,:,-1].detach()

                pred_adj = torch.bmm(preds[...,:-1].clone().transpose(1,2), preds[...,:-1].clone())
                target_adj = torch.bmm(targets[...,:-1].clone().transpose(1,2), targets[...,:-1].clone())
                f1 = metrics.f1_score(target_adj, pred_adj, type="adj")

                loss = loss_inc - f1.mean(0)
                loss_ = loss.detach()

                self.manual_backward(loss)

                if (batch_idx+1)%self.acc_grad_iter == 0:
                    opt.step()
                    opt.zero_grad()
                    loss_per_upd.append(loss_.detach().cpu().data)

                g.nodes['pflow_particles'].data['features'] = g.nodes['pflow_particles'].data['features'].detach()
                g.nodes['nodes'].data['features'] = g.nodes['nodes'].data['features'].detach()
                g.nodes['nodes'].data['features_0'] = g.nodes['nodes'].data['features_0'].detach()
                g.edges['pflow_particle_to_node'].data['incidence_val'] = g.edges['pflow_particle_to_node'].data['incidence_val'].detach()
                # g.edges['node_to_pflow_particle'].data['incidence_val'] = g.edges['node_to_pflow_particle'].data['incidence_val'].detach()

                if bptt_i == len(t_pre):
                    g.edges['node_to_pflow_particle'].data['incidence_val'] = g.edges['node_to_pflow_particle'].data['incidence_val'].detach()

                bptt_i += 1

            if (self.current_epoch > self.inc_only_epochs):
                hyperedge_pred, g = self.net.hyperedgenet(g)
                with torch.no_grad():
                    _, indices = metrics.LAP_loss(
                        preds_, target.detach(), has_track, n=min(N_RAY, target.size(0)), get_assignment=True, assgn_particles=-bs, bool_inc=self.config['bool_inc'])

                indicator  = torch.zeros_like(indicator_)

                regression_pred_, class_pred_ = hyperedge_pred
                regression_pred = torch.zeros_like(regression_pred_)
                
                class_pred_charged_, class_pred_neutral_, charged_mask_, neutral_mask_ = class_pred_
                class_pred_charged, class_pred_neutral = torch.zeros_like(class_pred_charged_), torch.zeros_like(class_pred_neutral_)
                charged_mask, neutral_mask = torch.zeros_like(charged_mask_), torch.zeros_like(neutral_mask_)
                for i in range(indices.shape[0]):
                    regression_pred[i] = regression_pred_[i][indices[i]]
                    class_pred_charged[i] = class_pred_charged_[i][indices[i]]
                    class_pred_neutral[i] = class_pred_neutral_[i][indices[i]]
                    charged_mask[i] = charged_mask_[i][indices[i]]
                    neutral_mask[i] = neutral_mask_[i][indices[i]]
                    indicator[i]    = indicator_[i][indices[i]]

                hyperedge_loss_charged = metrics.hyperedge_loss(
                    ptetaphi_truth, class_truth, regression_pred, class_pred_charged, indicator, mask=charged_mask, class_weights=[1,40,500])
                hyperedge_loss_neutral = metrics.hyperedge_loss(
                    ptetaphi_truth, class_truth, regression_pred, class_pred_neutral, indicator, mask=neutral_mask, class_weights=[3,1])
                hyperedge_loss = hyperedge_loss_charged + hyperedge_loss_neutral

                self.manual_backward(hyperedge_loss)

                if (batch_idx+1)%self.acc_grad_iter == 0:
                    opt.step()
                    opt.zero_grad()

            with torch.no_grad():
                if (self.current_epoch <= self.inc_only_epochs):
                    logs = {
                        "loss": np.mean(loss_per_upd),
                        "f1": f1[-bs:].mean(0),
                        **{f"loss_at{i}": l for i,l in enumerate(loss_per_upd)}}
                else:
                    logs = {
                        "loss": np.mean(loss_per_upd) + hyperedge_loss.item(),
                        "hyperedge_loss": hyperedge_loss.item(),
                        "f1": f1[-bs:].mean(0),
                        **{f"loss_at{i}": l for i,l in enumerate(loss_per_upd)}}                
                self.log_dict({f"train/{k}":v for k,v in logs.items()})
 
        # memory cleanup
        # del g, target, preds, targets #, pred_adj, target_adj #, indices
        # gc.collect()

        return loss
        

    def validation_step(self, batch, batch_idx):
        inputs, target, ptetaphi_truth, class_truth, has_track = batch
        target = torch.cat([target, target.bool().any(-1, keepdim=True).float()], dim=-1)

        bs = inputs.batch_size
        inc_preds, hyperedge_pred, _ = self(inputs)
        pred = inc_preds[-1]

        target_adj = torch.bmm(target[...,:-1].transpose(1,2), target[...,:-1])
        pred_adj = torch.bmm(pred[...,:-1].transpose(1,2), pred[...,:-1])

        if self.config['bool_inc']:
            bce_loss = metrics.LAP_loss(pred, target, has_track, n=min(N_RAY, target.size(0)), bool_inc=self.config['bool_inc']).mean(0)
        else:
            bce_loss, ind_loss, inc_loss = metrics.LAP_loss(
                pred, target, has_track, n=min(N_RAY, target.size(0)), bool_inc=self.config['bool_inc'], get_individual_losses=True)
            bce_loss, ind_loss, inc_loss = bce_loss.mean(0), ind_loss.mean(0), inc_loss.mean(0)

        f1 = metrics.f1_score(target_adj, pred_adj, type="adj").mean(0)

        with torch.no_grad():
            _, indices = metrics.LAP_loss(pred, target.detach(), has_track, n=min(N_RAY, target.size(0)), 
                                        get_assignment=True, assgn_particles=-bs, bool_inc=self.config['bool_inc'])

        indicator_ = pred[:,:,-1]
        indicator_bool_ = pred[:,:,-1] < self.indicator_threshold
        regression_pred_, class_pred_ = hyperedge_pred

        indicator = torch.zeros_like(indicator_)
        indicator_bool = torch.zeros_like(indicator_bool_).bool()
        regression_pred = torch.zeros_like(regression_pred_)

        # needed to compute metric for incidence prediction
        pred_incidence = torch.zeros_like(pred)

        class_pred_charged_, class_pred_neutral_, charged_mask_, neutral_mask_ = class_pred_
        class_pred_charged, class_pred_neutral = torch.zeros_like(class_pred_charged_), torch.zeros_like(class_pred_neutral_)
        charged_mask, neutral_mask = torch.zeros_like(charged_mask_), torch.zeros_like(neutral_mask_)

        for i in range(indices.shape[0]):
            regression_pred[i] = regression_pred_[i][indices[i]]
            class_pred_charged[i] = class_pred_charged_[i][indices[i]]
            class_pred_neutral[i] = class_pred_neutral_[i][indices[i]]
            charged_mask[i] = charged_mask_[i][indices[i]]
            neutral_mask[i] = neutral_mask_[i][indices[i]]
            indicator[i]      = indicator_[i][indices[i]]
            indicator_bool[i] = indicator_bool_[i][indices[i]]

            pred_incidence[i] = pred[i][indices[i]]


        hyperedge_loss_charged = metrics.hyperedge_loss(
            ptetaphi_truth, class_truth, regression_pred, class_pred_charged, indicator, mask=charged_mask, class_weights=[1,40,500])
        hyperedge_loss_neutral = metrics.hyperedge_loss(
            ptetaphi_truth, class_truth, regression_pred, class_pred_neutral, indicator, mask=neutral_mask, class_weights=[3,1])
        hyperedge_loss = hyperedge_loss_charged + hyperedge_loss_neutral

        # for plotting later
        class_pred_neutral = torch.cat([
            class_pred_neutral, torch.zeros((class_pred_neutral.shape[0], class_pred_neutral.shape[1], 1), device=class_pred_neutral.device) - 1
        ], dim=-1)
        class_pred = class_pred_charged * charged_mask + class_pred_neutral * neutral_mask

        logs = {
            "loss": bce_loss - f1 + hyperedge_loss,
            "bce": bce_loss,
            "f1": f1,
            "hyperedge_loss": hyperedge_loss
        }

        if self.config['bool_inc'] == False:
            logs['ind_loss'] = ind_loss
            logs['inc_loss'] = inc_loss

        # undo_scalling
        regression_pred = self.undo_scalling(regression_pred)
        ptetaphi_truth  = self.undo_scalling(ptetaphi_truth, ignore_defaults=True)

        regression_pred, class_pred = regression_pred.cpu(), class_pred.cpu()
        remapping = neutral_mask.squeeze(-1).detach().cpu().data * 3

        # just so that we can remove these points (truth and reco both being garbage) from the plots. inference code doesn't do this
        regression_pred[indicator_bool] = torch.FloatTensor([0,0,0], device=regression_pred.device)

        plot_data = {}
        if batch_idx == 0:
            _, indices_one = metrics.LAP_loss(pred, target, has_track, n=min(N_RAY, target.size(0)),
                get_assignment=True, assgn_particles=-1, bool_inc=self.config['bool_inc'])
            plot_data["inc_metrices"] = (target[-1], pred[-1][indices_one,:])

        # incidence prediction metrices

        # for the indicator
        truth_ind = target[:,:,-1]
        pred_ind  = pred_incidence[:,:,-1]
        logs['ind_mean'] = torch.abs(truth_ind - pred_ind).mean()
        logs['ind_std']  = torch.abs(truth_ind - pred_ind).std()

        # for all the incidence score diffs
        truth_inc = target[:,:,:-1]
        pred_inc  = pred_incidence[:,:,:-1]
        logs['inc_mean'] = torch.abs(truth_inc - pred_inc).mean()
        logs['inc_std']  = torch.abs(truth_inc - pred_inc).std()

        # for all the incidence score diffs, with truth > 0.01 (mean, sigma)
        mask = truth_inc > 0.01
        logs['inc_mean_1'] = torch.abs(truth_inc[mask] - pred_inc[mask]).mean()
        logs['inc_std_1']  = torch.abs(truth_inc[mask] - pred_inc[mask]).std()

        # for all the incidence score diffs, with truth < 0.01 (mean, sigma)
        mask = truth_inc < 0.01
        logs['inc_mean_2'] = torch.abs(truth_inc[mask] - pred_inc[mask]).mean()
        logs['inc_std_2']  = torch.abs(truth_inc[mask] - pred_inc[mask]).std()

        # keep the pflow objects with indicator > 0.5 (not doing it here)
        plot_data["ptetaphi"] = (ptetaphi_truth.detach().cpu().data, regression_pred.detach().cpu().data)
        plot_data["classes"]  = (class_truth.detach().cpu().data + remapping, torch.argmax(class_pred, dim=-1).detach().cpu().data + remapping)

        # memory cleanup
        del inputs, target, pred, target_adj, pred_adj, 
        ptetaphi_truth, class_truth, regression_pred_, class_pred_, regression_pred, class_pred
        gc.collect()

        self.log_dict({f"val/{k}":v for k,v in logs.items()})
        return {"logs": logs, "plot_data": plot_data}


    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.config['learningrate'])

        if self.config['lr_scheduler'] == None:
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config['lr_scheduler']['T_max'], 
                eta_min=self.config['lr_scheduler']['eta_min'],
                last_epoch=self.config['lr_scheduler']['last_epoch'],
                verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
 

    def train_dataloader(self):
        if 'reduce_ds_train' in self.config:
            reduce_ds = self.config['reduce_ds_train']
        else:
            reduce_ds = -1
        dataset = PflowDataset(self.config['path_to_train'], self.config, bool_inc=self.config['bool_inc'], reduce_ds=reduce_ds)

        batch_sampler = PflowSampler(dataset.n_nodes, batch_size=self.config['batchsize'])
        loader = DataLoader(dataset, num_workers=self.config['num_workers'], 
            collate_fn=collate_graphs, batch_sampler=batch_sampler, pin_memory=False)

        return loader

    
    def val_dataloader(self):
        if 'reduce_ds_val' in self.config:
            reduce_ds = self.config['reduce_ds_val']
        else:
            reduce_ds = -1
        dataset = PflowDataset(self.config['path_to_valid'], self.config, bool_inc=self.config['bool_inc'], reduce_ds=reduce_ds)        

        batch_sampler = PflowSampler(dataset.n_nodes, batch_size=self.config['batchsize'])
        loader = DataLoader(dataset, num_workers=self.config['num_workers'], 
            collate_fn=collate_graphs, batch_sampler=batch_sampler, pin_memory=False)

        return loader


    def training_epoch_end(self, outputs):
        self.lr_schedulers().step()


    def validation_epoch_end(self, outputs):

        logs = {}
        for key in outputs[0]['logs'].keys():
            avg_loss = torch.stack([x['logs'][key] for x in outputs]).mean()
            logs[key] = avg_loss

        self.log('lr', self.lr_schedulers().get_lr()[0])

        # scatter plot
        if plt.get_fignums():
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(12, 16), dpi=100, tight_layout=True)

        canvas = FigureCanvas(fig) 

        truth_inc = outputs[0]['plot_data']['inc_metrices'][0].cpu().T
        pred_inc  = outputs[0]['plot_data']['inc_metrices'][1].cpu().T.squeeze()

        num_inc = len(truth_inc)
        if self.config['frozen'] == False:
            axs = [None]*num_inc
            for i in range(num_inc):
                axs[i] = fig.add_subplot(np.ceil(num_inc/3).astype(int), 3, i+1)
                axs[i].plot(truth_inc[i], label='True', color='r')
                axs[i].plot(pred_inc[i], label='Pred', color='cyan')
                axs[i].legend()

            canvas.draw()
            w, h = fig.get_size_inches() * fig.get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)

            if self.comet_exp is not None:
                self.comet_exp.log_image(
                    image_data=image,
                    name='incidence performance',
                    overwrite=False, 
                    image_format="png",
                )
            else:
                plt.savefig('plot_incidence.png')
            plt.clf()

        if self.config['frozen'] or self.current_epoch >= self.inc_only_epochs:

            # truth incidence matrix
            ax1 = fig.add_subplot(4, 3, 1)
            im1 = ax1.imshow(truth_inc, vmin=0, vmax=1, aspect='auto', interpolation='none', cmap='plasma')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')
            ax1.set_title('truth incidence')
            ax1.set_xlabel('particles'); ax1.set_ylabel('nodes')

            # pred incidence matrix
            ax2 = fig.add_subplot(4, 3, 2)
            # im2 = ax2.imshow(pred_inc, vmin=0, vmax=1, aspect='auto', interpolation='none', cmap='plasma')
            im2 = ax2.imshow(pred_inc, aspect='auto', interpolation='none', cmap='plasma')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax, orientation='vertical')
            ax2.set_title('pred incidence')
            ax2.set_xlabel('pflow particles'); ax2.set_ylabel('nodes')

            # class confusion
            class_target = np.vstack([x['plot_data']['classes'][0] for x in outputs]).flatten()
            class_pred   = np.vstack([x['plot_data']['classes'][1] for x in outputs]).flatten()

            ax3 = fig.add_subplot(4, 3, 3)
            cm = confusion_matrix(class_target, class_pred, labels=[0,1,2,3,4])
            df_cm = pd.DataFrame(cm, index=['ch had', 'electron', 'muon', 'neut had', 'photon'],
                columns=['ch had', 'electron', 'muon', 'neut had', 'photon'])
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap=sn.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), cbar=False)
            ax3.set_title('classes')
            ax3.set_xlabel('Reco'); ax3.set_ylabel('Truth')


            # pt eta phi plots
            ptetaphi_target_ = np.vstack([np.vstack(x['plot_data']['ptetaphi'][0]) for x in outputs])
            ptetaphi_pred_   = np.vstack([np.vstack(x['plot_data']['ptetaphi'][1]) for x in outputs])

            # non_garbage_mask = (ptetaphi_target[:,0] != 0) + (ptetaphi_pred[:,0] != 0)
            # ptetaphi_target  = ptetaphi_target[non_garbage_mask]
            # ptetaphi_pred    = ptetaphi_pred[non_garbage_mask]


            # charged hadron: 0, electron: 1, muon: 2, neutral hadronphoton: 3, photon: 4
            for idx, plot_type in enumerate(["charged", "neutral", "photon", ]):
                if plot_type == "charged":
                    class_mask = ((class_target == 0) + (class_target == 1) + (class_target == 2)).astype(bool)
                elif plot_type == "photon":
                    class_mask = (class_target == 4).astype(bool)
                elif plot_type == "neutral":
                    class_mask = (class_target == 3).astype(bool)

                ptetaphi_target = ptetaphi_target_[class_mask]
                ptetaphi_pred   = ptetaphi_pred_[class_mask]

                non_garbage_mask = (ptetaphi_target[:,0] != 0) + (ptetaphi_pred[:,0] != 0)
                ptetaphi_target  = ptetaphi_target[non_garbage_mask]
                ptetaphi_pred    = ptetaphi_pred[non_garbage_mask]

                ax4 = fig.add_subplot(4, 3, 3*idx+4)
                mask_pT_ = (ptetaphi_target[:,0] != 0) * (ptetaphi_pred[:,0] != 0)
                self.fancy_scatter(fig, ax4, ptetaphi_target[:,0][mask_pT_], ptetaphi_pred[:,0][mask_pT_])
                ax4.set_title(plot_type + ' log(pT) -  zoomed in')
                ax4.set_xlabel('Truth'); ax4.set_ylabel('Reco')

                ax5 = fig.add_subplot(4, 3, 3*idx+5)
                self.fancy_scatter(fig, ax5, ptetaphi_target[:,1], ptetaphi_pred[:,1])
                ax5.set_title(plot_type + ' eta')
                ax5.set_xlabel('Truth'); ax5.set_ylabel('Reco')

                ax6 = fig.add_subplot(4, 3, 3*idx+6)
                self.fancy_scatter(fig, ax6, ptetaphi_target[:,2], ptetaphi_pred[:,2])
                ax6.set_title(plot_type + ' phi')
                ax6.set_xlabel('Truth'); ax6.set_ylabel('Reco')

            canvas.draw()
            w, h = fig.get_size_inches() * fig.get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)

            if self.comet_exp is not None:
                self.comet_exp.log_image(
                    image_data=image,
                    name='particle performance',
                    overwrite=False, 
                    image_format="png",
                )
            else:
                plt.savefig('plot_particles.png')

    def fancy_scatter(self, fig, ax, x, y):
        xy = np.vstack([x, y])
        try:
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            im = ax.scatter(x, y, s=3, c=z, cmap="cool")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        except:
            ax.scatter(x, y, s=3, c="cyan")            

    def undo_scalling(self, ptetaphi, ignore_defaults=False):
        default_value = 0
        if ignore_defaults == True:
            mask = (ptetaphi[:,:,0]==default_value) * (ptetaphi[:,:,1]==default_value) * (ptetaphi[:,:,2]==default_value)
        pt  = (ptetaphi[:,:,0] * self.pt_std  + self.pt_mean).unsqueeze(-1) 
        eta = (ptetaphi[:,:,1] * self.eta_std + self.eta_mean).unsqueeze(-1)
        phi = (ptetaphi[:,:,2] * self.phi_std + self.phi_mean).unsqueeze(-1)
     
        if ignore_defaults == True:
            pt[mask]=default_value; eta[mask]=default_value; phi[mask]=default_value

        return torch.cat([pt, eta, phi], dim=-1)
