
import torch
import torch.nn as nn
from dataloader import EdgeLabelFunction
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

def f1_loss(g):


    g.edata['y hat'] = torch.sigmoid(g.edata['edge prediction'])
    g.edata['tp'] = g.edata['y hat']*g.edata['edge label']
    g.edata['fn'] = (1.0-g.edata['y hat'])*g.edata['edge label']
    g.edata['fp'] = g.edata['y hat']*(1.0-g.edata['edge label'])


    tp = dgl.sum_edges(g,'tp')
    
    fn =  dgl.sum_edges(g,'fn')
    fp =  dgl.sum_edges(g,'fp')

    
    loss = - ((2 * tp) / (2 * tp + fp + fn + 1e-10)).mean()
    
    return loss


class EdgeLossBCE(nn.Module):
    def __init__(self):
        super(EdgeLossBCE, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self,g):

        g.edata['bce loss'] = self.BCE( g.edata['edge prediction'], g.edata['edge label'] )
        
        graph_bce = dgl.mean_edges(g, 'bce loss')
        #print('--------')
        #print(torch.sigmoid( g.edata['edge prediction'][:10]) )
        #print(g.edata['edge label'][:10])
        #print('--------')
        return torch.mean(graph_bce,dim=0)
        



class VertexFindingLoss(nn.Module):
    def __init__(self,config):
        super(VertexFindingLoss, self).__init__()

        self.config = config

        self.node_loss = nn.CrossEntropyLoss()
        
        self.edge_BCE = EdgeLossBCE()
        

        
    def forward(self, g, hetro_g):

        n_prediction = g.ndata['node prediction']
        n_label = g.ndata['node labels']

        g.apply_edges(EdgeLabelFunction)

        node_loss = self.node_loss(n_prediction,n_label)
        edge_bce = self.edge_BCE(g)
        edge_f1 = f1_loss(g)
        loss = node_loss*self.config['loss weights']['node loss'] + edge_bce*self.config['loss weights']['edge bce']+ edge_f1*self.config['loss weights']['edge f1']
        
        return {'loss':loss,'node loss':node_loss.item(),'edge bce' : edge_bce.item(), 'edge f1': edge_f1.item() }


class JetClassificationLoss(nn.Module):
    def __init__(self,config):
        super(JetClassificationLoss, self).__init__()

        self.config = config

        self.jet_loss = nn.CrossEntropyLoss()
        

        
    def forward(self, predictions,targets):

        loss = self.jet_loss(predictions,targets)
        
        return { 'loss':loss }


