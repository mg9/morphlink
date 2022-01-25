
import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init

torch.manual_seed(0)
np.random.seed(0)

# a deterministic linear output (update: add noise)
class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size) # mu
        self.encode_12 = nn.Linear(h_size, embedding_size) # lsgms

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, 15) # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).cuda()

        if False:#self.training:
            z = eps*z_sgm + z_mu
        else:
            z = z_mu
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
    def forward(self, x, adj):
        #x = F.dropout(x, 0.1, self.training)
        y = torch.matmul(adj, x)
        #y = F.dropout(y, 0.3, self.training)
        y = torch.matmul(y,self.weight)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'

class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum'):
        '''
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        '''
        super(GraphVAE, self).__init__()
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(max_num_nodes)
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(max_num_nodes)

        self.act = nn.ReLU()

        self.input_dim = input_dim
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2

        self.vae = MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        self.max_num_nodes = max_num_nodes
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def permute_adj(self, adj, curr_ind, target_ind):
        ''' Permute adjacency matrix.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def pool_graph(self, x):
        if self.pool == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, input_features, adj_in, g, logger, adj_out):
        # input_features: (1, maxnode, maxnode),  adj: (1, maxnode, maxnode), unmasked_adj: ()
        adj_out = adj_out.squeeze(0)

        # (1, maxnode, hiddendim)
        x = self.conv1(input_features, adj_in)
        #x = self.bn1(x)
        x = self.act(x)
        # (1, maxnode, hiddendim)
        x = self.conv2(x, adj_in)
        #x = self.bn2(x)
        x = x.squeeze(0)

        recon_adj, z_mu, z_lsgms = self.vae(x)
        realgraphn =len(g.nodes)
        
        recon_adj = recon_adj[:realgraphn, :realgraphn]
        adj_out = adj_out[:realgraphn, :realgraphn]
        
        adj_recon_loss = self.adj_recon_loss(recon_adj, adj_out)
        loss = adj_recon_loss 

        # report accuracy etc.
        acc,  gold_real_edges_num, real_edge_correct, pred_real_edges_num  = self.real_adj_acc(recon_adj.cpu(), adj_out.cpu(), g, logger)
        return loss, acc, gold_real_edges_num, real_edge_correct, pred_real_edges_num

    def adj_recon_loss(self, adj_pred, adj_truth):
        return F.binary_cross_entropy_with_logits(adj_pred, adj_truth)


    def real_adj_acc(self, pred, target, g, logger):
        sft = nn.Softmax(1)
        target = target.squeeze(0)
        
        preds = torch.max(sft(pred),dim=1).indices
        golds = torch.max(sft(target),dim=1).indices
        total_acc = (sum(preds == golds)/ len(preds)).item()

        gold_real_edges = [(i!=golds[i]).item() for i in range(len(golds))]
        pred_real_edges = [(i!=preds[i]).item() for i in range(len(preds))]


        gold_real_edges_num = sum(gold_real_edges)
        pred_real_edges_num = sum(pred_real_edges)
        
        real_edge_correct   =  sum(torch.tensor(gold_real_edges) * (preds == golds)).item()
        real_edge_recall    =  real_edge_correct / gold_real_edges_num
        if pred_real_edges_num == 0:
            real_edge_precision = 0
        else:    
            real_edge_precision =  real_edge_correct / pred_real_edges_num

        pred_adj = np.zeros((len(preds), len(preds)))
        for i in range(len(preds)):
            if pred_real_edges[i]:
                pred_adj[i][preds[i]] = 1

        graph_nodes = [node for node in g.nodes()]
        pred_adj = pred_adj[:len(graph_nodes),:len(graph_nodes)]
        predicted_edges = []
        for i in range(len(pred_adj)):
            for j in range(len(pred_adj)):
                if pred_adj[i][j] == 1 and (i!=j):
                    edge = (graph_nodes[i], graph_nodes[j])
                    if (graph_nodes[j], graph_nodes[i]) not in predicted_edges:
                        predicted_edges.append(edge)

        target = target[:len(graph_nodes),:len(graph_nodes)]
        gold_edges = []
        for i in range(len(target)):
            for j in range(len(target)):
                if target[i][j] == 1 and (i!=j):
                    edge = (graph_nodes[i], graph_nodes[j])
                    if (graph_nodes[j], graph_nodes[i]) not in gold_edges:
                        gold_edges.append(edge)

        #if len(predicted_edges) == 0:
        #    breakpoint()
        logger.write('\n')
        logger.write('gold_edges:')
        logger.write(gold_edges)
        logger.write('\npred_edges:')
        logger.write(predicted_edges)
        logger.write('\n')

        if np.isnan(real_edge_recall):
            breakpoint()

        return total_acc, gold_real_edges_num, real_edge_correct, pred_real_edges_num #real_edge_recall, real_edge_precision

    def real_adj_acc_old(self, pred, target, g, logger):
        pred = (pred > 0.5).to(torch.float32)
        target = target.squeeze(0)
        total_correct = torch.sum(pred == target).item()
        acc = total_correct/ (target.shape[0] * target.shape[1])
        pred_diagonal   = torch.diagonal(pred)
        target_diagonal = torch.diagonal(target)
       
        edge_recall = torch.sum((pred == target) * (target==1)) / torch.sum((target==1))
        edge_precision = torch.sum((pred == target) * (pred==1)) / torch.sum((pred==1))
        num_edges = torch.sum(target).item()
        num_diagonal = torch.sum(target_diagonal)
        
        correct_edge_except_diagonal = torch.sum((pred == target) * (target==1)) - torch.sum((pred_diagonal == target_diagonal) * (target_diagonal==1))
        total_edge_except_diagonal = torch.sum(target) - torch.sum(target_diagonal)
        real_edge = num_edges - num_diagonal

        if correct_edge_except_diagonal.item() == 0:
            real_edge_precision = 0 
            real_edge_recall = 0
        else:
            real_edge_precision = correct_edge_except_diagonal / (torch.sum((pred==1)) - torch.sum(pred_diagonal))
            real_edge_recall = correct_edge_except_diagonal / total_edge_except_diagonal

        graph_nodes = [node for node in g.nodes()]
        pred = pred[:len(graph_nodes),:len(graph_nodes)]
        predicted_edges = []
        for i in range(len(pred)):
            for j in range(len(pred)):
                if pred[i][j] == 1 and (i!=j):
                    edge = (graph_nodes[i], graph_nodes[j])
                    if (graph_nodes[j], graph_nodes[i]) not in predicted_edges:
                        predicted_edges.append(edge)

        target = target[:len(graph_nodes),:len(graph_nodes)]
        gold_edges = []
        for i in range(len(target)):
            for j in range(len(target)):
                if target[i][j] == 1 and (i!=j):
                    edge = (graph_nodes[i], graph_nodes[j])
                    if (graph_nodes[j], graph_nodes[i]) not in gold_edges:
                        gold_edges.append(edge)


        logger.write('\n')
        logger.write('gold_edges:')
        logger.write(gold_edges)
        logger.write('\npred_edges:')
        logger.write(predicted_edges)
        logger.write('\n')

        if np.isnan(real_edge_recall):
            breakpoint()

        return acc, edge_recall, edge_precision, num_edges, real_edge, real_edge_recall, real_edge_precision