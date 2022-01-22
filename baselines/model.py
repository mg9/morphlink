
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
        self.decode_2 = nn.Linear(embedding_size, y_size) # make edge prediction (reconstruct)
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
        #x = F.dropout(x, 0.2, self.training)
        y = torch.matmul(adj, x)
        #y = F.dropout(y, 0.3, self.training)
        y = torch.matmul(y,self.weight)
        return y

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
        #self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn1 = nn.BatchNorm1d(max_num_nodes)
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        #self.bn2 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(max_num_nodes)

        self.act = nn.ReLU()

        self.input_dim = input_dim
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2

        self.vae = MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        #self.vae = MLP_VAE_plain(input_dim * input_dim, latent_dim, output_dim)
        #self.vae = MLP_VAE_plain(input_dim * max_num_nodes, latent_dim, output_dim)

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

    def forward(self, input_features, adj, g, logger, unmasked_adj=None):
        # input_features: (1, maxnode, maxnode),  adj: (1, maxnode, maxnode)

        # (1, maxnode, hiddendim)
        x = self.conv1(input_features, adj)
        x = self.bn1(x)
        x = self.act(x)
        # (1, maxnode, hiddendim)
        x = self.conv2(x, adj)
        x = self.bn2(x)

        # pool over all nodes (1,hiddendim)
        graph_h = self.pool_graph(x)
        # (1, maxnode * maxnode)
        # graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes)
        #graph_h = input_features.view(-1, self.max_num_nodes * self.input_dim)

        # h_decode: (1, upper triangle of adj), z_mu: (1, 256), z_lsgms: (1,256)
        h_decode, z_mu, z_lsgms = self.vae(graph_h)
        
        # out: (1, upper triangle of adj)
        out = F.sigmoid(h_decode)
        out_tensor = out.cpu().data
        recon_adj_lower = self.recover_adj_lower(out_tensor)
        
        # (maxnode, maxnode)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)
        adj_data = adj.cpu().data[0]
        adj_permuted = adj_data
        
        # report accuracy etc.
        acc, edge_recall, edge_prec, num_edges, num_real_edges, real_edge_recall, real_edge_precision = self.real_adj_acc(recon_adj_tensor, adj.cpu(), g, logger, unmasked_target=unmasked_adj)

        # (upper triangle of adj)
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()
        adj_vectorized_var = Variable(adj_vectorized).cuda()

        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes # normalize
        loss = adj_recon_loss #+ loss_kl

        return loss, acc, edge_recall, edge_prec, num_edges, num_real_edges, real_edge_recall, real_edge_precision

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_pred, adj_truth)

    def real_adj_acc(self, pred, target, g, logger, unmasked_target=None):
        if unmasked_target is not None:
            target = unmasked_target
            pred = (pred > 0.5).to(torch.float32)
        else:
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


        if unmasked_target is not None:
            logger.write('\n')
            logger.write('gold_edges:')
            logger.write(gold_edges)
            logger.write('\npred_edges:')
            logger.write(predicted_edges)
            logger.write('\n')

        if np.isnan(real_edge_recall):
            breakpoint()

        return acc, edge_recall, edge_precision, num_edges, real_edge, real_edge_recall, real_edge_precision