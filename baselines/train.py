# ref: https://raw.githubusercontent.com/JiaxuanYou/graph-generation/master/baselines/graphvae/train.py

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os, sys
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init 
from torch.autograd import Variable 
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from baselines.model import GraphVAE
from baselines.data import GraphAdjSampler, get_morph_graphs

CUDA = 0

LR_milestones = [500, 1000]
torch.manual_seed(0)
np.random.seed(0)

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "w")

  def write(self, message):
    #print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def build_model(args):
    max_num_nodes = args.max_num_nodes
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == 'id':
        input_dim = max_num_nodes
    elif args.feature_type == 'deg':
        input_dim = 1
    elif args.feature_type == 'struct':
        input_dim = 2

    input_dim = 32
    model = GraphVAE(input_dim, 64, 256, max_num_nodes)
    return model

def test(args, model, graphs):
    model.eval()
    epoch_loss = 0; epoch_recall = 0; epoch_prec = 0; 
    epoch_num_real_edges = 0; epoch_real_edge_recall = 0 ; epoch_real_edge_prec = 0
    for batch_idx, data in enumerate(graphs): 
        # data['adj']: (1,maxnode,maxnode), data['adj_decoded']: (1, upper triangle of adj), features: (1,maxnode,maxnode)
        g = data['g']
        features = data['features'].float()
        adj_input = data['adj'].float()
        features = Variable(features).cuda()
        unmasked_adj = adj_input.detach().clone()
        # clean adj_input for test
        num_nodes = len(g.nodes())
        _adj_input = torch.tensor(np.identity(num_nodes))
        adj_input[:,:num_nodes, :num_nodes] = _adj_input
        adj_input = Variable(adj_input).cuda()
        
        loss, acc, edge_recall, edge_prec, num_edges, num_real_edges, real_edge_recall, real_edge_prec = model(features, adj_input, g, args.logger, unmasked_adj)
        epoch_loss += loss.item()
        epoch_recall += edge_recall
        epoch_prec += edge_prec
        epoch_num_real_edges += num_real_edges
        epoch_real_edge_recall += real_edge_recall
        epoch_real_edge_prec += real_edge_prec

    epoch_loss = epoch_loss / len(graphs)
    epoch_recall = epoch_recall / len(graphs)
    epoch_prec = epoch_prec / len(graphs)
    epoch_real_edge_recall = epoch_real_edge_recall / len(graphs)
    epoch_real_edge_prec = epoch_real_edge_prec / len(graphs)
    print('VAL Loss: %.4f, epoch_num_real_edges: %d, epoch_real_edge_recall: %.4f, epoch_real_edge_prec: %.4f' % (epoch_loss, epoch_num_real_edges, epoch_real_edge_recall, epoch_real_edge_prec))


def train(args, model, graphs):
    graphs_train, graphs_test = graphs
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = 0; epoch_recall = 0; epoch_prec = 0
        epoch_num_real_edges = 0; epoch_real_edge_recall = 0 ; epoch_real_edge_prec = 0
        args.logger.write('\n')

        for batch_idx, data in enumerate(graphs_train): 
            # data['adj']: (1,maxnode,maxnode), data['adj_decoded']: (1, upper triangle of adj), features: (1,maxnode,maxnode)
            model.zero_grad()
            g = data['g']
            features = data['features'].float()
            adj_input = data['adj'].float()
            features = Variable(features).cuda()
            adj_input = Variable(adj_input).cuda()
            loss, acc, edge_recall, edge_prec, num_edges, num_real_edges, real_edge_recall, real_edge_prec = model(features, adj_input, g, args.logger)
            loss.backward()
            epoch_loss += loss.item()
            epoch_recall += edge_recall
            epoch_prec += edge_prec
            epoch_num_real_edges += num_real_edges
            epoch_real_edge_recall += real_edge_recall
            epoch_real_edge_prec += real_edge_prec
            optimizer.step()
            #scheduler.step()
        epoch_loss      = epoch_loss    / len(graphs_train)
        epoch_recall    = epoch_recall  / len(graphs_train)
        epoch_prec      = epoch_prec    / len(graphs_train)
        epoch_real_edge_recall  = epoch_real_edge_recall / len(graphs_train)
        epoch_real_edge_prec    = epoch_real_edge_prec   / len(graphs_train)

        print('\nEpoch: %d, Loss: %.7f, epoch_num_real_edges: %d, epoch_real_edge_recall: %.4f, epoch_real_edge_prec: %.4f' % (epoch, epoch_loss, epoch_num_real_edges, epoch_real_edge_recall, epoch_real_edge_prec))
        args.logger.write('\n Epoch %d, VAL ' %(epoch))
        test(args, model, graphs_test)
        
        model.train()

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,
            help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')

    parser.set_defaults(dataset='grid',
                        feature_type='id',
                        lr=0.001,
                        batch_size=1,
                        num_workers=1,
                        max_num_nodes=-1)
    return parser.parse_args()

def main():
    prog_args = arg_parse()
    prog_args.logger = Logger('train.log')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    prog_args.max_num_nodes = 10
    prog_args.epochs = 1000
    model = build_model(prog_args).cuda()
    print(model)
    graphs = get_morph_graphs(prog_args.max_num_nodes)

    graphs_len = len(graphs)
    graphs_train = graphs[:int(0.8 * graphs_len)]
    graphs_test = graphs[int(0.8 * graphs_len):]

    graphs = (graphs_train, graphs_test)
    print('total graph num: {}, training set: {}, test set: {}'.format(graphs_len, len(graphs_train), len(graphs_test)))
    train(prog_args, model, graphs)

if __name__ == '__main__':
    main()

