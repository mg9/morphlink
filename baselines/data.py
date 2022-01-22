import networkx as nx
import numpy as np
import torch
import json, pickle

torch.manual_seed(0)
np.random.seed(0)
def get_morph_graphs(max_num_nodes=-1):
    graphs = []
    dropped_graphs = 0
    real_sizes =[]
    # read gold morphs
    with open('data/goldstdsample.tur', 'r') as reader:
        word_and_morphs = dict()
        for line in reader:
            line = line.split('\t')[1].split(',')[0]
            word = line.strip().replace(' ', '')
            word_and_morphs[word] = []
            word_morphemes = line.strip().split(' ')
            prev_word = ''
            for i in range(len(word_morphemes)):
                morph = word_morphemes[i]
                subword = prev_word + morph
                word_and_morphs[word].append(subword)
                prev_word = subword

    with open ('data/features', 'rb') as fp:
        vae_features = pickle.load(fp)

    with open('data/probs2.json', 'r') as json_file:
        logps = json.load(json_file)
        logps = dict(sorted(logps.items())) # fix order

        if max_num_nodes == -1:
            max_num_nodes = max([len(_) for (word, _) in logps.items()])

        for (word, _) in logps.items():
            node_features = []
            g = nx.Graph()
            # do not make graph for no-morphemed words
            if len(word_and_morphs[word]) ==1:
                continue
            for key in _.keys():
                g.add_node(key)
                g.add_edge(key, key)
                node_features.append(vae_features[key])

            if len(g.nodes()) > max_num_nodes:
                dropped_graphs += 1
                continue

            for i in range(len(word_and_morphs[word])-1):
                src = word_and_morphs[word][i]
                tgt = word_and_morphs[word][i+1]
                g.add_edge(src, tgt)

            adj = nx.adjacency_matrix(g).toarray()
            num_nodes = adj.shape[0]
            real_sizes.append(num_nodes)
            adj_padded = np.zeros((max_num_nodes, max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
            adj_decoded = np.zeros(max_num_nodes * (max_num_nodes + 1) // 2)
            adj_vectorized = adj_padded[np.triu(np.ones((max_num_nodes, max_num_nodes)) ) == 1]
           
            
            features = np.identity(max_num_nodes)
            # make features as probabilities
            diags =[d for d in _.values()]
            for i in range(len(diags)):
                features[i][i] = np.exp(diags[i])
            features = torch.tensor(np.expand_dims(features, axis=0))
            
            adj_padded = torch.tensor(np.expand_dims(adj_padded, axis=0))
            adj_vectorized = torch.tensor(np.expand_dims(adj_vectorized, axis=0))
           
            node_features = torch.stack(node_features).squeeze(1)
            feat_dim = node_features.shape[-1]
            _features = np.ones((max_num_nodes, feat_dim))
            _features[:num_nodes,:feat_dim] = node_features.cpu()
            _features = torch.tensor(_features).unsqueeze(0)

            graphs.append({'adj':adj_padded,
            'adj_decoded':adj_vectorized, 
            'features':features,
            #'features':_features,
            'word':word,
            'g':g})

        print('Number of graphs removed due to upper-limit of number of nodes: ', dropped_graphs)
        print('total graph num: {}'.format(len(graphs)))
        print('real_sizes: ', real_sizes)
        return graphs

class GraphAdjSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_nodes, features='id'):
        self.max_num_nodes = max_num_nodes
        self.adj_all = []
        self.len_all = []
        self.feature_all = []

        for G in G_list:
            adj = nx.to_numpy_matrix(G)
            # the diagonal entries are 1 since they denote node probability
            self.adj_all.append(
                    np.asarray(adj) + np.identity(G.number_of_nodes()))
            self.len_all.append(G.number_of_nodes())
            if features == 'id':
                self.feature_all.append(np.identity(max_num_nodes))
            elif features == 'deg':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'struct':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()],
                                             'constant'),
                                      axis=1)
                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                self.feature_all.append(np.hstack([degs, clusterings]))

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0
        
        adj_vectorized = adj_padded[np.triu(np.ones((self.max_num_nodes,self.max_num_nodes)) ) == 1]
        # the following 2 lines recover the upper triangle of the adj matrix
        #recovered = np.zeros((self.max_num_nodes, self.max_num_nodes))
        #recovered[np.triu(np.ones((self.max_num_nodes, self.max_num_nodes)) ) == 1] = adj_vectorized
        #print(recovered)
        
        return {'adj':adj_padded,
                'adj_decoded':adj_vectorized, 
                'features':self.feature_all[idx].copy()}
