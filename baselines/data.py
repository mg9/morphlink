import networkx as nx
import numpy as np
import torch
import json, pickle, math

torch.manual_seed(0)
np.random.seed(0)
def get_morph_graphs(max_num_nodes=-1):
    graphs = []
    dropped_graphs = 0
    real_sizes =[]
    # read gold morphs
    with open('data/goldstd_mc05-10aggregated.segments.tur', 'r') as reader:
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

    with open ('data/features2', 'rb') as fp:
        vae_features = pickle.load(fp)

    with open('data/probs.json', 'r') as json_file:
        logps = json.load(json_file)
        logps = dict(sorted(logps.items())) # fix order

        if max_num_nodes == -1:
            max_num_nodes = max([len(_) for (word, _) in logps.items()])

        for (word, _) in logps.items():
            node_features = []
            g_in = nx.Graph()
            g_out = nx.DiGraph()
            # do not make graph for no-morphemed words
            if len(word_and_morphs[word]) ==1:
                continue
            for key in _.keys():
                g_out.add_node(key)
                g_in.add_node(key)
                g_in.add_edge(key, key)
                node_features.append(vae_features[key])

            key_list = [key for key in _.keys()]
            for i, key in enumerate(key_list):
                if i<len(key_list)-1:
                    g_in.add_edge(key_list[i], key_list[i+1])

            if len(g_in.nodes()) > max_num_nodes:
                dropped_graphs += 1
                continue

            for i in range(len(word_and_morphs[word])-1):
                src = word_and_morphs[word][i]
                tgt = word_and_morphs[word][i+1]
                g_out.add_edge(src, tgt)


            adj_in = nx.adjacency_matrix(g_in).toarray()
            adj_out = nx.adjacency_matrix(g_out).toarray()

            for i in range(len(adj_out)):
                if 1 not in adj_out[i]:
                    adj_out[i][i] = 1

            num_nodes = adj_out.shape[0]
            real_sizes.append(num_nodes)
           
            adj_in_padded = np.zeros((max_num_nodes, max_num_nodes))
            adj_in_padded[:num_nodes, :num_nodes] = adj_in

            adj_out_padded = np.zeros((max_num_nodes, max_num_nodes))
            adj_out_padded[:num_nodes, :num_nodes] = adj_out
            
            features = np.identity(max_num_nodes)
            # make features as probabilities
            diags =[d for d in _.values()]
            for i in range(len(diags)):
                features[i][i] = np.exp(diags[i])
            features = torch.tensor(np.expand_dims(features, axis=0))
            
            adj_in_padded  = torch.tensor(np.expand_dims(adj_in_padded, axis=0))
            adj_out_padded = torch.tensor(np.expand_dims(adj_out_padded, axis=0))

            graphs.append({
            'adj_in': adj_in_padded,
            'adj_out':adj_out_padded,
            'features':features,
            #'features':_features,
            'word':word,
            'g':g_out})

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
