import math
import random
import time
from scipy import sparse
from numpy.linalg import norm
import threading
from util.estimate import rand_index
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import cluster as cluster_methods

class mix_scoring_parallel(threading.Thread):
    def __init__(self, threadID, rnns, data, references, R, row_names, supporting_nodes):
        threading.Thread.__init__(self)
        self.data = data
        self.threadID = threadID
        self.rnns = rnns
        self.data = data
        self.references = references
        self.R = R
        self.row_names = row_names
        self.supporting_nodes = supporting_nodes

    def get_degree(self, s):
        return self.R.getrow(s).sum()

    def get_bundary_distance(self, s):
        bundary_distance = 0
        for re in self.references:
            bundary_distance += abs(norm(self.data.values[s] - re[0]) - norm(
                self.data.values[s] - re[1]))
        return bundary_distance

    def get_ave_neighbor_degree(self, s):
        n = 0
        di = 0
        x = self.R.getrow(s).tocoo()
        # print(type(row))
        for i,j,v in zip(x.row, x.col, x.data):
            n += 1
            di += self.R.getrow(j).sum()
        return di / n

    def get_ave_step(self, s):
        n = 0
        di = 0
        searching_ = set([s])
        searched = set([s])
        t = 0
        while len(searching_) > 0:
            t += 1
            new_searching = set()
            for node_i in searching_:
                x = self.R.getrow(node_i).tocoo()
                for i,j,v in zip(x.row, x.col, x.data):
                    if j not in searched:
                        n += 1
                        di += t
                        searched.add(j)
                        new_searching.add(j)
            searching_ = new_searching
        return di / n

    def get_centrality(self, s):
        n = 0
        di = 0
        searching_ = set([s])
        searched = set([s])
        t = 0
        while len(searching_) > 0:
            t += 1
            new_searching = set()
            for node_i in searching_:
                x = self.R.getrow(node_i).tocoo()
                for i,j,v in zip(x.row, x.col, x.data):
                    if j not in searched:
                        n += 1
                        di += norm(self.data.values[i] - self.data.values[j]) / t
                        searched.add(j)
                        new_searching.add(j)
            searching_ = new_searching
        return di / n

    def run(self):
        s1, s2 = self.rnns
        degree_1 = self.get_degree(s1)
        degree_2 = self.get_degree(s2)

        # 如果是孤立的一堆RNNs，直接判断点对位置，忽略后续计算。
        if degree_1 == 2 and degree_2 ==2:
            bundary_distance_1 = self.get_bundary_distance(s1)
            bundary_distance_2 = self.get_bundary_distance(s2)
            if bundary_distance_1 >= bundary_distance_2:
                self.supporting_nodes.append(self.row_names[s1])
            else:
                self.supporting_nodes.append(self.row_names[s2])
            return

        ave_neighbor_degree_1 = self.get_ave_neighbor_degree(s1)
        ave_neighbor_degree_2 = self.get_ave_neighbor_degree(s2)

        centrality_1 = self.get_centrality(s1)
        centrality_2 = self.get_centrality(s2)

        if (centrality_1 + centrality_2) == 0:
            score = (ave_neighbor_degree_1 / (ave_neighbor_degree_1 + ave_neighbor_degree_2)) / 2
        else:
            score = (ave_neighbor_degree_1 / (ave_neighbor_degree_1 + ave_neighbor_degree_2) + \
                     centrality_2 / (centrality_1 + centrality_2)) / 2

        if score == 0.5:
            bundary_distance_1 = self.get_bundary_distance(s1)
            bundary_distance_2 = self.get_bundary_distance(s2)
            if bundary_distance_1 >= bundary_distance_2:
                self.supporting_nodes.append(self.row_names[s1])
            else:
                self.supporting_nodes.append(self.row_names[s2])
            return

        if score > 0.5:
            self.supporting_nodes.append(self.row_names[s1])
        else:
            self.supporting_nodes.append(self.row_names[s2])
        return

class mix_scoring():
    def __init__(self, threadID, data, references, R, row_names, supporting_nodes):
        self.data = data
        self.threadID = threadID
        self.data = data
        self.references = references
        self.supporting_nodes = supporting_nodes
        self.R = R
        self.row_names = row_names

    def get_degree(self, s):
        return self.R.getrow(s).sum()

    def get_bundary_distance(self, s):
        bundary_distance = 0
        for re in self.references:
            bundary_distance += abs(norm(self.data.values[s] - re[0]) - norm(
                self.data.values[s] - re[1]))
        return bundary_distance

    def get_ave_neighbor_degree(self, s):
        n = 0
        di = 0
        x = self.R.getrow(s).tocoo()
        # print(type(row))
        for i,j,v in zip(x.row, x.col, x.data):
            n += 1
            di += self.R.getrow(j).sum()
        return di / n

    def get_ave_step(self, s):
        n = 0
        di = 0
        searching_ = set([s])
        searched = set([s])
        t = 0
        while len(searching_) > 0:
            t += 1
            new_searching = set()
            for node_i in searching_:
                x = self.R.getrow(node_i).tocoo()
                for i,j,v in zip(x.row, x.col, x.data):
                    if j not in searched:
                        n += 1
                        di += t
                        searched.add(j)
                        new_searching.add(j)
            searching_ = new_searching
        return di / n

    def get_centrality(self, s):
        n = 0
        di = 0
        searching_ = set([s])
        searched = set([s])
        t = 0
        while len(searching_) > 0:
            t += 1
            new_searching = set()
            for node_i in searching_:
                x = self.R.getrow(node_i).tocoo()
                for i,j,v in zip(x.row, x.col, x.data):
                    if j not in searched:
                        n += 1
                        di += norm(self.data.values[i] - self.data.values[j]) / t
                        searched.add(j)
                        new_searching.add(j)
            searching_ = new_searching
        return di / n

    def get(self, rnns):
        s1, s2 = rnns
        degree_1 = self.get_degree(s1)
        degree_2 = self.get_degree(s2)

        # 如果是孤立的一堆RNNs，直接判断点对位置，忽略后续计算。
        if degree_1 == 2 and degree_2 ==2:
            bundary_distance_1 = self.get_bundary_distance(s1)
            bundary_distance_2 = self.get_bundary_distance(s2)
            if bundary_distance_1 >= bundary_distance_2:
                self.supporting_nodes.append(self.row_names[s1])
            else:
                self.supporting_nodes.append(self.row_names[s2])
            return

        ave_neighbor_degree_1 = self.get_ave_neighbor_degree(s1)
        ave_neighbor_degree_2 = self.get_ave_neighbor_degree(s2)


        centrality_1 = self.get_centrality(s1)
        centrality_2 = self.get_centrality(s2)

        if (centrality_1 + centrality_2) == 0:
            score = (ave_neighbor_degree_1 / (ave_neighbor_degree_1 + ave_neighbor_degree_2)) / 2
        else:
            score = (ave_neighbor_degree_1 / (ave_neighbor_degree_1 + ave_neighbor_degree_2) + \
                     centrality_2 / (centrality_1 + centrality_2)) / 2

        if score == 0.5:
            bundary_distance_1 = self.get_bundary_distance(s1)
            bundary_distance_2 = self.get_bundary_distance(s2)
            if bundary_distance_1 >= bundary_distance_2:
                self.supporting_nodes.append(self.row_names[s1])
            else:
                self.supporting_nodes.append(self.row_names[s2])
            return

        if score > 0.5:
            self.supporting_nodes.append(self.row_names[s1])
        else:
            self.supporting_nodes.append(self.row_names[s2])
        return

class PRS():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.clusters = []


    def get_clusters(self, K):
        self.K = K

        sample = random.sample(range(self.data.values.shape[0]), int(np.ceil(math.log2(self.data.values.shape[0]))))
        references = []
        ref_set = set()
        for sam_i in sample:
            max = 0
            a = sam_i
            for i in range(self.data.values.shape[0]):
                if i in ref_set:
                    continue
                d = norm(self.data.values[sam_i] - self.data.values[i])
                if d > max:
                    max = d
                    a = i

            max = 0
            b = a
            for i in range(self.data.values.shape[0]):
                if i in ref_set:
                    continue
                d = norm(self.data.values[a] - self.data.values[i])
                if d > max:
                    max = d
                    b = i

            references.append([self.data.values[a], self.data.values[b]])
            ref_set.add(a)
            ref_set.add(b)

        edges = self.aggregate(self.data, references)

        """
            shortestlinke
        """
        edges.update(self.shortestlinke(edges))

        """
            get tree
        """
        clustering_tree, roots = get_tree(edges)

        self.results = get_result(roots, clustering_tree)

    def shortestlinke(self,edges):

        roots = set()
        additional_edges = {}
        for c, p in edges.items():

            if c == p:
                roots.add(p)
        if len(roots) > self.K:
            data_roots = self.data[self.data.index.isin(roots)]
            row_names = data_roots._stat_axis.values.tolist()
            d = data_roots.values
            neighbors = NearestNeighbors(n_neighbors=2)
            neighbors.fit(d)

            distance_ = neighbors.kneighbors()[0][:, 0]
            index_ = neighbors.kneighbors()[1][:, 0]

            sorted_distance_index = [0]
            for i in range(1,len(index_)):
                flag = False
                for j in range(len(sorted_distance_index)):
                    if distance_[i] < distance_[sorted_distance_index[j]]:
                        sorted_distance_index.insert(j,i)
                        flag = True
                        break
                if flag == False:
                    sorted_distance_index.append(i)
            # print(top)
            additional_edges = {}
            for i in roots:
                additional_edges[i] = i

            n_aditional_edges = 0
            for i in range(len(sorted_distance_index)):
                c = row_names[i]
                p = row_names[index_[i]]

                if additional_edges[p] != c:
                    additional_edges[c] = p
                    n_aditional_edges += 1
                if len(roots) - n_aditional_edges == self.K:
                    break


        return additional_edges



    def aggregate(self, data: pd.DataFrame, references):

        row_names = data._stat_axis.values.tolist()

        t1 = time.process_time()
        # 1. get the adjacent matrix and the corresponding relational matrix
        A, R = get_adjacent_matrix(data)

        t2 = time.process_time()

        # 2. get supporting nodes
        sup_nodes = self.get_supporting_nodes(data, references, R, row_names)

        t3 = time.process_time()
        # 3. if the number of sn smaller than K, stop aggregating

        # if len(data) % 1000 == 0:
        #     # print('RNNs','Score')
        #     print(t2-t1, t3-t2)
        edges = {}
        if self.K <= len(sup_nodes):
            # 3-1.  对于不是根结点的节点，看作已经确定了邻居，此时就返回其指向，
            #       对于根结点就看作没有确定的节点，近一步探索，迭代到下一层。
            for i in range(A.shape[0]):
                if row_names[i] not in sup_nodes:
                    edges[row_names[i]] = row_names[A[i].argmax()]

            data_roots = data[data.index.isin(sup_nodes)]
            edges.update(self.aggregate(data_roots, references))

        else:
            # 3-2. otherwise, in A, roots direct to;
            #      如果下一次聚类得到的类个数小于预期就停止聚类，
            #      每跟根点指向自己作为根节点的标记。
            for i in row_names: edges[i] = i

        return edges

    def get_results(self):
        return self.results

    def get_supporting_nodes(self, data, references, R, row_names):

        # 随机
        # supporting_nodes = self.get_sn_random(R, row_names)

        # 边界
        # supporting_nodes = self.get_sn_boundary(data, R, row_names)

        # 度
        # supporting_nodes = self.get_sn_degree(data, R, row_names)

        # 邻居的平均度
        # supporting_nodes = self.get_sn_ave_neighbor_degree(data, R, row_names)

        # 基于步长的中心性
        # supporting_nodes = self.get_sn_ave_setp(data, R, row_names)

        # 基于距离的中心性
        # supporting_nodes = self.get_sn_centrality(data, R, row_names)


        # 混合选点
        supporting_nodes = self.get_sn_mixed(data, references, R, row_names)

        return supporting_nodes

    def get_sn_random(self, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                if random.random() >= 0.5:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[R[s1].argmax()])
                candidates.remove(R[s1].argmax())
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes

    def get_sn_boundary(self, data, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        RNNs = set()
        for i in range(R.shape[0]):
            if R[i].max() == 2:
                RNNs.add(i)
        sample = random.sample(RNNs, int(np.ceil(math.log2(len(RNNs)))))
        references = set()
        ref_set = set()
        for sam_i in sample:
            max = 0
            argmax = sam_i
            for i in RNNs:
                if i in ref_set:
                    continue
                d = norm(data.values[sam_i] - data.values[i])
                if d > max:
                    max = d
                    argmax = i

            max = 0
            argmaxargmax = argmax
            for i in RNNs:
                if i in ref_set:
                    continue
                d = norm(data.values[argmax] - data.values[i])
                if d > max:
                    max = d
                    argmaxargmax = i

            references.add((argmax, argmaxargmax))
            ref_set.add(argmax)
            ref_set.add(argmaxargmax)
            # print(len(RNNs),sam_i,argmax,argmaxargmax)

        # references = RNNs
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                s2 = R[s1].argmax()
                ave_d_1 = 0
                ave_d_2 = 0
                for re in references:
                    ave_d_1 += abs(norm(data.values[s1] - data.values[re[0]]) - norm(
                        data.values[s1] - data.values[re[1]]))
                    ave_d_2 += abs(norm(data.values[s2] - data.values[re[0]]) - norm(
                        data.values[s2] - data.values[re[1]]))

                if ave_d_1 >= ave_d_2:
                    supporting_nodes.add(row_names[s1])
                    # print(s1,s2,':',s1)
                else:
                    supporting_nodes.add(row_names[s2])
                    # print(s1, s2, ':', s2)
                candidates.remove(s2)
                candidates.remove(s1)

            else:
                continue
        return supporting_nodes

    def get_sn_degree(self, data, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
     

        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                s2 = R[s1].argmax()

                degree_1 = R[s1].sum()
                degree_2 = R[s2].sum()
                if degree_1 == degree_2:
                    degree_1 = random.random()
                    degree_2 = random.random()
                    
                if degree_1 >= degree_2:
                    supporting_nodes.add(row_names[s1])

                else:
                    supporting_nodes.add(row_names[s2])
                candidates.remove(s2)
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes

    def get_sn_ave_neighbor_degree(self, data, R, row_names):

        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
 
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                # s1和s2 都是supporting node
                s2 = R[s1].argmax()

                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0
                for i in range(np.size(R[s1])):
                    if R[s1, i] > 0:
                        n_1 += 1
                        di_1 += R[i].sum()
                for i in range(np.size(R[s2])):
                    if R[s2, i] > 0:
                        n_2 += 1
                        di_2 += R[i].sum()

                ave_neighbor_degree_1 = di_1 / n_1
                ave_neighbor_degree_2 = di_2 / n_2

       

                if ave_neighbor_degree_1 == ave_neighbor_degree_2:
                    ave_neighbor_degree_1 = random.random()
                    ave_neighbor_degree_2 = random.random()
                if ave_neighbor_degree_1 >= ave_neighbor_degree_2:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[s2])
                candidates.remove(s2)
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes

    def get_sn_ave_setp(self, data, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:

                s2 = R[s1].argmax()
                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0

                searching_ = set([s1])
                searched = set([s1])
                t = 0
                while len(searching_) > 0:
                    t += 1
                    new_searching = set()
                    for node_i in searching_:

                        for node_j in range(np.size(R[node_i])):
                            if  R[node_i, node_j] > 0 and node_j not in searched:
                                n_1 += 1
                                di_1 += t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                searching_ = set([s2])
                searched = set([s2])
                t = 0
                while len(searching_) > 0:
                    t += 1
                    new_searching = set()
                    for node_i in searching_:
                        for node_j in range(np.size(R[node_i])):
                            if R[node_i, node_j] > 0 and node_j not in searched:
                                n_2 += 1
                                di_2 += t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                ave_setp_1 = di_1 / n_1
                ave_setp_2 = di_2 / n_2

                if ave_setp_1 == ave_setp_2:
                    ave_setp_1 = random.random()
                    ave_setp_2 = random.random()
                if ave_setp_1 <= ave_setp_2:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[s2])
                candidates.remove(s2)
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes

    def get_sn_centrality(self, data, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            # 找到支撑节点
            if R[s1].max() == 2 and s1 in candidates:
                # s1和s2 都是supporting node
                s2 = R[s1].argmax()

                # 从两个节点开始广度优先搜索

                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0

                searching_ = set([s1])
                searched = set([s1])
                t = 0

                while len(searching_) > 0:
                    t += 1
                    new_searching = set()
                    for node_i in searching_:

                        for node_j in range(np.size(R[node_i])):
                            if R[node_i, node_j] > 0 and node_j not in searched:
                                n_1 += 1
                                di_1 += norm(data.values[node_i] - data.values[node_j]) / t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                searching_ = set([s2])
                searched = set([s2])
                t = 0
                while len(searching_) > 0:
                    t += 1
                    new_searching = set()
                    for node_i in searching_:
                        for node_j in range(np.size(R[node_i])):
                            if R[node_i, node_j] > 0 and node_j not in searched:
                                n_2 += 1
                                di_2 += norm(data.values[node_i] - data.values[node_j]) / t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                centrality_1 = di_1 / n_1
                centrality_2 = di_2 / n_2
                if centrality_1 == centrality_2:

                    centrality_1 = random.random()
                    centrality_2 = random.random()
                if centrality_1 <= centrality_2:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[s2])
                candidates.remove(s2)
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes

    def get_sn_mixed(self, data, references, R, row_names):
        R.eliminate_zeros()
        R_COO = R.tocoo()
        candidates = set()
        rnns_pairs = []
        for i,j,v in zip(R_COO.row, R_COO.col, R_COO.data):
            if i in candidates: continue
            if v == 2:
                candidates.add(i)
                candidates.add(j)
                rnns_pairs.append((i, j))

        supporting_nodes = []

        ms = mix_scoring(i, data, references, R_COO, row_names, supporting_nodes)

        for rnns in rnns_pairs:
            ms.get(rnns)

        return supporting_nodes


def get_tree(edges):
    clustering_tree = {}
    roots = set()
    for c, p in edges.items():

        if c == p:
            # print(c, ':', p)
            roots.add(p)

        if p not in clustering_tree.keys():
            nc = set()
            nc.add(c)
            clustering_tree[p] = nc
        else:
            clustering_tree[p].add(c)
    # print(clustering_tree)
    return clustering_tree, roots


def get_height_th_tree(tree, root):
    n = 1
    ps = set([root])

    while len(ps) > 0:
        ps_new = set()
        for p in ps:
            if p in tree.keys() and len(tree[p]) > 0:
                ps_new = ps_new.union(tree[p])

                n += len(tree[p])

        ps = ps_new

    return math.ceil(math.log2(n) / math.log2(2))


def get_result(roots, clustering_tree):
    result = {}
    for r in roots: result[r] = r
    parents_next = roots
    while len(parents_next) != 0:
        labels_new = set()
        for p in parents_next:
            if p in clustering_tree.keys():
                for c in clustering_tree[p]:
                    result[c] = result[p]
                    # print(c, '->', labels[p],'(',c,'->
                    # ,p,')')
                labels_new = clustering_tree[p] | labels_new
        if len(labels_new) > 0:
            parents_next = labels_new - parents_next
        else:
            break
    return result


def get_adjacent_matrix(data):
    d = data.values

    """
    普通的向量数据
    """
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(d)
    A = neighbors.kneighbors_graph(d) - sparse.eye(len(d))
    R = A + A.T

    return A, R


def adjacent_list_2_children_map(results):
    re_map = {}
    for p in results:
        re_map[p[0]] = p[1]
    print(len(results), ',', len(re_map))


def get_labels(clusters, data_size):
    labels = -1 * np.ones(data_size)
    for i, root in zip(range(len(clusters.keys())), clusters.keys()):
        for node in clusters[root].keys():
            labels[node] = i
    return labels

def ex_random_selecting(f):
    result = open('/Users/wenboxie/Data/FRS/Results/random_adult.txt', 'w')
    for t in range(100):

        data = pd.read_csv(f, header=None).iloc[:, 0:-1]
        label = pd.read_csv(f, header=None).iloc[:, -1]
        for i in range(1, 10):
            prs = PRS(data)
            prs.get_clusters(i)
            r = prs.get_results()
            # print('k =', len(set(r.values())))
            RS_labels_ = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
            if len(set(r.values())) == 1: break
            result.write(str(len(set(r.values()))) + '\t' + str(i) + '\n')

    result.close()

    return True

def ex_compare_selecting(f, K, Q):

    # wf = open('/Users/wenboxie/Documents/Manuscript/RS-IN/benchmarks_comp/iris_sc_revised.txt','w')
    # data = pd.read_csv(f, header=None).iloc[:, 0:-1]
    # label = pd.read_csv(f, header=None).iloc[:, -1]
    # data = (data - data.mean()) / (data.std())
    # data = (data - data.min()) / (data.max() - data.min())

    full_data = pd.DataFrame(np.random.random(size=(1000 * 2 ** 10, 10)))

    for k in range(K,Q):

        nmi = []
        rand = []
        data = full_data
        cpu_times = 0
        for times in range(10):
            start_time  = time.time()
            prs = PRS(data)
            prs.get_clusters(10)
            r = prs.get_results().head(1000 * 2 ** k)
            end_time = time.time()
            cpu_times += (end_time - start_time)
            # print('k =', len(set(r.values())))
        #     RS_labels_temp = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
        #     old_l = set(RS_labels_temp)
        #     new_l = {}
        #     i = 0
        #     for l in old_l:
        #         new_l[l] = i
        #         i+=1
        #     RS_labels_ = []
        #     for i in range(len(RS_labels_temp)):
        #         RS_labels_.append(new_l[RS_labels_temp[i]])
        #     # RS_labels_ = np.array(RS_labels_)
        #     benchmark = []
        #     for i in range(len(label)):
        #         benchmark.append(label[i])
        #
        #
        #     if len(set(r.values())) == 1:
        #         print('one root left!')
        #         # break
        #         # print(metrics.cluster.adjusted_rand_score(label, RS_labels_))
        #     # print(rand_index(benchmark, RS_labels_))
        #     nmi.append(metrics.cluster.normalized_mutual_info_score(benchmark, RS_labels_))
        #     rand.append(rand_index(benchmark, RS_labels_))
        #
        # nmi = np.array(nmi)
        # rand = np.array(rand)
        # print('rand','nmi')
        # print(np.mean(rand), np.mean(nmi))
        # print(np.std(rand), np.std(nmi))

        print(1000 * 2 ** k, cpu_times / 10)
    #         wf.write(str(rand_index(benchmark, RS_labels_))+'\n')
    # wf.close()
    # plot the results

        # print(RS_labels_)





if __name__ == '__main__':

    """
    Read data
    "breast-w", "ecoli", "glass", "ionosphere", "iris", "kdd_synthetic_control", "mfeat-fourier",
    "mfeat-karhunen","mfeat-zernike",
    "optdigits", "segment", "sonar", "vehicle", "waveform-5000", "letter", "kdd_synthetic_control"
    "adult"
    'avila'
    
    """
    k = 10
    data_name = 'mfeat-fourier'
    # f='/Users/wenboxie/Data/uci-20070111/exp_disturbed/'+data_name+'.txt'
    f='../exp_disturbed/'+data_name+'.txt'
    # f = '/Users/wenboxie/Data/FRS/Datasets/HTRU2/HTRU_2.csv'
    print(data_name)


    if k-5 < 2:
        K = 2
        Q = 13
    else:
        K = k - 5
        Q = k + 6
    K = 3
    Q = 10
    ex_compare_selecting(f, K, Q)


