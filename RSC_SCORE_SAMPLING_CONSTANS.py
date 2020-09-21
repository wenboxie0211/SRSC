import math
import random
from util.estimate import rand_index
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import cluster as cluster_methods

class PRS():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.clusters = []

    def get_clusters(self, K):
        self.K = K

        """
            heirachical nn
        """
        sample = random.sample(range(self.data.values.shape[0]), int(np.ceil(math.log2(self.data.values.shape[0]))))
        references = []
        ref_set = set()
        for sam_i in sample:
            max = 0
            argmax = sam_i
            for i in range(self.data.values.shape[0]):
                if i in ref_set:
                    continue
                d = np.linalg.norm(self.data.values[sam_i] - self.data.values[i])
                if d > max:
                    max = d
                    argmax = i

            max = 0
            argmaxargmax = argmax
            for i in range(self.data.values.shape[0]):
                if i in ref_set:
                    continue
                d = np.linalg.norm(self.data.values[argmax] - self.data.values[i])
                if d > max:
                    max = d
                    argmaxargmax = i

            references.append([self.data.values[argmax], self.data.values[argmaxargmax]])
            ref_set.add(argmax)
            ref_set.add(argmaxargmax)

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

        # 1. get the adjacent matrix and the corresponding relational matrix
        A, R = get_adjacent_matrix(data)

        # 2. get supporting nodes
        sup_nodes = self.get_supporting_nodes(data, references, R, row_names)

        # 3. if the number of sn smaller than K, stop aggregating
        edges = {}
        if self.K <= len(sup_nodes):
            for i in range(A.shape[0]):
                if row_names[i] not in sup_nodes:
                    edges[row_names[i]] = row_names[A[i].argmax()]

            data_roots = data[data.index.isin(sup_nodes)]
            edges.update(self.aggregate(data_roots, references))

        else:
            for i in row_names: edges[i] = i

        return edges

    def get_results(self):
        return self.results

    def get_supporting_nodes(self, data, references, R, row_names):

        # 随机
        # supporting_nodes = self.get_sn_random(R, row_names)

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
                d = np.linalg.norm(data.values[sam_i] - data.values[i])
                if d > max:
                    max = d
                    argmax = i

            max = 0
            argmaxargmax = argmax
            for i in RNNs:
                if i in ref_set:
                    continue
                d = np.linalg.norm(data.values[argmax] - data.values[i])
                if d > max:
                    max = d
                    argmaxargmax = i

            references.add((argmax, argmaxargmax))
            ref_set.add(argmax)
            ref_set.add(argmaxargmax)
        
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                s2 = R[s1].argmax()
                ave_d_1 = 0
                ave_d_2 = 0
                for re in references:
        
                    ave_d_1 += abs(np.linalg.norm(data.values[s1] - data.values[re[0]]) - np.linalg.norm(
                        data.values[s1] - data.values[re[1]]))
                    ave_d_2 += abs(np.linalg.norm(data.values[s2] - data.values[re[0]]) - np.linalg.norm(
                        data.values[s2] - data.values[re[1]]))

                if ave_d_1 >= ave_d_2:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[s2])
   
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
            #找到支撑节点
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
                            if R[node_i, node_j] > 0 and node_j not in searched:
                                n_1 += 1
                                di_1 += np.linalg.norm(data.values[node_i] - data.values[node_j]) / t
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
                                di_2 += np.linalg.norm(data.values[node_i] - data.values[node_j]) / t
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

        candidates = set(range(R.shape[0]))
        supporting_nodes = set()

        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                s2 = R[s1].argmax()

                degree_1 = R[s1].sum()
                degree_2 = R[s2].sum()

                if degree_1 == 2 and degree_2 ==2:
                    ave_d_1 = 0
                    ave_d_2 = 0
                    for re in references:
                        ave_d_1 += abs(np.linalg.norm(data.values[s1] - re[0]) - np.linalg.norm(
                            data.values[s1] - re[1]))
                        ave_d_2 += abs(np.linalg.norm(data.values[s2] - re[0]) - np.linalg.norm(
                            data.values[s2] - re[1]))

                    if ave_d_1 >= ave_d_2:
                        supporting_nodes.add(row_names[s1])
                    else:
                        supporting_nodes.add(row_names[s2])
                    candidates.remove(s2)
                    candidates.remove(s1)
                    continue

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
                                di_1 += np.linalg.norm(data.values[node_i] - data.values[node_j]) / t
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
                                di_2 += np.linalg.norm(data.values[node_i] - data.values[node_j]) / t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                centrality_1 = di_1 / n_1
                centrality_2 = di_2 / n_2

                # complete version #
                # score = (degree_1 / (degree_1 + degree_2) + \
                #                  ave_neighbor_degree_1/(ave_neighbor_degree_1+ave_neighbor_degree_2)+\
                #          ave_setp_2 / (ave_setp_1 + ave_setp_2) + centrality_2 / (centrality_1 + centrality_2))/4

                score = (ave_neighbor_degree_1 / (ave_neighbor_degree_1 + ave_neighbor_degree_2) + \
                         centrality_2 / (centrality_1 + centrality_2)) / 2

                if score == 0.5:
                    ave_d_1 = 0
                    ave_d_2 = 0
                    for re in references:
                        
                        ave_d_1 += abs(np.linalg.norm(data.values[s1] - re[0]) - np.linalg.norm(
                            data.values[s1] - re[1]))
                        ave_d_2 += abs(np.linalg.norm(data.values[s2] - re[0]) - np.linalg.norm(
                            data.values[s2] - re[1]))

                    if ave_d_1 >= ave_d_2:
                        supporting_nodes.add(row_names[s1])
                    else:
                        supporting_nodes.add(row_names[s2])
                    candidates.remove(s2)
                    candidates.remove(s1)
                    continue

                if score > 0.5:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[s2])
                candidates.remove(s2)
                candidates.remove(s1)
            else:
                continue

        return supporting_nodes


def get_tree(edges):
    clustering_tree = {}
    roots = set()
    for c, p in edges.items():

        if c == p:
            roots.add(p)

        if p not in clustering_tree.keys():
            nc = set()
            nc.add(c)
            clustering_tree[p] = nc
        else:
            clustering_tree[p].add(c)
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
    # print('roots:',roots)
    parents_next = roots
    while len(parents_next) != 0:
        labels_new = set()
        for p in parents_next:
            if p in clustering_tree.keys():
                for c in clustering_tree[p]:
                    result[c] = result[p]
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
    A = neighbors.kneighbors_graph(d) - np.eye(len(d))
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


def draw_matrix(m):
    data = np.array(m)
    # print(data[:,0])
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


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
    data = pd.read_csv(f, header=None).iloc[:, 0:-1]
    label = pd.read_csv(f, header=None).iloc[:, -1]
    data = (data - data.mean()) / (data.std())
    # data = (data - data.min()) / (data.max() - data.min())
    for times in range(50):
        for k in range(K,Q):
            prs = PRS(data)
            prs.get_clusters(k)
            r = prs.get_results()
            # print('k =', len(set(r.values())))
            RS_labels_temp = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
            old_l = set(RS_labels_temp)
            new_l = {}
            i = 0
            for l in old_l:
                new_l[l] = i
                i+=1
            RS_labels_ = []
            for i in range(len(RS_labels_temp)):
                RS_labels_.append(new_l[RS_labels_temp[i]])
            # RS_labels_ = np.array(RS_labels_)
            benchmark = []
            for i in range(len(label)):
                benchmark.append(label[i])


            if len(set(r.values())) == 1:
                print('one root left!')
                # break
                # print(metrics.cluster.adjusted_rand_score(label, RS_labels_))
            print(rand_index(benchmark, RS_labels_))
    #         wf.write(str(rand_index(benchmark, RS_labels_))+'\n')
    # wf.close()
    # plot the results

        # print(RS_labels_)
        # plt.scatter(data.values[:, 0], data.values[:, 1], c=RS_labels_, cmap='rainbow')
        # plt.show()

def ex_single_linkgae(f,K, Q):
    data = pd.read_csv(f, header=None).iloc[:, 0:-1]
    data = (data - data.mean()) / (data.std())
    # data = (data-data.min()) / (data.max()-data.min())
    # print(data)
    label = pd.read_csv(f, header=None).iloc[:, -1]

    for k in range(K, Q):
        single_model = cluster_methods.AgglomerativeClustering(n_clusters=k,linkage='average').fit(data)
    # print(metrics.cluster.supervised.adjusted_rand_score(label, single_model.labels_))
        benchmark = []
        prid = []
        labels_ =  single_model.labels_
        for i in range(len(label)):
            benchmark.append(label[i])
            prid.append(labels_[i])
        print(rand_index(benchmark, labels_))

    # print(single_model.labels_)
    # plt.scatter(data.values[:, 0], data.values[:, 1], c=single_model.labels_, cmap='rainbow')
    # plt.show()

def ex_BIRCH(f):
    data = pd.read_csv(f, header=None).iloc[:, 0:-1]
    label = pd.read_csv(f, header=None).iloc[:, -1]
    for k in range(2, 51):
        birch_model = cluster_methods.Birch(n_clusters=k,threshold=0.1).fit(data)
        print(metrics.cluster.adjusted_rand_score(label, birch_model.labels_))
    # print(single_model.labels_)
    plt.scatter(data.values[:, 1], data.values[:, -1], c=birch_model.labels_, cmap='rainbow')
    plt.show()

def ex_chameleon(f, k):
    data = pd.read_csv(f, header=None).iloc[:, 0:-1]
    label = pd.read_csv(f, header=None).iloc[:, -1]
    for k in range(K, Q):
        single_model = cluster_methods.AgglomerativeClustering(n_clusters=k,linkage='complete').fit(data)
        print(metrics.cluster.adjusted_rand_score(label, single_model.labels_))

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
    f='/Users/wenboxie/Data/uci-20070111/exp_disturbed/'+data_name+'.txt'
    # f = '/Users/wenboxie/Data/FRS/Datasets/HTRU2/HTRU_2.csv'
    print(data_name)
    """
    1. 在不同大小的数据上测试随机选择代表节点
    """
    # ex_random_selecting(f)

    """
    2. 比较不同选点方法的RI
    """

    # if k-5 < 2:
    #     K = 2
    #     Q = 13
    # else:
    #     K = k - 5
    #     Q = k + 6
    K = k
    Q = k + 1
    ex_compare_selecting(f, K, Q)

    """
    3. single的RI
    """
    # ex_single_linkgae(f, K, Q)

    """
    3. chameleon的RI
    """
    # ex_chameleon(f,K, Q)

    """
    5. BIRCH
    """
    # ex_BIRCH(f)



