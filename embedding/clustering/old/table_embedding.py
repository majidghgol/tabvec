import sys
import os
from scipy.spatial.distance import cosine as cos_dist
from sklearn.manifold import TSNE, isomap, LocallyLinearEmbedding
import numpy as np
import copy
from gensim.models import Word2Vec
from random_indexing import RandomIndexing
from itertools import product as cross_product
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn import metrics
import random
import gzip
from pyspark import SparkContext, SparkConf
import json


norm = lambda x: np.linalg.norm(x,2)
# vec_angle = lambda x,y: cosine_distances(x.reshape(1, -1), y.reshape(1, -1)).flatten()[0] * norm(x-y)
vec_angle = lambda x,y: cosine_distances(x.reshape(1, -1), y.reshape(1, -1)).flatten()[0]
my_normalize = lambda x: normalize(x.reshape(1,-1))[0]
# my_normalize = lambda x: x
# vec_angle = lambda x: cosine_distances(x.reshape(1, -1), np.ones(x.reshape(1, -1).shape)).flatten()[0]*norm(x)
# vec_angle = norm
eps = 1e-5
# not_found=lambda s: [np.random.uniform(-100, 100, s.d)]
not_found = lambda s: None


class TableEmbedding:
    def __init__(self, aggregate_method='avg',
                 cvs=None, d=100, factors=3,
                 token_thresh=50):
        self.aggregate_method = aggregate_method
        self.cell_token_threshold = token_thresh
        self.d = d
        self.factors = factors

    def generate_table_sentences(self, t, max_thresh, window, modes=('cells', 'None')):
        # TODO: add tokenizer as input
        #print 'here: {}'.format(t)
        t = json.loads(t)["tok_tarr"]
        sentences = []
        if 'cells' in modes:
            sentences.extend(self.generate_sentences_from_tarr_cell(t))
        if 'rows' in modes:
            sentences.extend(self.generate_sentences_from_tarr_row(t, window, max_thresh))
        if 'cols' in modes:
            sentences.extend(self.generate_sentences_from_tarr_col(t, window, max_thresh))
        if 'hrow' in modes:
            sentences.extend(self.generate_sentences_from_tarr_hrow(t, max_thresh))
        if 'hcol' in modes:
            sentences.extend(self.generate_sentences_from_tarr_hcol(t, max_thresh))
        return sentences

    def train_models(self, sc, tok_tables_rdd, text_sentences_rdd, sentences, window=2, max_thresh=50,
                     min_count=10, method='word2vec'):
        print(tok_tables_rdd)
        for k, v in sentences.items():
            print v
            s_temp = tok_tables_rdd.flatMap(lambda x: self.generate_table_sentences(x,  max_thresh, window, v))
            if 'text' in v:
                s_temp = s_temp + text_sentences_rdd
            def f(x):
                print x
            # s_temp.foreach(f)
            s_temp.cache()

            if method == 'word2vec':
                s = s_temp.reduce(lambda a, b: a + b)
                cvs = Word2Vec(s, size=self.d, window=window,
                                       min_count=min_count, workers=4)
                yield k, cvs
            else:
                ri = RandomIndexing(s_temp, size=self.d, window=window,
                               min_count=min_count)
                temp_base = s_temp.flatMap(lambda x: [x[i] for i in range(len(x))]).distinct().map(lambda w: (w, ri.get_base())).collect()
                # temp_base.foreach(f)
                # print(temp_base)
                base = dict(temp_base)
                counts = s_temp.flatMap(lambda x: [x[i] for i in range(len(x))]).map(lambda w: (w, 1)).reduceByKey(lambda v1, v2: v1 + v2).collect()
                counts = dict(counts)
                ri.init_base_vectors_from_dict(base, counts)
                del base
                del counts
                cvs = s_temp.flatMap(lambda x: ri.get_cvs_batch(x)).reduceByKey(lambda v1, v2: ri.add_vecs(v1,v2)).collect()
                cvs = dict(cvs)
                yield k, cvs
                ri.clear()
            s_temp.unpersist()
            del s_temp


    def calc_table_vector(self, toktarr, cvs, weights, fine_grained=False):
        v = self.calc_table_vector_ensemble_reduced(toktarr, cvs, weights, self.aggregate_method, fine_grained)
        if v is None:
            v = np.zeros(self.factors)
            # v = np.random.rand(self.factors*4)
        return v

    def generate_sentences_from_tarr_cell(self, tokenized_tarr):
        all_sentences = []
        for r in tokenized_tarr:
            for tokenized_c in r:
                all_sentences.append(tokenized_c)  # tokens in the cell
        # all_sentences.append(table_sentence)
        return all_sentences

    def generate_sentences_from_tarr_col(self, tokenized_tarr, window, max_thresh):
        all_sentences = []
        if len(tokenized_tarr) > 0:
            for j in range(len(tokenized_tarr[0])):
                try:
                    for i in range(0, len(tokenized_tarr)-window+1):
                        lists = []
                        n = 1
                        for k in range(window):
                            tl = len(tokenized_tarr[i+k][j])
                            if tl == 0:
                                continue
                            lists.append(tokenized_tarr[i+k][j])
                            n *= tl
                        if n > max_thresh:
                            all_sentences.extend(self.sample_cross_product_arrays(lists, max_thresh))
                        else:
                            all_sentences.extend(self.cross_product_arrays(lists))
                        # all_sentences += cross_product(tokenized_tarr[j][i], tokenized_tarr[j+1][i])
                except Exception as e:
                    print('i is:{}'.format(j))
                    print(e.message)

        # all_sentences.append(table_sentence)
        return all_sentences

    def generate_sentences_from_tarr_row(self, tokenized_tarr, window, max_thresh):
        all_sentences = []
        for i in range(len(tokenized_tarr)):
            for j in range(0, len(tokenized_tarr[i])-window+1):
                lists = []
                n = 1
                for k in range(window):
                    tl = len(tokenized_tarr[i][j + k])
                    if tl == 0:
                        continue
                    lists.append(tokenized_tarr[i][j+k])
                    n *= tl
                if n > max_thresh:
                    all_sentences.extend(self.sample_cross_product_arrays(lists, max_thresh))
                else:
                    all_sentences.extend(self.cross_product_arrays(lists))
                # all_sentences += cross_product(tokenized_tarr[i][j], tokenized_tarr[i][j+1])
        # all_sentences.append(table_sentence)
        return all_sentences

    def generate_sentences_from_tarr_hrow(self, tokenized_tarr, max_thresh):
        all_sentences = []
        if len(tokenized_tarr) > 1:
            header = tokenized_tarr[0]
            tokenized_tarr = tokenized_tarr[1:]
            for i in range(len(tokenized_tarr)):
                for j in range(len(header)):
                    tl = len(tokenized_tarr[i][j])
                    hl = len(header[j])
                    if tl == 0 or hl == 0:
                        continue
                    n = tl * hl
                    if n > max_thresh:
                        temp = self.sample_cross_product_arrays([tokenized_tarr[i][j], header[j]], max_thresh)
                    else:
                        temp = self.cross_product_arrays([tokenized_tarr[i][j], header[j]])
                    # print temp
                    all_sentences.extend(temp)
                    # all_sentences += cross_product(tokenized_tarr[j][i], tokenized_tarr[j+1][i])
        # all_sentences.append(table_sentence)
        return all_sentences

    def generate_sentences_from_tarr_hcol(self, tokenized_tarr, max_thresh):
        all_sentences = []
        if len(tokenized_tarr) > 0 and len(tokenized_tarr[0]) > 1:
            header = [tokenized_tarr[i][0] for i in range(len(tokenized_tarr))]
            for i in range(len(tokenized_tarr)):
                for j in range(1, len(tokenized_tarr[i])):
                    tl = len(tokenized_tarr[i][j])
                    hl = len(header[i])
                    # print header[i]
                    if tl == 0 or hl == 0:
                        continue
                    n = tl * hl
                    if n > max_thresh:
                        temp = self.sample_cross_product_arrays([tokenized_tarr[i][j], header[i]], max_thresh)
                    else:
                        temp = self.cross_product_arrays([tokenized_tarr[i][j], header[i]])
                    # if '21' in temp or '22' in temp or '23' in temp:
                    # print temp
                    all_sentences.extend(temp)
                    # all_sentences += cross_product(tokenized_tarr[j][i], tokenized_tarr[j+1][i])
        # all_sentences.append(table_sentence)
        return all_sentences

    def sample_cross_product_arrays(self, lists, thresh):
        res = []
        for i in range(thresh):
            temp = []
            for l in lists:
                j = random.randrange(0, len(l))
                temp.append(l[j])
            res.append(temp)
        return res

    def cross_product_arrays(self, lists):
        res = self.cross_product_arrays_helper([[]], lists)
        return res

    def cross_product_arrays_helper(self, current, lists):
        if len(lists) == 0:
            return current
        # print lists, current
        l = lists[0]
        rest = lists[1:]
        res = []
        for x in l:
            temp = []
            for i in range(len(current)):
                temp.append(current[i]+[x])
            r = self.cross_product_arrays_helper(temp, rest)
            res.extend(r)
        return res

    # def dev_vectarr2vec_fine_grained(self, vec_tarr):
    #     if len(vec_tarr) == 0:
    #         return None
    #     col_var = []
    #     row_var = []
    #     for j in range(0,len(vec_tarr[0])):#range(len(vec_tarr[0])):
    #         xx = [vec_tarr[i][j][k] for i in range(len(vec_tarr)) for k in range(len(vec_tarr[i][j]))
    #               if vec_tarr[i][j][k] is not None]
    #         if len(xx) > 0:
    #             xx = np.vstack(xx)
    #             col_var.append(np.var(xx, axis=0).flatten())
    #     for i in range(0,len(vec_tarr)):
    #         xx = [vec_tarr[i][j][k] for j in range(len(vec_tarr[i])) for k in range(len(vec_tarr[i][j]))
    #               if vec_tarr[i][j][k] is not None]
    #         if len(xx) > 0:
    #             xx = np.vstack(xx)
    #             row_var.append(np.var(xx, axis=0).flatten())
    #     xx = [vec_tarr[i][j][k] for i in range(len(vec_tarr))
    #           for j in range(len(vec_tarr[i]))
    #           for k in range(len(vec_tarr[i][j]))
    #           if vec_tarr[i][j] is not None]
    #
    #     if len(xx) == 0 or len(col_var) == 0 or len(row_var) == 0:
    #         return None
    #     tot_var = np.var(np.vstack(xx), axis=0)
    #     col_var = np.vstack(col_var)
    #     row_var = np.vstack(row_var)
    #     # res = np.mean(col_var, axis=0)
    #     res = np.concatenate((tot_var, np.mean(col_var, axis=0), np.mean(row_var, axis=0)))
    #     # print(np.linalg.norm(tot_var))
    #     # res = tot_var
    #     return res

    def dev_vectarr2vec_fine_grained_reduced(self, vec_tarr):
        if len(vec_tarr) == 0:
            return None
        col_var = []
        row_var = []
        for j in range(0,len(vec_tarr[0])):#range(len(vec_tarr[0])):
            xx = [vec_tarr[i][j][k] for i in range(len(vec_tarr)) for k in range(len(vec_tarr[i][j]))
                  if vec_tarr[i][j][k] is not None]
            if len(xx) > 0:
                xx = np.vstack(xx)
                col_var.append(np.var(xx, axis=0).flatten())
        for i in range(0,len(vec_tarr)):
            xx = [vec_tarr[i][j][k] for j in range(len(vec_tarr[i])) for k in range(len(vec_tarr[i][j]))
                  if vec_tarr[i][j][k] is not None]
            if len(xx) > 0:
                xx = np.vstack(xx)
                row_var.append(np.var(xx, axis=0).flatten())
        xx = [vec_tarr[i][j][k] for i in range(len(vec_tarr))
              for j in range(len(vec_tarr[i]))
              for k in range(len(vec_tarr[i][j]))
              if vec_tarr[i][j] is not None]

        if len(xx) == 0 or len(col_var) == 0 or len(row_var) == 0:
            return None
        tot_var = np.var(np.vstack(xx), axis=0)
        col_var = np.vstack(col_var)
        row_var = np.vstack(row_var)
        # res = np.mean(col_var, axis=0)
        res = np.array([norm(tot_var), norm(np.mean(col_var, axis=0)), norm(np.mean(row_var, axis=0))])
        # print(np.linalg.norm(tot_var))
        # res = tot_var
        return res

    # def dev_vectarr2vec(self, vec_tarr):
    #     if len(vec_tarr) == 0:
    #         return None
    #     col_var = []
    #     row_var = []
    #     for i in range(0,len(vec_tarr[0])):#range(len(vec_tarr[0])):
    #         xx = [vec_tarr[j][i] for j in range(len(vec_tarr)) if vec_tarr[j][i] is not None]
    #         if len(xx) > 0:
    #             xx = np.vstack(xx)
    #             col_var.append(np.var(xx, axis=0).flatten())
    #     for i in range(0,len(vec_tarr)):
    #         xx = [vec_tarr[i][j] for j in range(len(vec_tarr[i])) if vec_tarr[i][j] is not None]
    #         if len(xx) > 0:
    #             xx = np.vstack(xx)
    #             row_var.append(np.var(xx, axis=0).flatten())
    #     xx = [vec_tarr[i][j] for i in range(len(vec_tarr)) for j in range(len(vec_tarr[i])) if vec_tarr[i][j] is not None]
    #
    #     if len(xx) == 0 or len(col_var) == 0 or len(row_var) == 0:
    #         return None
    #     tot_var = np.var(np.vstack(xx), axis=0)
    #     col_var = np.vstack(col_var)
    #     row_var = np.vstack(row_var)
    #     # res = np.mean(col_var, axis=0)
    #     res = np.concatenate((tot_var, np.mean(col_var, axis=0), np.mean(row_var, axis=0)))
    #     # print(np.linalg.norm(tot_var))
    #     # res = tot_var
    #     return res
    def get_mean_dev(self, xx):
        xx = np.vstack(xx)
        m = np.mean(xx, axis=0)
        v = [vec_angle(xx[j], m) for j in range(xx.shape[0])]
        # var_temp = np.var(xx, axis=0).flatten()
        # col_var.append(vec_angle(var_temp))
        return np.mean(v)

    def dev_vectarr2vec_reduced(self, vec_tarr):
        if len(vec_tarr) == 0:
            return None
        col_var = []
        row_var = []
        for i in range(0,len(vec_tarr[0])):#range(len(vec_tarr[0])):
            xx = [vec_tarr[j][i] for j in range(len(vec_tarr)) if vec_tarr[j][i] is not None]
            if len(xx) > 0:
                col_var.append(self.get_mean_dev(xx))
        for i in range(0,len(vec_tarr)):
            xx = [vec_tarr[i][j] for j in range(len(vec_tarr[i])) if vec_tarr[i][j] is not None]
            if len(xx) > 0:
                row_var.append(self.get_mean_dev(xx))
        xx = [vec_tarr[i][j] for i in range(len(vec_tarr)) for j in range(len(vec_tarr[i])) if vec_tarr[i][j] is not None]

        if len(xx) == 0 or len(col_var) == 0 or len(row_var) == 0:
            return None
        tot_var = self.get_mean_dev(xx) + eps
        col_var = np.array(col_var)
        row_var = np.array(row_var)
        # res = np.mean(col_var, axis=0)
        mean_c = np.mean(col_var) + eps
        mean_r = np.mean(row_var) + eps
        median_c = np.median(col_var) + eps
        median_r = np.median(row_var) + eps
        res = np.array([tot_var, mean_c, mean_r,
                        median_c, median_r,
                        mean_c/tot_var, mean_r/tot_var, mean_r/mean_c,
                        median_c/tot_var, median_r/tot_var, median_r/median_c])
        # res = np.array([tot_var, mean_c, mean_r,
        #                 mean_r/mean_c, mean_c/mean_r,
        #                 median_c, median_r,
        #                 median_c/median_r , median_r/median_c])
        # res = np.array([tot_var, mean_c, mean_r,
        #                 median_c, median_r])
        # res = np.array([tot_var, mean_c, mean_r, median_c, median_r])
        self.factors = 11
        # print(np.linalg.norm(tot_var))
        # res = tot_var
        return res

    # def calc_table_vector_ensemble(self, tok_tarr, wcvs, wcv_weights, method, fine_grained):
    #     """
    #         get the tokenized table array, build a vec_tarr by calculating vectors for each cell,
    #         and then build a table vector by deviating the vectors in the table, along col, along row
    #     """
    #     # print('======================')
    #     if len(tok_tarr) < 1:
    #         return None
    #     vec_tarr = self.calc_vec_tarr_ensemble(tok_tarr, wcvs, wcv_weights, method=method, fine_grained=fine_grained)
    #     # print(vec_tarr)
    #     tv = self.dev_vectarr2vec_fine_grained(vec_tarr) if fine_grained else self.dev_vectarr2vec(vec_tarr)
    #     return tv

    def calc_table_vector_ensemble_reduced(self, tok_tarr, wcvs, wcv_weights, method, fine_grained):
        """
            get the tokenized table array, build a vec_tarr by calculating vectors for each cell,
            and then build a table vector by deviating the vectors in the table, along col, along row
        """
        # print('======================')
        if len(tok_tarr) < 1:
            return None
        vec_tarr = self.calc_vec_tarr_ensemble(tok_tarr, wcvs, wcv_weights, method=method, fine_grained=fine_grained)
        # print(vec_tarr)
        tv = self.dev_vectarr2vec_fine_grained_reduced(vec_tarr) if fine_grained else self.dev_vectarr2vec_reduced(vec_tarr)
        return tv

    def calc_vec_tarr_ensemble(self, tok_tarr, wcvs, wcv_weights, method='avg', fine_grained=True):
        vec_tarr = []
        for i in range(len(tok_tarr)):
            row = []
            for j in range(len(tok_tarr[i])):
                row.append(None)
            vec_tarr.append(row)
        for i, r in enumerate(tok_tarr):
            for j, c in enumerate(r):
                cv = []
                for x in c:
                    v = None
                    try:
                        vs = []
                        for wcv in wcvs:
                            v = wcv[x]
                            v = my_normalize(v)
                            vs.append(v)
                        if method == 'avg':
                            vs = np.vstack(vs)
                            v = np.average(vs, axis=0, weights=wcv_weights)
                        elif method == 'concat':
                            v = np.hstack(vs)
                        # print('XXX: {}'.format(v))
                    except Exception as e:
                        # print('SHAME: {}'.format(e.message))
                        # v = np.random.rand(self.d)
                        v = None
                        pass
                    if v is not None:
                        cv.append(v)
                if len(cv) == 0:
                    cv = not_found(self)
                if cv is not None and not fine_grained:
                    temp = np.vstack(cv)
                    cv = np.mean(temp, axis=0)
                vec_tarr[i][j] = cv
        return vec_tarr

    def calc_distance_matrix(self, vecs):
        vecs = np.matrix(vecs)
        return euclidean_distances(vecs)
        # return cosine_distances(vecs)

if __name__ == '__main__':
    te = TableEmbedding(None)
    for i in range(1000000):
        x = random.randrange(0, 100)
        if x == 100:
            print 'bingo!'