import json
import os
import sys
from jsonpath_rw import jsonpath, parse
import pickle
import gzip
import numpy as np
from table_embedding import TableEmbedding
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'util'))
    from toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit
from __init__ import sentences, regularize_bool, domains
# vec_angle = lambda x,y: cosine_distances(x.reshape(1, -1), y.reshape(1, -1)).flatten()[0]
vec_angle = lambda x,y: euclidean_distances(x.reshape(1, -1), y.reshape(1, -1)).flatten()[0]

def create_toktarr(t, put_extractions, regularize):
    tarr = tabletk.create_table_array(t, put_extractions, regularize)
    tabletk.clean_cells(tarr)
    res = tabletk.create_tokenized_table_array(tarr)
    return res

def calc_cell_distance(vec_tarr):
    N = len(vec_tarr)
    M = len(vec_tarr[0])
    print(N, M)

    # n*m
    # conf_matrix = np.zeros((M*N, M*N), dtype='float32')
    # indices = [(i,j) for i in range(N) for j in range(M)]
    # indices_text = [str(x) for x in indices]
    # for i in range(len(indices)):
    #     for j in range(len(indices)):
    #         ii = indices[i][0]
    #         ji = indices[i][1]
    #         ij = indices[j][0]
    #         jj = indices[j][1]
    #         print vec_tarr[ii][ji][0]
    #         conf_matrix[i][j] = vec_angle(vec_tarr[ii][ji][0], vec_tarr[ij][jj][0])
    # viztk.plot_confusion_matrix(conf_matrix,
    #                       indices_text,
    #                       show=True)
    # column std dev
    conf_matrix_column = np.zeros((N,M), dtype='float32')
    for j in range(0, len(vec_tarr[0])):  # range(len(vec_tarr[0])):
        xx = [vec_tarr[i][j][0] for i in range(len(vec_tarr)) if vec_tarr[i][j] is not None]

        xx = np.vstack(xx)
        m = np.median(xx, axis=0)
        for i in range(len(vec_tarr)):
            # print vec_tarr[i][j][0][:10]
            if vec_tarr[i][j] is not None:
                conf_matrix_column[i,j] = vec_angle(vec_tarr[i][j][0], m)
    viztk.plot_dist_matrix(conf_matrix_column,
                            show=True, vmax= 1.0,
                           save_to_file=output_path+'{}_col.pdf'.format(l))
    # row std dev
    conf_matrix_row = np.zeros((N, M), dtype='float32')
    for i in range(0, len(vec_tarr)):
        xx = [vec_tarr[i][j][0] for j in range(len(vec_tarr[i])) if vec_tarr[i][j] is not None]
        xx = np.vstack(xx)

        m = np.median(xx, axis=0)
        for j in range(len(vec_tarr[i])):
            if vec_tarr[i][j] is not None:
                conf_matrix_row[i,j] = vec_angle(vec_tarr[i][j][0], m)
    viztk.plot_dist_matrix(conf_matrix_row,
                            show=True, vmax= 1.0,
                           save_to_file=output_path+'{}_row.pdf'.format(l))
    # total std dev
    conf_matrix_total = np.zeros((N, M), dtype='float32')
    xx = [vec_tarr[i][j][0] for i in range(len(vec_tarr)) for j in range(len(vec_tarr[i])) if
          vec_tarr[i][j] is not None]
    xx = np.vstack(xx)
    m = np.median(xx, axis=0)
    for i in range(0, len(vec_tarr)):
        for j in range(len(vec_tarr[i])):
            if vec_tarr[i][j] is not None:
                conf_matrix_total[i,j] = vec_angle(vec_tarr[i][j][0], m)
    viztk.plot_dist_matrix(conf_matrix_total,
                           show=True, vmax= 1.0,
                           save_to_file=output_path+'{}_total.pdf'.format(l))


if __name__ == '__main__':
    viztk = VizToolkit()
    tabletk = TableToolkit()
    method = 'avg'
    data_path = '../../../data/'
    toshow = [274, 445, 3, 314, 453]
    toshow_label = ['non-data', 'entity', 'list', 'matrix', 'relational']
    # toshow = [3]
    # toshow_label = ['list']

    domain = 'HT'
    gt_tables = [json.loads(x) for x in open(data_path+'{}_sample_tables_tabletype_GT.jl'.format(domain))]

    cv_name = 'cells'
    for reg in ['noreg', 'reg']:
        output_path = '../../../result/figs/examples_{}/'.format(reg)
        te = TableEmbedding(aggregate_method=method,
                            d=200, token_thresh=100)
        cv = pickle.load(gzip.open(data_path+'word_embeddings/{}/{}.{}.pickle.gz'.format(domain,cv_name,reg)))
        regularize = True if reg =='reg' else False
        for i, l in zip(toshow, toshow_label):
            t = gt_tables[i]
            cdr_id = t['cdr_id']
            fingerprint = t['fingerprint']
            vec_tarr = te.calc_vec_tarr_ensemble(create_toktarr(t, False, regularize),[cv], [1.0])
            # v = te.calc_table_vector(create_toktarr(t, False, False), [cv], [1.0], False)
            calc_cell_distance(vec_tarr)



