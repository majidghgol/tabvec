import os
import sys
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import StringIO
from __init__ import easy_samples

if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'util'))
    from toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit
from __init__ import sentences, regularize, domains

def evaluate_clustering(orig_vectors, labels_text):
    n_clusters = len(set(labels_text))
    classes = list(set(labels_text))
    labels = [classes.index(x) for x in labels_text]

    km = KMeans(n_clusters)
    dbs = DBSCAN(eps=0.009, min_samples=5)
    agg = AgglomerativeClustering(n_clusters)

    normalized_vectors = normalize(orig_vectors, axis=0)
    vectors_tsne = mlToolkit.manifold_TSNE(orig_vectors, 5)
    normalized_vectors_tsne = mlToolkit.manifold_TSNE(normalized_vectors, 5)

    header = 'id\tdbs\tkm\tagg\tsvm\trf\ttruel\ttruel_t\n'
    for vectors, l in zip([orig_vectors, orig_vectors[:, 1:3], orig_vectors[:, 3:5], vectors_tsne, normalized_vectors,
                           normalized_vectors_tsne],
                          ['orig', 'orig_mean', 'orig_median', 'tsne5', 'norm', 'norm_tsne5']):
        # outfile = open(out_path+'.{}.res'.format(l), 'w')
        pred_labels = dict()
        pred_labels['dbs'] = mlToolkit.clustering_to_labels(dbs.fit_predict(vectors, labels), labels)
        pred_labels['km'] = mlToolkit.clustering_to_labels(km.fit_predict(vectors, labels), labels)
        pred_labels['agg'] = mlToolkit.clustering_to_labels(agg.fit_predict(vectors, labels), labels)
        svm_labels = SVC(kernel='rbf').fit(vectors, labels).predict(vectors)
        rf_labels = RandomForestClassifier().fit(vectors, labels).predict(vectors)
        for ll in ['km', 'dbs', 'agg']:
            temp = sorted(list(set(zip(labels, labels_text))), key=lambda x: x[0])
            print '{}: {}'.format(l,mlToolkit.get_score_report(pred_labels[ll], labels))
            vizToolkit.plot_confusion_matrix(mlToolkit.calc_conf_matrix(pred_labels[ll], labels, [xx[1] for xx in temp]), [xx[1] for xx in temp],save_to_file=out_path+'.{}.{}.png'.format(ll, l))
            f, p, r, fm, pm, rm, tp, fp, fn = mlToolkit.get_score_report([classes[pred_labels[ll][i]] for i in range(len(labels))],
                                                                         labels_text)
            res = dict()
            res['precision'] = p
            res['recall'] = r
            res['fscore'] = f
            res['tp'] = tp
            res['fp'] = fp
            res['fn'] = fn
            res['fmicro'] = {'fm': fm}
            res['pmicro'] = {'pm': pm}
            res['rmicro'] = {'rm': rm}
            res['classmap'] = dict([(i, x) for i,x in enumerate(classes)])
            res['conf_matrix'] = mlToolkit.calc_conf_matrix(pred_labels[ll], labels, classes).tolist()
            open('../../../result/output/HT/easy/HT_{}_{}_{}_easy.res'.format(cv_name, ll,l),'w').write(json.dumps(res))
        # outfile.write(header)
        # for uri, dbsl, kml, aggl, svml, rfl, truel, truel_t in zip(ids, dbs_labels, km_labels, agg_labels, svm_labels, rf_labels, labels, labels_text):
        #     outfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(uri, dbsl, kml, aggl, svml, rfl, truel, truel_t))
        # outfile.close()

def show_result(orig_vectors, labels_text):
    n_clusters = len(set(labels_text))
    classes = list(set(labels_text))
    labels = [classes.index(x) for x in labels_text]

    km = KMeans(n_clusters)
    dbs = DBSCAN(eps=0.009, min_samples=5)
    agg = AgglomerativeClustering(n_clusters)

    normalized_vectors = normalize(orig_vectors, axis=0)
    vectors_tsne = mlToolkit.manifold_TSNE(orig_vectors, 5)
    normalized_vectors_tsne = mlToolkit.manifold_TSNE(normalized_vectors, 5)

    header = 'id\tdbs\tkm\tagg\tsvm\trf\ttruel\ttruel_t\n'
    # for vectors, l in zip([orig_vectors, orig_vectors[:, 1:3], orig_vectors[:, 3:5], vectors_tsne, normalized_vectors,
    #                        normalized_vectors_tsne],
    #                       ['orig', 'orig_mean', 'orig_median', 'tsne5', 'norm', 'norm_tsne5']):

    vizToolkit.plot_x_pca_v5(orig_vectors[:, 1:3], labels_text, title='{}.{}.{}'.format(domain, cv_name,reg), save_to_file=out_path+'.median.pca.png')

def get_GT_tables(gt_annotations, return_hard_samples=False):
    ann_tables = []
    for t in gt_annotations:
        ll = t['labels']
        if 'THROW' in ll:
            continue
        if not return_hard_samples and t['ishard']:
            continue
        # t['id_'] = (t['cdr_id'], t['fingerprint'])
        ann_tables.append(t)
    return ann_tables

if __name__ == '__main__':
    mlToolkit = MLToolkit()
    vizToolkit = VizToolkit()
    do_hard = False
    GT_path = '../../../result/all_output/'
    res_path = '../../../result/output_easy/'
    for domain in ['HT']:
        easy = []
        for l in easy_samples[domain].values():
            easy += l
        for cv_name in sentences.keys():
            for reg in regularize:
                GT = [json.loads(x) for x in open(GT_path+'{}/{}.{}.out.jl'.format(domain, reg, cv_name))]
                # GT = get_GT_tables(GT, return_hard_samples=do_hard)
                print len(GT)
                GT = [GT[i] for i in easy]
                vectors = [np.array(x['vector']) for x in GT]
                vectors = np.array(vectors).reshape(len(GT), vectors[0].shape[0])
                ids = [(x['cdr_id'], x['fingerprint']) for x in GT]
                labels = [x['labels'][2] for x in GT]

                out_path = res_path + '{}/{}.{}'.format(domain, reg, cv_name)
                show_result(vectors, labels)
                evaluate_clustering(vectors, labels)


