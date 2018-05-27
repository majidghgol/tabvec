import os
import sys
import json
from sklearn.metrics import silhouette_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from toolkit.toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit
import numpy as np


def calc_cluster_coherency(table_vecs, table_clusters):
    return silhouette_score(table_vecs, table_clusters)


def evaluate_classification(true_labels, pred_labels):
    mltoolkit = MLToolkit()
    eval_res = dict(entity=dict(p=0,r=0,f=0,tp=0,fp=0,fn=0),
                    matrix=dict(p=0, r=0, f=0,tp=0,fp=0,fn=0),
                    relational=dict(p=0, r=0, f=0,tp=0,fp=0,fn=0),
                    list=dict(p=0, r=0, f=0,tp=0,fp=0,fn=0),
                    nondata=dict(p=0, r=0, f=0,tp=0,fp=0,fn=0),
                    acc=0)
    fscore, precision, recall, f_micro, p_micro, r_micro, tp, fp, fn = mltoolkit.get_score_report(pred_labels, true_labels)
    for l in fscore.keys():
        eval_res[l]['p'] = precision[l]
        eval_res[l]['r'] = recall[l]
        eval_res[l]['f'] = fscore[l]
        eval_res[l]['tp'] = tp[l]
        eval_res[l]['fp'] = fp[l]
        eval_res[l]['fn'] = fn[l]
    eval_res['acc'] = f_micro
    return eval_res

def cluster_to_label(clusters, true_labels):
    cl2label = dict()
    for c,l in zip(clusters, true_labels):

        if c not in cl2label:
            cl2label[c] = []
        cl2label[c].append(l)
    for c in cl2label.keys():
        # find majority
        ll = cl2label[c]
        counts = dict()
        for l in ll:
            if l not in counts:
                counts[l] = 0
            counts[l] += 1
        label = ll[0]
        for l in ll:
            if counts[label] < counts[l]:
                label = l
        # print '{} won in {} to {} race'.format(label, counts[label], len(ll))
        cl2label[c] = label
    pred_labels = [cl2label[c] for c in clusters]
    return pred_labels


def evaluate_clustering(table_vecs, cl_model, gt):
    # calculate cluster coherency on the whole tables
    table_vecs = np.vstack([x['vec'] for x in table_vecs])
    # print table_vecs[:10]
    clusters = cl_model.predict(table_vecs)
    # clusters = [cl_model.predict(np.array(x['vec']).reshape(1, -1)) for x in table_vecs]
    # print clusters
    # coherrency = calc_cluster_coherency(table_vecs, clusters)
    # prune tables not in GT, same order as GT
    # clusters_gt, gt = get_GT_tables(table_clusters, gt)

    # convert cluster numbers to labels
    true_labels = [x['label'] for x in gt]
    pred_clusters = clusters
    pred_labels = cluster_to_label(pred_clusters, true_labels)
    # evaluate
    return evaluate_classification(true_labels, pred_labels)