import os
import sys
import json
from sklearn.metrics import silhouette_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'toolkit'))
from toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit


def load_GT(GT_path):
    gt = []
    with open(GT_path) as gt_file:
        for line in gt_file:
            jobj = json.loads(line)
            if 'THROW' in jobj['labels']:
                continue
            l = jobj['labels'][0]
            if l == 'NON-DATA':
                l = 'nondata'
            l = l.lower()

            gt.append(dict(cdr_id=jobj['cdr_id'],
                           fingerprint=jobj['fingerprint'],
                           label=l))
    return gt

def calc_cluster_coherency(table_vecs, table_clusters):
    return silhouette_score(table_vecs, table_clusters)

def get_GT_tables(table_clusters, gt):
    gt_ids = set([(x['cdr_id'], x['fingerprint']) for x in gt])
    tables = dict()
    for t in table_clusters:
        cdr_id = t['cdr_id']
        fingerprint = t['fingerprint']
        if (cdr_id, fingerprint) in gt_ids:
            if cdr_id not in tables:
                tables[cdr_id] = dict()
            tables[cdr_id][fingerprint] = t
    count = 0
    for x in gt:
        if x['cdr_id'] not in tables:
            # print 'cdr id not found: {}'.format(x['cdr_id'])
            count +=1
        elif x['fingerprint'] not in tables[x['cdr_id']]:
            # print 'fingerprint not found {}'.format(x['fingerprint'])
            count +=1
    print '{} tables not found out of {}'.format(count, len(gt))
    # exit(0)
    new_gt = []
    res = []
    for x in gt:
        if x['cdr_id'] in tables and x['fingerprint'] in tables[x['cdr_id']]:
            new_gt.append(x)
            res.append(tables[x['cdr_id']][x['fingerprint']])
    return res, new_gt

def get_GT_tables2(table_clusters, gt):
    gt_ids = set([x['fingerprint'] for x in gt])
    tables = dict()
    for t in table_clusters:
        fingerprint = t['fingerprint']
        if fingerprint in gt_ids:
            tables[fingerprint] = t
    count = 0
    for x in gt:
        if x['fingerprint'] not in tables:
            print 'fingerprint not found {}'.format(x['fingerprint'])
            count += 1
    print '{} tables not found out of {}'.format(count, len(gt))
    exit(0)
    return [tables[x['cdr_id']][x['fingerprint']] for x in gt]

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


def evaluate_clustering(table_vecs, table_clusters, GT_path):
    # load GT
    gt = load_GT(GT_path)

    # calculate cluster coherency on the whole tables
    clusters = [x['cluster'] for x in table_clusters]
    # coherrency = calc_cluster_coherency(table_vecs, clusters)
    # prune tables not in GT, same order as GT
    clusters_gt, gt = get_GT_tables(table_clusters, gt)

    # convert cluster numbers to labels
    true_labels = [x['label'] for x in gt]
    pred_clusters = [x['cluster'] for x in clusters_gt]
    pred_labels = cluster_to_label(pred_clusters, true_labels)
    # evaluate
    return evaluate_classification(true_labels, pred_labels)