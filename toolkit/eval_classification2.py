import sys
import json
from toolkit import MLToolkit
import pickle
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

wc_mapping = {
    'RELATION': 'relational',
    'ENTITY': 'entity',
    'LAYOUT': 'nondata',
    'MATRIX': 'matrix',
    'LIST': 'list',
    'FORM': 'nondata',
    'OTHER': 'nondata',
    'RELATIONAL': 'relational',
    'NON-DATA': 'nondata'
}

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
    classes = ['relational', 'entity', 'matrix', 'list', 'nondata']
    true_labels = [classes.index(x) for x in true_labels]
    pred_labels = [classes.index(x) for x in pred_labels]
    eval_res['conf'] = mltoolkit.calc_conf_matrix(pred_labels, true_labels, classes).tolist()
    return eval_res

def get_GT_tables(table_clusters, gt):
    gt_ids = set([(x['cdr_id'], x['fingerprint']) for x in gt])
    tables = dict()
    for t in table_clusters:
        cdr_id = t['cdr_id']
        fingerprint = t['fingerprint']
        # print fingerprint
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
        else:
            res.append(dict(tableType='NON-DATA'))
    return res, new_gt

def get_GT_tables2(t, gt_ids):
    cdr_id = t['cdr_id']
    fingerprint = t['fingerprint']
    # print fingerprint
    if (cdr_id, fingerprint) in gt_ids:
        return [t]
    return []

if __name__ == '__main__':

    configs = {
        'spark': {
            'spark_master': 'local[20]',  # 'spark://kg2018a.isi.edu:7077',  # spark://majid-Precision-Tower-3420:7077
            # 'spark_master': 'spark://kg2018a.isi.edu:7077',
            'executor_mem': '15g',
            'driver_mem': '15g',
            'driver_max_res_size': '15g',
            'exec_instances': '1',
            'exec_cores': '5'
        }
    }
    mltoolkit = MLToolkit()
    model_file = open(sys.argv[1])
    tabvecs_path = sys.argv[2]
    gt = load_GT(sys.argv[3])
    outfile = open(sys.argv[4], 'w')

    gt_ids = set([(x['cdr_id'], x['fingerprint']) for x in gt])

    conf = SparkConf().setAppName("evaluateClusteringApp") \
        .setMaster(configs['spark']['spark_master']) \
        .set("spark.executor.memory", configs['spark']['executor_mem']) \
        .set('spark.driver.memory', configs['spark']['driver_mem']) \
        .set('spark.driver.maxResultSize', configs['spark']['driver_max_res_size'])

    sc = SparkContext(conf=conf)
    _ = SparkSession(sc)
    sc.addPyFile('tabvec_src.zip')

    intables = sc.textFile(tabvecs_path).flatMap(lambda x: get_GT_tables2(json.loads(x), gt_ids)).collect()


    pred_labels = [wc_mapping[x['tableType']] for x in tables_in]
    true_labels = [x['label'] for x in gt]
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
    classes = ['relational', 'entity', 'matrix', 'list', 'nondata']
    true_labels = [classes.index(x) for x in true_labels]
    pred_labels = [classes.index(x) for x in pred_labels]
    eval_res['conf'] = mltoolkit.calc_conf_matrix(pred_labels, true_labels, classes).tolist()
    json.dump(eval_res, outfile)






