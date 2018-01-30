import json
import os
import sys
from jsonpath_rw import jsonpath, parse
import pickle
import gzip
from table_embedding import TableEmbedding
if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'util'))
    from toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit
from __init__ import sentences, regularize_bool, domains

def create_toktarr(t, put_extractions, regularize):
    tarr = tabletk.create_table_array(t, put_extractions, regularize)
    tabletk.clean_cells(tarr)
    res = tabletk.create_tokenized_table_array(tarr)
    return res

if __name__ == '__main__':
    tabletk = TableToolkit()
    method = 'avg'
    data_path = '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/'
    output_path = '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/all_output/'
    for domain in domains:
        gt_tables = [json.loads(x) for x in open(data_path+'{}_sample_tables_tabletype_GT.jl'.format(domain))]
        for regularize_toke0ns in regularize_bool:
            reg = 'reg' if regularize_tokens else 'noreg'
            for cv_name in sentences.keys():
                te = TableEmbedding(aggregate_method=method,
                                    d=200, token_thresh=100)
                cv = pickle.load(gzip.open(data_path+'word_embeddings/{}/{}.{}.pickle.gz'.format(domain,cv_name,reg)))
                outfile = open(output_path+'{}/{}.{}.out.jl'.format(domain, reg, cv_name), 'w')
                for t in gt_tables:
                    cdr_id = t['cdr_id']
                    fingerprint = t['fingerprint']
                    v = te.calc_table_vector(create_toktarr(t, False, regularize_tokens), [cv], [1.0], False)
                    t['vector'] = v.tolist()
                    outfile.write(json.dumps(t) + '\n')



