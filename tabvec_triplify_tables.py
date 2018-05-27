import os,sys,json
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


from hashlib import sha256

from embedding.TableEmbedding import run_table_embedding2
from embedding.TokenizeTableText import run_tokenize_table, run_tokenize_text
from embedding.WordCount import run_word_count
from embedding.WordEmbedding import run_word_embedding, create_occurrences, read_occurrences
from embedding.clustering.TableClustering import run_clustering, run_clustering2
import gzip

def create_triple(st, si, sj, ot, oi, oj, p, cdr_id):
    sid = '{}_r{}_c{}_{}'.format(cdr_id, si, sj, st)
    if ot is None:
        if p == 'left_most':
            oid = 'LEFT_PAD'
        elif p == 'right_most':
            oid = 'RIGHT_PAD'
        elif p == 'top_most':
            oid = 'TOP_PAD'
        elif p == 'bottom_most':
            oid = 'BOTTOM_PAD'
    else:
        oid = '{}_r{}_c{}_{}'.format(cdr_id, oi, oj, ot)
    if p == 'top':
        return (sid, 'TOP', oid)
    elif p == 'bottom':
        return (sid, 'BOTTOM', oid)
    elif p == 'right':
        return (sid, 'RIGH', oid)
    elif p == 'left':
        return (sid, 'LEFT', oid)
    else:
        return (sid, 'UNARY', oid)


def triplify(tok_tarr, cdr_id):
    triples = list()
    n = len(tok_tarr)
    if n == 0:
        return triples
    m = len(tok_tarr[0])
    if m == 0:
        return triples
    for i, row in enumerate(tok_tarr):
        for j, c in enumerate(row):
            ct = '_'.join(c)
            # create unary triples
            if i == 0:
                triples.append(create_triple(ct,i,j,None,-1,-1,'top_most', cdr_id))
            if j == 0:
                triples.append(create_triple(ct,i,j,None,-1,-1,'left_most', cdr_id))
            if i == n-1:
                triples.append(create_triple(ct,i,j,None,-1,-1,'bottom_most', cdr_id))
            if j == m-1:
                triples.append(create_triple(ct,i,j,None,-1,-1,'right_most', cdr_id))
            # only store right and bottom, it is symmetric relation
            if i<n-1:
                ot = '_'.join(tok_tarr[i+1][j])
                triples.append(create_triple(ct, i, j, ot, i+1, j, 'bottom', cdr_id))
            if j<m-1:
                ot = '_'.join(tok_tarr[i][j+1])
                triples.append(create_triple(ct, i, j, ot, i, j+1, 'right', cdr_id))
    return triples


if __name__ == '__main__':
    outfile = gzip.open('/Users/majid/Desktop/elicit_triples.tar.gz', 'w')
    cl_path = '/Users/majid/Desktop/elicit_data_clusters/kmeans_12.tar.gz'
    with gzip.open(cl_path) as infile:
        for line in infile:
            doc = json.loads(line)
            cdr_id = doc['cdr_id']
            tok_tarr = doc['tok_tarr']
            triples = triplify(tok_tarr, cdr_id)
            for t in triples:
                outfile.write('({}\t{}\t{})\n'.format(t[0], t[2], t[1]))

