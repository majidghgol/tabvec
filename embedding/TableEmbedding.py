import json
from jsonpath_rw import parse

from data_processing import *
from random_indexing import RandomIndexing
from random_indexing_wrapper import wrap_random_indexing_spark
from table_embedding import TableEmbedding

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


config = {
    'master': 'local[*]',
    'vector_dim': 100,
    'nbits': 2,
    'window': 2,
    'put_extractions': False,
    'regularize_tokens': False,
    'cut-off': 0.05,
    'table_path': '*.table.tables[*]',
    'max_product_th': 100
}

def put_table_vector(doc, vec):
    return json.dumps(dict(fingerprint=doc['fingerprint'],
                cdr_id=doc['cdr_id'],
                html=doc['html'],
                vec=vec.tolist()))

if __name__ == '__main__':
    conf = SparkConf().setAppName("tableEmbeddingApp") \
        .setMaster(config['master'])
    sc = SparkContext(conf=conf)
    _ = SparkSession(sc)
    sc.addPyFile('../toolkit/toolkit.py')
    sc.addPyFile('data_processing.py')
    sc.addPyFile('random_indexing.py')
    sc.addPyFile('random_indexing_wrapper.py')
    sc.addPyFile('__init__.py')

    input_path = sys.argv[1]
    word_embeddings_path = sys.argv[2]
    # get table_counts
    table_counts = sc.textFile(input_path). \
        map(lambda x: ('xx', count_tables(json.loads(x), config['table_path'], 2, 2))). \
        reduceByKey(lambda v1, v2: v1 + v2).collect()

    table_counts = table_counts[0][1]
    print 'done with table count, {} tables found'.format(table_counts)
    # get word_counts
    word_counts = sc.textFile(input_path). \
        flatMap(lambda x: get_table_from_jpath(json.loads(x), config['table_path'], 2, 2)). \
        map(lambda x: create_tokenized_table(x, config['put_extractions'], config['regularize_tokens'])). \
        flatMap(lambda x: count_table_words(x)). \
        reduceByKey(lambda v1, v2: v1 + v2).collect()
    print 'done with word count, {} different words found'.format(len(word_counts))
    # prune word counts by cutoff freq
    word_counts = prune_words(word_counts, config['cut-off'], table_counts)
    words = [x[0] for x in word_counts]
    print 'done with word pruning, {} different words remained'.format(len(words))

    # generate word embeddings
    # indexer = RandomIndexing(size=config['vector_dim'], d=config['nbits'], window=config['window'], min_count=0)
    # indexer.init_base_vectors(words)
    # indexer.init_vecs()
    word_embeddings = wrap_random_indexing_spark(sc, input_path,
                               words, config['table_path'], config['put_extractions'],
                               config['regularize_tokens'],
                               size=config['vector_dim'], window=config['window'], d=config['nbits'],
                               max_product_th=config['max_product_th'])
    # word_embeddings = json.dumps(dict(word_embeddings))
    word_embeddings = dict(word_embeddings)

    # dump word embedding model
    open(word_embeddings_path, 'w').write(str(word_embeddings))

    # generate table vectors
    te = TableEmbedding()
    sc.textFile(input_path). \
        flatMap(lambda x: get_table_from_jpath(json.loads(x), config['table_path'], 2, 2)). \
        map(lambda x: create_tokenized_table(x, config['put_extractions'], config['regularize_tokens'])). \
        map(lambda x: put_table_vector(x, te.calc_table_vector(x['tok_tarr'], [word_embeddings], [1]))). \
        saveAsTextFile(sys.argv[3])


        #saveAsTextFile(outpath, compressionCodecClass='org.apache.hadoop.io.compress.GzipCodec')
