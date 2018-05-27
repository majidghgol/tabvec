import json
from jsonpath_rw import parse

from data_processing import *
from random_indexing import RandomIndexing
from random_indexing_wrapper import wrap_random_indexing_spark
from pyspark.mllib.feature import Word2Vec


def run_word_embedding_ri(sc, config, occ, doc_counts, word_counts, word_embeddings_path):
    # input_path = sys.argv[1]
    # word_embeddings_path = sys.argv[2]
    # prune word counts by cutoff freq
    word_counts = prune_words(word_counts, config['cut-off'], doc_counts)
    words = [x[0] for x in word_counts]
    print 'done with word pruning, {} different words remained'.format(len(words))

    wrap_random_indexing_spark(sc, occ,
                               words,
                               size=config['vector_dim'], window=config['window'], d=config['nbits']).\
        map(lambda x: json.dumps(dict(word=x[0], vector=x[1].tolist()))).\
            saveAsTextFile(word_embeddings_path)


def run_word_embedding_word2vec(sc, sentences, wepath):
    word2vec = Word2Vec()
    model = word2vec.fit(sentences)
    # model.transform(sentences).saveAsTextFile(wepath)
    print 'saving model to output: {}'.format(wepath)
    model.save(sc, wepath)



def read_occurrences(sc, occ_path, s):
    return sc.union([read_occurrence(sc,occ_path, context) for context in s]).\
                map(lambda x: create_dummy_key(x)).reduceByKey(lambda x,y: (x[0],x[1],x[2]+y[2])).\
                map(lambda x: x[1])


def read_occurrence(sc, occ_path, c):
    return sc.textFile('{}_{}'.format(occ_path, c)).map(lambda x: eval(x))


def create_occurrences(sc, tok_table_path, tok_text_path, occ_path,
                       context, window=2, max_product_th=100):
    if context == 'text':
        occ = sc.textFile(tok_text_path). \
            flatMap(lambda x: get_text_occurrences(json.loads(x), window))
    else:
        occ = sc.textFile(tok_table_path). \
            flatMap(lambda x: get_occurrences(json.loads(x), window, max_product_th, [context]))
    occ.map(lambda x: create_dummy_key(x)).reduceByKey(lambda x,y: (x[0],x[1],x[2]+y[2])).\
        map(lambda x: x[1]).saveAsTextFile(occ_path)
    return occ


def create_context_sentences(sc, tok_table_path, tok_text_path, outpath,
                             context, window=2, max_product_th=100):
    if context == 'text':
        ss = sc.textFile(tok_text_path). \
            flatMap(lambda x: json.loads(x)['text_sentences']). \
            saveAsTextFile(outpath)
    else:
        ss = sc.textFile(tok_table_path). \
            flatMap(lambda x: get_sentences(json.loads(x), window, max_product_th, [context])). \
            saveAsTextFile(outpath)
    # ss.map(lambda x: create_dummy_key(x)).reduceByKey(lambda x,y: (x[0],x[1],x[2]+y[2])).\
    #     map(lambda x: x[1]).saveAsTextFile(occ_path)
    return ss


def read_context_sentences(sc, csentence_path, s):
    return sc.union([read_context_sentence(sc, csentence_path, context) for context in s])


def read_context_sentence(sc, csentence_path, c):
    return sc.textFile('{}_{}'.format(csentence_path, c)).map(lambda x: TextToolkit.tokenize_text(x))

def create_dummy_key(occ):
    w1 = occ[0]
    w2 = occ[1]
    weight = occ[2]
    k = w1+w2 if w1<w2 else w2+w1
    return (k, (w1,w2,weight))
