from random_indexing import RandomIndexing
from data_processing import *
import json
def f(x):
    print x


def wrap_random_indexing_spark(sc, input_path,
                               words, table_path, text_path, put_extractions, regularize_tokens,
                               sentences,
                               word_embeddings_path,
                               size=100, window=2, d=2,
                               min_count=5, max_product_th=100):
    ri = RandomIndexing(size=size, window=window,
                        min_count=min_count, d=d)
    ri.init_base_vectors(words)

    embeddings = sc.textFile(input_path). \
        flatMap(lambda x: get_table_from_jpath(json.loads(x), table_path, 2, 2)). \
        map(lambda x: create_tokenized_table(x, put_extractions, regularize_tokens)).\
        flatMap(lambda x: get_occurrences(x, window, max_product_th, sentences)).\
        flatMap(lambda x: address_occurrence(x, ri))
    if 'text' in sentences:
        text_emb = sc.textFile(input_path). \
            flatMap(lambda x: get_text_occurrences(json.loads(x), text_path, window)).\
            flatMap(lambda x: address_occurrence(x, ri))
        embeddings = sc.union([embeddings, text_emb])
    embeddings = embeddings.reduceByKey(lambda v1, v2: v1 + v2)

    return embeddings

def calc_embedding_partition(occurrences, ri):
    embeddings = dict()
    for x in occurrences:
        # print next(x)
        (w1, w2, alpha) = next(x)
        b1 = ri.get_word_base(w1)
        b2 = ri.get_word_base(w2)
        if w1 not in embeddings:
            embeddings[w1] = ri.get_init_embedding()
        if w2 not in embeddings:
            embeddings[w2] = ri.get_init_embedding()
        embeddings[w1] = ri.add_vecs(embeddings[w1], alpha*b2)
        embeddings[w2] = ri.add_vecs(embeddings[w2], alpha * b1)
    res = embeddings.items()
    for x in res:
        yield x
    # print 'here: {}'.format(res)
    # return res

def address_occurrence(o, ri):
    # print next(x)
    (w1, w2, alpha) = o
    b1 = ri.get_word_base(w1)
    b2 = ri.get_word_base(w2)
    return [(w2, alpha * b1),(w1, alpha * b2)]
