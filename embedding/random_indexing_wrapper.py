from random_indexing import RandomIndexing
from data_processing import *
import json
def f(x):
    print x


def wrap_random_indexing_spark(sc, occ,
                               words,
                               size=100, window=2, d=2,
                               min_count=5):
    ri = RandomIndexing(size=size, window=window,
                        min_count=min_count, d=d)
    ri.init_base_vectors(words)

    embeddings = occ. \
        flatMap(lambda x: address_occurrence(x, ri)).\
        reduceByKey(lambda v1, v2: v1 + v2)

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
    if b1 is not None and b2 is not None:
        return [(w2, alpha * b1), (w1, alpha * b2)]
    else:
        return []
