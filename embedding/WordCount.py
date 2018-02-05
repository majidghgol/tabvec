import json
from jsonpath_rw import parse

from data_processing import *
from random_indexing import RandomIndexing
from random_indexing_wrapper import wrap_random_indexing_spark

def get_text_tokens(doc):
    return doc['text_tokens']

def run_word_count(sc, tok_table_path, tok_text_path, wcpath, use_text):
    # get word_counts
    word_counts = sc.textFile(tok_table_path). \
        map(lambda x: json.loads(x)).\
        flatMap(lambda x: count_table_words(x))

    if use_text:
        words_in_page = sc.textFile(tok_text_path). \
            flatMap(lambda x: get_text_tokens(x)). \
            map(lambda x: (x, 1))

        word_counts = sc.union([words_in_page, word_counts])

    word_counts = word_counts.reduceByKey(lambda v1, v2: v1 + v2)

    no_words = word_counts.count()

    print 'done with word count, {} different words found'.format(no_words)
    word_counts.saveAsTextFile(wcpath)