import json
from jsonpath_rw import parse

from data_processing import *
from random_indexing import RandomIndexing
from random_indexing_wrapper import wrap_random_indexing_spark


def run_word_embedding(sc, config, input_path, word_embeddings_path):
    # input_path = sys.argv[1]
    # word_embeddings_path = sys.argv[2]
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
        flatMap(lambda x: count_table_words(x))

    if 'text' in config['sentences']:
        words_in_page = sc.textFile(input_path). \
            flatMap(lambda x: get_text_words(x, config['text_path'])). \
            map(lambda x: (x, 1))

        word_counts = sc.union([words_in_page, word_counts])
    word_counts = word_counts.reduceByKey(lambda v1, v2: v1 + v2).collect()

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
                               words, config['table_path'], config['text_path'], config['put_extractions'],
                               config['regularize_tokens'],
                               config['sentences'],
                               word_embeddings_path,
                               size=config['vector_dim'], window=config['window'], d=config['nbits'],
                               max_product_th=config['max_product_th'])
    word_embeddings.map(lambda x: json.dumps(dict(word=x[0], vector=x[1].tolist()))).\
        saveAsTextFile(word_embeddings_path)