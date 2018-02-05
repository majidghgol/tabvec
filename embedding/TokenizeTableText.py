import json
from jsonpath_rw import parse

from data_processing import *
from random_indexing import RandomIndexing
from random_indexing_wrapper import wrap_random_indexing_spark


def run_tokenize_table(sc, config, input_path, tok_tables_path):
    # get word_counts
    word_counts = sc.textFile(input_path). \
        flatMap(lambda x: get_table_from_jpath(json.loads(x), config['table_path'], 2, 2)). \
        map(lambda x: create_tokenized_table(x, config['put_extractions'], config['regularize_tokens'])). \
        map(lambda x: json.dumps(x)).\
        saveAsTextFile(tok_tables_path)
    print 'done with creating tokenized tables'

def wrap_text_tokens(doc, text_path):
    res = dict()
    if doc is None:
        return dict(cdr_id="NONE", text_tokens=[])
    if '_id' in doc:
        cdr_id = doc['_id']
    elif 'cdr_id' in doc:
        cdr_id = doc['cdr_id']
    else:
        raise Exception('cdr_id not found in {}'.format(jobj))
    res['cdr_id'] = cdr_id
    res['text_tokens'] = get_text_words(doc, text_path)
    return res

def run_tokenize_text(sc, config, input_path, text_tok_path):
    words_in_page = sc.textFile(input_path). \
        map(lambda x: wrap_text_tokens(json.loads(x), config['text_path'])). \
        map(lambda x: json.dumps(x)). \
        saveAsTextFile(text_tok_path)
    print 'done with creating tokenized text'