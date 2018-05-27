import json
from jsonpath_rw import parse

from data_processing import *
from random_indexing import RandomIndexing
from random_indexing_wrapper import wrap_random_indexing_spark


def run_tokenize_table(sc, config, input_path, tok_tables_path):
    # tokenize tables
    tok_tables = sc.textFile(input_path). \
        flatMap(lambda x: get_table_from_jpath(json.loads(x), config['table_path'], 2, 2)). \
        map(lambda x: create_tokenized_table(x, config['put_extractions'], config['regularize_tokens'])). \
        map(lambda x: json.dumps(x))
    if tok_tables_path is not None:
        tok_tables.saveAsTextFile(tok_tables_path, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
    print 'done with creating tokenized tables'
    return tok_tables

def wrap_text_sentences(doc, text_path):
    res = dict()
    if doc is None:
        return dict(cdr_id="NONE", text_sentences=[])
    if '_id' in doc:
        cdr_id = doc['_id']
    elif 'cdr_id' in doc:
        cdr_id = doc['cdr_id']
    elif 'doc_id' in doc:
        cdr_id = doc['doc_id']
    elif 'document_id' in doc:
        cdr_id = doc['document_id']
    else:
        raise Exception('cdr_id not found in {}'.format(json.dumps(doc, sort_keys=True, indent=4)))
    res['cdr_id'] = cdr_id
    sentences = []
    my_parser = parse(text_path)
    for match in my_parser.find(doc):
        val = match.value
        sentences.extend(TextToolkit.get_text_sentences(val))
    res['text_sentences'] = sentences
    return res

def run_tokenize_text(sc, config, input_path, text_tok_path):
    words_in_page = sc.textFile(input_path). \
        map(lambda x: wrap_text_sentences(json.loads(x), config['text_path'])). \
        map(lambda x: json.dumps(x)). \
        saveAsTextFile(text_tok_path, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
    print 'done with creating tokenized text'