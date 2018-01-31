import sys
import os
from jsonpath_rw import parse
import random
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'toolkit'))
from toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit

tabletk = TableToolkit()


def get_table_dim(t):
    return t['features']['no_of_rows'], t['features']['max_cols_in_a_row']


def get_text_words(jobj, text_path):
    my_parser = parse(text_path)
    res = []
    for match in my_parser.find(jobj):
        val = match.value
        res.extend(TextToolkit.tokenize_text(val))
    return res

def get_table_from_jpath(jobj, jpath, minrow, mincol):
    if jobj is None:
        return []
    if '_id' in jobj:
        cdr_id = jobj['_id']
    elif 'cdr_id' in jobj:
        cdr_id = jobj['cdr_id']
    else:
        raise Exception('cdr_id not found in {}'.format(jobj))
    my_parser = parse(jpath)
    res = []
    for match in my_parser.find(jobj):
        val = match.value
        if val is not None:
            row,col = get_table_dim(val)
            if row >= minrow or col>=mincol:
                val['cdr_id'] = cdr_id
                res.append(val)
    return res

def count_tables(jobj, jpath, minrow, mincol):
    # print 'here we are!'
    if jobj is None:
        return 0
    if '_id' in jobj:
        cdr_id = jobj['_id']
    elif 'cdr_id' in jobj:
        cdr_id = jobj['cdr_id']
    else:
        raise Exception('cdr_id not found in {}'.format(jobj))
    count = 0
    my_parser = parse(jpath)
    for match in my_parser.find(jobj):
        val = match.value
        if val is not None:
            row,col = get_table_dim(val)
            if row >= minrow or col>=mincol:
                count +=1
    return count


def create_tokenized_table(t, put_extractions, regularize):
    res = dict()
    res['cdr_id'] = t['cdr_id']
    res['fingerprint'] = t['fingerprint']
    tarr = tabletk.create_table_array(t, put_extractions, regularize)
    tabletk.clean_cells(tarr)
    res['tok_tarr'] = tabletk.create_tokenized_table_array(tarr)
    res['html'] = t['html']
    return res


def count_table_words(tok_tarr):
    counts = dict()
    for row in tok_tarr['tok_tarr']:
        for c in row:
            for w in c:
                counts[w] = 1 if w not in counts else counts[w] + 1
    return counts.items()

def prune_words(word_tuples, cutoff, num_docs):
    min_cutoff = cutoff * float(num_docs)
    max_cutoff = (1.0-cutoff)*float(num_docs)
    res = list()
    for x in word_tuples:
        if x[1] < min_cutoff or x[1] > max_cutoff:
            continue
        res.append(x)
    return res

def get_occurrences(tok_tarr, window, thresh, sentences):
    if 'cell' in sentences:
        # within cell
        tok_tarr = tok_tarr['tok_tarr']
        for row in tok_tarr:
            for c in row:
                for i in range(len(c)):
                    for j in range(max(0, i-window), min(len(c), i+window)):
                        if i != j:
                            yield (c[i], c[j], 1.0)
    if 'adjcell' in sentences:
        # adjacent rows
        for i in range(len(tok_tarr)-1):
            for j in range(len(tok_tarr[i])):
                for x in cross_product_arrays(tok_tarr[i][j], tok_tarr[i+1][j], thresh):
                    yield (x[0], x[1], 1.0)

        # adjacent cols
        for i in range(len(tok_tarr)):
            for j in range(len(tok_tarr[i])-1):
                for x in cross_product_arrays(tok_tarr[i][j], tok_tarr[i][j+1], thresh):
                    yield (x[0], x[1], 1.0)

    if 'hrow' in sentences:
        # first row
        for i in range(1, len(tok_tarr)):
            for j in range(len(tok_tarr[i])):
                for x in cross_product_arrays(tok_tarr[0][j], tok_tarr[i][j], thresh):
                    yield (x[0], x[1], 1.0)

    if 'hcol' in sentences:
        # first col
        for i in range(len(tok_tarr)):
            for j in range(1, len(tok_tarr[i])):
                for x in cross_product_arrays(tok_tarr[i][0], tok_tarr[i][j], thresh):
                    yield (x[0], x[1], 1.0)


def get_text_occurrences(jobj, text_path, window):
    text_parser = parse(text_path)
    for match in text_parser.find(jobj):
        text = match.value
        tokens = TextToolkit.tokenize_text(text)
        for i in range(0, len(tokens)-window):
            for k in range(1, window+1):
                yield (tokens[i], tokens[i+k], 1.0)


def cross_product_arrays(l1, l2, thresh):
    if len(l1) * len(l2) < thresh:
        return itertools.product(l1, l2)
    res = []
    for _ in range(thresh):
        i = random.randrange(0, len(l1))
        j = random.randrange(0, len(l2))
        res.append((l1[i], l2[j]))
    return res

