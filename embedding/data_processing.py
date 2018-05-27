import sys
import os
from jsonpath_rw import parse
import random
import itertools
import json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'toolkit'))
# print sys.path
from toolkit.toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit

tabletk = TableToolkit()

def put_attr(doc, attr_val, attr_name):
    doc[attr_name] = attr_val
    return doc

def load_GT(GT_path):
    gt = []
    gt_ids = set()
    with open(GT_path) as gt_file:
        for line in gt_file:
            jobj = json.loads(line)
            if 'THROW' in jobj['labels']:
                continue
            l = jobj['labels'][0]
            if l == 'NON-DATA':
                l = 'nondata'
            l = l.lower()
            cdr_id = jobj['cdr_id']
            fp = jobj['fingerprint']
            uid = (cdr_id, fp)
            if uid in gt_ids:
                continue
            gt_ids.add(uid)
            gt.append(dict(cdr_id=cdr_id,
                           fingerprint=fp,
                           label=l))

    return gt

def get_unique(tables):
    t_ids = set()
    new_tables = []
    for t in tables:
        cdr_id = t['cdr_id']
        fp = t['fingerprint']
        uid = (cdr_id, fp)
        if uid in t_ids:
            continue
        t_ids.add(uid)
        new_tables.append(t)
    return new_tables

def get_GT_tables(t, gt_ids):
    cdr_id = t['cdr_id']
    fingerprint = t['fingerprint']
    # print fingerprint
    if (cdr_id, fingerprint) in gt_ids:
        return [t]
    return []

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
    elif 'doc_id' in jobj:
        cdr_id = jobj['doc_id']
    elif 'document_id' in jobj:
        cdr_id = jobj['document_id']
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
    tarr_temp = tabletk.create_table_array(t, False, False)
    tabletk.clean_cells(tarr)
    res['tarr'] = tarr_temp
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
    # TODO: prune high freq words as well
    min_cutoff = cutoff * float(num_docs)
    print 'min cutoff: {}'.format(min_cutoff)
    # max_cutoff = (1.0-cutoff)*float(num_docs)
    res = list()
    for x in word_tuples:
        if x[1] < min_cutoff:
            continue
        res.append(x)
    return res

def get_occurrences(tok_tarr, window, thresh, sentences):
    tok_tarr = tok_tarr['tok_tarr']
    if 'cell' in sentences:
        # within cell
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


def get_sample_sentences(cells, thresh):
    if np.prod([len(x) for x in cells]) <= thresh:
        return my_cross_product(cells)
    else:
        res = []
        for _ in range(thresh):
            tempres = ''
            for x in cells:
                i = random.randrange(0, len(x))
                tempres += '\t'+x[i]
            tempres = tempres.strip()
            res.append(tempres)
        return res


def get_sentences(tok_tarr, window, thresh, contexts):
    ttarr = np.array(tok_tarr['tok_tarr'])
    n = len(ttarr)
    if n == 0:
        return
    m = len(ttarr[0])
    if 'cell' in contexts:
        # within cell
        for row in ttarr:
            for c in row:
                yield '\t'.join(c)

    if 'adjcell' in contexts:
        # adjacent rows
        for i in range(n-window):
            for j in range(m):
                try:
                    yield get_sample_sentences(ttarr[i:i+window, j].tolist(), thresh)
                except Exception as e:
                    print e
                    print '---i={},j={},m={},n={},len={}'.format(i, j, m, n, len(ttarr[i]))
                    print ttarr.tolist()
                    sys.exit(-1)

        # adjacent cols
        for i in range(n):
            for j in range(m-window):
                yield get_sample_sentences(ttarr[i, j:j + window], thresh)

    if 'hrow' in contexts:
        # first row
        for i in range(1, n):
            for j in range(m):
                yield get_sample_sentences([ttarr[0, j], ttarr[i, j]], thresh)

    if 'hcol' in contexts:
        # first col
        for i in range(n):
            for j in range(1, m):
                yield get_sample_sentences([ttarr[i, 0], ttarr[i, j]], thresh)

    if 'border' in contexts:
        for i in range(n):
            yield get_sample_sentences([ttarr[i, 0], ['LEFT']], thresh)
        for i in range(n):
            yield get_sample_sentences([ttarr[i, m-1], ['RIGHT']], thresh)
        for j in range(m):
            yield get_sample_sentences([ttarr[0, j], ['TOP']], thresh)
        for j in range(m):
            yield get_sample_sentences([ttarr[n-1, j], ['BOTTOM']], thresh)


def get_text_sentences(jobj):
    if type(jobj) is not dict:
        return
    for s in jobj['text_sentences']:
        yield '\t'.join(TextToolkit.tokenize_text(s))


def get_text_occurrences(jobj, window):
    if type(jobj) is not dict:
        return 
    tokens = jobj['text_sentences']
    for s in tokens:
        s = TextToolkit.tokenize_text(s)
        for i in range(0, len(s)-window):
            for k in range(1, window+1):
                yield (s[i], s[i+k], 1.0)


def cross_product_arrays(l1, l2, thresh):
    if len(l1) * len(l2) < thresh:
        return itertools.product(l1, l2)
    res = []
    for _ in range(thresh):
        i = random.randrange(0, len(l1))
        j = random.randrange(0, len(l2))
        res.append((l1[i], l2[j]))
    return res


def my_cross_product(arrays):
    if len(arrays) == 0:
        return []
    return my_cross_product_helper(arrays[0], arrays[1:])


def my_cross_product_helper(res, arrays):
    if len(arrays) == 0:
        return res
    tempres = [x + '\t' + y for x in res for y in arrays[0]]
    return my_cross_product_helper(tempres, arrays[1:])