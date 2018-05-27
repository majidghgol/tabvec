__author__ = 'majid'
import json
import os
import sys
import re
import numpy as np
from jsonpath_rw import jsonpath, parse
if __name__ =='__main__' and __package__ is None:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from baselines.nnet.tablenet import TCM
    from toolkit import MLToolkit
    from eval_classification import wc_mapping, evaluate_classification
mltoolkit = MLToolkit()

def tokenize_cell(text, i, j):
    text = re.sub('[<>]', ' ', text)
    text = re.sub('[\w\-]+="[\w/\.\s\-_]+"', ' ', text)
    text = re.sub('"', ' ', text)
    text = re.sub('\s+', ' ', text)

    text = text.strip()
    text = text.lower()

    return ['row{}'.format(i), 'col{}'.format(j)] + re.split('\s+', text)


def tokenize_table(t, num_token=50):
    tok_tarr = []
    num_col = 0
    for i, r in enumerate(t['rows']):
        new_row = []
        for j, c in enumerate(r['cells']):
            if j+1 > num_col:
                num_col = j+1
            text = c['cell']
            tokens = tokenize_cell(text, i, j)
            if len(tokens) < num_token:
                tokens += ['DUMMY']*(num_token-len(tokens))
            else:
                tokens = tokens[:num_token]
            new_row.append(tokens)
        tok_tarr.append(new_row)
    num_row = len(t['rows'])
    return tok_tarr, num_row, num_col

def pad_table(tok_t, rr, cc, num_toks):
    # adjust rows
    if len(tok_t) < rr:
        tok_t += [[['DUMMY']*num_toks]*cc]*(rr-len(tok_t))
    else:
        tok_t = tok_t[:rr]
    # adjust num columns
    for i, row in enumerate(tok_t):
        if len(row) < cc:
            tok_t[i] += [['DUMMY']*num_toks] * (cc-len(row))
        else:
            tok_t[i] = row[:cc]
    for i, row in enumerate(tok_t):
        for j, cell in enumerate(row):
            if len(tok_t[i][j]) < num_toks:
                tok_t[i][j] += ['DUMMY']*(num_toks-len(tok_t[i][j]))
            elif len(tok_t[i][j]) > num_toks:
                tok_t[i][j] = tok_t[i][j][:num_toks]
    for i, row in enumerate(tok_t):
        for j, cell in enumerate(row):
            tok_t[i][j] = ['row{}'.format(i+1),'col{}'.format(j+1)] + tok_t[i][j]
    # print tok_t
    return tok_t



def prepare_tables(tables, num_rows, num_cols, num_tokens):
    new_tables = []
    for t in tables:
        tok_t, r, c = t['tok_tarr'], len(t['tok_tarr']), len(t['tok_tarr'][0])
        tok_t = pad_table(tok_t, num_rows, num_cols, num_tokens)
        new_tables.append(tok_t)
    return new_tables


def get_GT_tables(gt_annotations):
    ann_tables = []
    for t in gt_annotations:
        ll = t['labels']
        if 'THROW' in ll:
            continue
        ann_tables.append(t)
    return ann_tables

def create_vocab(tok_tables):
    vocab = dict()
    counter = 0
    for t in tok_tables:
        for r in t:
            for c in r:
                for tok in c:
                    if tok not in vocab:
                        vocab[tok] = counter
                        counter += 1
    return vocab, counter

def encode_tokens(tok_tables, vocab):
    new_tables = []
    for t in tok_tables:
        new_t = []
        for r in t:
            new_row = []
            for c in r:
                new_c = []
                for tok in c:
                    if tok not in vocab:
                        raise Exception("sharaf baba add kardam ke!")
                    new_c.append(vocab[tok])
                new_row.append(new_c)
            new_t.append(new_row)
        new_tables.append(new_t)
    return new_tables

def prune_tables(tables, gt):
    tables_dict = dict()
    new_gt = []
    res = []
    for t in tables:
        tables_dict[(t['cdr_id'], t['fingerprint'])] = t

    for t in gt:
        k = (t['cdr_id'], t['fingerprint'])
        if k in tables_dict:
            new_gt.append(t)
            res.append(tables_dict[k])
    print '{} tables not found out of {}'.format(len(gt)-len(new_gt), len(gt))
    return res, new_gt


if __name__ == '__main__':
    # text = '<td class=\"popup_menu_content\"><a class=\"nd\" href=\"/ubbthreads/ubbthreads.php/ubb/addfavuser/User/2426/n/12100921/p/1/f/1\" rel=\"nofollow\"><i aria-hidden=\"true\" class=\"fa fa-check-square-o fa-fw\"></i> Follow User</a></td>'
    # print(text)
    # print tokenize_cell(text,1,2)
    # exit(0)

    for domain in ['ATF','HT','SEC', 'WCC']:
        print '=========== doing {} ============='.format(domain)
        num_tokens = 5
        num_row = num_col = 5
        # load annotated tables
        gt_annotations_file = '/Users/majid/DIG/data/{}_annotated.jl'.format(domain)
        path = '/Users/majid/DIG/data/{}_annotated_tables.jl'.format(domain)
        out_path = '/Users/majid/DIG/tabvec/output/evaluation/{}/tabNet_result.json'.format(domain)
        if gt_annotations_file != '':
            # load GT
            gt_annotations = [json.loads(x) for x in open(gt_annotations_file)]
            tables_toshow = get_GT_tables(gt_annotations)
            tables = [json.loads(x) for x in open(path)]
            tables, tables_toshow = prune_tables(tables, tables_toshow)

            tokenized_tables = prepare_tables(tables, num_row,num_col,num_tokens)
            vocab, vocab_size = create_vocab(tokenized_tables)
            tokenized_tables = encode_tokens(tokenized_tables, vocab)
            # print('vocab_size: {}'.format(vocab_size))

            ll = [x['labels'][0] for x in tables_toshow]
            classes = list(set(ll))
            Y = np.array([classes.index(x) for x in ll])
            X = np.array(tokenized_tables, dtype='int32')

            print(X.shape, Y.shape)

            nn = TCM(vocab_size=vocab_size, word_embed_size=100, cell_embed_size=100, num_maps=50,
                     num_blocks=10, hidden_size=100,
                     table_size=5, class_size=len(classes), embed_unchain=True)
            nn.set_library(use_gpu=False)
            nn.set_embed_unchain(True)
            nn.encode(X, False)
            # print X.shape
            # print Y.shape
            # print Y

            loss, acc, report, y_pred = nn.train_classify(X,Y)
            y_pred = y_pred.data
            y_pred = [np.argmax(x) for x in y_pred]
            # print y_pred
            # y_pred, y2 = nn.test_classify(X)
            # print y_pred
            # print y2
            # exit(0)


            pred_labels = [classes[i] for i in y_pred]
            true_labels = [classes[i] for i in Y]
            pred_labels = [wc_mapping[x] for x in pred_labels]
            true_labels = [wc_mapping[x] for x in true_labels]
            eval_res = evaluate_classification(pred_labels, true_labels)
            print eval_res
            json.dump(eval_res, open(out_path, 'w'))
