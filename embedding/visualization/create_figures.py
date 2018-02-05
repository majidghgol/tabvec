import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'toolkit'))

def transform_context_label(label):
    res = []
    if 'text' in label:
        res.append('T')
    if 'cell' in label and 'adjcell' in label:
        res.append('C')
        res.append('A')
    elif 'cell' in label:
        res.append('C')
    if 'hrow' in label and 'hcol' in label:
        res.append('H')
    return '+'.join(res)

def plot_subfig(data):
    plt.plot(data['reg'], 'r+')
    plt.plot(data['noreg'], 'b*')
    plt.ylim((0.5,0.85))
    plt.grid(b=None, which='both', axis='y')
    # plt.grid(b=None, which='minor', axis='y')
    labels = [transform_context_label(x) for x in data['labels']]
    plt.xticks(np.arange(len(labels)), labels)

if __name__ == '__main__':
    infile = '/home/majid/my_drive/DIG/tabvec/output/evaluation/HT.json'
    eval_res = json.load(open(infile))
    print eval_res
    # plt.figure(1)
    dd = [100,200]
    nn = [4,6,8,10,12]

    sentences = [
        ['text'],
        ['cell'],
        ['text', 'cell'],
        ['cell', 'hrow', 'hcol'],
        ['cell', 'hrow', 'hcol', 'adjcell'],
        ['text', 'hrow', 'cell', 'adjcell', 'hcol']
     ]

    num_plots = len(dd)* len(nn)
    for i, n in enumerate(nn):
        for j, d in enumerate(dd):
            # get the data
            data = dict(labels=['_'.join(s) for s in sentences], reg=[], noreg=[])
            for r in ['reg', 'noreg']:
                for c in data['labels']:
                    k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n,r,d,c)
                    data[r].append(eval_res[k]['acc'])
            plt.subplot(len(nn), len(dd), i*len(dd)+j+1)
            plot_subfig(data)

    plt.show()

    # plt.plot([1, 2, 3, 4])
    # plt.subplot(2,1,2)
    # plt.plot([1, 2, 3, 4])
    # plt.show()
