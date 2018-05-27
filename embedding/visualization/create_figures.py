import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'toolkit'))
from toolkit import VizToolkit, MLToolkit

d_map = {
    'HT': 200,
    'SEC': 200,
    'WCC': 200,
    'ATF': 200
}

n_map = {
    'HT': 12,
    'SEC': 12,
    'WCC': 12,
    'ATF': 12
}

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

def plot_subfig(data, ll, x_title=None, y_title=None,
                datalabels=None, legend_cols=None,
                xlim=None, ylim=None, marker_size=None,
                putlegend=True):
    plots = []
    colors = ['r', 'b', 'g', 'o', 'y']
    markers = ['+', '*', 'o', '>', 'x', 's', 'X']

    for i, l in enumerate(ll):
        plots.append(plt.plot(data[l], '{}-'.format(markers[i]),
                              label=datalabels[i] if datalabels else str(l),
                              markersize=12 if not marker_size else marker_size)[0])
    if putlegend:
        plt.legend(plots, ll, bbox_to_anchor=(0.5, 0.85),
                   ncol=4 if not legend_cols else legend_cols,
                   fontsize=16, frameon=True,
                   edgecolor='black', loc='lower center', shadow=True)
    if x_title:
        plt.xlabel(x_title, fontsize=18)
    if y_title:
        plt.ylabel(y_title, fontsize=18)
    plt.tick_params(axis='both', labelsize=16)
    # plt.plot(data['noreg'], 'b*')
    # plt.ylim((0.4,0.85))
    if ylim:
        plt.ylim(ylim)
    plt.grid(b=None, which='both', axis='y')
    # plt.grid(b=None, which='minor', axis='y')
    labels = data['labels']
    plt.xticks(np.arange(len(labels)), labels)

def plot_domains_regularization():
    data = dict(labels=domains)
    d = 200
    n = 10
    c = 'cell'
    ll = ['reg', 'noreg']
    plt.figure(figsize=(6, 3))
    for i, l in enumerate(ll):
        r = l
        data[l] = []
        for dd in data['labels']:
            eval_res = json.load(open(infile.format(dd)))
            k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n, r, d, c)
            data[l].append(eval_res[k]['acc'])
    ii = data['labels'].index('SEC')
    data['labels'][ii] = 'microcap'
    print data['labels']
    plot_subfig(data, ll,
                y_title='F1 Micro',
                ylim=(0.4,0.85))
    plt.tight_layout()
    plt.savefig('figs/domains_reg.pdf')

def plot_domains_sentences():
    data = dict(labels=['_'.join(s) for s in sentences])

    ll = domains
    r = 'reg'
    plt.figure(figsize=(8, 3))
    for i, l in enumerate(ll):
        eval_res = json.load(open(infile.format(l)))
        d = d_map[l]
        n = n_map[l]
        if l == 'SEC':
            l = 'microcap'
        data[l] = []
        for c in data['labels']:
            k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n, r, d, c)
            data[l].append(eval_res[k]['acc'])
    data['labels'] = [transform_context_label(x) for x in data['labels']]
    plot_subfig(data, ['microcap' if l == 'SEC' else l for l in ll],
                y_title='F1 Micro',
                ylim=(0.4,0.85))
    plt.tight_layout()
    plt.savefig('figs/domains_sentences.pdf')

def plot_ncluster_HT():
    data = dict(labels=nn)

    ll = nn
    d = 200
    r = 'reg'
    c = 'cell'
    plt.figure(figsize=(8, 4))
    ll2 = ['ATF', 'HT', 'microcap', 'WCC']
    ll = ['ATF', 'HT', 'SEC', 'WCC']
    for domain in ll:
        eval_res = json.load(open(infile.format('{}_cl'.format(domain))))
        if domain == 'SEC':
            domain = 'microcap'
        data[domain] = []
        for n in data['labels']:
            k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n, r, d, c)
            data[domain].append(eval_res[k]['distortion'])
    plot_subfig(data, ll2, putlegend=True,
                y_title='Sum of Squared Errors')
    plt.tight_layout()
    plt.savefig('figs/nclusters_elbow_HT.pdf')
    plt.clf()
    data = dict(labels=nn)
    plt.figure(figsize=(8, 3))
    for domain in ll:
        eval_res = json.load(open(infile.format('{}_cl'.format(domain))))
        if domain == 'SEC':
            domain = 'microcap'
        data[domain] = []
        for n in data['labels']:
            k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n, r, d, c)
            data[domain].append(eval_res[k]['sil'])
    plot_subfig(data, ll2, putlegend=True,
                ylim=(0.3,0.8),
                y_title='Silhouette Score',
                x_title='n')
    plt.tight_layout()
    plt.savefig('figs/nclusters_sil_HT.pdf')

def plot_vecsize_only_HT():
    data = dict(labels=dd)

    ll = dd
    n = 10
    r = 'reg'
    c = 'cell'
    plt.figure(figsize=(8, 4))
    ll2 = ['ATF', 'HT', 'microcap', 'WCC']
    ll = ['ATF', 'HT', 'SEC', 'WCC']
    for domain in ll:
        eval_res = json.load(open(infile.format('domains_d/{}_cl'.format(domain))))
        print eval_res
        if domain == 'SEC':
            domain = 'microcap'
        data[domain] = []
        for d in data['labels']:
            k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n, r, d, c)
            if k in eval_res:
                data[domain].append(eval_res[k]['acc'])
            else:
                data[domain].append(0)
    plot_subfig(data, ll2, putlegend=True,
                ylim=(0.4, 0.9),
                y_title='F1 Micro',
                x_title='d')
    plt.tight_layout()
    plt.savefig('figs/vecsize_HT.pdf')
    plt.clf()

def plot_vecsize_HT():

    ll = ['reg', 'noreg']
    d = 200
    r = 'reg'
    n = 10
    ll = [20, 50, 100, 200, 400, 500]
    plt.figure(figsize=(9, 5))
    for i, d in enumerate(['HT']):
        data = dict(labels=['_'.join(s) for s in sentences])
        eval_res = json.load(open(infile.format(d)))
        plt.subplot(1, 1, i + 1)
        for l in ll:
            kk = 'd={}'.format(l)
            data[kk] = []
            d = l
            for c in data['labels']:
                k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n, r, d, c)
                data[kk].append(eval_res[k]['acc'])
        data['labels'] = [transform_context_label(x) for x in data['labels']]
        plot_subfig(data, ['d={}'.format(l) for l in ll])
    plt.tight_layout()
    plt.savefig('figs/vecsize_n_HT.pdf')
    plt.clf()

def plot_vecsize_ncluster_HT():
    ll = ['reg', 'noreg']
    r = 'reg'
    c = 'text_cell'
    ll = [20, 50, 100, 200, 400, 500]

    plt.figure(figsize=(9, 5))
    for i, d in enumerate(['HT']):
        data = dict(labels=nn)
        eval_res = json.load(open(infile.format(d)))
        plt.subplot(1, 1, i + 1)
        for l in ll:
            kk = 'd={}'.format(l)
            data[kk] = []
            d = l
            for n in data['labels']:
                k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n, r, d, c)
                data[kk].append(eval_res[k]['acc'])
        plot_subfig(data, ['d={}'.format(l) for l in ll],
                    legend_cols=3, y_title='Accuracy',
                    ylim=(0.55, 0.85), marker_size=16)
    plt.tight_layout()
    plt.savefig('figs/vecsize_ncluster_HT.pdf')
    plt.clf()

def plot_conf_matrixes():
    plt.figure(figsize=(8, 8))
    for i, d in enumerate(['ATF', 'HT', 'SEC', 'WCC']):
        # plot confusion matrix
        for j, base in enumerate(['wc', 'eberius', 'tabNet', 'tabvec']):
            if base == 'tabvec':
                k = 'cl_kmeans_n{}_0.0001_reg_{}_cell.jl.tar.gz'.format(n_map[d], d_map[d])
                cm = json.load(open(infile.format(d)))[k]['conf']
            else:
                cm = json.load(open(infile2.format(d, base)))['conf']
            cm = np.matrix(cm)
            plt.subplot(4, 4, i * 4 + j + 1)
            viztk.plot_confusion_matrix(cm, ['a', 'b', 'c', 'd', 'e'], show=False, vmax=1.0)
            # plt.figure(1)
            # plt.axes().set_aspect('equal')

    plt.tight_layout()
    plt.savefig('figs/conf_matrices.pdf')
    plt.clf()

def plot_reg_HT():
    eval_res = json.load(open(infile.format('HT')))
    n = 10
    d = 200
    fscores = dict()
    categories = ['w/ regularization', 'w/o regularization']
    k = 'cl_kmeans_n{}_0.0001_reg_{}_cell.jl.tar.gz'.format(n_map['HT'], d_map['HT'])
    fscores['w/ regularization'] = dict(
        R=eval_res[k]['relational']['f'],
        M=eval_res[k]['matrix']['f'],
        E=eval_res[k]['entity']['f'],
        L=eval_res[k]['list']['f'],
        ND=eval_res[k]['nondata']['f'],
        Acc=eval_res[k]['acc']
    )
    k = 'cl_kmeans_n{}_0.0001_noreg_{}_cell.jl.tar.gz'.format(n_map['HT'], d_map['HT'])
    fscores['w/o regularization'] = dict(
        R=eval_res[k]['relational']['f'],
        M=eval_res[k]['matrix']['f'],
        E=eval_res[k]['entity']['f'],
        L=eval_res[k]['list']['f'],
        ND=eval_res[k]['nondata']['f'],
        Acc= eval_res[k]['acc']
    )
    labels = ['R', 'E', 'M', 'L', 'ND', 'Acc']
    viztk.plot_categorical_multibar(fscores,
                                     categories,
                                     labels,
                                     max_y=1.0,
                                     save_to_file='figs/reg_HT.pdf')

def plot_sentences_HT():
    ss = ['_'.join(s) for s in sentences]
    categories = [transform_context_label(x) for x in ss]
    eval_res = json.load(open(infile.format('HT')))
    n = 10
    d = 200
    fscores = dict()
    for s in ss:
        k = 'cl_kmeans_n12_0.0001_reg_200_{}.jl.tar.gz'.format(s)
        fscores[transform_context_label(s)] = dict(
            R=eval_res[k]['relational']['f'],
            M=eval_res[k]['matrix']['f'],
            E=eval_res[k]['entity']['f'],
            L=eval_res[k]['list']['f'],
            ND=eval_res[k]['nondata']['f'],
            Acc=eval_res[k]['acc']
        )
    labels = ['R', 'E', 'M', 'L', 'ND', 'Acc']
    viztk.plot_categorical_multibar(fscores,
                                     categories,
                                     labels,
                                     max_y=1.0,
                                     save_to_file='figs/sentences_HT.pdf')

def tabvecs_2d_plot():
    gt = [json.loads(x) for x in open(gtfile.format('HT'))]
    tables = [json.loads(x) for x in open(gttables.format('HT'))]
    tables = dict([((x['cdr_id'],x['fingerprint']),x) for x in tables])
    vecs = []
    ll = []
    for x in gt:
        if 'THROW' in x['labels']:
            continue
        v = tables[(x['cdr_id'], x['fingerprint'])]['vec']
        if sum([vv**2 for vv in v]) == 0:
            continue
        vecs.append(v)
        ll.append(x['labels'][0])

    vecs = np.array(vecs)
    vecs = mltk.manifold_TSNE(vecs)
    plt.figure(figsize=(12, 10)).set_dpi(400)
    plt.ylim(-20, 20)

    viztk.plot_x_pca_v5(vecs, ll,
                        colors=['blue', 'green', 'crimson', 'purple', 'black'],
                        markers=['o', '*', 'v', 'X', 'P'],
                        classes=['RELATIONAL', 'ENTITY', 'MATRIX', 'LIST', 'NON-DATA'],
                        save_to_file='figs/scatter_HT.pdf')


if __name__ == '__main__':
    gtfile = '/Users/majid/DIG/data/{}_annotated.jl'
    gttables = '/Users/majid/DIG/data/{}_annotated_tables_cl.jl'

    infile = '/Users/majid/DIG/tabvec/output/evaluation/{}.json'
    infile2 = '/Users/majid/DIG/tabvec/output/evaluation/{}/{}_result.json'

    viztk = VizToolkit()
    mltk = MLToolkit()

    # exit(0)

    dd = [20, 50, 100,200, 400]
    nn = [4,6,8,10,12,14]

    sentences = [
        ['text'],
        ['cell'],
        ['text', 'cell'],
        ['cell', 'hrow', 'hcol'],
        ['cell', 'hrow', 'hcol', 'adjcell'],
        ['text', 'hrow', 'cell', 'adjcell', 'hcol']
     ]
    domains = ['ATF', 'HT', 'SEC', 'WCC']


    # num_plots = len(dd)* len(nn)
    # for i, n in enumerate(nn):
    #     for j, d in enumerate(dd):
    #         # get the data
    #         data = dict(labels=['_'.join(s) for s in sentences], reg=[], noreg=[])
    #         for r in ['reg', 'noreg']:
    #             for c in data['labels']:
    #                 k = 'cl_kmeans_n{}_0.0001_{}_{}_{}.jl.tar.gz'.format(n,r,d,c)
    #                 data[r].append(eval_res[k]['acc'])
    #         plt.subplot(len(nn), len(dd), i*len(dd)+j+1)
    #         plot_subfig(data)
    # plot_domains_sentences()
    # plot_conf_matrixes()
    # plot_vecsize_HT()
    # plot_domains_regularization()
    plot_ncluster_HT()
    # plot_vecsize_only_HT()
    # plot_reg_HT()
    # plot_sentences_HT()
    # plot_vecsize_ncluster_HT()
    # tabvecs_2d_plot()

    # plt.plot([1, 2, 3, 4])
    # plt.subplot(2,1,2)
    # plt.plot([1, 2, 3, 4])
    # plt.show()
