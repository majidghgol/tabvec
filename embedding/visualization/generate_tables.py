import json
from StringIO import StringIO
from create_figures import transform_context_label, n_map, d_map

if __name__ == '__main__':
    res = StringIO()
    res.write('''
        \\begin{table}
            \\begin{tabular}{|c|c|c|c|c|c|c|c|}
            \hline
            &\multirow{2}{*}{system}& \multicolumn{5}{c|}{per class F1 score}&\multirow{2}{*}{F1-M}\\\\
            \cline{3-7}
            && R & E & M & L & ND & \\\\
    ''')
    for d in ['ATF', 'HT', 'SEC', 'WCC']:
    # for d in ['HT']:
        jpath = 'cl_kmeans_n{}_0.0001_reg_{}_cell.jl.tar.gz'.format(n_map[d], d_map[d])
        inpath_tabvec = '../../output/evaluation/{}.json'.format(d)
        inpath_eberius = '../../output/evaluation/{}/eberius_result.json'.format(d)
        inpath_wc = '../../output/evaluation/{}/wc_result.json'.format(d)
        inpath_tabnet = '../../output/evaluation/{}/tabNet_result.json'.format(d)
        tabvec_res = json.load(open(inpath_tabvec))[jpath]
        eberius_res = json.load(open(inpath_eberius))
        wc_res = json.load(open(inpath_wc))
        tabnet_res = json.load(open(inpath_tabnet))
        # print tabvec_res
        # exit(0)

        eb_f_r = eberius_res['relational']['f']
        eb_f_m = eberius_res['matrix']['f']
        eb_f_e = eberius_res['entity']['f']
        eb_f_l = eberius_res['list']['f']
        eb_f_n = eberius_res['nondata']['f']
        eb_acc = eberius_res['acc']

        wc_f_r = wc_res['relational']['f']
        wc_f_m = wc_res['matrix']['f']
        wc_f_e = wc_res['entity']['f']
        wc_f_l = wc_res['list']['f']
        wc_f_n = wc_res['nondata']['f']
        wc_acc = wc_res['acc']

        # tabnet_f_r = wc_res['relational']['f']
        # tabnet_f_m = wc_res['matrix']['f']
        # tabnet_f_e = wc_res['entity']['f']
        # tabnet_f_l = wc_res['list']['f']
        # tabnet_f_n = wc_res['nondata']['f']
        # tabnet_acc = wc_res['acc']
        tabnet_f_r = tabnet_res['relational']['f']
        tabnet_f_m = tabnet_res['matrix']['f']
        tabnet_f_e = tabnet_res['entity']['f']
        tabnet_f_l = tabnet_res['list']['f']
        tabnet_f_n = tabnet_res['nondata']['f']
        tabnet_acc = tabnet_res['acc']

        tabvec_f_r = tabvec_res['relational']['f']
        tabvec_f_m = tabvec_res['matrix']['f']
        tabvec_f_e = tabvec_res['entity']['f']
        tabvec_f_l = tabvec_res['list']['f']
        tabvec_f_n = tabvec_res['nondata']['f']
        tabvec_acc = tabvec_res['acc']

        res.write(
            '\hline'+
        	'\parbox[t]{2mm}{\multirow{4}{*}{\\rotatebox[origin=c]{90}{'+d+'}}}'+
            '&webc&'+
            '{0:.2f}&{1:.2f}&{2:.2f}&{3:.2f}&{4:.2f}&{5:.2f}\\\\\n'.format(wc_f_r, wc_f_e, wc_f_m, wc_f_l, wc_f_n, wc_acc) +
            '\cline{2-8}'+
            '&DWTC&'+
            '{0:.2f}&{1:.2f}&{2:.2f}&{3:.2f}&{4:.2f}&{5:.2f}\\\\\n'.format(eb_f_r, eb_f_e, eb_f_m, eb_f_l, eb_f_n, eb_acc)+
            '\cline{2-8}'+
            '&TabNet&'+
            '{0:.2f}&{1:.2f}&{2:.2f}&{3:.2f}&{4:.2f}&{5:.2f}\\\\\n'.format(tabnet_f_r, tabnet_f_e, tabnet_f_m, tabnet_f_l, tabnet_f_n, tabnet_acc)+
            '\cline{2-8}'+
            '&TabVec&'+
            '{0:.2f}&{1:.2f}&{2:.2f}&{3:.2f}&{4:.2f}&{5:.2f}\\\\\n'.format(tabvec_f_r, tabvec_f_e, tabvec_f_m, tabvec_f_l, tabvec_f_n, tabvec_acc)+
            '\hline\n'
        )
    res.write('\end{tabular}')
    print res.getvalue()

    print '=================== GT STAT ===================\n\n'

    res = StringIO()
    res.write(
        '\\begin{tabular}{ | c | c | c | c | c | c | c |}\n'+
        '\hline\n'+
         '& entity & relational & matrix & list & non - data & sum\\\\\n'+
        '\hline\n'+
        '\hline\n'
    )
    for d in ['ATF', 'HT', 'SEC', 'WCC']:
        counts = dict()
        counts['LIST'] = 0
        with open('/Users/majid/DIG/data/{}_annotated.jl'.format(d)) as ann_file:
            for line in ann_file:
                obj = json.loads(line)
                if 'THROW' in obj['labels']:
                    continue
                l = obj['labels'][0]
                if l not in counts:
                    counts[l] = 0
                counts[l] += 1
        res.write(
            '{} & {} & {} & {} & {} & {} & {}\\\\\n'.format(d,
                                                            counts['ENTITY'],
                                                            counts['RELATIONAL'],
                                                            counts['MATRIX'],
                                                            counts['LIST'],
                                                            counts['NON-DATA'],
                                                            sum(counts.values()))
        )
    res.write(
        '\hline\n'+
        '\end{tabular}\n'
    )
    print res.getvalue()

    print '================== CONTEXT SENTENCES HT ========='
    sentences = [
        ['text'],
        ['cell'],
        ['text', 'cell'],
        ['cell', 'hrow', 'hcol'],
        ['cell', 'hrow', 'hcol', 'adjcell'],
        ['text', 'hrow', 'cell', 'adjcell', 'hcol']
    ]
    res = StringIO()
    infile = '/Users/majid/DIG/tabvec/output/evaluation/{}.json'
    eval_res = json.load(open(infile.format('HT')))
    res.write(
        '\\begin{tabular}{ | c | c | c | c | c | c | c |}\n'+
        '\hline\n'+
        '\multirow{2}{*}{context} & \multicolumn{5}{c |}{perclass F1 score} & \multirow{2}{*}{F1-M}\\\\\n'+
        '\cline{2-6}\n'+
        '& R & E & M & L & ND &  \\\\\n'+
        '\hline\n'
    )

    for s in sentences:
        s = '_'.join(s)
        x = transform_context_label(s)
        k = 'cl_kmeans_n{}_0.0001_reg_{}_{}.jl.tar.gz'.format(n_map['HT'], d_map['HT'], s)
        res.write(
            '{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\\n'.format(x,
                                                                              eval_res[k]['relational']['f'],
                                                                              eval_res[k]['entity']['f'],
                                                                              eval_res[k]['matrix']['f'],
                                                                              eval_res[k]['list']['f'],
                                                                              eval_res[k]['nondata']['f'],
                                                                              eval_res[k]['acc'])+
            '\hline\n'
        )

    res.write('\end{tabular}\n')
    print res.getvalue()

    print '================== REG HT ========='
    res = StringIO()
    res.write(
        '\\begin{tabular}{ | c | c | c | c | c | c | c |}\n' +
        '\hline\n' +
        '\multirow{2}{*}{} & \multicolumn{5}{c |}{perclass F1 score} & \multirow{2}{*}{F1-M}\\\\\n' +
        '\cline{2-6}\n' +
        '& R & E & M & L & ND &  \\\\\n' +
        '\hline\n'
    )

    s = '_'.join(s)
    x = transform_context_label(s)
    k = 'cl_kmeans_n{}_0.0001_reg_{}_cell.jl.tar.gz'.format(n_map['HT'], d_map['HT'])
    res.write(
        'w/ regularization & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\\n'.format(
                                                                                eval_res[k]['relational']['f'],
                                                                                eval_res[k]['entity']['f'],
                                                                                eval_res[k]['matrix']['f'],
                                                                                eval_res[k]['list']['f'],
                                                                                eval_res[k]['nondata']['f'],
                                                                                eval_res[k]['acc']) +
        '\hline\n'
    )
    k = 'cl_kmeans_n{}_0.0001_noreg_{}_cell.jl.tar.gz'.format(n_map['HT'], d_map['HT'])
    res.write(
        'w/o regularization & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\\n'.format(
                                                                               eval_res[k]['relational']['f'],
                                                                               eval_res[k]['entity']['f'],
                                                                               eval_res[k]['matrix']['f'],
                                                                               eval_res[k]['list']['f'],
                                                                               eval_res[k]['nondata']['f'],
                                                                               eval_res[k]['acc']) +
        '\hline\n'
    )

    res.write('\end{tabular}\n')
    print res.getvalue()



