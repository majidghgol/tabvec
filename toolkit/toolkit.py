__author__ = 'majid'
import json
from jsonpath_rw import jsonpath, parse
import StringIO
from itertools import cycle, product
import re
import sys
from nltk.tokenize.api import StringTokenizer, TokenizerI
import nltk
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from copy import copy
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import string
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, isomap, LocallyLinearEmbedding
from matplotlib.ticker import AutoMinorLocator

sys.path.append('/Users/majid/DIG/pycharm-projects-ubuntu-mayank/memex-CP4/')
# from wordEmbeddings.RandomIndexer import RandomIndexer
# from wordEmbeddings.TextAnalyses import TextAnalyses

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# class Embedding:
#     def init_csvs(self, attr_df, include_dummies=False, d=200, non_zero_ratio=0.01):
#         token_cvs = RandomIndexer._generate_context_vectors_for_idf_v2(attr_df, include_dummies=include_dummies,
#                                                                        d=d, non_zero_ratio=non_zero_ratio)
#         return token_cvs
#
#     def create_embeddings(self, token_tuples, attr_val_csv, attr_name_csv,
#                           context_window=(0,2), include_dummies=False, normalize_res=True):
#
#         attr_name_embedding = RandomIndexer.generate_unigram_embeddings_v2(token_tuples, attr_name_csv, attr_val_csv,
#                                                                            output_file=None, context_window=context_window,
#                                                                            include_dummies=include_dummies)
#         if normalize_res:
#             normalize_embeddings(attr_name_embedding)
#
#         attr_val_embedding = RandomIndexer.generate_unigram_embeddings_v2(token_tuples, attr_val_csv, attr_name_csv,
#                                                                           output_file=None, include_dummies=include_dummies,
#                                                                           context_window=(-context_window[1], context_window[0]))
#         if normalize_res:
#             normalize_embeddings(attr_val_embedding)
#         return attr_name_embedding, attr_val_embedding
#
#
#     def tokenize_string(self, s):
#         return s.split()
class TextToolkit:
    def clean_text(self, text, level=1, lower=True):
        lev2_regex = '\\+' # non-printable and \
        lev3_regex = '<[^>,]+>' # html tags
        text = re.sub('\s+', ' ', text)  # white spaces and \n's
        if lower:
            text = text.lower()
        if level > 1:
            text = re.sub(lev2_regex, ' ', text)  # lev2 set of characters
            text = ''.join([x if x in string.printable else ' ' for x in text])
        if level > 2:
            text = re.sub(lev3_regex, ' ', text)  # lev3
        text = re.sub('\s+', ' ', text)  # white spaces and \n's
        return text

    @staticmethod
    def tokenize_text(text):
        res = nltk.word_tokenize(text)
        return res



class TableToolkit:
    def create_table_array(self, t, put_extractions=False, regularize=False):
        rows = t['rows']
        tt = []
        max_cols = t['features']['max_cols_in_a_row']
        for r in rows:
            new_r = ['' for xx in range(max_cols)]
            for i, c in enumerate(r['cells']):
                # id_ = c['id']
                # print(id_)
                # ii,jj = re.findall('row_(\d+)_col_(\d+)', id_)[0]
                # ii = int(ii)
                # jj = int(jj)
                # print(ii,jj)
                text = c['text']
                text = text.lower()
                text = text.strip()
                if put_extractions and 'data_extraction' in c:
                    data_extractions = c['data_extraction']
                    # text = ''
                    for key in data_extractions.keys():
                        text += ' DUMMY' + key.upper()
                new_r[i] = text.strip()
            tt.append(new_r)

        if regularize:
            self.regulize_cells(tt)
        return tt

    def create_tokenized_table_array(self, tarr, threshold=-1):
        tokenized_tarr = []
        for r in tarr:
            new_r = []
            for c in r:
                tok_c = self.tokenize_cell(c)
                if len(tok_c) > threshold > 0:
                    new_r.append(tok_c[:threshold])
                else:
                    new_r.append(tok_c)
            tokenized_tarr.append(new_r)
        return tokenized_tarr

    def create_html_from_array(self, t):
        sio = StringIO.StringIO()
        sio.write('<table border="1">\n')
        sio.write('<tbody>\n')
        for r in t:
            sio.write('<tr>')
            for c in r:
                sio.write('<td>' + str(c) + '</td>')
            sio.write('</tr>')
        sio.write('</tbody>\n')
        sio.write('</table>\n')
        return sio.getvalue()

    def create_dropdown_html(self, items, selected_item, style=None):
        sio = StringIO.StringIO()
        if style is not None:
            sio.write('<select style="{0}">'.format(style))
        else:
            sio.write('<select>')
        for i, x in enumerate(items):
            if selected_item == x:
                sio.write('<option selected="selected" value="{0}">{0}</option>'.format(x))
            else:
                sio.write('<option value="{0}">{0}</option>'.format(x))
        sio.write('</select>')
        return sio.getvalue()

    def create_type_annotation_div(self, l=None, ishard=False):
        text = """
        <table_type__ style="border:2px solid black;width: 100%;float:left;display: table-cell" >
            annotate table type: \n
            <div style="width: 100%;float:left">
            <form>\n
                <input type="checkbox" name="not_good" value="THROW">  throw away this table\n
                <input type="checkbox" name="hard_job" value="HARD">  hard to detect\n
            </form>\n
            </div>
            <div style="width: 30%;float:left">
            <form>\n
                <input type="radio" name="domain_" value="IN-DOMAIN"> IN-DOMAIN<br>\n
                <input type="radio" name="domain_" value="OUT-DOMAIN"> OUT-DOMAIN<br>\n
            </form>
            </div>
            <div style="width: 30%;float:left">
            <form>\n
                <input type="radio" name="layout_" value="LAYOUT"> LAYOUT<br>\n
                <input type="radio" name="layout_" value="NOT-LAYOUT"> NOT-LAYOUT\n
            </form>
            </div>
            <div style="width: 30%;float:left">
            <form>\n
                <input type="radio" name="type_" value="ENTITY"> ENTITY<br>\n
                <input type="radio" name="type_" value="RELATIONAL"> RELATIONAL<br>\n
                <input type="radio" name="type_" value="MATRIX"> MATRIX<br>\n
                <input type="radio" name="type_" value="LIST"> LIST\n
            </form>\n
            </div>
        </table_type__>
        """
        if l is not None:
            for x in l:
                x = '"' + x + '"'
                if x in text:
                    args_ = text.split(x)
                    text = args_[0] + x + ' checked' + args_[1]
        if ishard:
            args_ = text.split('"HARD"')
            text = args_[0] + '"HARD"' + ' checked' + args_[1]
        return text

    def create_html_from_array_with_labels(self, t, l=None, tl=None, valid_labels=None, table_type=None,
                                           cdr_id=None, fingerprint=None, table_index=0, ishard=False,
                                           meta={}):
        sio = StringIO.StringIO()
        sio.write('<table_annotation__ style="display: table-row;width: 100%;float:left;">\n')
        sio.write(self.create_type_annotation_div(table_type, ishard) + '\n')
        sio.write('<table_itself__ style="display: table-cell">\n')
        sio.write('<table border="1">\n')
        sio.write('<tbody>\n')
        for i, r in enumerate(t):
            sio.write('<tr>')
            for c in r:
                if len(c) > 50:
                    c = c[:50]
                    c += '....'
                sio.write('<td>' + c.encode('utf-8') + '</td>')
            sio.write('</tr>')
        sio.write('</tbody>\n')
        sio.write('</table>\n')
        sio.write('</table_itself__>\n')
        # predicted semantic labels
        if l is not None:
            sio.write('<predicted_labels style="display: table-cell" name="predicted_labels">\n')
            sio.write('<table border="1">\n')
            for i, r in enumerate(t):
                sio.write('<tr><td>' + str(l[i].encode('utf-8')) + '</td></tr>')
            sio.write('</table>\n')
            sio.write('</predicted_labels>\n')
        sio.write('<meta__ style="display: table-row;width: 100%;float:left;">\n')
        sio.write('table index: {}'.format(table_index))
        if cdr_id is not None:
            sio.write('<cdr_id__ name="{}"></cdr_id__>\n'.format(cdr_id))
            sio.write('cdr_id: {}'.format(cdr_id))
            # sio.write('<cdr_id__></__cdr_id__>\n')
        if fingerprint is not None:
            sio.write('<fingerprint__ name="{}"></fingerprint__>\n'.format(fingerprint))
            sio.write('fingerprint: {}'.format(fingerprint))
        for k, v in meta.items():
            sio.write('<{0}__ name="{0}"></{1}__>\n'.format(k, v))
            sio.write('{}: {}'.format(k, v))
            # sio.write('<fingerprint__></__fingerprint__>\n')
        sio.write('</meta__>\n')
        sio.write('</table_annotation__>\n')

        # true semantic labels
        if valid_labels is not None:
            if tl is not None:
                sio.write('<true_labels style="display: table-cell" name="true_labels">\n')
                sio.write('<table border="1">\n')
                for i, r in enumerate(t):
                    sio.write('<tr><td>' + self.create_dropdown_html(valid_labels, tl[i] if tl[
                                                                                           i] in valid_labels else None) + '</td></tr>')
                sio.write('</table>\n')
                sio.write('</true_labels>\n')
            elif l is not None:
                sio.write('<true_labels style="display: table-cell" name="true_labels">\n')
                sio.write('<table border="1">\n')
                for i, r in enumerate(t):
                    sio.write('<tr><td>' + self.create_dropdown_html(valid_labels,
                                                                l[i] if l[i] in valid_labels else None) + '</td></tr>')
                sio.write('</table>\n')
                sio.write('</true_labels>\n')

        return sio.getvalue()

    def create_html_from_html_with_labels(self, t_html, l=None, tl=None, valid_labels=None, table_type=None,
                                           cdr_id=None, fingerprint=None, table_index=0, ishard=False,
                                           meta={}):
        sio = StringIO.StringIO()
        sio.write('<table_annotation__ style="display: table-row;width: 100%;float:left;">\n')
        sio.write(self.create_type_annotation_div(table_type, ishard) + '\n')
        sio.write('<table_itself__ style="display: table-cell">\n')
        sio.write('<table border="1">\n')
        sio.write(t_html+'\n')

        sio.write('</table_itself__>\n')
        # predicted semantic labels
        if l is not None:
            sio.write('<predicted_labels style="display: table-cell" name="predicted_labels">\n')
            sio.write('<table border="1">\n')
            for i, r in enumerate(l):
                sio.write('<tr><td>' + str(l[i].encode('utf-8')) + '</td></tr>')
            sio.write('</table>\n')
            sio.write('</predicted_labels>\n')
        sio.write('<meta__ style="display: table-row;width: 100%;float:left;">\n')
        sio.write('table index: {}'.format(table_index))
        if cdr_id is not None:
            sio.write('<cdr_id__ name="{}"></cdr_id__>\n'.format(cdr_id))
            sio.write('cdr_id: {}'.format(cdr_id))
            # sio.write('<cdr_id__></__cdr_id__>\n')
        if fingerprint is not None:
            sio.write('<fingerprint__ name="{}"></fingerprint__>\n'.format(fingerprint))
            sio.write('fingerprint: {}'.format(fingerprint))
        for k, v in meta.items():
            sio.write('<{0}__ name="{0}"></{1}__>\n'.format(k, v))
            sio.write('{}: {}'.format(k, v))
            # sio.write('<fingerprint__></__fingerprint__>\n')
        sio.write('</meta__>\n')
        sio.write('</table_annotation__>\n')

        # true semantic labels
        if valid_labels is not None:
            if tl is not None:
                sio.write('<true_labels style="display: table-cell" name="true_labels">\n')
                sio.write('<table border="1">\n')
                for i, r in enumerate(tl):
                    sio.write('<tr><td>' + self.create_dropdown_html(valid_labels, tl[i] if tl[
                                                                                           i] in valid_labels else None) + '</td></tr>')
                sio.write('</table>\n')
                sio.write('</true_labels>\n')
            elif l is not None:
                sio.write('<true_labels style="display: table-cell" name="true_labels">\n')
                sio.write('<table border="1">\n')
                for i, r in enumerate(t):
                    sio.write('<tr><td>' + self.create_dropdown_html(valid_labels,
                                                                l[i] if l[i] in valid_labels else None) + '</td></tr>')
                sio.write('</table>\n')
                sio.write('</true_labels>\n')

        return sio.getvalue()

    def create_html_page(self, tables, towrite, semantic_labels=None):
        html_file = StringIO.StringIO()
        html_file.write('<html>\n')
        html_file.write('''
            <head>
              <script src = "https://code.jquery.com/jquery-1.10.2.js"></script>
            <script>
                $(function() {
                    $("#submitter").click(function(){
                        if($.trim($('#out_file').val()) == ''){
                            alert("insert out file name!");
                            return;
                        }
                        if($.trim($('#orig_file').val()) == ''){
                            alert("insert input file name!");
                            return;
                        }
                        var cdrid, fingerprint, type, domain, layout, not_good, sem_dict;
                        var alltext = "";
                        tables = document.getElementsByName('table_annotation__');
                        sem_dict = {};
                        for (index = 0; index < tables.length; ++index){

                            cdrid = tables[index].getElementsByTagName("cdrid")[0].getAttribute("value");
                            fingerprint = tables[index].getElementsByTagName("fingerprint")[0].getAttribute("value");
                            not_good = null;
                            layout = null;
                            domain = null;
                            type = null;
                            // deal with inputs[index] element.
                            inputs = tables[index].getElementsByTagName("input");
                            for (i = 0 ; i < inputs.length ; ++i){
                                if(inputs[i].name == "type_")
                                    if(inputs[i].checked)
                                         type = inputs[i].value;
                                if(inputs[i].name == "domain_")
                                    if(inputs[i].checked)
                                         domain = inputs[i].value;
                                if(inputs[i].name == "layout_")
                                    if(inputs[i].checked)
                                         layout = inputs[i].value;
                                if(inputs[i].name == "not_good")
                                    if(inputs[i].checked)
                                         not_good = inputs[i].value;
                            }
                            sem_labels = []
                            true_sem_labels_div = tables[index].getElementsByTagName('true_labels')[0];
                            true_sem_labels_rows = true_sem_labels_div.getElementsByTagName("select");
                            for(i = 0 ; i < true_sem_labels_rows.length ; ++i){
                                e = true_sem_labels_rows[i]
                                sem_labels.push(e.options[e.selectedIndex].value);
                            }
                            labels = [];
                            if(not_good != null)
                                labels.push(not_good);
                            if(layout != null)
                                labels.push(layout);
                            if(domain != null)
                                labels.push(domain);
                            if(type != null)
                                labels.push(type);
                            jobj = {};
                            jobj["cdr_id"] = cdrid;
                            jobj["fingerprint"] = fingerprint;
                            jobj["labels"] = labels;
                            jobj["sem_labels"] = sem_labels;
                            alltext += JSON.stringify(jobj) + "\\n";
                        }
                        var ajaxurl = "http://localhost:8000";
                        data = {};
                        data["data"] = alltext;
                        data["dict"] = sem_dict;
                        data["out_file_name"] = $.trim($('#out_file').val());
                        data["in_file_name"] = $.trim($('#orig_file').val());

                        //data = "data="+alltext+"&file_name=temp.txt";
                        //data = $( "form" ).serialize();
                        $.post(ajaxurl, data, function(res){
                            alert(res);
                        })
                            .fail(function(xhr, status, error) {
                                alert(status+error);
                            });
                       // txtFile.close();
                    });
                });
            </script>
            </head>\n
        ''')
        html_file.write('<body>\n')
        # Extractor(table_extractor_init, 'raw_content', 'extractors.tables.text')
        for t_num, t in enumerate(tables):

            # if t_num > 100:
            #     print("ended: {0}".format(line_num))
            #     break
            if t_num % 1000 == 0:
                print("working: {0}".format(t_num))
            # html_file.write('##############################################################################<br>\n'+
            #                 '##############################################################################<br>\n'+
            #                 '##############################################################################<br>\n')
            html_file.write('<div style="border:4px solid red;" name="table_annotation__">\n')
            html_file.write('<div style="border:1px solid brown;">\n')
            html_file.write('line_num: ' + str(t_num + 1) + '<br>\n')
            html_file.write('cdr_id: <cdrid value="' + str(t['cdr_id']) + '">' + str(t['cdr_id']) + '</cdrid><br>\n')
            html_file.write('fingerprint: <fingerprint value="' + str(t['fingerprint']) + '">' + str(
                t['fingerprint']) + '</fingerprint><br>\n')
            labels = []

            for x in towrite:
                if x in t:
                    html_file.write('x: ' + str(t[x]) + '<br>\n')
            if 'labels' in t:
                labels = t['labels']
            text = """
            <div style="border:2px solid black;width: 100%;float:left">
                annotate table type: \n
                <div style="width: 100%;float:left">
                <form>\n
                    <input type="checkbox" name="not_good" value="THROW">  throw away this table\n
                </form>\n
                </div>
                <div style="width: 30%;float:left">
                <form>\n
                    <input type="radio" name="domain_" value="IN-DOMAIN"> IN-DOMAIN<br>\n
                    <input type="radio" name="domain_" value="OUT-DOMAIN"> OUT-DOMAIN<br>\n
                </form>
                </div>
                <div style="width: 30%;float:left">
                <form>\n
                    <input type="radio" name="layout_" value="LAYOUT"> LAYOUT<br>\n
                    <input type="radio" name="layout_" value="NOT-LAYOUT"> NOT-LAYOUT\n
                </form>
                </div>
                <div style="width: 30%;float:left">
                <form>\n
                    <input type="radio" name="type_" value="ENTITY"> ENTITY<br>\n
                    <input type="radio" name="type_" value="RELATIONAL"> RELATIONAL<br>\n
                    <input type="radio" name="type_" value="MATRIX"> MATRIX<br>\n
                    <input type="radio" name="type_" value="LIST"> LIST\n
                </form>\n
                </div>
            </div>
            """
            for x in labels:
                x = '"' + x + '"'
                if x in text:
                    args_ = text.split(x)
                    text = args_[0] + x + ' checked' + args_[1]

            html_file.write(text)
            # html_file.write('header rows: ' + str(line['header_rows']) + '<br>\n')
            # html_file.write('header columns: ' + str(line['header_cols']) + '<br>\n')
            html_file.write('</div>')
            html_file.write('<div style="display: table-row">\n')
            if semantic_labels:
                if 'true_sem_labels' in t:
                    temp = self.create_html_from_array_with_labels(t['table_array'], t['table_row_sem_labels'],
                                                              t['true_sem_labels'], semantic_labels)
                else:
                    temp = self.create_html_from_array_with_labels(t['table_array'], t['table_row_sem_labels'],
                                                              None, semantic_labels)
            else:
                temp = self.create_html_from_array(t['table_array'])
            html_file.write(temp + '\n')
            html_file.write('</div>\n')
            html_file.write('</div>')
        html_file.write('''
            <div style="display: table-row">
            out file name: <input style="display: table-cell" rows="4" cols="50" id="out_file">
            </input> <br>
            all tables file name: <input style="display: table-cell" rows="4" cols="50" id="orig_file">
            </input> <br>
            <button style="display: table-cell" id="submitter"> submit </button>

            </div>
        ''')
        return html_file.getvalue()

    def write_as_jl(self, j_arr, filename):
        f = open(filename, 'w')
        for l in j_arr:
            f.write(json.dumps(l) + '\n')
        f.close()

    def regulize_text(self, text):
        # for x in re.findall('([0-9])', text):
        #     int_x = int(x)
        #     if int_x < 5:
        #         text = re.sub(x, 'SSSS', text)
        #     else:
        #         text = re.sub(x, 'LLLL', text)
        text = re.sub('[0-9]', 'NUM', text)
        for x in re.findall('([a-z][a-z][a-z]+@)', text):
            text = re.sub(x, 'EMAILNAME ', text)
        return text

    def regulize_cells(self, t):
        for r in t:
            for i in range(len(r)):
                if type(r[i]) is list:
                    for j in range(len(r[i])):
                        r[i][j] = self.regulize_text(r[i][j])
                elif type(r[i] is str):
                    r[i] = self.regulize_text(r[i])
                else:
                    print('not supported!!')
                    exit(-1)

    def clean_cells(self, t, level=1):  # modifies t
        for r in t:
            for i in range(len(r)):
                if level >= 1:
                    r[i] = filter(lambda x: x in set(string.printable), r[i])
                if level >= 2:
                    r[i] = re.sub(r'[^\x00-\x7F]', ' ', r[i])  # remove unicodes
                    r[i] = re.sub('[^\s\w\.\-\$_%\^&*#~+@"\']', ' ', r[i])  # remove annoying puncts
                if level >= 3:
                    for x in re.findall('(\.[a-z])', r[i]):
                        r[i] = re.sub('\.{0}'.format(x[1]), ' {0}'.format(x[1]), r[i])
                r[i] = re.sub('\s+', ' ', r[i])
                r[i] = r[i].strip()
        return t

    def has_data(self, text):
        res = copy(text)
        res = re.sub('[^\w]', '', res)
        if res == '':
            return False
        return True

    def tokenize_cell(self, text):
        res = TextToolkit.tokenize_text(text)
        res = [x for x in res if self.has_data(x)]
        return res

    def create_token_pairs(self, t, table_type='entity', max_cell_size=None, tokenize_header=False):
        res = []
        new_table_array = []
        if table_type == 'entity':
            for r in t:
                new_r = []
                new_table_array.append(new_r)
                attr_name = r[0].strip()
                if tokenize_header:
                    attr_name_tokens = self.tokenize_cell(attr_name)
                else:
                    if not self.has_data(attr_name):
                        continue
                    attr_name_tokens = [attr_name]
                attr_val = r[1].strip()
                attr_val_tokens = self.tokenize_cell(attr_val)
                if attr_name == '' or attr_val == '':
                    continue
                if max_cell_size is not None:
                    if len(attr_name_tokens) > max_cell_size or len(attr_val_tokens) > max_cell_size:
                        continue
                new_r.append(attr_name_tokens)
                new_r.append(attr_val_tokens)
                new_tokens = list(product(attr_name_tokens, attr_val_tokens))
                res += new_tokens
                # res.append((attr_name, attr_val))
            return res, new_table_array
        else:
            print('only entity tables are allowed for now!')

    def update_vectors(self, cvs, embeddings):
        for key in embeddings.keys():
            if np.sum(embeddings[key] ** 2) > 0:
                cvs[key] = embeddings[key]

    def normalize_embeddings(self, embeddings):
        for k in embeddings.keys():
            temp = normalize(embeddings[k].reshape(1, -1))[0]
            temp = list(temp)
            # temp = [0.0 if x < 0.5 and x > -0.5 else
            #         -1.0 if x < -0.5 else 1.0 for x in temp]
            embeddings[k] = np.array(temp, dtype='float64')

    def write_as_csv(self, l):
        sio = StringIO.StringIO()
        for row in l:
            for cell in row:
                sio.write(str(cell) + '\t')
            sio.write('\n')
        return sio.getvalue()



class VizToolkit:
    def plot_confusion_matrix(self, cm, classes, title='', x_label='Predicted label',
                              y_label='True label', cmap=plt.cm.Blues, save_to_file=None, vmax=None,
                              sub_plot='111', show=True, fignum=1):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure(fignum)
        if vmax is None:
            vmax = cm.max()
        vmax = float(vmax)
        ax = plt.subplot(sub_plot)

        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=vmax)
        x = np.arange(-1,len(classes)+1, 0.01)
        plt.title(title)
        # cbar = plt.colorbar()
        # cbar.ax.tick_params(labelsize=13)
        tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=45)
        # plt.yticks(tick_marks, classes)
        for i in range(len(classes)):
            plt.plot(x, [0.5+i]*len(x), color='black')
            plt.plot([0.5 + i] * len(x), x, color='black')
        plt.tick_params(axis='both', which='major', labelsize=13)
        thresh = vmax/2.0
        # for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        #     plt.text(j, i, "%.2f" % round(cm[i, j],2),
        #              horizontalalignment="center",
        #              color="white" if cm[i, j] > thresh else "black", fontsize=12)

        # plt.tight_layout()
        # plt.ylabel(y_label)
        # plt.xlabel(x_label)
        plt.ylim(ymax=len(classes)-0.5)
        plt.ylim(ymin=-0.5)
        plt.xlim(xmax=len(classes)-0.5)
        plt.xlim(xmin=-0.5)

        if save_to_file is not None and show:
            # plt.figure(fignum).savefig(save_to_file)
            # plt.figure(fignum).set_size_inches(15, 10)
            # plt.figure(fignum).set_dpi(100)
            ax.set_aspect('equal')
            plt.figure(fignum).tight_layout()
            plt.figure(fignum).savefig(save_to_file)
            # plt.figure(fignum).savefig(save_to_file+'.png')
            plt.clf()
            # plt.figure(fignum).clf()
        elif show:
            print 'here we are!'
            plt.show()

    def plot_dist_matrix(self, cm, title='', x_label='Predicted label',
                              y_label='True label', cmap=plt.cm.Blues, save_to_file=None, vmax=None,
                              sub_plot='111', show=True, fignum=1):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if vmax is None:
            vmax = cm.max()
        vmax = float(vmax)

        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=vmax)
        # plt.title(title)
        # plt.colorbar()
        N = cm.shape[0]
        M = cm.shape[1]
        tick_marks_x = np.arange(M)
        tick_marks_y = np.arange(N)
        plt.xticks(tick_marks_x, tick_marks_x, rotation=90)
        plt.yticks(tick_marks_y, tick_marks_y)
        plt.tick_params(axis='both', which='major', labelsize=10)
        thresh = vmax/2.0
        # for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        #     plt.text(j, i, "%.2f" % round(cm[i, j],2),
        #              horizontalalignment="center",
        #              color="white" if cm[i, j] > thresh else "black", fontsize=8)

        # plt.tight_layout()
        # plt.ylabel(y_label)
        # plt.xlabel(x_label)

        if save_to_file is not None and show:
            # plt.figure(fignum).savefig(save_to_file)
            # plt.figure(fignum).set_size_inches(15, 10)
            # plt.figure(fignum).set_dpi(100)
            plt.tight_layout()
            plt.savefig(save_to_file)
            # plt.figure(fignum).savefig(save_to_file+'.png')
            plt.clf()
            # plt.figure(fignum).clf()
        elif show:
            print 'here we are!'
            plt.show()

    def plot_x_pca(self, X_pca, pca_methods, words=None, show_label=None):
        fig = plt.figure(figsize=(15, 8))
        n_plots = len(X_pca)
        counter = 0
        colors_ = ['red', 'blue']
        for ii, X in enumerate(X_pca):
            plt.subplot(n_plots, counter / n_plots + 1, counter % n_plots + 1)

            plt.title("Dimensionality reduction using {0}".format(pca_methods[ii]), fontsize=14)

            # colors_ = [xx[1] for xx in zip(range(len(classes)), colors_)]
            colors = [colors_[0] if yy else colors_[1] for yy in show_label]
            ####### 2d
            plt.scatter([x[0] for x in X], [x[1] for x in X], c=colors, cmap=plt.cm.Spectral)
            plt.legend(loc="lower right", prop={'size': 10})
            if words is not None:
                for label, v, show, c in zip(words, X, show_label, colors):
                    # print(v)
                    # if not show:
                    #     continue
                    plt.annotate(
                        label,
                        xy=v, xytext=(-20, 20),
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc=c, alpha=0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            counter += 1

        plt.show()


    def plot_x_pca_v2(self, X_pca, pca_methods, clusters, cluster_nums, labels=None,
                      show_label=None, specific_clusters=None, save_to_file=None,
                      legend_labels=None):
        if legend_labels is None:
            legend_labels = specific_clusters
        n_plots = len(X_pca)
        counter = 0
        colors_ = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        r = lambda: random.randint(0, 255)

        colors_ = ['#%02X%02X%02X' % (r(), r(), r()) for xx in cluster_nums]
        # colors_ = [xx[1] for xx in zip(cluster_nums, colors_)]
        colors = [colors_[cluster_nums.index(yy)] for yy in clusters]
        markers_ = ['o', '*', 's', 'v']
        my_markers = [markers_[cluster_nums.index(yy)] for yy in clusters]
        for ii, X in enumerate(X_pca):
            temp_ll = []
            plt.subplot(n_plots, counter / n_plots + 1, counter % n_plots + 1)

            plt.title("Dimensionality reduction using {0}".format(pca_methods[ii]), fontsize=14)

            ####### 2d
            for _s, c, _x, _y, cl in zip(my_markers, colors, [x[0] for x in X], [x[1] for x in X], clusters):
                if specific_clusters is not None and cl not in specific_clusters:
                    continue
                ind = specific_clusters.index(cl)
                l = str(legend_labels[ind])
                if l in temp_ll:
                    plt.scatter(_x, _y, marker=_s, c=c, s=150)
                else:
                    plt.scatter(_x, _y, marker=_s, c=c, s=150, label=l)
                    temp_ll.append(l)
            # plt.scatter([x[0] for x in X], [x[1] for x in X], c=colors, cmap=plt.cm.Spectral, marker=my_markers)

            counter += 1

            if labels is not None:
                for l, show, v, cl in zip(labels, show_label, X, clusters):
                    if specific_clusters is not None and cl not in specific_clusters:
                        continue
                    if show is False:
                        continue
                    plt.annotate(
                        l,
                        xy=v, xytext=(-20, 20),
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.4', fc='yellow', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
                       ncol=3, fancybox=True, shadow=True)
        if save_to_file is not None:
            plt.savefig(save_to_file, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()


    def plot_x_pca_v3(self, X_pca, clusters, cluster_names,
                      specific_clusters=None, save_to_file=None):
        n_plots = len(X_pca)
        counter = 0
        colors_ = cycle(['red', 'blue', 'green', 'yellow', 'orange'])
        r = lambda: random.randint(0, 255)

        # colors_ = ['#%02X%02X%02X' % (r(), r(), r()) for xx in cluster_names]
        colors_ = [xx[1] for xx in zip(cluster_names, colors_)]

        colors = [colors_[cluster_names.index(yy)] for yy in clusters]
        markers_ = ['o', '*', 's', 'v', 'p']
        # markers_ = [xx[1] for xx in zip(cluster_names, markers_)]
        my_markers = [markers_[x/5] for x in range(len(cluster_names))]
        # print(zip(cluster_names,colors_,my_markers))
        for ii, X in enumerate(X_pca):
            temp_ll = []
            plt.subplot(n_plots, counter / n_plots + 1, counter % n_plots + 1)

            plt.title("Dimensionality reduction using {0}".format('TSNE'), fontsize=14)

            ####### 2d
            for c, _x, _y, cl in zip(colors, [x[0] for x in X], [x[1] for x in X], clusters):
                if specific_clusters is not None and cl not in specific_clusters:
                    continue
                size = 25
                l = str(cl)
                a = 1.0
                m = my_markers[cluster_names.index(cl)]
                if cl == 'other':
                    a = 0.3
                    c = 'grey'
                    m = 'o'
                if m == '*':
                    size = size*3
                if l in temp_ll:
                    plt.scatter(_x, _y, c=c, s=size, alpha=a, marker=m)
                else:
                    plt.scatter(_x, _y, c=c, s=size, label=l, alpha=a, marker=m)
                    temp_ll.append(l)
            # plt.scatter([x[0] for x in X], [x[1] for x in X], c=colors, cmap=plt.cm.Spectral, marker=my_markers)

            counter += 1
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
                       ncol=3, fancybox=True, shadow=True)
        if save_to_file is not None:
            plt.savefig(save_to_file, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()


    def plot_x_pca_v4(self, X_pca, title='', save_to_file=None):
        n_plots = len(X_pca)
        # print(n_plots)
        plt.title("Dimensionality reduction using {0}".format('TSNE'), fontsize=14)
        print('putting points')
        plt.scatter([x[0] for x in X_pca], [x[1] for x in X_pca], cmap=plt.cm.Spectral)
        print('done putting points')
        if save_to_file is not None:
            plt.savefig(save_to_file, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()

    def plot_x_pca_v5(self, X_pca, labels, title='', save_to_file=None, cl_model=None, fignum=1):
        plt.figure(fignum)
        classes = list(set(labels))
        colors_ = cycle(['firebrick', 'darkgreen', 'darkblue', 'darkmagneta', 'deeppink'])
        my_cm = plt.cm.Paired
        colors_ = [xx[1] for xx in zip(classes, colors_)]
        points = dict()
        for x, l in zip(X_pca, labels):
            if l in points:
                points[l].append(x)
            else:
                points[l] = [x]
        plots = []
        legends = []
        for i, (l, p) in enumerate(points.items()):
            # print l, classes.index(l) * 255 / len(classes), my_cm(classes.index(l) * 255 / len(classes))
            plots.append(plt.scatter([x[0]for x in p], [x[1] for x in p], color=my_cm(classes.index(l) * 11 / len(classes))))
            legends.append(l)
        plt.title(title, fontsize=14)
        print('putting points')
        print('done putting points')
        plt.legend(plots, legends, loc="upper right", prop={'size': 10})

        if cl_model is not None:
            X = X_pca[: , 0]
            Y = X_pca[: , 1]
            x_min, x_max = X.min() - 1, X.max() + 1
            y_min, y_max = Y.min() - 1, Y.max() + 1
            h = 0.02
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = cl_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.cm.Pastel1,
                       aspect='auto', origin='lower')

        if save_to_file is not None:
            plt.savefig(save_to_file, bbox_inches='tight')
            plt.savefig(save_to_file+'.png', bbox_inches='tight')
            plt.clf()
        else:
            plt.show()

    def plot_x_pca_v6(self, X_pca, labels, title='', save_to_file=None, fignum=1):
        plt.figure(fignum)
        classes = list(set(labels))
        colors_ = cycle(['firebrick', 'darkgreen', 'darkblue', 'darkmagneta', 'deeppink'])
        my_cm = plt.cm.Paired
        colors_ = [xx[1] for xx in zip(classes, colors_)]
        points = dict()
        for x, l in zip(X_pca, labels):
            if l in points:
                points[l].append(x)
            else:
                points[l] = [x]
        plots = []
        legends = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, (l, p) in enumerate(points.items()):
            # print l, classes.index(l) * 255 / len(classes), my_cm(classes.index(l) * 255 / len(classes))
            plots.append(ax.scatter([x[0] for x in p], [x[1] for x in p], [x[2] for x in p],
                                     color=my_cm(classes.index(l) * 11 / len(classes))))
            legends.append(l)
        plt.title(title, fontsize=14)
        print('putting points')
        print('done putting points')
        plt.legend(plots, legends, loc="upper right", prop={'size': 10})
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if save_to_file is not None:
            plt.savefig(save_to_file, bbox_inches='tight')
            plt.savefig(save_to_file + '.png', bbox_inches='tight')
            plt.clf()
        else:
            plt.show()
        # return ax


    def plot_heat_surface(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def plot_categorical_multibar(self, data, categories, data_names, xlabel='', ylabel='', max_y=None, save_to_file=None, width=0.1):
        fig, ax = plt.subplots()
        N = len(categories)
        M = len(data_names)
        cmap = plt.cm.summer
        rects = []
        ind = np.arange(N)  # the x locations for the groups
        width = 0.12
        colors = ['r', 'y', 'b', 'g', 'purple', 'orange']
        patterns = ('/', '.', '\\\\\\', 'O', '\\', '////')
        max_height = -1
        for c in categories:
            temp_max = max(data[c].values())
            if temp_max > max_height:
                max_height = float(int(temp_max*10.0+0.5))/10.0

        for i, n in enumerate(data_names):
            temp = []
            for c in categories:
                temp.append(data[c][n])
            # print temp
            # rects.append(ax.bar(ind+i*width, temp, width,
            #                      color=cmap(i*255 / M), edgecolor=['black']*N, zorder=3))
            bars = ax.bar(ind + i * width, temp, width,
                   color=cmap(i*255 / M), edgecolor=['black'] * N, zorder=3)
            rects.append(bars)

            for bar in bars:
                bar.set_hatch(patterns[i])
        for r in rects:
            for rect in r:
                height = rect.get_height()
                if height < 0.001:
                    ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                            '{}'.format(int(height*100)),
                            ha='center', va='bottom')
        # add some
        ax.yaxis.grid(b=True, which='major', color='black', lw=0.8, linestyle='--',  zorder=0)
        plt.ylabel(xlabel)
        plt.title(ylabel)
        if 'WTDC' in categories:
            categories[categories.index('WTDC')] = 'DWTC'
        plt.xticks(ind + width*M/2, categories)
        plt.ylim(0.0, max_height+0.1)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # plt.tick_params(which='minor', length=5)
        ax.yaxis.grid(b=True, which='minor', color='grey', lw=0.5, linestyle='--')
        # if max_y is not None:
        #     plt.ylim(ymax=max_y)

        # plt.legend(rects, data_names, loc='upper center', bbox_to_anchor=(0.5, 1.35),
        #   ncol=3, fancybox=True, shadow=True)
        plt.legend(rects, data_names, loc='lower center',bbox_to_anchor=(0.5, 0.9),
                   ncol=3, fancybox=True, shadow=True, handlelength=3, handleheight=2)

        if save_to_file is not None:
            temp = float(N)
            if N == 2:
                a = 0.6
            elif N == 3:
                a = 1.3
            elif N ==4:
                a = 2
            elif N ==5:
                a = 2.5
            elif N == 6:
                a = 3
            ax.set_aspect(a)
            # plt.savefig(save_to_file, bbox_inches='tight')
            plt.savefig(save_to_file + '.png', bbox_inches='tight')
            plt.savefig(save_to_file, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()

    def get_cluster_names(self, cl, vecs, vec_names):
        if hasattr(cl, 'cluster_centers_indices_'):
            centers = cl.cluster_centers_indices_
        elif hasattr(cl, 'core_sample_indices_'):
            centers = cl.core_sample_indices_
        elif hasattr(cl, 'cluster_centers_'):
            print(cl.cluster_centers_)

            centers = [min(enumerate([np.linalg.norm(x-xx) for xx in vecs]), key=lambda s:s[1])[0] for x in cl.cluster_centers_]
        else:
            exit(-1)
        return [vec_names[i] for i in centers]


class MLToolkit:
    def manifold_isomap(self, vectors):
        X = np.matrix(vectors, dtype='float64')
        X_pca = isomap.Isomap(n_components=2).fit_transform(X)
        return X_pca

    def manifold_llembedding(self, vectors):
        X = np.matrix(vectors, dtype='float64')
        X_pca = LocallyLinearEmbedding(n_components=2).fit_transform(X)
        return X_pca

    def manifold_TSNE(self, vectors, n_components=2, verbose=0, method='exact'):
        X = np.matrix(vectors, dtype='float64')
        X_pca = TSNE(n_components=n_components, init='pca', method='exact', verbose=verbose).fit_transform(X)
        # X_pca = TSNE(n_components=2, random_state=0, init='pca',verbose=2).fit_transform(X)
        return X_pca

    def svd_pca(self, vectors, n_components=2, verbose=2):
        return PCA(n_components=n_components).fit_transform(vectors)

    def score_clustering(self, pred_labels, labels, data):
        # simple_score = cl_model.score(vectors, labels)
        homogenity = metrics.homogeneity_score(labels, pred_labels)
        completeness = metrics.completeness_score(labels, pred_labels)
        v_measure = metrics.v_measure_score(labels, pred_labels)
        rand_score = metrics.adjusted_rand_score(labels, pred_labels)
        adjusted_mutual_info = metrics.adjusted_mutual_info_score(labels, pred_labels)
        silhouette = metrics.silhouette_score(data, labels,
                                              metric='euclidean',
                                              sample_size=100)
        # return '''
        #     clustering score:
        #         simple_score: {},
        #         homogenity: {},
        #         completeness: {},
        #         v_measure: {},
        #         {} rand_score: {} {},
        #         adjusted mutual info: {},
        #         silhouette: {}
        # '''.format(simple_score, homogenity, completeness, v_measure,
        #            color.BLUE, rand_score, color.END ,
        #            adjusted_mutual_info, silhouette)

        return 'clustering score: \n rand_score: {}'.format(rand_score)

    def get_rand_score(self, pred_labels, labels):
        res = metrics.adjusted_rand_score(labels, pred_labels)
        return round(res, 2)

    def clustering_to_labels(self, pred_labels, labels):
        clusters = {}
        for i, l in enumerate(pred_labels):
            if l not in clusters:
                clusters[l] = list()
            clusters[l].append(labels[i])
        cluster_labels = {}
        for l, v in clusters.items():
            count = {}
            for x in v:
                if x not in count:
                    count[x] = 0
                count[x] += 1
            majority = sorted(count.items(), key=lambda x: x[1], reverse=True)[0][0]
            cluster_labels[l] = majority
        return [cluster_labels[x] for x in pred_labels]

    def get_score_report(self, pred_labels, labels):
        classes = set(labels)
        tp = dict([(x, 0.0) for x in classes])
        fn = dict([(x, 0.0) for x in classes])
        fp = dict([(x, 0.0) for x in classes])
        total = dict([(x, 0.0) for x in classes])

        for x in zip(labels, pred_labels):
            if x[0] == x[1]:
                tp[x[0]] += 1.0
            else:
                fp[x[1]] += 1.0
                fn[x[0]] += 1.0
            total[x[0]] += 1.0
        # print tp
        precision = dict([(k, tp[k] / (tp[k] + fp[k] + 1e-5)) for k in classes])
        recall = dict([(k, tp[k] / (tp[k] + fn[k] + 1e-5)) for k in classes])
        fscore = dict([(k, 2.0 * precision[k] * recall[k] / (precision[k] + recall[k] + 1e-10)) for k in classes])
        tp_sum = float(sum(tp.values()))
        fp_sum = float(sum(fp.values()))
        fn_sum = float(sum(fn.values()))
        p_micro = tp_sum / (tp_sum + fp_sum + 1e-5)
        r_micro = tp_sum / (tp_sum + fn_sum + 1e-5)
        f_micro = 2.0 * p_micro * r_micro / (p_micro + r_micro + 1e-10)
        # class_map = dict([(i, classes[i]) for i in range(len(classes))])
        # self.calc_conf_matrix([classes.index(x) for x in pred_labels], [classes.index[x] for x in labels], classes)
        return fscore, precision, recall, f_micro, p_micro, r_micro, tp, fp, fn

    def calc_conf_matrix(self, pred_y, true_y, classes):
        '''
        :param pred_y: integer encoded predicted labels
        :param true_y: integer encoded true labels
        :param classes: class names
        :return: conf_matrix: the confusion matrix, numpy array (2X2)
        '''
        n = len(classes)
        conf_matrix = np.zeros((n, n), dtype=np.float)
        for x,y in zip(pred_y, true_y):
            if type(x) is not int:
                x = int(x)
            if type(y) is not int:
                y = int(y)
            if x == y:
                conf_matrix[x,y] += 1
            else:
                # conf_matrix[x,y] += 1
                conf_matrix[y,x] += 1
        for i in range(n):
            conf_matrix[i] = conf_matrix[i] / np.sum(conf_matrix[i])
        return conf_matrix

    def calc_conf_matrix_clustering(self, sem_label_to_cluster, do_normalize=True):
        num_keys = len(sem_label_to_cluster.keys())
        conf_matrix = np.zeros((num_keys, num_keys), dtype=np.float)

        for ind, x in enumerate(product(sem_label_to_cluster.keys(), repeat=2)):
            if ind%num_keys == ind/num_keys:
                # print(x[])
                conf_matrix[ind/num_keys][ind%num_keys] = len(sem_label_to_cluster[x[0]])
                continue
            c1 = sem_label_to_cluster[x[0]]
            c2 = sem_label_to_cluster[x[1]]

            # remove majority cluster label for row
            # remove non-majority cluster label for col
            temp = dict()
            for cl in c1:
                if cl in temp:
                    temp[cl] += 1
                else:
                    temp[cl] = 1
            maj_c1 = sorted(temp.items(), key=lambda xx:xx[1])[0][1]
            temp = dict()
            for cl in c1:
                if cl in temp:
                    temp[cl] += 1
                else:
                    temp[cl] = 1
            maj_c2 = sorted(temp.items(), key=lambda xx:xx[1])[0][1]

            index_c1 = np.argwhere(c1 == maj_c1)
            index_c2 = np.argwhere(c2 != maj_c2)
            c1 = np.delete(c1, index_c1)
            c2 = np.delete(c2, index_c2)

            c1 = sorted(c1)
            c2 = sorted(c2)
            i = 0
            j = 0
            overlap = 0
            while i < len(c1) and j < len(c2):
                if c1[i] == c2[j]:
                    overlap += 1
                    i += 1
                    j += 1
                elif c1[i] > c2[j]:
                    i += 1
                else:
                    j += 1
            conf_matrix[ind/num_keys][ind%num_keys] = overlap
        if do_normalize:
            for i in range(num_keys):
                conf_matrix[i] = conf_matrix[i]/np.max(conf_matrix[i])
        return conf_matrix

if __name__ == '__main__':
    viz = VizToolkit()
    res = {'ours':{
      'matrix': 0.99,
      'relational': 0.22
    },
    'theirs': {
        'matrix': 0.89,
        'relational': 0.72
    }}
    # viz.plot_categorical_multibar(res, ['ours', 'theirs'], ['matrix', 'relational'])
    viz.plot_confusion_matrix(np.array([[0,.05,1.0], [1.0,.3,.7], [.4,.2,.5]]), ['1', '2', '3'], show=True)