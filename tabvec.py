import sys, os
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from hashlib import sha256
from jsonpath_rw import parse
import json
sys.path.append('/Users/majid/DIG/tabvec')

from embedding.TableEmbedding import run_table_embedding2
from embedding.TokenizeTableText import run_tokenize_table, run_tokenize_text
from embedding.WordCount import run_word_count
from embedding.WordEmbedding import run_word_embedding_word2vec, create_context_sentences, read_context_sentences
from embedding.clustering.TableClustering import run_clustering, run_clustering2
from toolkit.toolkit import VizToolkit
from embedding.data_processing import put_attr
import pyspark

configs = {
        'spark': {
            'num_cores': 6,
            'spark_master': 'local[6]',  # spark://majid-Precision-Tower-3420:7077
            'executor_mem': '3g',
            'driver_mem': '3g',
            'driver_max_res_size': '10g'
        },
        'tokenizer': {
            'put_extractions': False,
            'regularize_tokens': False,
            'table_path': '*.table.tables[*]',
            'text_path': '*.table.html_text'
        },
        'tabvec': {
            'working_dir': '__temp_files__'
        },
        'embeddings': {
            'context': ['text', 'cell', 'border'],
            'max_product_th': 100,
            'window': 2,
            'vector_dim': 200,
            'nbits': 2,
            'cut-off': 0.05,
        },
        'clustering': {
            'method': 'kmeans',
            'nclusters': 12
        }
    }

conf = SparkConf().setAppName("tableEmbeddingApp") \
            .setMaster(configs['spark']['spark_master']) \
            .set("spark.executor.memory", configs['spark']['executor_mem']) \
            .set('spark.driver.memory', configs['spark']['driver_mem']) \
            .set('spark.driver.maxResultSize', configs['spark']['driver_max_res_size']) \
            .set('spark.executor.extraClassPath', '/Users/majid/DIG/tabvec') \
            .set('spark.executorEnv.PYTHONPATH', '/Users/majid/DIG/tabvec:$PYTHONPATH')

sc = SparkContext(conf=conf)
spsession = SparkSession(sc)


class TabVec:
    def release_list(self, a):
        del a[:]
        del a

    def get_tokenizer_trace(self):
        return '{}_{}'.format(configs['tokenizer']['put_extractions'],
                              configs['tokenizer']['regularize_tokens'])

    def get_embedding_trace(self):
        return '{}_{}_{}'.format(configs['embeddings']['max_product_th'],
                                 configs['embeddings']['window'],
                                 '_'.join(configs['embeddings']['context']))

    def get_clustering_trace(self):
        return '{}_{}'.format(configs['clustering']['method'],
                              configs['clustering']['nclusters'])

    def tokenize_tables(self, input_path, input_id):
        """
        Calculate the word embeddings given the corpus of web pages
        :param input_path: the path to the folder/jl_file that etk output resides
        :param input_id: the id of input (sha256 of input path)
        :return: none
        """
        tok_table_path = '{}/{}/tok_tarr_{}'.format(configs['tabvec']['working_dir'],
                                                    input_id, self.get_tokenizer_trace())
        tok_text_path = '{}/{}/tok_text'.format(configs['tabvec']['working_dir'], input_id)

        if os.path.exists(tok_table_path):
            print '"{}" existed .... continuing ...'.format(tok_table_path)
        else:
            run_tokenize_table(sc, configs['tokenizer'], input_path, tok_table_path)
        if os.path.exists(tok_text_path):
            print '"{}" existed .... continuing ...'.format(tok_text_path)
        else:
            run_tokenize_text(sc, configs['tokenizer'], input_path, tok_text_path)

    def count_words(self, input_id):
        wcpath = '{}/{}/word_count_{}'.format(configs['tabvec']['working_dir'], input_id,
                                              self.get_tokenizer_trace())
        if os.path.exists(wcpath):
            print('{} existed .... continuing ...'.format(wcpath))
        else:
            tok_table_path = '{}/{}/tok_tarr_{}'.format(configs['tabvec']['working_dir'], input_id,
                                                        self.get_tokenizer_trace())
            tok_text_path = '{}/{}/tok_text'.format(configs['tabvec']['working_dir'], input_id)
            run_word_count(sc, tok_table_path, tok_text_path, wcpath, 'text' in configs['embeddings']['context'])

    # def calc_occurrences(self, input_id):
    #     print('creating word occurrences for {}'.format(input_id))
    #     tok_table_path = '{}/{}/tok_tarr_{}'.format(configs['tabvec']['working_dir'], input_id,
    #                                                 self.get_tokenizer_trace())
    #     tok_text_path = '{}/{}/tok_text'.format(configs['tabvec']['working_dir'], input_id)
    #     for context in ['text', 'cell', 'adjcell', 'hrow', 'hcol']:
    #         occ_path = '{}/{}/occurrences_{}_{}'.format(configs['tabvec']['working_dir'], input_id,
    #                                                     self.get_tokenizer_trace(), context)
    #         if os.path.exists(occ_path):
    #             print('{} existed .... loading ...'.format(occ_path))
    #         else:
    #             create_occurrences(sc, tok_table_path, tok_text_path, occ_path,
    #                                context,
    #                                window=configs['embeddings']['window'],
    #                                max_product_th=configs['embeddings']['max_product_th'])

    def calc_word_embeddings(self, input_path, output_path, input_id):
        #         input_id = sha256(input_path).hexdigest()
        wepath = '{}_{}'.format(output_path, self.get_embedding_trace())
        print wepath
        if os.path.exists(wepath):
            print('{} existed .... continuing ...'.format(wepath))
        else:
            self.tokenize_tables(input_path, input_id)
            self.count_words(input_id)
            self.calc_context_sentences(input_id)
            # wcpath = '{}/{}/word_count_{}'.format(configs['tabvec']['working_dir'], input_id,
            #                                       self.get_tokenizer_trace())
            context_sentences_path = '{}/{}/sentences_{}'.format(configs['tabvec']['working_dir'], input_id,
                                                     self.get_tokenizer_trace())
            print wepath
            # doc_counts = sc.textFile(input_path).count()
            # word_counts = sc.textFile(wcpath).map(lambda x: eval(x)). \
            #     reduceByKey(lambda v1, v2: v1 + v2).collect()
            context_sentences = read_context_sentences(sc, context_sentences_path, configs['embeddings']['context']).persist()
            # context_sentences = spsession.createDataFrame(context_sentences, ["text"])
            print('creating word embeddings ...')
            run_word_embedding_word2vec(sc, context_sentences, wepath)

    def calc_context_sentences(self, input_id):
        print('creating context sentences for {}'.format(input_id))

        tok_table_path = '{}/{}/tok_tarr_{}'.format(configs['tabvec']['working_dir'], input_id,
                                                    self.get_tokenizer_trace())
        tok_text_path = '{}/{}/tok_text'.format(configs['tabvec']['working_dir'], input_id)

        for context in ['text', 'cell', 'adjcell', 'hrow', 'hcol', 'border']:
            sentencespath = '{}/{}/sentences_{}_{}'.format(configs['tabvec']['working_dir'], input_id,
                                                           self.get_tokenizer_trace(), context)
            if os.path.exists(sentencespath):
                print('{} existed .... loading ...'.format(sentencespath))
            else:
                create_context_sentences(sc, tok_table_path, tok_text_path, sentencespath,
                                         context,
                                         window=configs['embeddings']['window'],
                                         max_product_th=configs['embeddings']['max_product_th'])

    def calc_table_vectors(self, input_path, we_path, out_path):
        # tokenize input tables
        if os.path.exists(out_path):
            print('{} existed .... continuing ...'.format(out_path))
        else:
            tok_tables = run_tokenize_table(sc, configs['tokenizer'], input_path, None)
            we = spsession.read.parquet(we_path + '/data/')
            we = we.rdd.collect()
            we = dict(we)
            run_table_embedding2(sc, tok_tables, we, out_path)


    def cluster_tables(self, tabvec_path, output_path):
        outpath = '{}/{}.tar.gz'.format(output_path, self.get_clustering_trace())
        if os.path.exists(outpath):
            print('{} existed .... continuing ...'.format(outpath))
        else:
            tables = sc.textFile(tabvec_path).map(lambda x: json.loads(x)).collect()
            table_vecs = [x['vec'] for x in tables]
            table_vecs_matrix = np.matrix(table_vecs)
            self.release_list(table_vecs)
            print(table_vecs_matrix.shape)
            cls = run_clustering(table_vecs_matrix,
                                 configs['clustering']['method'],
                                 configs['clustering']['nclusters'],
                                 outpath)
            # print('writing clusters to output ...')
            # self.sc.parallelize(zip(tables, cls), numSlices=600). \
            #     map(lambda x: put_attr(x[0], str(x[1]), 'cluster')).\
            #     map(lambda x: json.dumps(x)). \
            #     saveAsTextFile(outpath, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # for t, c in zip(tables, clusters):
            #     t['cluster'] = str(c)
            #     outfile.write(json.dumps(t) + '\n')
            # outfile.close()
            del table_vecs_matrix
            self.release_list(tables)

    def cluster_cells(self, tabvec_path, output_path):
        tables = self.sc.textFile(tabvec_path).map(lambda x: json.loads(x)).collect()
        cell_path = parse('$.vec_tarr[*][*]')
        cell_text_path = parse('$.tarr[*][*]')
        cell_vecs = list()
        for t in tables:
            vecs_text = [(match_vec.value, match_text.value) for (match_vec, match_text) in zip(cell_path.find(t), cell_text_path.find(t))
                    if match_vec.value is not None]
            vecs = [x[0] for x in vecs_text]
            texts = [x[1] for x in vecs_text]
            if t['cdr_id'] == 'Sheet86ma':
                print vecs
                print texts
                print t['tarr']
                cls = run_clustering2(texts, vecs, 'kmeans', 6)
                cl2text = dict()
                for c,txt in zip(cls, texts):
                    c = str(c)
                    if c not in cl2text:
                        cl2text[c] = []
                    cl2text[c].append(txt)
                print json.dumps(cl2text, indent=2)
                # viztk = VizToolkit()
                # viztk.plot_highdim_2d(vecs, texts)
                # exit(0)
            cell_vecs.extend(vecs)
        cell_vecs_matrix = np.matrix(cell_vecs)

        self.release_list(cell_vecs)
        print(cell_vecs_matrix.shape)
        # outpath = '{}/{}.tar.gz'.format(output_path, self.get_clustering_trace())
        # if os.path.exists(outpath):
        #     print('path existed .... continuing ...')
        # else:
        #     run_clustering(tables, cell_vecs_matrix,
        #                    self.configs['clustering']['method'],
        #                    self.configs['clustering']['nclusters'],
        #                    outpath)
        del cell_vecs_matrix
        self.release_list(tables)

if __name__ == '__main__':
    dataset = 'HTSAMPLE'
    reg = 'reg' if configs['tokenizer']['regularize_tokens'] else 'noreg'
    input_path = '/Users/majid/DIG/tabvec/data/{}.etk.out'.format(dataset)
    we_path = '/Users/majid/DIG/tabvec/data/{}_{}.we'.format(reg, dataset)
    tvec_path = '/Users/majid/DIG/tabvec/data/{}_{}.tabvec'.format(reg, dataset)
    cl_path = '/Users/majid/DIG/tabvec/data/{}_{}.cl'.format(reg, dataset)
    tvec = TabVec()

    tvec.calc_word_embeddings(input_path, we_path, dataset)
    tvec.calc_table_vectors(input_path, we_path + '_{}'.format(tvec.get_embedding_trace()), tvec_path)
    # tvec.cluster_tables(tvec_path, cl_path)