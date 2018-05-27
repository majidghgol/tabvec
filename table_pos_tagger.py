import unittest
import sys
import os
from jsonpath_ng import jsonpath, parse
from nltk import word_tokenize, pos_tag

import json
import codecs
from pyspark import SparkContext
from pyspark.sql import Row, SparkSession

def do_pos_tag(x):
    table_expr = parse('$.content_extraction.table.tables[*].rows[*].cells[*]')
    for match in table_expr.find(x):
        c = match.value
        pos_res = pos_tag(word_tokenize(c['text']))
        c['pos_tags'] = pos_res
        match.full_path.update(x, c)
    return json.dumps(x)


if __name__ == '__main__':
    # infile = open('/Users/majid/DIG/etk/etk/unit_tests/ground_truth/table.jl')
    # infile = open(sys.argv[1])
    sc = SparkContext(master="local[*]", appName="majid test")
    spark = SparkSession(sc)
    textFile = sc.textFile(sys.argv[1])

    output = textFile.map(lambda x: json.loads(x)). \
        map(lambda x: do_pos_tag(x))
    output.saveAsTextFile(sys.argv[2], compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
