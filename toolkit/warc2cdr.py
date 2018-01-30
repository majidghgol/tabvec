__author__ = 'majid'
import warc
import warcio
import gzip
from warc.utils import FilePart
import json

import fileinput
import gzip
from bs4 import BeautifulSoup
# f = warc.open("/Users/majid/Downloads/CC-MAIN-20140423032007-00002-ip-10-147-4-33.ec2.internal.warc.gz", "rb")
# outfile = open("/Users/majid/DIG/dig-table-extractor/experiments/data/commoncrawl_data.jl", "w")
filename = "/Users/majid/Downloads/CC-MAIN-20150728002301-00001-ip-10-236-191-2.ec2.internal.warc.gz"
# f = warc.open("/home/majid/Downloads/DARTMOUTH-NBER-RESEARCH-2017-WARCS-20170721000000-PART-00000-000000.warc", "rb")
outfile = open("/Users/majid/Downloads/test1.jl", "w")


from warcio.archiveiterator import ArchiveIterator
import warcio
# print(warcio)
# exit(0)
counter = 0
good_counter = 0
with open(filename, 'rb') as stream:
    for record in ArchiveIterator(stream):
        if record.rec_type == 'response':
            if counter % 100 == 0:
                print counter
            counter += 1
            html = record.content_stream().read().decode('latin1')
            bs = BeautifulSoup(html, "lxml")
            if len(bs.findAll('table')) == 0:
                continue
            url = cdr_id = record.rec_headers.get_header('WARC-Target-URI')
            line = dict()
            line['cdr_id'] = cdr_id
            line['raw_content'] = html
            line['url'] = url
            outfile.write(json.dumps(line) + '\n')
            good_counter += 1
            # print(record.rec_headers.get_header('WARC-Target-URI'))
print counter
print good_counter
# with gzip.open(filename, mode='rb') as gzf:
#     for record in warc.WARCFile(fileobj=gzf):
#         print record.payload.read()

# for record in f:
#     try:
#         if record.header['WARC-Type'] == 'response':
#             url = record.url
#             date = record.date
#             html = record.payload.read()
#             html = html.decode('utf-8')
#
#             outfile.write(json.dumps({
#                 'cdr_id': url,
#                 'date': date,
#                 'url': url,
#                 'raw_content': html
#             }) + '\n')
#     except Exception as e:
#         print(e)



# warc.WARCRecord.
# FilePart.