import csv
import sys
import json
from bs4 import BeautifulSoup
from StringIO import StringIO
import pyexcel_io
import pyexcel_xlsx
import os

def wrap_html(html, doc_id):
    doc = dict(
        doc_id=doc_id,
        url='table.com',
        tld='table.com',
        raw_content=html
    )
    return doc

if __name__ == "__main__":
    infile_path = sys.argv[1]
    outpath = sys.argv[2]
    outfile = open(outpath, 'w')
    fn, extention = os.path.splitext(infile_path)
    if extention == ".csv":
        get_data = pyexcel_io.get_data
    elif extention == ".xls":
        get_data = pyexcel_xlsx.get_data
    elif extention == ".xlsx":
        get_data = pyexcel_xlsx.get_data
    else:
        print "file extension can not read"

    data = get_data(infile_path, auto_detect_datetime=False)


    for key in data.keys():
        sheet = data[key]
        html = StringIO()
        html.write('<html><body>')
        html.write('<table>')
        for row in sheet:
            html.write('<tr>')
            for c in row:
                if isinstance(c, basestring):
                    c = c.encode('ascii', 'ignore')
                html.write('<td>{}</td>'.format(c))
            html.write('</tr>')

        html.write('</table>')
        html.write('</body></html>')
        url = infile_path
        cdr_id = key
        obj = dict(
            cdr_id=cdr_id,
            url=url,
            doc_id=cdr_id,
            _id=cdr_id,
            raw_content=html.getvalue()
        )
        outfile.write(json.dumps(obj) + '\n')
    outfile.close()
    # print(json.dumps(data, default=str))
    # exit(0)
    #
    # with open(infile_path, 'rb') as infile:
    #     csv_reader = csv.reader(infile, delimiter=',', quotechar='"')
    #     html = StringIO()
    #     html.write('<html><body>')
    #     html.write('<table>')
    #     for i, row in enumerate(csv_reader):
    #         # if i > 10:
    #         #     break
    #         html.write('<tr>')
    #         for x in row:
    #             html.write('<td>{}</td>'.format(x))
    #         html.write('</tr>')
    #     html.write('</table>')
    #     html.write('</body></html>')
    #     url = cdr_id = infile_path
    #     obj = dict(
    #         cdr_id=cdr_id,
    #         url=url,
    #         doc_id=cdr_id,
    #         _id=cdr_id,
    #         raw_content=html.getvalue()
    #     )
    #     outfile = open(sys.argv[2], 'w')
    #     outfile.write(json.dumps(obj))


