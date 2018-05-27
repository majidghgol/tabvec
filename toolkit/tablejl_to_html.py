import json
import sys


if __name__ == '__main__':
    infile = open(sys.argv[1])
    html_file = open(sys.argv[2], 'w')
    html_file.write('<html>\n')
    html_file.write('<body>\n')
    # Extractor(table_extractor_init, 'raw_content', 'extractors.tables.text')
    for line_num, line in enumerate(infile):
        t = json.loads(line)
        fp = t['fingerprint'] if 'fingerprint' in t else None
        id_ = t['cdr_id'] if 'cdr_id' in t else None
        html = t['html'] if 'html' in t else ""
        vec = t['vec'] if 'vec' in t else None
        cl = t['cluster'] if 'cluster' in t else None
        html_file.write('<div>{}\ncluster: {}\nvector: {}<div>'.format(html.encode('utf-8', errors='ignore'), cl, vec))
    html_file.write('\n</body>\n</html>\n')
    html_file.close()
    infile.close()