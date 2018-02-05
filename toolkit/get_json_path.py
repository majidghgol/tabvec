from jsonpath_rw import parse
import sys
import json

if __name__ == '__main__':
    filepath = sys.argv[1]
    jpath = sys.argv[2]
    parser = parse(jpath)
    jobj = json.load(open(filepath))
    for match in parser.find(jobj):
        print '{}: {}'.format(match.value, match.full_path)