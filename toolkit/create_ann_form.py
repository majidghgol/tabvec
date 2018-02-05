__author__ = 'majid'
import sys
import json
import re
import os
import io
from toolkit import TableToolkit
from jsonpath_rw import jsonpath, parse

all_labels = [
    "LAYOUT",
    "IN-DOMAIN",
    "ENTITY",
    "RELATIONAL",
    "MATRIX",
    "LIST"
]

# sys.path.append('/Users/majid/DIG/dig-table-extractor/')
# from digTableExtractor.table_extractor import TableExtractor
# from digExtractor.extractor_processor import ExtractorProcessor
# from digExtractor.extractor import Extractor

if __name__ == '__main__':
    tabletk = TableToolkit()
    if len(sys.argv) < 3:
        print('USAGE: INFILE OUTFILE')
        exit(-1)
    infile = open(sys.argv[1])
    html_file = io.open(sys.argv[2], 'w', encoding='utf-8')
    html_file.write(u'<html>\n')
    html_file.write(u'''
        <head>
          <script src = "https://code.jquery.com/jquery-1.10.2.js"></script>
        <script>
            $(function() {
                $("#submitter").click(function(){
                    var cdrid, fingerprint, type, domain, layout, not_good;
                    var alltext = "";
                    var num_lines = 0;
                    tables = document.getElementsByTagName('data__')[0].getElementsByTagName('table_annotation__');
                    for (index = 0; index < tables.length; ++index){
                        cdrid = tables[index].getElementsByTagName("meta__")[0]
                                             .getElementsByTagName("cdr_id__")[0]
                                             .getAttribute("name");
                        fingerprint = tables[index].getElementsByTagName("meta__")[0]
                                                   .getElementsByTagName("fingerprint__")[0]
                                                   .getAttribute("name");
                        not_good = null;
                        type = null;
                        hard = false;
                        // deal with inputs[index] element.
                        inputs = tables[index].getElementsByTagName("table_type__")[0].getElementsByTagName("input");
                        //alltext += inputs + "\\n";
                        for (i = 0 ; i < inputs.length ; ++i){
                            if(inputs[i].name == "type_")
                                if(inputs[i].checked)
                                     type = inputs[i].value;
                            if(inputs[i].name == "not_good")
                                if(inputs[i].checked)
                                     not_good = inputs[i].value;
                            if(inputs[i].name == "hard_job")
                                if(inputs[i].checked)
                                     hard = true;
                        }
                        labels = [];
                        if(not_good != null)
                            labels.push(not_good);
                        else{
                            if(type != null)
                                labels.push(type);
                        }
                        jobj = {};
                        jobj["cdr_id"] = cdrid;
                        jobj["fingerprint"] = fingerprint;
                        jobj["ishard"] = hard;
                        jobj["labels"] = labels;
                        if(labels.length > 0){
                            if(!labels.includes('THROW'))
                                num_lines = num_lines + 1;
                            alltext += JSON.stringify(jobj) + "\\n";
                        }
                        //alltext += fingerprint + "\\n";
                    }
                   //alltext += 'majid';
                   $('#output_temp').val(alltext);
                   $('#ann_nums').val(num_lines.toString());
                   // txtFile.close();
                });
            });
        </script>
        </head>\n
    ''')
    html_file.write(u'<body>\n')
    html_file.write(u'<data__ style="border:2px solid black;float:left;width: 50%;display: table-cell" >\n')
    # Extractor(table_extractor_init, 'raw_content', 'extractors.tables.text')
    for line_num, line in enumerate(infile):
        t = json.loads(line)
        ll = t['labels'] if 'labels' in t else None
        fp = t['fingerprint'] if 'fingerprint' in t else None
        id_ = t['cdr_id'] if 'cdr_id' in t else None
        if 'ishard' in t:
            ishard = t['ishard']
        else:
            ishard = False
        tarr = [[' '.join(cell) for cell in row] for row in t['tok_tarr']]
        html_file.write(unicode(tabletk.create_html_from_array_with_labels(tarr, fingerprint=fp, cdr_id=id_,
                                                           table_index=line_num, ishard=False) + '\n'))
    html_file.write(u'</data__>')
    html_file.write(u'<submit__ style="width: 46%;border:2px solid black;float:left;display: table-cell" >\n')
    html_file.write(u'''
        <input type="text" id="ann_nums"><br>
        <button id="submitter" style="display: table-cell;float: center;width: 20%; height: 50"> submit </button>
        <textarea rows="100" cols="100" id="output_temp" style="display: table-cell;float: bottom; width: 100%">
        </textarea>
    ''')
    html_file.write(u'</submit__>')
    html_file.write(u'\n</body>\n</html>\n')
    html_file.close()
    infile.close()

