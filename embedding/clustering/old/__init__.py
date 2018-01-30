config = {
    'ATF1': {
        'etk_out': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/ATF_index2_etk_out.json.sample',
        'path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/pickle_files/ATF/',
        'gt': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/ATF_sample_tables_tabletype_GT.jl',
        'res_path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/output/ATF/'
    },
    'ATF': {
        'etk_out': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/dataset1.json',
        'path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/pickle_files/ATF/',
        'gt': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/ATF_sample_tables_tabletype_GT.jl',
        'res_path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/output/ATF/'
    },
    'HT': {
        'etk_out': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/HT_index.json',
        'path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/pickle_files/HT/',
        'gt': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/HT_sample_tables_tabletype_GT.jl',
        'res_path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/output/HT/'
    },
    'HT_sample': {
        'etk_out': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/HT_index.json.sample',
        'path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/pickle_files/HT/',
        'gt': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/HT_sample_tables_tabletype_GT.jl',
        'res_path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/output/HT/'
    },
    'SEC': {
        'etk_out': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/dataset4.json',
        'path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/pickle_files/SEC/',
        'gt': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/SEC_sample_tables_tabletype_GT.jl',
        'res_path': '/home/majid/my_drive/DIG/dig-table-extractor/experiments/result/output/SEC/'
    }
}

table_path = 'content_extraction.table[*]'
readibility_path = 'content_extraction.content_strict.text'

regularize = ['noreg', 'reg']
regularize_bool = [False, True]
domains = ['HT', 'ATF', 'SEC']
#domains = ['HT_sample']
sentences = {
                'cells': ['cells'],
                'text': ['text'],
                'cellstext': ['cells', 'text'],
                'cells_text_headers': ['cells', 'text', 'hrow', 'hcol'],
                'all': ['cells', 'rows', 'cols', 'hrow', 'hcol', 'text'],
                'headers': ['hrow', 'hcol'],
                'hrow': ['hrow'],
                'hcol': ['hcol']
            }

easy_samples = {
    'HT': {
        'LIST': [3,5,6,8,9,10,14,16,18,20],
        'ENTITY': [445,462,480,483,487,491,407,408,412,427],
        'RELATIONAL': [100,106,107,115,121,126,130,137,149,151],
        'MATRIX': [300,301,306,307,311,314,317,320,321,325],
        'NON-DATA': [274,283,233,237,235,248,249,267,273,217]
    },
    'ATF': {
        'LIST': [],
        'ENTITY': [],
        'RELATIONAL': [],
        'MATRIX': [],
        'NON-DATA': []
    },
    'SEC': {
        'LIST': [],
        'ENTITY': [],
        'RELATIONAL': [],
        'MATRIX': [],
        'NON-DATA': []
    }
}