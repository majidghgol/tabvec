import numpy as np
from pyspark.sql import Row

class RandomIndexing(object):
    def __init__(self, size=100, window=2, d=2, min_count=5):
        self.d = size
        self.w = window
        self.minc = min_count
        self.rseed = 1
        self.positions = d
        self.base = None
        self.embeddings = None

        #self.init_base_vectors(s)
        
        #self.calc_context_vectors(s)

    def clear(self):
        if self.base is not None:
            del self.base

    def get_base(self):
        v = np.zeros(self.d, dtype='float64')
        ind = np.random.choice(range(self.d), self.positions * 2, replace=False)
        v[ind[0:self.positions]] = 1.0
        v[ind[self.positions:]] = -1.0
        # v = v.tolist()
        return v

    def get_word_base(self, w):
        if w not in self.base:
            return np.zeros(self.d)
        return self.base[w]

    def get_init_embedding(self):
        # return np.ones(self.d, dtype='float64') / float(self.d)
        return np.zeros(self.d, dtype='float64')

    def init_base_vectors(self, words):
        self.base = dict()
        for k in words:
            self.base[k] = self.get_base()

    def init_vecs(self):
        self.embeddings = dict()
        for k in self.base.keys():
            self.embeddings[k] = np.ones(self.d, dtype='float64')/float(self.d)

    def record_occurrence(self, w1, w2, alpha=1):
        self.embeddings[w1] += alpha*self.base[w2]
        self.embeddings[w2] += alpha * self.base[w1]

    def add_vecs(self,v1, v2):
        # return (np.array(v1).reshape(self.d) +
        #         np.array(v2).reshape(self.d)).tolist()
        return v1+v2

    def merge_cvs(self, cv1, cv2):
        res = dict()
        for k in cv1.keys() + cv2.keys():
            if k in cv1 and k in cv2:
                res[k] = cv1[k] + cv2[k]
            elif k in cv1:
                res[k] = cv1[k]
            else:
                res[k] = cv2[k]
        return res
