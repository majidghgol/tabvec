import pickle
import sys
import numpy

cl_model = pickle.load(open(sys.argv[1]))
centers = [str(x) for x in cl_model.cluster_centers_.tolist()]
print '\n'.join(centers)