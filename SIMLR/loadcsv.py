import csv
import pickle
#import numpy

files = {'in_X': [], 'Genes': [], 'true_labs': []}

for filename in files.keys():
	with open(filename+'.csv') as f:
		content = csv.reader(f)
		files[filename] = list(content)
		#print(numpy.array(files[filename]))

name = 'Usoskin'
with open(name+'.pkl', 'wb') as pkl:
	pickle.dump(files, pkl, pickle.HIGHEST_PROTOCOL)

print("Successfully transformed")