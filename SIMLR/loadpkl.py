import pickle

def loadpkl(name):
	with open(name+'.pkl', 'rb') as f:
		return pickle.load(f)