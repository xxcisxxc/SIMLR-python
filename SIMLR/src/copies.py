import os, shutil

path = os.path.join(os.getcwd(), 'build')
plist = os.listdir(path)
for i in plist.copy():
	if 'temp' in i:
		plist.remove(i)
path = os.path.join(path, plist[0])
for i in os.listdir(path):
	shutil.copy(os.path.join(path, i), os.getcwd())