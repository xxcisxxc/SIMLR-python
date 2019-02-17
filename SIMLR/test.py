from SIMLR import SIMLR
from loadpkl import loadpkl
X=loadpkl("mECS")['in_X']
SIMLR(X, 3)