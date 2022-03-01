'''
module for some frequently used statistic measures
save efforts for importing packages
'''
from numpy.linalg import norm
from numpy.linalg import eigh

def fnorm(A, B):
    return norm((A - B), 'fro')

def isPSD(matrix):
    if(sum(eigh(matrix)[0] > -1e-8) < matrix.shape[0]):
        print("matrix not PSD")
    else: 
        print("matrix IS PSD")
        