import numpy as np 
from scipy.misc import imsave
from sklearn.preprocessing import StandardScaler
import numpy.linalg as linalg
import pylab as pl



def computeVarianceExplained(evals=None):
    return evals/evals.sum()

def plotCumSumVariance(var=None,filename=None):
    pl.figure()
    pl.plot(np.arange(var.shape[0]),np.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    #Save File
    if filename!=None:
        pl.savefig(filename)

x = np.load('latent_coords.npy').astype('float32')
x = x[8:]
n = x.shape[0]
X = x - np.mean(x,axis=0)

# We want to perform PCA analysis, let us do it manually

sigma = 1/(n-1) * np.dot(X.T,X)

[eig_vals, eig_vecs] = linalg.eig(sigma)

indices = np.argsort(-eig_vals)

eigenvalues = np.real(eig_vals[indices])
eigenvectors = np.real(eig_vecs[:,indices])


cum_var = computeVarianceExplained(eig_vals)
a = np.where(np.cumsum(cum_var)*100>=95)
print(a)
plotCumSumVariance(cum_var,'cumulative_var.pdf')






'''
# PCA Analysis via SVD

print(X.shape)

[U,S,Vh] = linalg.svd(1/(np.sqrt(X.shape[0]-1))*X.T,full_matrices=False)
D = np.diag(S**2)
print(U[:,0])
'''
