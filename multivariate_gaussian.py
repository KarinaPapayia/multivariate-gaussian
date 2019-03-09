import numpy as np
from math import exp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting
import matplotlib.cm as cm
from pylab import figure, plot, show, grid, axis, xlabel, ylabel, title, hold, draw
import scipy.stats as stats
import seaborn as sns
from sympy import plot_implicit
plt.style.use('seaborn-whitegrid')
from sklearn.metrics import mean_squared_error

#parameters for 2dimensional brownian brownian_paths


#time parameters
T = 1
N = 1000
dt = T/N

#zero mean
mu = np.zeros(2)

#varaiances
sigma1  = 4.32
sigma2 = 1.25

#correletaion
rho = 0.00001

#covariance matrix
c = np.array([[dt * pow(sigma1 , 2), rho * sigma1 * sigma2], [rho * sigma1 * sigma2, dt * pow(sigma2 , 2)]])
C = np.array([[2, 1], [1, 1]])
def box_muller(n):
    """Generate n random standard normal bivariate with box-muller transformation."""

    u1 = np.random.random((n+1)//2) #floor division returns integer instead of float number
    u2 = np.random.random((n+1)//2)
    r_squared = -2 * np.log(u1)
    r = np.sqrt(r_squared)
    theta = 2 * np.pi*u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.empty(n)
    z[:((n+1)//2)] = x
    z[((n+1)//2):] = y
    return z[:n]


def brownian_paths(mu, sigma, N = 1):
    """Generate n samples from bivariate normal with mean mu and covariance sigma."""

    A = np.linalg.cholesky(sigma) #cholesky decomposition for covariance matrix
    p = len(mu)

    zs = np.zeros((N, p))
    for i in range(N):
        z = box_muller(p)
        zs[i] = mu + A @ z
    return zs
#x, y = brownian_paths(mu, sigma, N).T
#g = sns.jointplot(x, y, kind='scatter')
pass
pass
G_T = brownian_paths(mu, C , N).T
BM_T = [np.cumsum(G_T[i]) for i in range(2)]

#for i in range(2):
#    plt.plot(np.linspace(0, 1, N), BM_T[i])
#plt.show()
