import numpy as np
import matplotlib.pyplot as plt
import platform
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from scipy import argmin, inner

def cvt(samples, times, rnum):
    s_num = rnum * rnum
    p = np.random.random((samples, 2))
    interval = np.linspace(0.0, 1.0, rnum)
    sx = np.zeros (s_num)
    sy = np.zeros (s_num)
    s = np.zeros ([s_num, 2])
    
    k = 0
    for j in range(rnum):
        for i in range(rnum):
            sx[k], sy[k] = interval[i], interval[j]
            s[k,0], s[k,1]  = interval[i], interval[j]
            k += 1

    for it in range(times):
        if it % 3 == 0:
            vor = Voronoi(p)
            voronoi_plot_2d(vor)
            plt.title('Iteration Time: ' + str(it))
            plt.show()
        
        # find the index of the nearest
        k = [argmin([inner(gg - ss, gg - ss) for gg in p]) for ss in s]
        # counts the number of sample points it is nearest to
        w = np.ones(s_num)
        m = np.bincount(k, weights = w)
        
        #  G, i.e. the average of the nearest points
        new_x = np.bincount(k, weights = sx)
        new_y = np.bincount(k, weights = sy)

        for i in range(samples):
            if m[i] > 0:
                new_x[i] = new_x[i] / float (m[i])
                new_y[i] = new_y[i] / float (m[i])
        
        p[:,0], p[:,1] = new_x[:], new_y[:]

if __name__ == '__main__':
  cvt(20, 20, 100)

