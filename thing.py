#%%
import numpy as np

def data_for_cylinder_along_z(center_x,center_y,radius,height_z, zpoints,tpoints):
    z = np.linspace(0, height_z, zpoints)
    theta = np.linspace(0, 2*np.pi, tpoints)
    theta_grid, x_grid=np.meshgrid(theta, z)
    z_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# for x,y,z in zip(Xc,Yc,Zc):
#     ax.scatter(Xc, Yc, Zc, c='k')
with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1,50,50)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.3)
    Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1,6,6)
    #plt.title('This is the lattice')
    for xs,ys,zs in zip(Xc,Yc,Zc):
        for i, x in enumerate(xs):
            for k, z in enumerate(zs):
                for j, y in enumerate(ys):
                    ax.plot([x,x],[ys[j],ys[min(len(ys)-1, j+1)]],[z,z],c='k' )
                    ax.plot([xs[i],xs[min(len(xs)-1, i+1)]],[y,y],[z,z],c='k')
                    ax.plot([x,x],[y,y],[zs[k],zs[min(len(zs)-1, k+1)]],c='k' )

plt.show()
