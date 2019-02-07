import glob
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bresenham import bresenham
from PIL import Image


ymin=-1.0
ymax=1.0
xmin=-1.0
xmax=1.0
nx = 512
ny = 512
sc=8
snx=int(nx/sc)
sny=int(ny/sc)
uv_size=[snx, sny]

density=np.zeros((nx, ny), dtype=int)
U=np.zeros((nx, ny), dtype=float)
V=np.zeros((nx, ny), dtype=float)
for file in glob.glob('C:\\Users\\cazin\\Dropbox\\share_pablo\\ABCDE\\result\\scenario2*raw_position_predicted*trial_4*.csv'):
    x, y = np.loadtxt(file, skiprows =1, delimiter=',', unpack=True)
    ix = ((nx - 1) * (x - xmin) / (xmax - xmin)).astype(int)
    iy = ((ny - 1) * (y - ymin) / (ymax - ymin)).astype(int)

    dq=1.0/len(ix)

    for k in range(0, len(ix)-1):
        x0=ix[k]
        y0=iy[k]
        x1=ix[k+1]
        y1=iy[k+1]
        px=list(bresenham(x0, y0, x1, y1))
        if len(px) > 1 :
            px.pop()
            indices = np.array((px)).T.tolist()
            rows=indices[1]
            cols=indices[0]
    
          
            du=(x[k+1]-x[k])*dq;
            dv=(y[k+1]-y[k])*dq;

            U[rows, cols] = U[rows, cols] + du;
            V[rows, cols] = V[rows, cols] + dv;    

            density[rows, cols] = density[rows, cols] + 1;

d = np.amax(np.hypot(U, V))
U = U / d;
V = V / d;
density = density / np.amax(density)

ax = plt.imshow(density,  cmap="bone_r", interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax])
cbar = plt.colorbar(ax, ticks=[0, 1], orientation='vertical', label='Density')
cbar.ax.set_yticklabels(['Low', 'High'])  # horizontal colorbar

#plt.hist2d(x, y, bins=(nx, ny), range=((xmin, xmax), (ymin, ymax)))
xr=np.linspace(xmin, xmax, snx)
yr=np.linspace(ymin, ymax, sny)
X, Y = np.meshgrid(xr, yr)
U=np.array(Image.fromarray(U).resize(uv_size, resample=Image.BICUBIC))
V=np.array(Image.fromarray(V).resize(uv_size, resample=Image.BICUBIC))
#plt.quiver(X, Y,  U, V,  scale_units='xy',  scale=1/10, color="blue", minlength=0, width=0.001, linewidth=1)
plt.quiver(X, Y,  U, V, scale_units='width', scale=4, color="blue", minlength=0, width=0.001, linewidth=1)
plt.xlabel('X', fontsize=18)
plt.ylabel('Y', fontsize=18)
plt.title('Generated trajectories density plot', fontsize=20)
#plt.legend()
plt.show()