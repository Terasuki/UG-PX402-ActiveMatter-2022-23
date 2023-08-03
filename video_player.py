import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
"""
data_500_l1 = np.loadtxt('./Data/com-500_l1.dat')
data_500_l2 = np.loadtxt('./Data/com-500_l2.dat')
data_500_l3 = np.loadtxt('./Data/com-500_l3.dat')
data_500_l4 = np.loadtxt('./Data/com-500_l4.dat')
data_500_l5 = np.loadtxt('./Data/com-500_l5.dat')

data_500 = np.vstack((data_500_l1, data_500_l2, data_500_l3, data_500_l4, data_500_l5))
split_500 = np.array_split(data_500, 1280)
dpt_500 = np.subtract(split_500, 500)
msd_500 = np.var(dpt_500, axis=0)
msd = msd_500[:, 0] + msd_500[:, 1]
time = np.linspace(1, 8000, 8000)
norm = data_500-500
norm_split = np.array_split(norm, 1280)
"""

"""

data_500 = np.loadtxt('./Data/com-vd0.dat')
split_500 = np.array_split(data_500, 256)
dpt_500 = np.subtract(split_500, 500)
msd_500 = np.var(dpt_500, axis=0)
time = np.linspace(1, 6000, 6000)
f, ax = plt.subplots(2, 1)

for timestep in range(8000):

    ax[1].clear()
    ax[0].clear()
    ax[1].set_ylim(0, 400)
    ax[1].set_xlim(0, 400)
    ax[0].set_ylim(-5, 10)
    ax[0].set_xlim(0, 13)
    fin = data_500[timestep::6000]-500
    ax[1].scatter(fin[:, 0], fin[:, 1], label=timestep)
    ax[0].plot(np.log2(time)[:timestep], np.log2(msd_500)[:timestep], label='Norm')
    ax[1].grid()
    ax[0].grid()
    plt.pause(0.0001)
"""
"""
f, ax = plt.subplots(1, 1)
mean = np.zeros(8000)[:350]
var = np.zeros(8000)[:350]
def gaussian(x, v, D, t):

    frac = np.sqrt(1/(4*np.pi*D*t))
    ex = np.exp(-np.divide((x-v*t)**2, 4*D*t))
    if t>300:
        ex = np.exp(-np.divide((x)**2, 4*D*t))
    return frac*ex

x = np.linspace(0, 500, 1400)

for timestep in range(1, 300):

    ax.clear()
    ax.set_ylim(0, 0.2)
    ax.set_xlim(0, 500)
    ax.hist(norm[:, 0][timestep::8000], label=timestep, bins=40, density=True)
    ax.plot(x, gaussian(x, 0.5, msd_500[:, 0][timestep]/(2*timestep), timestep))
    ax.grid()
    plt.legend()
    plt.pause(0.0001)

plt.show()
"""
"""
data_pol_1 = np.loadtxt('./Data/com-pol_1.dat')
data_pol_2 = np.loadtxt('./Data/com-pol_2.dat')
data_pol_3 = np.loadtxt('./Data/com-pol_3.dat')
data_pol_4 = np.loadtxt('./Data/com-pol_4.dat')

data_500 = np.vstack((data_pol_1, data_pol_2, data_pol_3, data_pol_4))
"""
data_500 = np.loadtxt('./Data/com-d5.dat')
split_500 = np.array_split(data_500, 1024)
dpt_500 = np.subtract(split_500, 1000)
msd_500 = np.var(dpt_500, axis=0)
time = np.linspace(1, 6000, 6000)

f, ax = plt.subplots(1, 1)

for timestep in range(6000):

    ax.clear()
    ax.set_ylim(0, 500)
    ax.set_xlim(0, 500)
    fin = data_500[timestep::6000]-1000
    ax.scatter(fin[:, 0], fin[:, 1], label=timestep)
    plt.legend()
    ax.grid()
    plt.pause(0.0001)

