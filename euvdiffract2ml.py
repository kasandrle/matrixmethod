from matrixmethod_numba import *
import xray_compounds as xc
import pint
u = pint.UnitRegistry()

import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt

mat = ['Si'] + 40 * ['Si', 'MoSi2', 'Mo', 'MoSi2'] + ['Si']
n_layers = len(mat) + 1  # including vacuum
rough = np.array([0] * (n_layers-1), dtype=np.float64)

Si_cap, Mo, MoSi2_1, Si, MoSi2_2 = 4.5629366565530356, 1.9965001372912379, 1.5102102800286887, 2.2030971866431175, 1.3931918631238953
thick = np.array([Si_cap] + 40 * [Si, MoSi2_2, Mo, MoSi2_1])

wl_nm = np.arange(12.5, 14.51, 0.02)

ang = np.deg2rad(np.array([90-6]))

ns = []
for wl in wl_nm:
    n_Si = np.conj(xc.refractive_index('Si', energy=wl*u.nm))
    n_Mo = np.conj(xc.refractive_index('Mo', energy=wl*u.nm))
    n_MoSi2 = np.conj(xc.refractive_index('MoSi2', energy=wl*u.nm))
    n = np.array([1] + [n_Si] + 40 * [n_Si, n_MoSi2, n_Mo, n_MoSi2] + [n_Si])
    ns.append(n)
ns = np.array(ns)

def calc():
    res = []
    for i, wl in enumerate(wl_nm):
        n = ns[i]
        r, t = reflec_and_trans(n, wl, ang, thick, rough)
        res.append((r, t))
    return np.array(res).reshape((len(wl_nm), 2))
