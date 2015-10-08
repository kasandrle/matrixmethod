"""Matrix method algorithm for x-ray reflectivity and transmittivity as described by A. Gibaud and G. Vignaud in
J. Daillant, A. Gibaud (Eds.), "X-ray and Neutron Reflectivity: Principles and Applications", Lect. Notes Phys. 770
(Springler, Berlin Heidelberg 2009), DOI 10.1007/978-3-540-88588-7, chapter 3.2.1 "The Matrix Method".

Conventions used:
I'm following the conventions of A. Gibaud and G. Vignaud cited above:
There is a stack of j=0..N media on a substrate S, with j=0 and S being infinite. The interface between j and j+1
is Z_{j+1}, so Z_1 is the interface between the topmost layer (i.e. usually air or vacuum) and the first sample layer.
Electromagnetic waves are represented by their electric field \vec{E}, which is divided in one part travelling
downwards, \vec{E}^- and one travelling upwards, \vec{E}^+.

\vec{E}^{-/+} = A^{-/+} \exp\( +i(\omega t - k_{\text{in}x,j} x - k_\{\text{in}z,j} z) \) \, \hat{e}_y

The magnitude of the electric fields (which is time-independent) is denoted by:
U(-/+ k_{\text{in}z,j}, z) = A^{-/+}_j \exp(-/+ ik_{\text{in}z,j} z)

using
p_{j, j+1} = \frac{k_{z,j} + k_{z,j+1}}{2k_{z,j}}
m_{j, j+1} = \frac{k_{z,j} - k_{z,j+1}}{2k_{z,j}}

the refraction matrix \RR_{j, j+1} is given by:
 \( \begin{pmatrix}
  U(k_{z,j}, Z_{j+1}) \\
  U(-k_{z,j}, Z_{j+1})
 \end{pmatrix} \)
 =
 \( \begin{pmatrix}  % this is \RR_{j, j+1}
  p_{j, j+1} & m_{j, j+1} \\
  m_{j, j+1} & p_{j, j+1}
 \end{pmatrix} \)
 \( \begin{pmatrix}
  U(k_{z,j+1}, Z_{j+1}) \\
  U(-k_{z,j+1}, Z_{j+1})
 \end{pmatrix} \)

while the translation matrix \TT is defined as
 \( \begin{pmatrix}
  U(k_{z,j}, z) \\
  U(-k_{z,j}, z)
 \end{pmatrix} \)
 =
 \( \begin{pmatrix}  % this is \TT_{j}
  exp(-ik_{z,j} h) & 0 \\
  0 & exp(ik_{z,j} h)
 \end{pmatrix} \)
 \( \begin{pmatrix}
  U(k_{z,j}, z+h) \\
  U(-k_{z,j}, z+h)
 \end{pmatrix} \)

such that the transfer matrix \MM is
 \MM = \prod_{j=0}^N \( \RR_{j,j+1} \TT_{j+1} \) \RR_{N,s}
 =
 \( \begin{pmatrix}
  M_{11} & M_{12} \\
  M_{21} & M_{22}
 \end{pmatrix} \)

with this, the reflection coefficient is:
 r = \frac{M_{12}}{M_{22}}
and the transmission coefficient is:
 t = \frac{1}{M_{22}}

"""

import cmath
import math
import numpy as np
import scipy as sp

import numba


@numba.jit(nopython=True, cache=True)
def p_m(k_z, l):
    """matrix elements of the refraction matrices
    p[j] is p_{j, j+1}
    p_{j, j+1} = k_{z, j} + k_{z, j+1} / (2 * k_{z, j})  for all j=0..N-1
    """
    p = k_z[l] + k_z[l+1]
    m = k_z[l] - k_z[l+1]
    o = 2 * k_z[l+1]
    return p/o, m/o


@numba.jit(nopython=True, cache=True)
def reflec_and_trans_inner(k2n2, k2, theta, thick):
    # wavevectors in the different layers
    k2_x = k2 * math.cos(theta)**2  # k_x is conserved due to snell's law
    k_z = -np.sqrt(k2n2 - k2_x)  # k_z is different for each layer.

    # the transfer matrix MM is obtained as
    # \RR_{0, 1} \prod_{j=1}^N-1 \TT_{j} \RR_{j, j+1}
    # with
    # \RR{j, j+1} = ((p_{j, j+1}, m_{j, j+1}), (m_{j, j+1}, p_{j, j+1}))
    #             = ((pj, mj), (mj, pj))
    # and
    # \TT{j} = ((exp(-w_j), 0), (0, exp(w_j))
    # so MM * \TT_j * RR_j is (using wm=exp(-w_j), wp=exp(w_j))
    # ((M11*pj*wm + M12*mj*wp, M11*mj*wm + M12*pj*wp),
    #  (M21*pj*wm + M22*mj*wp, M21*mj*wm + M22*pj*wp))
    p0, m0 = p_m(k_z, 0)
    M11 = p0
    M12 = m0
    M21 = m0
    M22 = p0
    for l in range(len(thick)):
        pl, ml = p_m(k_z, l+1)
        wl = 1.j * k_z[l+1] * thick[l]
        wm = cmath.exp(-wl)
        wp = cmath.exp(wl)
        plwm = pl*wm
        plwp = pl*wp
        mlwm = ml*wm
        mlwp = ml*wp
        M11, M12 = M11*plwm + M12*mlwp, M11*mlwm + M12*plwp
        M21, M22 = M21*plwm + M22*mlwp, M21*mlwm + M22*plwp

    # reflection coefficient
    r = M12 / M22

    # transmission coefficient
    t = 1 / M22

    return r, t


@numba.jit(nopython=True, cache=True)
def reflec_and_trans(n, lam, thetas, thick):
    """Calculate the reflection coefficient and the transmission coefficient for a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: vector of refractive indices n = 1 - \delta - i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param theta: incident angle in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :return: (reflec, trans)
    """
    k2 = (2 * math.pi / lam)**2  # k is conserved
    k2n2 = k2 * n**2
    rs = []
    ts = []
    for i, theta in enumerate(thetas):
        r, t = reflec_and_trans_inner(k2n2, k2, theta, thick)
        rs.append(r)
        ts.append(t)
    return rs, ts


if __name__ == '__main__':
    n = np.array([1] + [1-1e-5+1e-6j]*100 + [1-2e-5+2e-6j])
    thick = np.array([1.]*100)
    wl = 0.15
    ang_deg = np.linspace(0, 1, 10001)[1:]
    ang = np.deg2rad(ang_deg)
    r, t = reflec_and_trans(n, wl, ang, thick)
    pass

















