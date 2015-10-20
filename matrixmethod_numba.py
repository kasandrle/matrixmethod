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
def p_m(k_z, l, s2h):
    """matrix elements of the refraction matrices
    p[j] is p_{j, j+1}
    p_{j, j+1} = (k_{z, j} + k_{z, j+1}) / (2 * k_{z, j}) * exp(-(k_{z,j} - k_{z,j+1})**2 sigma_j**2/2) for all j=0..N-1
    m_{j, j+1} = (k_{z, j} - k_{z, j+1}) / (2 * k_{z, j}) * exp(-(k_{z,j} + k_{z,j+1})**2 sigma_j**2/2) for all j=0..N-1
    """
    p = k_z[l] + k_z[l+1]
    m = k_z[l] - k_z[l+1]
    rp = cmath.exp(-m**2*s2h[l])
    rm = cmath.exp(-p**2*s2h[l])
    o = 2 * k_z[l]
    return p*rp/o, m*rm/o

@numba.jit(nopython=True, cache=True)
def reflec_and_trans_inner(k2n2, k2, theta, thick, s2h):
    # wavevectors in the different layers
    k2_x = k2 * math.cos(theta)**2  # k_x is conserved due to snell's law
    k_z = -np.sqrt(k2n2 - k2_x)  # k_z is different for each layer.
    N = len(thick)  # number of layers

    pS, mS = p_m(k_z, N, s2h)
    # RR over interface to substrate
    mm11 = pS
    mm12 = mS
    mm21 = mS
    mm22 = pS
    for l in range(N):
        j = N - l - 1  # j = N-1 .. 0
        # transition through layer j
        vj = 1j * k_z[j+1] * thick[j]
        # transition through interface between j-1 an j
        pj, mj = p_m(k_z, j, s2h)
        m11 = pj * cmath.exp(-vj)
        m12 = mj * cmath.exp(+vj)
        m21 = mj * cmath.exp(-vj)
        m22 = pj * cmath.exp(+vj)

        mm11, mm21 = m11*mm11 + m12*mm21, m21*mm11 + m22*mm21
        mm12, mm22 = m11*mm12 + m12*mm22, m21*mm12 + m22*mm22

    # reflection coefficient
    r = mm12 / mm22

    # transmission coefficient
    t = 1 / mm22

    return r, t

@numba.jit(nopython=True, cache=True)
def fields_inner(k2n2, k2, theta, thick, s2h):
    # wavevectors in the different layers
    k2_x = k2 * math.cos(theta)**2  # k_x is conserved due to snell's law
    k_z = -np.sqrt(k2n2 - k2_x)  # k_z is different for each layer.
    N = len(thick)  # number of layers

    pS, mS = p_m(k_z, N, s2h)
    # entries of the transition matrix MM
    mm11 = np.empty(N+1, dtype=np.complex128)
    mm12 = np.empty(N+1, dtype=np.complex128)
    mm21 = np.empty(N+1, dtype=np.complex128)
    mm22 = np.empty(N+1, dtype=np.complex128)
    # RR over interface to substrate
    mm11[N] = pS
    mm12[N] = mS
    mm21[N] = mS
    mm22[N] = pS
    for l in range(N):
        j = N - l - 1  # j = N-1 .. 0
        # transition through layer j
        vj = 1j * k_z[j+1] * thick[j]
        # transition through interface between j-1 an j
        pj, mj = p_m(k_z, j, s2h)
        m11 = pj * cmath.exp(-vj)
        m12 = mj * cmath.exp(+vj)
        m21 = mj * cmath.exp(-vj)
        m22 = pj * cmath.exp(+vj)

        mm11[j] = m11*mm11[j+1] + m12*mm21[j+1]
        mm12[j] = m11*mm12[j+1] + m12*mm22[j+1]
        mm21[j] = m21*mm11[j+1] + m22*mm21[j+1]
        mm22[j] = m21*mm12[j+1] + m22*mm22[j+1]

    # reflection coefficient
    r = mm12[0] / mm22[0]

    # transmission coefficient
    t = 1 / mm22[0]

    fields_r = np.empty(N+2, dtype=np.complex128)
    fields_t = np.empty(N+2, dtype=np.complex128)
    fields_t[0] = 1  # in the vacuum layer
    fields_r[0] = r  # in the vacuum layer
    for l in range(N):
        j = l + 1  # j = 1 .. N
        fields_t[j] = mm22[l] * t
        fields_r[j] = mm12[l] * t
    fields_t[N+1] = t  # in the substrate
    fields_r[N+1] = 0

    return fields_r, fields_t



@numba.jit(nopython=True, cache=True)
def reflec_and_trans(n, lam, thetas, thick, rough):
    """Calculate the reflection coefficient and the transmission coefficient for a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: vector of refractive indices n = 1 - \delta - i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param theta: incident angle in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :return: (reflec, trans)
    """
    k2 = (2 * math.pi / lam)**2  # k is conserved
    k2n2 = k2 * n**2
    s2h = rough**2 / 2
    rs = []
    ts = []
    for theta in thetas:
        r, t = reflec_and_trans_inner(k2n2, k2, theta, thick, s2h)
        rs.append(r)
        ts.append(t)
    return rs, ts


#@numba.jit(nopython=True, cache=True)
def fields(n, lam, thetas, thick, rough):
    """Calculate the reflection coefficient and the transmission coefficient for a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: vector of refractive indices n = 1 - \delta - i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param theta: incident angle in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :return: (reflec, trans)
    """
    k2 = (2 * math.pi / lam)**2  # k is conserved
    k2n2 = k2 * n**2
    s2h = rough**2 / 2
    rs = []
    ts = []
    for theta in thetas:
        r, t = fields_inner(k2n2, k2, theta, thick, s2h)
        rs.append(r)
        ts.append(t)
    return rs, ts


if __name__ == '__main__':
    _n = np.array([1] + [1-1e-5+1e-6j]*10 + [1-2e-5+2e-6j] + [0.99])
    _rough = np.array([.2]*12)
    _thick = np.array([1.]*11)
    _wl = 0.15
    _ang_deg = np.linspace(0, 1, 101)[1:]
    _ang = np.deg2rad(_ang_deg)
    _r, _t = reflec_and_trans(_n, _wl, _ang, _thick, _rough)
    pass

















