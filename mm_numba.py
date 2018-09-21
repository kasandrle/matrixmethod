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
    rp = cmath.exp(-sq(m) * s2h[l])
    rm = cmath.exp(-sq(p) * s2h[l])
    o = 2 * k_z[l]
    return p*rp/o, m*rm/o


@numba.jit(nopython=True, cache=True)
def sq(input):
    return input*input


@numba.jit(nopython=True, cache=True)
def reflec_and_trans_inner(k2n2, k2, theta, thick, s2h):
    # wavevectors in the different layers
    k2_x = k2 * sq(math.cos(theta))  # k_x is conserved due to snell's law
    k_z = -np.sqrt(k2n2 - k2_x)  # k_z is different for each layer.
    N = len(thick)  # number of layers

    pS, mS = p_m(k_z, N, s2h)
    # RR over interface to substrate
    #mm11 = pS
    mm12 = mS
    #mm21 = mS
    mm22 = pS
    for l in range(N):
        j = N - l - 1  # j = N-1 .. 0
        # transition through layer j
        vj = cmath.exp(1j * k_z[j+1] * thick[j])
        # transition through interface between j-1 an j
        pj, mj = p_m(k_z, j, s2h)
        m11 = pj / vj
        m12 = mj * vj
        m21 = mj / vj
        m22 = pj * vj

        #mm11, mm21 = m11*mm11 + m12*mm21, m21*mm11 + m22*mm21
        mm12, mm22 = m11*mm12 + m12*mm22, m21*mm12 + m22*mm22

    # reflection coefficient
    r = mm12 / mm22

    # transmission coefficient
    t = 1 / mm22

    return r, t


@numba.jit(nopython=True, cache=True)
def reflec_and_trans(n, lam, thetas, thick, rough):
    """Calculate the reflection coefficient and the transmission coefficient for a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: vector of refractive indices n = 1 - \delta + i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param theta: incident angle in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :return: (reflec, trans)
    """
    if not len(n) == len(rough) + 1 == len(thick) + 2:
        raise ValueError('array lengths do not match: len(n) == len(rough) + 1 == len(thick) + 2 does not hold.')
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


@numba.jit(nopython=True, cache=True)
def fields_inner(k2n2, k2, theta, thick, s2h, mm11, mm12, mm21, mm22, rs, ts, kt, N):
    # wavevectors in the different layers
    k2_x = k2 * math.cos(theta)**2  # k_x is conserved due to snell's law
    k_z = -np.sqrt(k2n2 - k2_x)  # k_z is different for each layer.

    pS, mS = p_m(k_z, N, s2h)

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

    ts[kt][0] = 1  # in the vacuum layer
    rs[kt][0] = r  # in the vacuum layer
    for j in range(1, N+1):  # j = 1 .. N
        ts[kt][j] = mm22[j] * t
        rs[kt][j] = mm12[j] * t
    ts[kt][N+1] = t  # in the substrate
    rs[kt][N+1] = 0


@numba.jit(nopython=True, cache=True)
def fields(n, lam, thetas, thick, rough):
    """Calculate the reflection coefficient and the transmission coefficient for a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: vector of refractive indices n = 1 - \delta + i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param theta: incident angle in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :return: (reflec, trans)
    """
    N = len(thick)
    T = len(thetas)

    k2 = (2 * math.pi / lam)**2  # k is conserved
    k2n2 = k2 * n**2
    s2h = rough**2 / 2

    # preallocate temporary arrays
    mm11 = np.empty(N + 1, dtype=np.complex128)
    mm12 = np.empty(N + 1, dtype=np.complex128)
    mm21 = np.empty(N + 1, dtype=np.complex128)
    mm22 = np.empty(N + 1, dtype=np.complex128)
    # preallocate whole result arrays
    rs = np.empty((T, N+2), dtype=np.complex128)
    ts = np.empty((T, N+2), dtype=np.complex128)

    for kt, theta in enumerate(thetas):
        fields_inner(k2n2, k2, theta, thick, s2h, mm11, mm12, mm21, mm22, rs, ts, kt, N)
    return rs, ts


@numba.jit(nopython=True, cache=True)
def fields_positions_inner(thick, s2h, kt, rs, ts, mm11, mm12, mm21, mm22, N, k_z):
    pS, mS = p_m(k_z[kt], N, s2h)
    # entries of the transition matrix MM
    # mm11, mm12, mm21, mm22
    # RR over interface to substrate
    mm11[N] = pS
    mm12[N] = mS
    mm21[N] = mS
    mm22[N] = pS
    for l in range(N):
        j = N - l - 1  # j = N-1 .. 0
        # transition through layer j
        uj = 1j * k_z[kt][j+1] * thick[j]
        vj = cmath.exp(-uj)
        wj = cmath.exp(+uj)
        # transition through interface between j-1 an j
        pj, mj = p_m(k_z[kt], j, s2h)
        m11 = pj * vj
        m12 = mj * wj
        m21 = mj * vj
        m22 = pj * wj

        mm11[j] = m11*mm11[j+1] + m12*mm21[j+1]
        mm12[j] = m11*mm12[j+1] + m12*mm22[j+1]
        mm21[j] = m21*mm11[j+1] + m22*mm21[j+1]
        mm22[j] = m21*mm12[j+1] + m22*mm22[j+1]

    # reflection coefficient
    r = mm12[0] / mm22[0]

    # transmission coefficient
    t = 1 / mm22[0]

    ts[kt][0] = 1  # in the vacuum layer
    rs[kt][0] = r  # in the vacuum layer
    for l in range(N):
        j = l + 1  # j = 1 .. N
        ts[kt][j] = mm22[j] * t
        rs[kt][j] = mm12[j] * t
    ts[kt][N+1] = t  # in the substrate
    rs[kt][N+1] = 0


@numba.jit(nopython=True, cache=True)
def fields_positions_inner_positions(kp, pos, Z, pos_rs, pos_ts, k_z, ts, rs, T):
    # MM_j * (0, t) is the field at the interface between the layer j and the layer j+1.
    # thus, if pos is within layer j, we need to use the translation matrix
    # TT = exp(-ik_{z,j} h), 0 \\ 0, exp(ik_{z,j} h)
    # with h the distance between the interface between the layer j and the layer j+1 (the "bottom" interface if
    # the vacuum is at the top and the z-axis is pointing upwards) and pos.

    # first find out within which layer pos lies
    for j, zj in enumerate(Z):  # checking from the top
        if pos > zj:
            break
    else:  # within the substrate
        # need to special-case the substrate since we have to propagate "down" from the substrate interface
        # all other cases are propagated "up" from their respective interfaces
        dist_j = 1j * (pos - Z[-1])  # common for all thetas: distance from interface to evaluation_position
        for ko in range(T):  # iterate through all thetas
            pos_rs[ko][kp] = 0.
            pos_ts[ko][kp] = ts[ko][-1] * cmath.exp(k_z[ko][-1] * dist_j)
        return

    # now zj = Z[j] is the layer in which pos lies
    dist_j = 1j * (pos - zj)  # common for all thetas: distance from interface to evaluation_position
    for ko in range(T):
        # now propagate the fields through the layer
        uj = k_z[ko][j] * dist_j

        # fields at position
        pos_ts[ko][kp] = ts[ko][j] * cmath.exp(+uj)
        pos_rs[ko][kp] = rs[ko][j] * cmath.exp(-uj)


@numba.jit(nopython=True, cache=True)
def fields_positions(n, lam, thetas, thick, rough, evaluation_positions):
    """Calculate the electric field intensities in a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: vector of refractive indices n = 1 - \delta + i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param theta: incident angle in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :param evaluation_positions: positions (in nm) at which the electric field should be evaluated. Given in distance
           from the surface, with the axis poiting away from the layer (i.e. negative positions are within the stack)
    :return: (reflec, trans, pos_reflec, pos_trans)
    """
    N = len(thick)
    T = len(thetas)
    P = len(evaluation_positions)

    # wavevectors in the different layers
    k2 = (2 * math.pi / lam)**2  # k is conserved
    k2n2 = k2 * n**2
    s2h = rough**2 / 2
    k2_x = k2 * np.cos(thetas)**2  # k_x is conserved due to snell's law, i.e. only dependent on theta
    k_z = np.empty((T, N+2), dtype=np.complex128)
    for kt in range(T):
        for kl in range(N+2):
            k_z[kt][kl] = -np.sqrt(k2n2[kl] - k2_x[kt])  # k_z is different for each layer.

    # calculate absolute interface positions from thicknesses
    Z = np.empty(N + 1, dtype=np.float64)
    Z[0] = 0.
    cs = -np.cumsum(thick)
    for i in range(0, N):
        Z[i+1] = cs[i]

    # preallocate temporary arrays
    mm11 = np.empty(N + 1, dtype=np.complex128)
    mm12 = np.empty(N + 1, dtype=np.complex128)
    mm21 = np.empty(N + 1, dtype=np.complex128)
    mm22 = np.empty(N + 1, dtype=np.complex128)
    # preallocate whole result arrays
    rs = np.empty((T, N+2), dtype=np.complex128)
    ts = np.empty((T, N+2), dtype=np.complex128)
    pos_rs = np.empty((T, P), dtype=np.complex128)
    pos_ts = np.empty((T, P), dtype=np.complex128)

    # first calculate the fields at the interfaces
    for kt in range(T):
        fields_positions_inner(thick, s2h, kt, rs, ts, mm11, mm12, mm21, mm22, N, k_z)

    # now calculate the fields at the given evaluation positions
    for kp, pos in enumerate(evaluation_positions):
        fields_positions_inner_positions(kp, pos, Z, pos_rs, pos_ts, k_z, ts, rs, T)

    return rs, ts, pos_rs, pos_ts


@numba.jit(nopython=True, cache=True)
def fields_positions_fields(n, lam, thetas, thick, rough):
    """Calculate the electric field intensities in a stack of N layers, with the incident
    wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    differs from Gibaud & Vignaud.
    :param n: vector of refractive indices n = 1 - \delta + i \beta of all layers, so n[0] is usually 1.
    :param lam: x-ray wavelength in nm
    :param theta: incident angle in rad
    :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    :param evaluation_positions: positions (in nm) at which the electric field should be evaluated. Given in distance
           from the surface, with the axis poiting away from the layer (i.e. negative positions are within the stack)
    :return: (reflec, trans, (*args)) with args for fields_positions_positions
    """
    N = len(thick)
    T = len(thetas)

    # wavevectors in the different layers
    k2 = (2 * math.pi / lam)**2  # k is conserved
    k2n2 = k2 * n**2
    s2h = rough**2 / 2
    k2_x = k2 * np.cos(thetas)**2  # k_x is conserved due to snell's law, i.e. only dependent on theta
    k_z = np.empty((T, N+2), dtype=np.complex128)
    for kt in range(T):
        for kl in range(N+2):
            k_z[kt][kl] = -np.sqrt(k2n2[kl] - k2_x[kt])  # k_z is different for each layer.

    # calculate absolute interface positions from thicknesses
    Z = np.empty(N + 1, dtype=np.float64)
    Z[0] = 0.
    cs = -np.cumsum(thick)
    for i in range(0, N):
        Z[i+1] = cs[i]

    # preallocate temporary arrays
    mm11 = np.empty(N + 1, dtype=np.complex128)
    mm12 = np.empty(N + 1, dtype=np.complex128)
    mm21 = np.empty(N + 1, dtype=np.complex128)
    mm22 = np.empty(N + 1, dtype=np.complex128)
    # preallocate whole result arrays
    rs = np.empty((T, N+2), dtype=np.complex128)
    ts = np.empty((T, N+2), dtype=np.complex128)

    # first calculate the fields at the interfaces
    for kt in range(T):
        fields_positions_inner(thick, s2h, kt, rs, ts, mm11, mm12, mm21, mm22, N, k_z)

    return rs, ts, k_z, Z


@numba.jit(nopython=True, cache=True)
def fields_positions_positions(evaluation_positions, rs, ts, k_z, Z):
    P = len(evaluation_positions)
    pos_rs = np.empty(P, dtype=np.complex128)
    pos_ts = np.empty(P, dtype=np.complex128)
    # now calculate the fields at the given evaluation positions
    for kp, pos in enumerate(evaluation_positions):
        # first find out within which layer pos lies
        for j, zj in enumerate(Z):  # checking from the top
            if pos > zj:
                break
        else:  # within the substrate
            # need to special-case the substrate since we have to propagate "down" from the substrate interface
            # all other cases are propagated "up" from their respective interfaces
            dist_j = 1j * (pos - Z[-1])  # common for all thetas: distance from interface to evaluation_position
            pos_rs[kp] = 0.
            pos_ts[kp] = ts[-1] * cmath.exp(k_z[-1] * dist_j)
            continue

        # now zj = Z[j] is the layer in which pos lies
        dist_j = 1j * (pos - zj)  # distance from interface to evaluation_position
        # now propagate the fields through the layer
        uj = k_z[j] * dist_j

        # fields at position
        pos_ts[kp] = ts[j] * cmath.exp(+uj)
        pos_rs[kp] = rs[j] * cmath.exp(-uj)

    return pos_rs, pos_ts


if __name__ == '__main__':
    _n_layers = 1001
    _n = np.array([1] + [1-1e-5+1e-6j, 1-2e-5+2e-6j]*int((_n_layers-1)/2))
    _thick = np.array([.1]*(_n_layers-2))
    _rough = np.array([.02]*(_n_layers-1))
    _wl = 0.15
    _ang_deg = np.linspace(0.1, 2., 10001)
    _ang = np.deg2rad(_ang_deg)
    #print('ang_deg')
    #for _i in _ang_deg:
    #    print(_i)
    _f = fields(_n, _wl, _ang, _thick, _rough)
    #_ar = np.abs(_r)**2
    #_at = np.abs(_t)**2
    #print('# ang_deg abs(r)**2 abs(t)**2 r.real r.imag t.real t.imag')
    #for _a, _iar, _iat, _ir, _it in zip(_ang_deg, _ar, _at, _r, _t):
    #    print('{} {} {} {} {} {} {}'.format(_a, _iar, _iat, _ir.real, _ir.imag, _it.real, _it.imag))








