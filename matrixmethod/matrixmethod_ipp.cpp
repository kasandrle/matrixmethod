/*Matrix method algorithm for x-ray reflectivity and transmittivity as described by A. Gibaud and G. Vignaud in
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
*/

#include <iostream>
#include "ipp.h"

using namespace std;

typedef Ipp64fc c;
typedef Ipp64f d;

const c j(0., 1.);

// number of finite layers
#define N 9999
// number of angles
#define T 10001

typedef c mat2x2_t[4];

void mul_inplace(c * a, const c * b) {
    c temp_am11 = b[0] * a[0] + b[1] * a[2];
    a[2] = b[2] * a[0] + b[3] * a[2];
    a[0] = temp_am11;

    c temp_am12 = b[0] * a[1] + b[1] * a[3];
    a[3] = b[2] * a[1] + b[3] * a[3];
    a[1] = temp_am12;
}


struct rt_t {
    c r;
    c t;
    rt_t(c r, c t): r(r), t(t) {}
};


struct rt_t reflec_and_trans_inner(const c * k2n2, const d & k2, const d & theta,
                                   const d * thick, const d * s2h) {

    mat2x2_t mm[N];

    const Ipp64f k2_x = k2 * sq(cos(theta));  // k_x is conserved due to snells law

    c k_z[N+2];
    ippsAddC_64fc(k2n2, k2_x, k_z, N+2);
    ippsSqrt_64fc_I(k_z, N+2);
    c * k_z = vzSqrt(N+2, k2n2);
    c k_z_p[N+1];
    ippsCopy_64fc(k_z+1*sizeof(c), k_z_p, N+1);

    // pre-compute the coefficients for construction of the matrix elements of the refraction matrices
    // pre-compute the coefficients for construction of the transition matrices
    c tp[N+1];
    c tm[N+1];
    ippsAdd_64fc(k_z, k_z_p, tp, N+1);
    ippsSub_64fc(k_z, k_z_p, tp, N+1);
    c rp[N+1];
    c rm[N+1];
    ippsSqr_64fc(tm, rp, N+1);
    ippsMul_64fc_I(s2h, tm, N+1);
    ipps

    for (uint l=0; l < N; l++) {
        const c tp = k_z[l] + k_z_p[l];
        const c tm = k_z[l] - k_z_p[l];
        const c rp = exp(-sq(tm) * s2h[l]);
        const c rm = exp(-sq(tp) * s2h[l]);
        const c o = two_times(k_z[l]);
        const c p = tp * rp/o;
        const c m = tm * rm/o;
        const c epv = exp(j * k_z_p[l] * thick[l]);
        mm[l][0] = p / epv;
        mm[l][1] = m * epv;
        mm[l][2] = m / epv;
        mm[l][3] = p * epv;
    }

    const c tp = k_z[N] + k_z_p[N];
    const c tm = k_z[N] - k_z_p[N];
    const c rp = exp(-sq(tm) * s2h[N]);
    const c rm = exp(-sq(tp) * s2h[N]);
    const c o = two_times(k_z[N]);
    const c p = tp * rp/o;
    const c m = tm * rm/o;


    // initialize transition matrix MM to RR over interface to substrate
    mat2x2_t MM;
    MM[0] = p;
    MM[1] = m;
    MM[2] = m;
    MM[3] = p;
    for (uint l = 0; l < N; l++) {
        uint i = N - l - 1; // i = N-1 .. 0
        // transition through interface between i-1 and i and through layer i
        // TT * RR
        mul_inplace(MM, mm[i]);
    }

    struct rt_t rt (
            // reflection coefficient
            MM[1] / MM[3],
            // transmission coefficient
            1. / MM[3]
    );

    return rt;
}

struct arr_rt_t {
    c r[T];
    c t[T];
};


struct arr_rt_t reflec_and_trans(const c * n, const d & lam, const d * thetas,
                                 const d * thick, const d * rough) {
    // Calculate the reflection coefficient and the transmission coefficient for a stack of N layers, with the incident
    // wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    // Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    // differs from Gibaud & Vignaud.
    // :param n: vector of refractive indices n = 1 - \delta - i \beta of all layers, so n[0] is usually 1.
    // :param lam: x-ray wavelength in nm
    // :param theta: incident angle in rad
    // :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    // :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    // :return: (reflec, trans)
    const d k2 = sq(2. * M_PI / lam);  // k is conserved
    c k2n2[N+2];
    for (uint i=0; i < N+2; i++) {
        k2n2[i] = k2 * sq(n[i]);
    }
    d s2h[N+1];
    for (uint i=0; i < N+1; i++) {
        s2h[i] = sq(rough[i]) / 2.;
    }

    // results
    struct arr_rt_t result;
    for (uint i=0; i < T; i++) {
        struct rt_t rt = reflec_and_trans_inner(k2n2, k2, thetas[i], thick, s2h);
        result.r[i] = rt.r;
        result.t[i] = rt.t;
    }
    return result;
}

int main() {
    c n[N+2];
    n[0] = 1.;
    for (uint i = 1; i < N+2; i++) {
        if (i%2 == 0) {
            n[i] = 1. - 1e-5 + 1e-6 * j;
        }
        else {
            n[i] = 1. - 2e-5 + 2e-6 * j;
        }
    }
    d thick[N];
    for (uint i = 0; i < N; i++) {
        thick[i] = 0.1;
    }
    d rough[N+1];
    for (uint i = 0; i < N+1; i++) {
        rough[i] = 0.02;
    }

    const d wl = 0.15;

    const d start = 0.1;
    const d end = 2.0;
    const d step = (end-start) / (T-1);
    d ang_deg[T];
    for (uint i = 0; i < T; i++) {
        ang_deg[i] = start + step * i;
    }
    d ang[T];
    for (uint i = 0; i < T; i++) {
        ang[i] = ang_deg[i] * M_PI / 180.;
    }

    struct arr_rt_t rt = reflec_and_trans(n, wl, ang, thick, rough);

    cout << "# ang_deg abs(r)**2 abs(t)**2 r.real r.imag t.real t.imag" << endl;
    for (uint i = 0; i < T; i++) {
        cout << ang_deg[i] << " " << sq(abs(rt.r[i])) << " " << sq(abs(rt.t[i])) << " " << rt.r[i].real() << " "
        << rt.r[i].imag() << " " << rt.t[i].real() << " " << rt.t[i].imag() << " " << endl;
    }

}
