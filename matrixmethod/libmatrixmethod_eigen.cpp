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

#include <complex>
#include <tuple>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef complex<double> c;

const c j(0., 1.);

typedef Array<c, Dynamic, 1> ArrXcd;
typedef Array<double, Dynamic, 1> ArrXd;

tuple<c, c> p_m(const ArrXcd& k_z, const uint l, const ArrXd& s2h) {
    // matrix elements of the refraction matrices
    // p[j] is p_{j, j+1}
    // p_{j, j+1} = (k_{z, j} + k_{z, j+1}) / (2 * k_{z, j}) * exp(-(k_{z,j} - k_{z,j+1})**2 sigma_j**2/2) for all j=0..N-1
    // m_{j, j+1} = (k_{z, j} - k_{z, j+1}) / (2 * k_{z, j}) * exp(-(k_{z,j} + k_{z,j+1})**2 sigma_j**2/2) for all j=0..N-1

    c p = k_z(l) + k_z(l + 1);
    c m = k_z(l) - k_z(l + 1);
    c rp = exp(-pow(m, 2) * s2h(l));
    c rm = exp(-pow(p, 2) * s2h(l));
    c o = 2. * k_z(l);
    return make_tuple(p * rp / o, m * rm / o);
}

tuple<c, c> reflec_and_trans_inner(const ArrXcd& k2n2, const c k2, const double theta,
                                   const ArrXd& thick, const ArrXd& s2h) {
    c k2_x = k2 * pow(cos(theta), 2);  // k_x is conserved due to snells law
    ArrXcd k_z = -sqrt(k2n2 - k2_x);
    uint N = thick.size();

    c pS, mS;
    tie(pS, mS) = p_m(k_z, N, s2h);
    // RR over interface to substrate
    Matrix2cd MM;
    MM << pS, mS,
          mS, pS;
    for (uint l = 0; l < N; l++) {
        uint i = N - l - 1; // i = N-1 .. 0
        // transition through layer i
        c vi = j * k_z[i + 1] * thick[i];
        Matrix2cd TT;
        TT << exp(-vi), 0.,
              0, exp(+vi);
        // transition through interface between i-1 and i
        c pi, mi;
        tie(pi, mi) = p_m(k_z, i, s2h);
        Matrix2cd RR;
        RR << pi, mi,
              mi, pi;

        MM *= TT * RR;
    }

    // reflection coefficient
    c r = MM(0, 1) / MM(1, 1);

    // transmission coefficient
    c t = 1. / MM(1, 1);

    return make_tuple(r, t);
}

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
tuple<ArrXcd, ArrXcd> reflec_and_trans(const ArrayBase<DerivedA>& n, const double lam,
                                       const ArrayBase<DerivedB>& thetas,
                                       const ArrayBase<DerivedC>& thick, const ArrayBase<DerivedD>& rough) {
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
    uint T = thetas.size();

    c k2 = pow(2. * M_PI / lam, 2);  // k is conserved
    ArrXcd k2n2(n.size());
    k2n2 = k2 * n.pow(2);
    ArrXd s2h(rough.size());
    s2h = rough.pow(2) / 2.;

    ArrXcd rs(T), ts(T);
    for (uint i = 0; i < T; i++) {
        c r, t;
        tie(r, t) = reflec_and_trans_inner(k2n2, k2, thetas(i), thick, s2h);
        rs(i) = r;
        ts(i) = t;
    }
    return make_tuple(rs, ts);
}

/*
tuple<vector<c>, vector<c>> fields_inner(const vector<c> k2n2, const c k2, const double theta,
                                         const vector<double> thick, const vector<double> s2h,
                                         vector<c> mm11, vector<c> mm12, vector<c> mm21, vector<c> mm22,
                                         const uint N) {
    c k2_x = k2 * pow(cos(theta), 2);  // k_x is conserved due to snells law
    vector<c> k_z;
    k_z.reserve(k2n2.size());
    for (auto k2n2_i: k2n2) {  // k_z is different for each layer
        k_z.push_back(-sqrt(k2n2_i - k2_x));
    }

    c pS, mS;
    tie(pS, mS) = p_m(k_z, N, s2h);

    // RR over interface to substrate
    mm11[N] = pS;
    mm12[N] = mS;
    mm21[N] = mS;
    mm22[N] = pS;

    for (uint l = 0; l < N; l++) {
        uint i = N - l - 1; // i = N-1 .. 0
        // transition through layer i
        c vi = j * k_z[i + 1] * thick[i];
        // transition through interface between i-1 and i
        c pi, mi;
        tie(pi, mi) = p_m(k_z, i, s2h);
        c m11 = pi * exp(-vi);
        c m12 = mi * exp(+vi);
        c m21 = mi * exp(-vi);
        c m22 = pi * exp(+vi);

        mm11[i] = m11*mm11[i+1] + m12*mm21[i+1];
        mm12[i] = m11*mm12[i+1] + m12*mm22[i+1];
        mm21[i] = m21*mm11[i+1] + m22*mm21[i+1];
        mm22[i] = m21*mm12[i+1] + m22*mm22[i+1];
    }

    // reflection coefficient
    c r = mm12[0] / mm22[0];

    // transmission coefficient
    c t = 1. / mm22[0];

    vector<c> fields_r;
    vector<c> fields_t;
    fields_r.reserve(N+2);
    fields_t.reserve(N+2);

    fields_r.push_back(r);  // in the vacuum layer
    fields_t.push_back(1.);

    for (uint l = 0; l < N; l++) {
        uint i = l + 1;  // i = 1 .. N
        fields_r.push_back(mm12[i] * t);
        fields_t.push_back(mm22[i] * t);
    }

    fields_r.push_back(0);  // in the substrate
    fields_t.push_back(t);

    return make_tuple(fields_r, fields_t);
}
 */

/*
tuple<vector<vector<c>>, vector<vector<c>>> fields(const vector<c> n, const double lam, const vector<double> thetas,
                                                   const vector<double> thick, const vector<double> rough) {
    // Calculate the reflection and transmission fields for a stack of N layers, with the incident
    // wave coming from layer 0, which is reflected into layer 0 and transmitted into layer N.
    // Note that N=len(n) is the total number of layers, including the substrate. That is the only point where the notation
    // differs from Gibaud & Vignaud.
    // :param n: vector of refractive indices n = 1 - \delta - i \beta of all layers, so n[0] is usually 1.
    // :param lam: x-ray wavelength in nm
    // :param theta: incident angle in rad
    // :param thick: thicknesses in nm, len(thick) = N-2, since layer 0 and layer N are assumed infinite
    // :param rough: rms roughness in nm, len(rough) = N-1 (number of interfaces)
    // :return: (reflec, trans)

    c k2 = pow(2. * M_PI / lam, 2);  // k is conserved
    vector<c> k2n2;
    k2n2.reserve(n.size());
    for (auto n_i: n) {
        k2n2.push_back(k2 * pow(n_i, 2));
    }
    vector<double> s2h;
    s2h.reserve(rough.size());
    for (auto rough_i: rough) {
        s2h.push_back(pow(rough_i, 2) / 2.);
    }

    uint N = thick.size();

    // entries of the transition matrix MM
    // pre-fill with NaN for re-usage in the inner loop
    vector<c> mm11, mm12, mm21, mm22;
    mm11.reserve(N+2);
    mm12.reserve(N+2);
    mm21.reserve(N+2);
    mm22.reserve(N+2);
    for (uint i = 0; i < N+1; i++) {
        mm11.push_back(NAN);
        mm12.push_back(NAN);
        mm21.push_back(NAN);
        mm22.push_back(NAN);
    }

    vector<vector<c>> rs, ts;
    rs.reserve(thetas.size());
    ts.reserve(thetas.size());
    for (auto theta: thetas) {
        vector<c> r, t;
        tie(r, t) = fields_inner(k2n2, k2, theta, thick, s2h, mm11, mm12, mm21, mm22, N);
        rs.push_back(r);
        ts.push_back(t);
    }
    return make_tuple(rs, ts);
}
*/
/*
tuple<vector<vector<c>>, vector<vector<c>>> fields_positions(vector<c> n, double lam, vector<double> thetas,
                                                             vector<double> thick, vector<double> rough,
                                                             vector<double> evaluation_positions) {


    c k2 = pow(2. * M_PI / lam, 2);  // k is conserved
    vector<c> k2n2;
    for (auto n_i: n) {
        k2n2.push_back(k2 * pow(n_i, 2));
    }
    vector<double> s2h;
    for (auto rough_i: rough) {
        s2h.push_back(pow(rough_i, 2) / 2.);
    }

    vector<vector<c>> rs, ts;
    for (auto theta: thetas) {
        vector<c> r, t;
        tie(r, t) = fields_inner(k2n2, k2, theta, thick, s2h);
        rs.push_back(r);
        ts.push_back(t);



    uint P = evaluation_positions.size();
    uint N = thick.size();

    vector<double> Z;
    Z.push_back(0.);
    double cs = 0.;
    for (uint i = 0; i < N; i++) {
        cs += thick[i];
        Z.push_back(cs);
    }

    vector<c> pos_rs, pos_ts;

    for (uint kp = 0; kp < P; kp++) {
        double pos = evaluation_positions[kp];
        // first find out within which layer pos lies
        uint i;
        for (i = 0; i < N+1; i++) {  // checking from the top
            if (pos > Z[i]) break;
        }
        if (pos < Z[i]) {  // within the substrate
            // need to special-case the substrate since we have to propagate "down" from the the substrate interface
            // all other cases are propagated "up" from their respective interfaces
            c dist_j = j * (pos - Z[i]);  // common for all thetas: distance from interface to evaluation position
            c k2_x = k2 * pow(cos(theta), 2);  // k_x is conserved due to snells law
            vector<c> k_z;
            for (auto k2n2_i: k2n2) {  // k_z is different for each layer
                k_z.push_back(-sqrt(k2n2_i - k2_x));
            }
            pos_rs.push_back(0.);  // by definition
            pos_ts.push_back(ts[ts.end()] * exp(k_z * dist_j));
        }
        else {  // within the layer Z[i]

        }

    }

};
*/
/*
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
    _n = np.array([1] + [1-1e-5+1e-6j]*10 + [1-2e-5+2e-6j] + [1])
    _rough = np.array([.2]*12)
    _thick = np.array([1.]*11)
    _wl = 0.15
    _ang_deg = np.linspace(0, 1, 101)[1:]
    _ang = np.deg2rad(_ang_deg)
    _r, _t, _k_z, _Z = fields_positions_fields(_n, _wl, _ang, _thick, _rough)
    _pos_rs, _pos_ts = fields_positions_positions(np.linspace(-10, 20, 31), _r[-1], _t[-1], _k_z[-1], _Z)
    print(_pos_rs)
    print(_pos_ts)
    pass





*/











