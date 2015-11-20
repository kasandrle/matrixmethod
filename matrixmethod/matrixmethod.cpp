#include <complex>
#include <vector>
#include <iostream>

#include "libmatrixmethod.cpp"

using namespace std;

void print_vec(vector<c> vec) {
    for (c v: vec) {
        cout << "(" << v.real() << v.imag() << "j)" << endl;
    }
}

void print_vec(vector<double> vec) {
    for (double v: vec) {
        cout << v << endl;
    }
}

int main() {
    const uint n_layers = 10001;
    vector<c> n(n_layers);
    n[0] = 1.;
    for (uint i = 1; i < n_layers; i++) {
        if (i%2 == 0) {
            n[i] = 1. - 1e-5 + 1e-6 * j;
        }
        else {
            n[i] = 1. - 2e-5 + 2e-6 * j;
        }
    }
    const vector<double> thick(n_layers-2, 0.1);
    const vector<double> rough(n_layers-1, 0.02);

    const double wl = 0.15;

    const uint n_ang = 10001;
    const double start = 0.1;
    const double end = 2.0;
    const double step = (end-start) / (n_ang-1);
    vector<double> ang_deg(n_ang);
    for (uint i = 0; i < n_ang; i++) {
        ang_deg[i] = start + step * i;
    }
    vector<double> ang(n_ang);
    for (uint i = 0; i < n_ang; i++) {
        ang[i] = ang_deg[i] * M_PI / 180.;
    }

    struct vec_rt_t rt = reflec_and_trans(n, wl, ang, thick, rough);

    cout << "# ang_deg abs(r)**2 abs(t)**2 r.real r.imag t.real t.imag" << endl;
    for (uint i = 0; i < n_ang; i++) {
        cout << ang_deg[i] << " " << pow(abs(rt.r[i]), 2) << " " << pow(abs(rt.t[i]), 2) << " " << rt.r[i].real() << " "
        << rt.r[i].imag() << " " << rt.t[i].real() << " " << rt.t[i].imag() << " " << endl;
    }

}
