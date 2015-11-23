#include <complex>
#include <vector>
#include <iostream>

#include "libmatrixmethod.cpp"

using namespace std;

void print_vec(vector<cf> vec) {
    for (cf v: vec) {
        cout << "(" << v.real() << v.imag() << "j)" << endl;
    }
}

void print_vec(vector<float> vec) {
    for (float v: vec) {
        cout << v << endl;
    }
}

int main() {
    const uint n_layers = 10001;
    vector<cf> n(n_layers);
    n[0] = 1.f;
    for (uint i = 1; i < n_layers; i++) {
        if (i%2 == 0) {
            n[i] = 1.f - 1e-5f + 1e-6f * j;
        }
        else {
            n[i] = 1.f - 2e-5f + 2e-6f * j;
        }
    }
    const vector<float> thick(n_layers-2, 0.1f);
    const vector<float> rough(n_layers-1, 0.02f);

    const float wl = 0.15f;

    const uint n_ang = 10001;
    const float start = 0.1f;
    const float end = 2.0f;
    const float step = (end-start) / (n_ang-1);
    vector<float> ang_deg(n_ang);
    for (uint i = 0; i < n_ang; i++) {
        ang_deg[i] = start + step * i;
    }
    vector<float> ang(n_ang);
    for (uint i = 0; i < n_ang; i++) {
        ang[i] = ang_deg[i] * M_PI / 180.f;
    }

    struct vec_rt_t rt = reflec_and_trans(n, wl, ang, thick, rough);

    cout << "# ang_deg abs(r)**2 abs(t)**2 r.real r.imag t.real t.imag" << endl;
    for (uint i = 0; i < n_ang; i++) {
        cout << ang_deg[i] << " " << pow(abs(rt.r[i]), 2.f) << " " << pow(abs(rt.t[i]), 2.f) << " " << rt.r[i].real() << " "
        << rt.r[i].imag() << " " << rt.t[i].real() << " " << rt.t[i].imag() << " " << endl;
    }

}
