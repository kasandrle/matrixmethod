#include <tuple>
#include <iostream>
#include <Eigen/Dense>

#include "libmatrixmethod_eigen.cpp"

using namespace Eigen;
using namespace std;

int main() {
    ArrXcd n(100001);
    n(0) = 1.;
    for (uint i = 0; i < 50000; i++) {
        n(2*i+1) = 1. - 1e-5 + 1e-6 * j;
        n(2*i+2) = (1. - 2e-5 + 2e-6 * j);
    }
    ArrXd rough(100000);
    rough = VectorXd::Constant(100000, 0.2);
    ArrXd thick(99999);
    thick = VectorXd::Constant(99999, 1.0);
    double wl = 0.15;

    ArrXd ang_deg = ArrXd::LinSpaced(100, 0.1, 2.);
    ArrXd ang = ang_deg * M_PI / 180.;

    //cout << "ang_deg" << endl;
    //cout << ang_deg << endl;

    ArrXcd rs, ts;
    tie(rs, ts) = reflec_and_trans(n, wl, ang, thick, rough);


    cout << "rs" << endl;
    cout << rs << endl;
    cout << "ts" << endl;
    cout << ts << endl;
}
