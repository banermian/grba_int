#include "r0_int.h"

RootFuncR0::RootFuncR0(const double Y, const double THV, const double KAP, const double SIG, const double K, const double P, const double GA) : GrbaIntegrator(THV, KAP, SIG, K, P, GA), y(Y) {}

RootFuncR0::RootFuncR0(const double Y, params& p) : GrbaIntegrator(p), y(Y) {}

// Test switching from r0 / y to r0, y
double RootFuncR0::F(double r0) {
  double eng0 = EnergyProfile(0.0, r0 / y);
  double lhs = gk*eng0*(r0 + y*tan_thv)*(r0 + y*tan_thv);
  double rhs = y - pow(y, 5.0 - k);
  return lhs - rhs;
}

double RootFuncR0::DF(double r0) {
  double thp0 = ThetaPrime(0.0, r0 / y);
  double frac = kap*log(2.0)*pow(thp0 / sig, 2.0*kap)*(y*(r0 + y*tan_thv) / (r0 * (y + r0*sin_thv*cos_thv)));
  double exponent = 2.0*gk*(r0 + y*tan_thv)*EnergyProfile(0.0, r0 / y);
  return (1.0 - frac)*exponent;
}

int fcnR0(void *p, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag)
{
    /*      subroutine fcnR0 for hybrj example. */
    (void)p;

    if (iflag != 2)
    {
        fvec[0] = ((RootFuncR0*)p)->F(x[0]);
    }
    else
    {
        fjac[0] = ((RootFuncR0*)p)->DF(x[0]);
    }
    //printf("phi = %02.3f, \t f = %02.5f, \t df = %02.5f \n", (double)((RootFuncR0*)p)->phiVal() / M_PI, (double)fvec[0], (double)fjac[0]);
    return 0;
}

double RootR0(RootFuncR0& func, double g, const double xacc) {
    int n, ldfjac, info, lwa;
    double tol; // , fnorm;
    double x[1], fvec[1], fjac[1 * 1], wa[99];

    n = 1;
    ldfjac = 1;
    lwa = 99;

    //tol = sqrt(__cminpack_func__(dpmpar)(1));
    tol = xacc;

    x[0] = g;

    void *p = NULL;
    p = &func;

    info = __cminpack_func__(hybrj1)(fcnR0, p, n, x, fvec, fjac, ldfjac, tol, wa, lwa);

    return (double)x[0];
}
