#ifndef PHI_INT_H
#define PHI_INT_H

#include <cminpack.h>

#include "../grba_int.h"

#define EXPORT extern "C"

class RootFuncPhi: GrbaIntegrator
{
public:
  RootFuncPhi(const double R0, const double THV, const double KAP, const double SIG, const double K, const double P, const double GA);
  RootFuncPhi(const double R0, params& p);
  double F(double r);
  double DF(double r);
  void SetPhi(double phi_val);
  double rPrime();
  const double r0;

private:
  double phi, cos_phi;
};

int fcnPhi(void *p, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag);
double RootPhi(RootFuncPhi& func, const double g, const double xacc);
double SimpsPhi(RootFuncPhi& func, const double a, const double b, const double eps = 1.0e-9);
double SimpsPhiAlt(RootFuncPhi& func, const double a, const double b, const double eps = 1.0e-9);
EXPORT double PhiIntegrate(const double r0, const double thv, const double kap, const double sig, const double k, const double p, const double ga);
EXPORT double PhiIntegrateAlt(const double r0, const double thv, const double kap, const double sig, const double k, const double p, const double ga);

// double f(double x, void *p);
// double R0Integrate(const double thv, const double kap, const double sig);

#endif
