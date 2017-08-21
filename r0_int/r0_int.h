#ifndef R0_INT_H
#define R0_INT_H

#include <cminpack.h>

#include "../grba_int.h"

#define EXPORT extern "C"

class RootFuncR0: GrbaIntegrator
{
public:
  RootFuncR0(const double Y, const double THV, const double KAP, const double SIG, const double K, const double P, const double GA);
  RootFuncR0(const double Y, params& p);
  double F(double r0);
  double DF(double r0);
  const double y;
};

int fcnR0(void *p, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag);
double RootR0(RootFuncR0& func, const double g, const double xacc);

#endif
