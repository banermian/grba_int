#ifndef GRBA_INT_H
#define GRBA_INT_H

#define EXPORT extern "C"

#include <math.h>
#include <stddef.h>
#include <iostream>
#include <cmath>

// EXPORT double ScipyCallableTest(double x);

struct params {
    const double THV;
    const double KAP;
    const double SIG;
    const double K;
    const double P;
    const double GA;
};

class GrbaIntegrator
{
public:
  GrbaIntegrator(const double THV, const double KAP, const double SIG, const double K, const double P, const double GA);
  GrbaIntegrator(params& p);
  double ThetaPrime(double phi, double r);
  double EnergyProfile(double phi, double r);
  double Chi(double r0, double y);
  double IntensityG(double y, double chi);
  int IntegrandG(double *vals, double r0, const double y);
  double R0Max(double y, double g, double xacc);
  // double IntG(int n, double *x, void *user_data);
  // double RootFuncR0(double r0, const double y);
  // double RootJacR0(double r0, const double y);
  double FluxG(double r0, const double y);
  const double thv, kap, sig, k, p, ga, gk, bg, tan_thv;

protected:
  const double tan_thv_sq, sin_2thv, cos_thv, sin_thv, chi_exp, y_exp;
};

class IntG : public GrbaIntegrator
{
public:
  IntG(double R0, const double Y, const double THV, const double KAP, const double SIG, const double K, const double P, const double GA);
  IntG(double R0, const double Y, params& p);
  double IntegrandY();
  double IntegrandChi();
  double IntegrandFac();
  double Integrand();
  double chi;

private:
  void SetChi();
  double r0, y, thp0;
};

#endif
