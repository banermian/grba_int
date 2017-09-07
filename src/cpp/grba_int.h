#ifndef GRBA_INT_H
#define GRBA_INT_H

#define EXPORT extern "C"

#include <math.h>
#include <stddef.h>
#include <iostream>
#include <cmath>

static const double TORAD = M_PI / 180.0;

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
  double IntegrandY(double y);
  double IntegrandChi(double r0, double y);
  double IntegrandChi(double chi);
  double IntegrandFac(double r0, double y);
  double IntegrandPhi(double r0, double y);
  double IntegrandPhiAlt(double r0, double y);
  double Integrand(double r0, double y);
  int IntegrandG(double *vals, double r0, const double y);
  double R0Max(double y, double g, double xacc);
  double FluxG(double r0, const double y);
  const double thv, kap, sig, k, p, ga, gk, bg, tan_thv;

protected:
  const double tan_thv_sq, sin_2thv, cos_thv, sin_thv, chi_exp, y_exp;
};

struct intparams {
  const double Y;
  const double THV;
  const double KAP;
  const double SIG;
  const double K;
  const double P;
  const double GA;
};
double Integrand(double x, void *int_params);
double Integrate(const double y,  GrbaIntegrator& grb);

#endif
