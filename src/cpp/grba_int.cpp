#include "grba_int.h"
#include "./phi_int/phi_int.h"
#include "./r0_int/r0_int.h"
#include <gsl/gsl_integration.h>

GrbaIntegrator::GrbaIntegrator(const double THV, const double KAP, const double SIG, const double K, const double P, const double GA) : thv(THV), kap(KAP), sig(SIG), k(K), p(P), ga(GA), gk((4.0 - k)*ga*ga), bg((1.0 - p) / 2.0), tan_thv(tan(thv)), tan_thv_sq(tan(thv)*tan(thv)), sin_2thv(sin(2.0*thv)), cos_thv(cos(thv)), sin_thv(sin(thv)), chi_exp((7.0*k - 23.0 + bg*(13.0 + k)) / (6.0*(4.0 - k))), y_exp(0.5*(bg*(4.0 - k) + 4.0 - 3.0*k)) {}

GrbaIntegrator::GrbaIntegrator(params& ps) : thv(ps.THV), kap(ps.KAP), sig(ps.SIG), k(ps.K), p(ps.P), ga(ps.GA), gk((4.0 - k)*ga*ga), bg((1.0 - p) / 2.0), tan_thv(tan(thv)), tan_thv_sq(tan(thv)*tan(thv)), sin_2thv(sin(2.0*thv)), cos_thv(cos(thv)), sin_thv(sin(thv)), chi_exp((7.0*k - 23.0 + bg*(13.0 + k)) / (6.0*(4.0 - k))), y_exp(0.5*(bg*(4.0 - k) + 4.0 - 3.0*k)) {}

double GrbaIntegrator::ThetaPrime(double phi, double r) {
  double cos_phi = cos(phi);
  double numer = r*pow(pow(cos_thv, 2) - 0.25*pow(sin_2thv, 2)*pow(cos_phi, 2), 0.5);
  double denom = 1.0 + 0.5*r*sin_2thv*cos_phi;
  return numer / denom;
}

double GrbaIntegrator::EnergyProfile(double phi, double r) {
  double thp = ThetaPrime(phi, r);
  return exp2(-pow(thp / sig, 2.0*kap));
}

double GrbaIntegrator::Chi(double r0, double y) {
  double eng0 = EnergyProfile(0.0, r0 / y);
  double chi = (y - gk*eng0*(y*tan_thv + r0)*(y*tan_thv + r0)) / (pow(y, 5.0 - k));
  return chi;
}

double GrbaIntegrator::IntegrandY(double y) {
  return pow(y, y_exp);
}

double GrbaIntegrator::IntegrandChi(double r0, double y) {
  double chi = Chi(r0, y);
  return pow(chi, chi_exp);
}

double GrbaIntegrator::IntegrandChi(double chi) {
  return pow(chi, chi_exp);
}

double GrbaIntegrator::IntegrandFac(double r0, double y) {
  double chi = Chi(r0, y);
  return pow((7.0 - 2.0*k)*chi*pow(y, 4.0 - k) + 1.0, bg - 2.0);
}

double GrbaIntegrator::IntegrandPhi(double r0, double y) {
  RootFuncPhi rfunc(r0 / y, thv, kap, sig, k, p, ga);
  return SimpsPhi(rfunc, 0.0, 2.0*M_PI, 1.0e-7);
}

double GrbaIntegrator::IntegrandPhiAlt(double r0, double y) {
  RootFuncPhi rfunc(r0 / y, thv, kap, sig, k, p, ga);
  return SimpsPhiAlt(rfunc, 0.0, 2.0*M_PI, 1.0e-7);
}

double GrbaIntegrator::Integrand(double r0, double y) {
  return r0*IntegrandY(y)*IntegrandChi(r0, y)*IntegrandFac(r0, y)*IntegrandPhi(r0, y);
}

int GrbaIntegrator::IntegrandG(double *vals, double r0, const double y) {
  double chi = Chi(r0, y);
  RootFuncPhi rfunc(r0 / y, thv, kap, sig, k, p, ga);
  vals[0] = pow(y, y_exp);
  vals[1] = pow(chi, chi_exp);
  vals[2] = pow((7.0 - 2.0*k)*chi*pow(y, 4.0 - k) + 1.0, bg - 2.0);
  vals[3] = SimpsPhi(rfunc, 0.0, 2.0*M_PI, 1.0e-7);
  vals[4] = r0*vals[0]*vals[1]*vals[2]*vals[3];
  return 0;
}

double GrbaIntegrator::FluxG(double r0, const double y) {
  double chi = Chi(r0, y);
  RootFuncPhi rfunc(r0 / y, thv, kap, sig, k, p, ga);
  return r0*pow(y, y_exp)*pow(chi, chi_exp)*pow((7.0 - 2.0*k)*chi*pow(y, 4.0 - k) + 1.0, bg - 2.0)*SimpsPhi(rfunc, 0.0, 2.0*M_PI, 1.0e-7);
}

double GrbaIntegrator::R0Max(double y, double g, double xacc) {
  RootFuncR0 rfunc(y, thv, kap, sig, k, p, ga);
  double root = RootR0(rfunc, g, xacc);
  return root;
}

double Integrand(double x, void *int_params) {
  struct intparams * p = (struct intparams *)int_params;
  const double y = p->Y;
  const double thv = p->THV;
  const double kap = p->KAP;
  const double sig = p->SIG;
  const double k = p->K;
  const double pp = p->P;
  const double ga = p->GA;
  GrbaIntegrator grb(thv*TORAD, kap, sig, k, pp, ga);
  return grb.FluxG(x, y);
}

double Integrate(const double y,  GrbaIntegrator& grb) {
  double result, error;
  double min = 1.0e-9;
  double max = grb.R0Max(y, 0.21, 1.0e-7);
  double chi_max = grb.Chi(max, y);
  if ((chi_max > 10.0) || (chi_max < 0.0)) {
    return 0.0;
  }

  struct intparams IP = { y, grb.thv, grb.kap, grb.sig, grb.k, grb.p, grb.ga };

  gsl_integration_workspace * w
  = gsl_integration_workspace_alloc (100);

  gsl_function F;
  F.function = &Integrand;
  F.params = &IP;

  gsl_integration_qags (&F, min, max, 0, 1e-7, 100, w, &result, &error);

return result;
}

// double IntegrandG(double x, void *int_params) {
//   struct params * p = (struct params *)int_params;
//   GrbaIntegrator grb(p->THV*TORAD, p->KAP, p->SIG, p->K, p->P, p->GA);
// }