#include "grba_int.h"
#include "./phi_int/phi_int.h"
#include "./r0_int/r0_int.h"

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

double GrbaIntegrator::IntensityG(double y, double chi) {
  double ys = pow(y, y_exp);
  double chis = pow(chi, chi_exp);
  double fac = pow((7.0 - 2.0*k)*chi*pow(y, 4.0 - k) + 1.0, bg - 2.0);
  return ys*chis*fac;
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

IntG::IntG(double R0, const double Y, const double THV, const double KAP, const double SIG, const double K, const double P, const double GA) : GrbaIntegrator(THV, KAP, SIG, K, P, GA), chi(0.0), r0(R0), y(Y), thp0(ThetaPrime(0.0, R0/Y)) {
  SetChi();
}

IntG::IntG(double R0, const double Y, params& p) : GrbaIntegrator(p), chi(0.0), r0(R0), y(Y), thp0(ThetaPrime(0.0, R0/Y)) {
  SetChi();
}

double IntG::IntegrandY() {
  return pow(y, y_exp);
}

double IntG::IntegrandChi() {
  return pow(chi, chi_exp);
}

double IntG::IntegrandFac() {
  return pow((7.0 - 2.0*k)*chi*pow(y, 4.0 - k) + 1.0, bg - 2.0);
}

double IntG::Integrand() {
  RootFuncPhi rfunc(r0 / y, thv, kap, sig, k, p, ga);
  return r0*IntegrandY()*IntegrandChi()*IntegrandFac()*SimpsPhi(rfunc, 0.0, 2.0*M_PI, 1.0e-7);
}

void IntG::SetChi() {
  // chi = (y - gk*exp2(-exp0)*(y*tan_thv + r0)*(y*tan_thv + r0)) / (pow(y, 5.0 - k));
  chi = Chi(r0, y);
}
