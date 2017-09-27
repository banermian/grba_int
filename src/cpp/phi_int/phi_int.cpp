#include "phi_int.h"
// #include <gsl/gsl_integration.h>

RootFuncPhi::RootFuncPhi(const double R0, const double THV, const double KAP, const double SIG, const double K, const double P, const double GA) : GrbaIntegrator(THV, KAP, SIG, K, P, GA), r0(R0), phi(0.0), cos_phi(1.0) {}

RootFuncPhi::RootFuncPhi(const double R0, params& p) : GrbaIntegrator(p), r0(R0), phi(0.0), cos_phi(1.0) {}

double RootFuncPhi::F(double r) {
  double eng = EnergyProfile(phi, r);
  double eng0 = EnergyProfile(0.0, r);
  double lhs = (pow(r, 2) + 2.0*r*tan_thv*cos_phi + tan_thv_sq)*eng;
  double rhs = pow(r0 + tan_thv, 2)*eng0;
  return lhs - rhs;
}

double RootFuncPhi::DF(double r) {
  double one = r + tan_thv*cos_phi;
  double thp = ThetaPrime(phi, r);
  double two = pow(r, 2) + 2.0*r*tan_thv*cos_phi + tan_thv_sq;
  double top = kap*log(2.0)*pow(thp / sig, 2.0*kap);
  double bot = r*(1.0 + 0.5*r*sin_2thv*cos_phi);
  double expo = 2.0*EnergyProfile(phi, r);
  return expo*(one - two*top / bot);
}

void RootFuncPhi::SetPhi(double phi_val) {
  phi = phi_val;
  cos_phi = cos(phi_val);
}

double RootFuncPhi::rPrime() {
  return -tan_thv*cos_phi + sqrt(r0*r0 + 2*r0*tan_thv + cos_phi*cos_phi*tan_thv_sq);
}

int fcnPhi(void *p, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag)
{
    /*      subroutine fcnPhi for hybrj example. */
    (void)p;

    if (iflag != 2)
    {
        fvec[0] = ((RootFuncPhi*)p)->F(x[0]);
    }
    else
    {
        fjac[0] = ((RootFuncPhi*)p)->DF(x[0]);
    }
    //printf("phi = %02.3f, \t f = %02.5f, \t df = %02.5f \n", (double)((RootFuncPhi*)p)->phiVal() / M_PI, (double)fvec[0], (double)fjac[0]);
    return 0;
}

double RootPhi(RootFuncPhi& func, double g, const double xacc) {
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

    try {
      info = __cminpack_func__(hybrj1)(fcnPhi, p, n, x, fvec, fjac, ldfjac, tol, wa, lwa);
      return (double)x[0];
    }
    catch (...) {
      return 0.0;
    }
};

double SimpsPhi(RootFuncPhi& root_func, const double a, const double b, const double eps) {
  const int NMAX = 25;
  try {
  double sum, osum = 0.0, r0 = root_func.r0;
    for (int n = 3; n < NMAX; n++) {
      int it, j;
      double h, s, x, g, tnm;
      for (it = 2, j = 1; j<n - 1; j++) it <<= 1;
      tnm = it;
      h = (b - a) / tnm;
      s = 2.0;
      g = r0;
      x = a;
      //printf("SimpsPhi: Nsteps = %d, step size = %02.3e, starting guess = %02.3e \n", it, h, g);
      for (int i = 1; i < it; i++, x += h) {
        root_func.SetPhi(x);
        double rp = RootPhi(root_func, g, 1.0e-9);
        g = rp;
        double fx = pow(rp / r0, 2.0);
        if (i % 2) {
          s += 4.0*fx;
        }
        else {
          s += 2.0*fx;
        }
      }
      //system("pause");
      sum = s*h / 3.0;
      // if (n > 3)
      if (std::abs(sum - osum) < eps*std::abs(osum) || (sum == 0.0 && osum == 0.0)) {
        //std::cout << "n = " << n << ",\tNum Steps = " << it << ",\tSum = " << sum << std::endl;
        return sum;
      }
      osum = sum;
    }
    throw("Maximum number of iterations exceeded in simpsPhi");
  }
  catch (char message) {
    std::cout << "An exception occurred" << message << std::endl;
    return 0.0;
  }
}

double SimpsPhiAlt(RootFuncPhi& root_func, const double a, const double b, const double eps) {
  const int NMAX = 25;
  try {
  double sum, osum = 0.0, r0 = root_func.r0;
    for (int n = 3; n < NMAX; n++) {
      int it, j;
      double h, s, x, g, tnm;
      for (it = 2, j = 1; j<n - 1; j++) it <<= 1;
      tnm = it;
      h = (b - a) / tnm;
      s = 2.0;
      g = r0;
      x = a;
      //printf("SimpsPhi: Nsteps = %d, step size = %02.3e, starting guess = %02.3e \n", it, h, g);
      for (int i = 1; i < it; i++, x += h) {
        root_func.SetPhi(x);
        double rp = root_func.rPrime();
        double fx = pow(rp / r0, 2.0);
        if (i % 2) {
          s += 4.0*fx;
        }
        else {
          s += 2.0*fx;
        }
      }
      //system("pause");
      sum = s*h / 3.0;
      // if (n > 3)
      if (std::abs(sum - osum) < eps*std::abs(osum) || (sum == 0.0 && osum == 0.0)) {
        //std::cout << "n = " << n << ",\tNum Steps = " << it << ",\tSum = " << sum << std::endl;
        return sum;
      }
      osum = sum;
    }
    throw("Maximum number of iterations exceeded in simpsPhi");
  }
  catch (char message) {
    std::cout << "An exception occurred" << message << std::endl;
    return 0.0;
  }
}

EXPORT double PhiIntegrate(const double r0, const double thv, const double kap, const double sig, const double k, const double p, const double ga) {
  RootFuncPhi rfunc(r0, thv, kap, sig, k, p, ga);
  double sum = SimpsPhi(rfunc, 0.0, 2.0*M_PI, 1.0e-7);
  return sum;
}

EXPORT double PhiIntegrateAlt(const double r0, const double thv, const double kap, const double sig, const double k, const double p, const double ga) {
  RootFuncPhi rfunc(r0, thv, kap, sig, k, p, ga);
  double sum = SimpsPhiAlt(rfunc, 0.0, 2.0*M_PI, 1.0e-7);
  return sum;
}

// double f(double x, void *p) {
//   params &ps = *reinterpret_cast<params *>(p);
//   RootFuncPhi rfunc(x, ps);
//   return SimpsPhi(rfunc, 0.0, 2.0*M_PI, 1.0e-7);
// }
//
// double R0Integrate(const double thv, const double kap, const double sig) {
//   gsl_integration_workspace *work_ptr = gsl_integration_workspace_alloc (1000);
//   params PS = { thv, kap, sig, 0.0, 2.2, 1.0 };
//   gsl_function F;
//   F.function = &f;
//   F.params = reinterpret_cast<void *>(&PS);
//   double result, error;
//   const double xlow = 1.0e-9;
//   const double xhigh = 0.1;
//   const double abs_err = 1.0e-5;
//   const double rel_err = 1.0e-5;
//   gsl_integration_qags (&F, xlow, xhigh, abs_err, rel_err, 1000, work_ptr, &result, &error);
//   return result;
// }

// int main(void) {
//   double THV, KAP;
//   std::cout << "THETA_V, and KAPPA: \n" << std::endl;
//   std::cin >> THV >> KAP;
//   // double val = PhiIntegrate(R0, THV*TORAD, KAP, 2.0, 2.0, 2.2, 1.0);
//   // std::cout << val << std::endl;
//   double val = R0Integrate(THV*TORAD, KAP, 2.0);
//   std::cout << val << std::endl;
//   return 0;
// }
