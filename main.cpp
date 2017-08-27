#include "grba_int.h"
#include "phi_int/phi_int.h"
#include "r0_int/r0_int.h"

// const double TORAD = M_PI / 180.0;

int main() {
  double R0, Y, THV, KAP;
  std::cout << "Enter Y, THV, and KAP" << std::endl;
  std::cin >> Y >> THV >> KAP;
  std::cout << "Enter R0" << std::endl;
  std::cin >> R0;

  GrbaIntegrator grb(THV*TORAD, KAP, 2.0, 0.0, 2.2, 1.0);
  double int_vals[4];
  grb.IntegrandG(int_vals, R0, Y);
  std::cout << "chi = " << grb.Chi(R0, Y) << std::endl;
  // std::cout << "IntY = " << int_vals[0] << std::endl;
  // std::cout << "IntChi = " << int_vals[1] << std::endl;
  // std::cout << "IntFac = " << int_vals[2] << std::endl;
  // std::cout << "IntegrandPhi = " << int_vals[3] << std::endl;
  // std::cout << "Integrand = " << int_vals[4] << std::endl;
  std::cout << "IntY = " << grb.IntegrandY(Y) << std::endl;
  std::cout << "IntChi = " << grb.IntegrandChi(R0, Y) << std::endl;
  std::cout << "IntFac = " << grb.IntegrandFac(R0, Y) << std::endl;
  std::cout << "IntegrandPhi = " << grb.IntegrandPhi(R0, Y) << std::endl;
  std::cout << "Integrand = " << grb.Integrand(R0, Y) << std::endl;

  RootFuncR0 r0func(Y, THV*TORAD, KAP, 2.0, 0.0, 2.2, 1.0);
  // double R0MAX = RootR0(r0func, 0.21, 1.0e-7);
  double R0MAX = grb.R0Max(Y, 0.21, 1.0e-7);
  std::cout << "R0MAX = " << R0MAX << std::endl;
  std::cout << "Chi(R0MAX) = " << grb.Chi(R0MAX, Y) << std::endl;
  std::cout << "RootFuncR0.F(R0MAX) = " << r0func.F(R0MAX) << std::endl;
  std::cout << "RootFuncR0.DF(R0MAX) = " << r0func.DF(R0MAX) << std::endl;
  grb.IntegrandG(int_vals, R0MAX-R0, Y);
  std::cout << "Integrand(R0MAX) = " << int_vals[4] << std::endl;

  double r0_val = Integrate(0.1, grb);
  std::cout << "Integral R0 = " << r0_val << std::endl;
  return 0;
}
