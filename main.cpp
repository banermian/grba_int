#include "grba_int.h"
#include "phi_int/phi_int.h"
#include "r0_int/r0_int.h"

void phi_int_test() {
  double R0, R0MIN, Y, THV, KAP;
  std::cout << "Enter Y, THV, and KAP" << std::endl;
  std::cin >> Y >> THV >> KAP;
  std::cout << "Enter R0_MIN" << std::endl;
  std::cin >> R0MIN;
  GrbaIntegrator grb(THV*TORAD, KAP, 2.0, 0.0, 2.2, 1.0);
  double R0MAX = grb.R0Max(Y, 0.25, 1.0e-7);
  std::cout << "R0_MAX = " << R0MAX << std::endl;
  R0 = R0MIN;
  int N = 10;
  double inc = (R0MAX - R0MIN) / N;
  for (int i = 0; i <= N; i++) {
    // std::cout << "R0 = " << R0 << std::endl;
    // std::cout << " phi_int = " << grb.IntegrandPhiAlt(R0, Y) << std::endl;
    double phi_int = grb.IntegrandPhi(R0,Y);
    double phi_int_alt = grb.IntegrandPhiAlt(R0,Y);
    double error = std::abs(phi_int_alt - phi_int) / phi_int * 100.0;
    std::cout << "Absolute difference (%): " << error << "\n";
    R0 += inc - 1.0e-9;
  }
}

void grba_int_test() {
  double R0, Y, THV, KAP;
  std::cout << "Enter Y, THV, and KAP" << std::endl;
  std::cin >> Y >> THV >> KAP;
  std::cout << "Enter R0" << std::endl;
  std::cin >> R0;
  GrbaIntegrator grb(THV*TORAD, KAP, 2.0, 0.0, 2.2, 1.0);
  double int_vals[4];
  grb.IntegrandG(int_vals, R0, Y);
  std::cout << "chi = " << grb.Chi(R0, Y) << std::endl;
  std::cout << "IntY = " << int_vals[0] << std::endl;
  std::cout << "IntChi = " << int_vals[1] << std::endl;
  std::cout << "IntFac = " << int_vals[2] << std::endl;
  std::cout << "IntegrandPhi = " << int_vals[3] << std::endl;
  std::cout << "Integrand = " << int_vals[4] << std::endl;
  std::cout << "IntY = " << grb.IntegrandY(Y) << std::endl;
  std::cout << "IntChi = " << grb.IntegrandChi(R0, Y) << std::endl;
  std::cout << "IntFac = " << grb.IntegrandFac(R0, Y) << std::endl;
  std::cout << "IntegrandPhi = " << grb.IntegrandPhi(R0, Y) << std::endl;
  std::cout << "Integrand = " << grb.Integrand(R0, Y) << std::endl;
}

void root_func_test() {
  double Y, THV, KAP;
  std::cout << "Enter Y, THV, and KAP" << std::endl;
  std::cin >> Y >> THV >> KAP;
  GrbaIntegrator grb(THV*TORAD, KAP, 2.0, 0.0, 2.2, 1.0);
  RootFuncR0 r0func(Y, THV*TORAD, KAP, 2.0, 0.0, 2.2, 1.0);
  double R0MAX = RootR0(r0func, 0.21, 1.0e-7);
  // double R0MAX = grb.R0Max(Y, 0.21, 1.0e-7);
  std::cout << "R0MAX = " << R0MAX << std::endl;
  std::cout << "Chi(R0MAX) = " << grb.Chi(R0MAX, Y) << std::endl;
  std::cout << "RootFuncR0.F(R0MAX) = " << r0func.F(R0MAX) << std::endl;
  std::cout << "RootFuncR0.DF(R0MAX) = " << r0func.DF(R0MAX) << std::endl;
}

int main() {
  phi_int_test();

  // double r0_val = Integrate(0.1, grb);
  // std::cout << "Integral R0 = " << r0_val << std::endl;

  return 0;
}
