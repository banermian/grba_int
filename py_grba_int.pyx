cimport cython
import numpy as np
cimport numpy as np
# from c_grba_int cimport params, GrbaIntegrator, IntG, PhiIntegrate, ScipyCallableTest
from c_grba_int cimport GrbaIntegrator, PhiIntegrate
# cdef double phi_integrate(double r0, void *user_data):
#     return PhiIntegrate(r0, (<double *>user_data)[0], (<double *>user_data)[1], (<double *>user_data)[2], (<double *>user_data)[3], (<double *>user_data)[4], (<double *>user_data)[5])
# def scipy_callable_test(double x):
#     return ScipyCallableTest(x)

#cdef class IG:
#    cdef IntG* c_intg
#
#    def __cinit__(self, double R0, double Y, double THV, double KAP, double SIG, double K, double P, double GA):
#        self.c_intg = new IntG(R0, Y, THV, KAP, SIG, K, P, GA)
#
#    def __dealloc__(self):
#        del self.c_intg
#
#    def int_y(self):
#        return self.c_intg.IntegrandY()
#
#    def int_chi(self):
#        return self.c_intg.IntegrandChi()
#
#    def int_fac(self):
#        return self.c_intg.IntegrandFac()
#
#    def int_phi(self):
#        return self.c_intg.IntegrandPhi()
#
#    def int(self):
#        return self.c_intg.Integrand()
#
#    def get_chi(self):
#        return self.c_intg.chi
#
#    def test_vals(self):
#        print self.c_intg.thv, self.c_intg.kap, self.c_intg.sig, self.c_intg.k, self.c_intg.p, self.c_intg.ga

cdef class GrbaInt:
    cdef GrbaIntegrator* c_grb

    def __cinit__(self, double thv, double kap, double sig, double k, double p, double ga):
        self.c_grb = new GrbaIntegrator(thv, kap, sig, k, p, ga)

    def __dealloc__(self):
        del self.c_grb

    def THV(self):
        return self.c_grb.thv

    def KAP(self):
        return self.c_grb.kap

    def SIG(self):
        return self.c_grb.sig

    def K(self):
        return self.c_grb.k

    def P(self):
        return self.c_grb.p

    def GA(self):
        return self.c_grb.ga

    def theta_prime(self, double phi, double r):
        return self.c_grb.ThetaPrime(phi, r)

    def energy_profile(self, double phi, double r):
        return self.c_grb.EnergyProfile(phi, r)

    def chi(self, double r0, double y):
        return self.c_grb.Chi(r0, y)

    def r0_max(self, double y, double g, double xacc):
        return self.c_grb.R0Max(y, g, xacc)
    
    def r0_int_y(self, double y):
        return self.c_grb.IntegrandY(y)
    
    def r0_int_chi(self, double r0, double y):
        return self.c_grb.IntegrandChi(r0, y)
    
    def r0_int_fac(self, double r0, double y):
        return self.c_grb.IntegrandFac(r0, y)
    
    def r0_int_phi(self, double r0, double y):
        return self.c_grb.IntegrandPhi(r0, y)
    
    def r0_int(self, double r0, double y):
        return self.c_grb.Integrand(r0, y)

    def integrand(self, np.ndarray[np.double_t, ndim=1] vals, double r0, double y):
        return self.c_grb.IntegrandG(<double*> vals.data, r0, y)

    # def r0_max_val(self, r, y):
    #     Gk = self.c_grb.gk
    #     thP0 = self.theta_prime(0.0, r / y)
    #     rExp = -np.power(np.divide(thP0, self.c_grb.sig), 2.0*self.c_grb.kap)
    #     lhs = np.divide(y - np.power(y, 5.0 - self.c_grb.k), Gk)
    #     rhs = (np.tan(self.c_grb.thv) + r)**2.0*np.exp2(rExp)
    #     return rhs - lhs

    def phi_int(self, r0):
        return PhiIntegrate(r0, self.c_grb.thv, self.c_grb.kap, self.c_grb.sig, self.c_grb.k, self.c_grb.p, self.c_grb.ga)

    # def flux(self, double r0, double y):
    #     return self.c_grb.FluxG(r0, y)

    # def root_func_r0(self, r0, y):
    #     thp0 = self.theta_prime(0.0, r0 / y)
    #     exp0 = np.power(thp0 / self.c_grb.sig, 2.0*self.c_grb.kap)
    #     lhs = (r0 / y + self.c_grb.tan_thv)*(r0 / y + self.c_grb.tan_thv)*exp0
    #     rhs = (y - np.power(y, 5.0 - self.c_grb.k)) / self.c_grb.gk
    #
    # def root_jac_r0(self, r0, y):
    #     thp0 = self.theta_prime(0.0, r0 / y)
    #     frac = self.c_grb.kap*np.log(2.0)*np.power(thp0 / self.c_grb.sig, 2.0*self.c_grb.kap)*((r0 + self.c_grb.tan_thv) / (r0 * (1.0 + r0*np.sin(self.c_grb.thv)*np.cos(self.c_grb.thv))))
    #     exponent = 2.0*self.energy_profile(0.0, r0 / y)
    #     return (1.0 - frac)*exponent
