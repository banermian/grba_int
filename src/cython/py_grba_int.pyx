cimport cython
import numpy as np
cimport numpy as np
from c_grba_int cimport GrbaIntegrator, PhiIntegrate, RootFuncPhi

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
        try:
            int_val = self.c_grb.IntegrandPhi(r0, y)
        except RuntimeError:
            int_val = 0.0

        return int_val

    def r0_int_phi_alt(self, double r0, double y):
        return self.c_grb.IntegrandPhiAlt(r0, y)

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

cdef class RootPhi:
    cdef RootFuncPhi* c_rf

    def __cinit__(self, double r0, double thv, double kap, double sig, double k, double p, double ga):
        self.c_rf = new RootFuncPhi(r0, thv, kap, sig, k, p, ga)

    def __dealloc__(self):
        del self.c_rf

    def root_func(self, double r):
        return self.c_rf.F(r)

    def root_jac(self, double r):
        return self.c_rf.DF(r)

    def set_phi(self, double phi):
        self.c_rf.SetPhi(phi)
