

import numpy as np
from scipy.integrate import tplquad

class GrbaInt(object):
    '''Class to hold parameter values and perform integration.'''

    def __init__(self, thv, kap, sig=2.0, k=0.0, p=2.2, ga=1.0):
        self.thv = thv
        self.kap = kap
        self.sig = sig
        self.k = k
        self.p = p
        self.ga = ga

        self.gk = (4.0 - k)*ga*ga
        self.ck = (4.0 - k)*np.power(5.0 - k, (k - 5.0)/(4.0 - k))
        self.bg = (1.0 - p) / 2.0

        self.tan_thv = np.tan(thv)
        self.tan_thv_sq = np.power(self.tan_thv, 2.0)
        self.sin_2thv = np.sin(2.0*thv)
        self.cos_thv = np.cos(thv)
        self.sin_thv = np.sin(thv)
        self.chi_exp = (7.0*k - 23.0 + self.bg*(13.0 + k)) / (6.0*(4.0 - k))
        self.y_exp = 0.5*(self.bg*(4.0 - k) + 4.0 - 3.0*k)

    def theta_prime(self, phi, r):
        cos_phi = np.cos(phi)
        numer = r*np.power(np.power(self.cos_thv, 2) - 0.25*np.power(self.sin_2thv, 2)*np.power(cos_phi, 2), 0.5)
        denom = 1.0 + 0.5*r*self.sin_2thv*cos_phi
        return numer / denom

    def energy_profile(self, phi, r):
        thp = self.theta_prime(phi, r)
        return np.exp2(-np.power(thp / self.sig, 2.0*self.kap))

    def chi(self, r0, y):
        eng0 = self.energy_profile(0.0, r0 / y)
        chi = (y - self.gk*eng0*np.power(y*self.tan_thv + r0, 2.0)) / np.power(y, 5.0 - self.k)
        return chi

    def _r_prime(self, phi, r0, y):
        cos_phi = np.cos(phi)
        return np.sqrt(r0*r0 + 2*r0*self.tan_thv + cos_phi*cos_phi*self.tan_thv_sq) - self.tan_thv*cos_phi

    def _phi_integrand(self, phi, r0, y):
        return np.power(self._r_prime(phi, r0, y) / r0, 2.0)

    def _r0_max(self, y):
        return np.sqrt((y - np.power(y, 5.0 - self.k)) / self.ck) - y*self.tan_thv

    def _integrand(self, phi, r0, y):
        chi = self.chi(r0, y)
        return r0*np.power(y, self.y_exp)*np.power(chi, self.chi_exp)*np.power((7.0 - 2.0*self.k)*chi*np.power(y, 4.0 - self.k) + 1.0, self.bg - 2.0)*self._phi_integrand(phi, r0, y)

    def _phi_lim_lo(self, r0, y):
        return 0.0

    def _phi_lim_hi(self, r0, y):
        return 2.0*np.pi

    def _r0_lim_lo(self, y):
        return 1.0e-9

    def _r0_lim_hi(self, y):
        r0min = self._r0_lim_lo(y)
        r0max = self._r0_max(y)
        if r0max > r0min:
            return r0max
        else:
            return r0min

    # def _y_lims(self):
    #     return [0.0, 1.0]

    def integrate(self):
        return tplquad(self._integrand, 0.0, 1.0, self._r0_lim_lo, self._r0_lim_hi, self._phi_lim_lo, self._phi_lim_hi)
