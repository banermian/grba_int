from __future__ import print_function
from __future__ import division

import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import quad, tplquad, nquad
from scipy.optimize import fsolve, root
from cubature import cubature


class GrbaIntBase(object):
    def __init__(self, thv, kap, sig=2.0, k=0.0, p=2.2, ga=1.0):
        self.TINY = 1.0e-33

        self.thv = np.radians(thv)
        self.kap = kap
        self.sig = sig
        self.k = k
        self.p = p
        self.ga = ga

        self.gk = (4.0 - k)*self.ga*self.ga
        self.ck = (4.0 - self.k)*np.power(5.0 - self.k, (self.k - 5.0)/(4.0 - self.k))
        self.bg = (1.0 - self.p) / 2.0

        self.tan_thv = np.tan(self.thv)
        self.tan_thv_sq = np.power(self.tan_thv, 2.0)
        self.sin_2thv = np.sin(2.0*self.thv)
        self.cos_thv = np.cos(self.thv)
        self.sin_thv = np.sin(self.thv)
        self.chi_exp = (7.0*self.k - 23.0 + self.bg*(13.0 + self.k)) / (6.0*(4.0 - self.k))
        self.y_exp = 0.5*(self.bg*(4.0 - self.k) + 4.0 - 3.0*self.k)


    def theta_prime(self, phi, r):
        cos_phi = np.cos(phi)
        numer = r*np.sqrt(np.power(self.cos_thv, 2) - 0.25*np.power(self.sin_2thv, 2)*np.power(cos_phi, 2))
        denom = 1.0 + 0.5*r*self.sin_2thv*cos_phi
        return numer / denom


    def energy_profile(self, phi, r):
        thp = self.theta_prime(phi, r)
        return np.exp2(-np.power(thp / self.sig, 2.0*self.kap))


    def gammaL(self, phi, r):
        return self.ga*np.sqrt(self.energy_profile(phi, r))


    def chi(self, phi, r, y):
        if type(y) == np.ndarray:
            y[np.equal(0.0, y)] = self.TINY
        elif y == 0.0:
            y = self.TINY

        # if type(r) == np.ndarray:
        #     r[np.equal(0.0, r)] = self.TINY
        # elif r == 0.0:
        #     r = self.TINY

        cos_phi = np.cos(phi)
        chi = (y - self.ck*(r**2 + y**2*self.tan_thv_sq + 2.0*y*self.tan_thv*cos_phi*r)) / np.power(y, 5.0 - self.k)
        # chi = (y - self.ck*np.power(y*self.tan_thv + r0, 2.0)) / np.power(y, 5.0 - self.k)
        return chi


    def x_sq(self, phi, r, y):
        xsq = r**2 + y**2*self.tan_thv_sq + 2.0*y*self.tan_thv*cos_phi*r
        return xsq


    def chi(self, y, x):
        chi = (y - self.ck*x**2) / np.power(y, 5.0 - self.k)
        return chi


    def intG(self, y, chi):
        try:
            ig = np.power(y, self.y_exp)*np.power(chi, self.chi_exp)*np.power((7.0 - 2.0*self.k)*chi*np.power(y, 4.0 - self.k) + 1.0, self.bg - 2.0)

        except FloatingPointError:
            print(y[y<0.0], chi[chi<1.0])
            print(np.power(y, self.y_exp), np.power(chi, self.chi_exp))
            print((7.0 - 2.0*self.k), np.power(y, 4.0 - self.k))
            ig = 0.0

        return ig


class GrbaIntStandard(GrbaIntBase):
    """Class for integration using standard, on-axis methods."""

    def __init__(self, thv=0.0, kap=0.0, sig=2.0, k=0.0, p=2.2, ga=1.0):
        super(GrbaIntStandard, self).__init__(thv, kap, sig, k, p, ga)
        self.TINY = 1.0e-13


    def _y_roots(self, x, g):
        # if type(x) == np.ndarray:

        def func(y):
            return y - np.power(y, (5.0 - self.k)) - self.ck*x**2

        # root = fsolve(func, g)[0]
        root_val = root(func, g)['x']  # [0]
        return root_val


    def _y_range(self, x):
        min_y = self._y_roots(x, 0.1)[0]
        max_y = self._y_roots(x, 0.9)[0]
        return (min_y, max_y)


    def _integrand(self, y, x):
        # chi = (y - self.ck*x**2) / np.power(y, 5.0 - self.k)
        chi = self.chi(0.0, x, y)
        return x*self.intG(y, chi)


    def integrate(self):
        return 2.0*np.pi*nquad(self._integrand, [self._y_range, [0.0, 1.0]])[0]

    # def integrand(self, x_array, *args):
    #     y = np.array(x_array[:, 0])
    #     x = np.array(x_array[:, 1])
    #     y[np.equal(0.0, y)] = self.TINY
    #     ymin = self._y_min(x)
    #     ymax = self._y_max(x)
    #     # y[np.logical_or(y < ymin, y > ymax)] = self.TINY
    #     y[y < ymin] = ymin[y < ymin]
    #     y[y > ymax] = ymax[y > ymax]
    #     chi = (y - self.ck*x**2) / np.power(y, 5.0 - self.k)
    #     print(chi[chi<1])
    #     int = 2.0*np.pi*x*self.intG(y, chi)
    #     # int[chi<1] = 0.0
    #     return int
    #
    #
    # def integrate(self):
    #     ndim = 2
    #     fdim = 1
    #     xmin = np.array([0., 0.], np.float64)
    #     xmax = np.array([1., 1.], np.float64)
    #
    #     val, err = cubature(self.integrand, ndim, fdim, xmin, xmax, vectorized=True)
    #     return (val, err)


class GrbaIntCuba(GrbaIntBase):
    """Class to perform integration using cubature package"""

    def __init__(self, thv, kap, sig=2.0, k=0.0, p=2.2, ga=1.0):
        super(GrbaIntCuba, self).__init__(thv, kap, sig, k, p, ga)
        np.seterr(all='raise')
        # warnings.filterwarnings('error')
        np.set_printoptions(precision=15)

        self.TINY = 1.0e-33


    def _r_prime(self, phi, r0, y):
        cos_phi = np.cos(phi)
        rp = np.sqrt(r0**2 + 2*r0*y*self.tan_thv + y**2*cos_phi**2*self.tan_thv_sq) - y*self.tan_thv*cos_phi

        return rp


    def _r0_max(self, y):
        return np.sqrt((y - np.power(y, 5.0 - self.k)) / self.ck) - y*self.tan_thv


    def _int_r0_y(self, x_array, y):
        phi, r0 = x_array

        if r0 == 0.0:
            r0 += self.TINY

        rp = self._r_prime(phi, r0, y)
        f = np.power(np.divide(rp, r0), 2.0)
        chi = self.chi(r0, y)
        val = np.array([(r0 + y*self.tan_thv)*f*self.intG(y, chi)*np.power(self.energy_profile(phi, rp), 4.0*(1.0 - self.bg))])
        # print(val)

        return val


    def _int_r0_y_v(self, x_array, y):
        phi = np.array(x_array[:, 0], dtype=np.float128)
        r0 = np.array(x_array[:, 1], dtype=np.float128)
        r0[np.equal(0.0, r0)] = self.TINY
        rp = self._r_prime(phi, r0, y)
        try:
            # f = np.power(np.divide(rp, r0), 2.0)
            f = (rp / r0)**2
        except Warning:
            r0_min = np.min(r0)
            print(y, self._r0_max(y), r0_min, phi[r0==r0_min], self._r_prime(phi[r0==r0_min], r0_min, y))
            sys.exit(1)

        chi = self.chi(0.0, r0, y)
        val = (r0 + y*self.tan_thv)*f*self.intG(y, chi)*np.power(self.energy_profile(phi, rp), 4.0*(1.0 - self.bg))
        return val


    def integrate_r0_y(self, y, eps, vectorized=False):
        r0_max = self._r0_max(y)
        # if r0_max < self.TINY:
        if r0_max < eps:
            return (np.array([0.0]), np.array([0.0]))

        ndim = 2
        fdim = 1
        # xmin = np.array([0., self.TINY], np.float64)
        xmin = np.array([0., eps*r0_max], dtype=np.float128)
        xmax = np.array([2.*np.pi, r0_max], dtype=np.float128)
        if vectorized:
            func = self._int_r0_y_v
        else:
            func = self._int_r0_y

        val, err = cubature(func, ndim, fdim, xmin, xmax, args=(y,), vectorized=vectorized)
        return (val, err)


    def _int_r0(self, x_array, *args):
        phi = np.array(x_array[:, 0], dtype=np.float64)
        r0 = np.array(x_array[:, 1], dtype=np.float64)
        y = np.array(x_array[:, 2], dtype=np.float64)
        y[np.equal(0.0, y)] = self.TINY
        r0[np.equal(0.0, r0)] = self.TINY
        r0_max = self._r0_max(y)
        r0[r0 > r0_max] = r0_max[r0 > r0_max] - self.TINY
        rp = self._r_prime(phi, r0, y)
        f = np.power(np.divide(rp, r0), 2.0)
        chi = self.chi(0.0, r0, y)
        chi_check = np.abs(chi - 1.0) < 1.0e-5
        chi[chi_check] = 1.0
        if np.any(chi < 1.0):
            mask = chi < 1.0
            chi[mask] = 1.0
            val = (r0 + y*self.tan_thv)*f*self.intG(y, chi)*np.power(self.gammaL(phi, rp), 4.0*(1.0 - self.bg))
            val[mask] = 0.0
        else:
            val = (r0 + y*self.tan_thv)*f*self.intG(y, chi)*np.power(self.gammaL(phi, rp), 4.0*(1.0 - self.bg))  # np.power(self.energy_profile(phi, rp), 4.0*(1.0 - self.bg))

        val[r0 > r0_max] = 0.0
        return val


    def integrate_r0(self):
        ndim = 3
        fdim = 1
        xmin = np.array([0., 0.0, 0.0], dtype=np.float64)
        xmax = np.array([2.*np.pi, 1.0, 1.0], dtype=np.float64)
        val, err = cubature(self._int_r0, ndim, fdim, xmin, xmax, vectorized=True)
        return (val, err)


class GrbaIntRP(GrbaIntBase):
    """Class for r-prime integration"""

    def __init__(self, thv, kap, sig=1.0, k=0.0, p=2.2, ga=1.0):
        super(GrbaIntRP, self).__init__(thv, kap, sig, k, p, ga)
        self.TINY = 1.0e-19

        np.seterr(all='raise')


    def _y_root_func(self, y, phi, rp):
        cos_phi = np.cos(phi)
        root_func_val = y - np.power(y, 5.0 - self.k) - self.ck*(rp**2 + 2.0*y*rp*self.tan_thv*cos_phi + y**2*self.tan_thv_sq)
        # root_func_jac = 1.0 - (5.0 - self.k)*np.power(y, 4.0 - self.k) - 2.0*self.ck*self.tan_thv*(cos_phi*rp + y*self.tan_thv)
        return root_func_val  # , root_func_jac


    def _y_roots(self, phi, rp, g):
        root_val = root(self._y_root_func, g, args=(phi, rp), jac=False)
        return (root_val.x, root_val['success'])
        # return root_val.x


    def _y_min(self, phi, r):
        return self._y_roots(phi, r, 0.1)[0]


    def _y_max(self, phi, r):
        return self._y_roots(phi, r, 0.9)[0]


    def _r_min(self, phi):
        return 0.0


    def _r_max(self, phi):
        return 1.0


    def _integrand(self, y, r, phi):
            cos_phi = np.cos(phi)
            chi = self.chi(phi, r, y)
            val = (r + y*self.tan_thv*cos_phi)*self.intG(y, chi)  # *np.power(self.gammaL(phi, r), 4.0*(1.0 - self.bg))
            return val


    def _integrand_cuba(self, x_array, *args):
        y = np.array(x_array[:, 0], dtype=np.float64)
        r = np.array(x_array[:, 1], dtype=np.float64)
        phi = np.array(x_array[:, 2], dtype=np.float64)
        ymin = self._y_roots(phi, r, np.repeat(0.1, len(phi)))[0]
        ymax = self._y_roots(phi, r, np.repeat(0.9, len(phi)))[0]
        # print(y, "\n", r, "\n", phi)
        # print(ymin)
        # print(ymax)
        # sys.exit(1)
        cos_phi = np.cos(phi)
        y[np.less(y, ymin)] = ymin[np.less(y, ymin)]
        y[np.greater(y, ymax)] = ymax[np.greater(y, ymax)]
        y[np.equal(0.0, y)] = self.TINY
        chi = self.chi(phi, r, y)
        chi_check = np.abs(chi - 1.0) < 1.0e-5
        chi[chi_check] = 1.0
        # print(phi.min(), phi.max(), r.min(), r.max(), y.min(), y.max(), chi.min(), chi.max())
        if np.any(chi < 1.0):
            mask = chi < 1.0
            chi[mask] = 1.0
            val = (r + y*self.tan_thv*cos_phi)*self.intG(y, chi)  # *np.power(self.gammaL(phi, rp), 4.0*(1.0 - self.bg))
            val[mask] = 0.0
        else:
            val = (r + y*self.tan_thv*cos_phi)*self.intG(y, chi)  # *np.power(self.gammaL(phi, rp), 4.0*(1.0 - self.bg))

        # print(val)
        return val


    def integrate(self):
        # ymin = self._y_roots(phi, r, 0.1)[0]
        # ymax = self._y_roots(phi, r, 0.9)[0]
        # int_val = quad(self._integrand, ymin, ymax, args=(r, phi))
        int_val = tplquad(self._integrand, 0.0, 2.0*np.pi, self._r_min, self._r_max, self._y_min, self._y_max, epsabs=1.0e-5, epsrel=1.0e-5)
        return int_val


    def integrate_cuba(self):
        ndim = 3
        fdim = 1
        xmin = np.array([0.0, self.TINY, 0.0], dtype=np.float64)
        xmax = np.array([1.0, 1.0, 2.0*np.pi], dtype=np.float64)
        val, err = cubature(self._integrand_cuba, ndim, fdim, xmin, xmax, abserr=1e-03, relerr=1e-03, adaptive='h', vectorized=True)
        return (val, err)



class GrbaInt(object):
    """Class to hold parameter values and perform integration."""

    def __init__(self, thv, kap, sig=2.0, k=0.0, p=2.2, ga=1.0):
        self.TINY = 1.0e-13

        self.thv = np.radians(thv)
        self.kap = kap
        self.sig = sig
        self.k = k
        self.p = p
        self.ga = ga

        self.gk = (4.0 - k)*self.ga*self.ga
        self.ck = (4.0 - self.k)*np.power(5.0 - self.k, (self.k - 5.0)/(4.0 - self.k))
        self.bg = (1.0 - self.p) / 2.0

        self.tan_thv = np.tan(self.thv)
        self.tan_thv_sq = np.power(self.tan_thv, 2.0)
        self.sin_2thv = np.sin(2.0*self.thv)
        self.cos_thv = np.cos(self.thv)
        self.sin_thv = np.sin(self.thv)
        self.chi_exp = (7.0*self.k - 23.0 + self.bg*(13.0 + self.k)) / (6.0*(4.0 - self.k))
        self.y_exp = 0.5*(self.bg*(4.0 - self.k) + 4.0 - 3.0*self.k)

    def theta_prime(self, phi, r):
        cos_phi = np.cos(phi)
        numer = r*np.power(np.power(self.cos_thv, 2) - 0.25*np.power(self.sin_2thv, 2)*np.power(cos_phi, 2), 0.5)
        denom = 1.0 + 0.5*r*self.sin_2thv*cos_phi
        return numer / denom

    def energy_profile(self, phi, r):
        thp = self.theta_prime(phi, r)
        return np.exp2(-np.power(thp / self.sig, 2.0*self.kap))

    def chi(self, phi, r, y):
        cos_phi = np.cos(phi)
        if y == 0.0:
            y += self.TINY

        # eng0 = self.energy_profile(0.0, r0 / y)
        # chi = (y - self.gk*eng0*np.power(y*self.tan_thv + r0, 2.0)) / np.power(y, 5.0 - self.k)
        # chi = (y - self.ck*np.power(y*self.tan_thv + r0, 2.0)) / np.power(y, 5.0 - self.k)
        chi = (y - self.ck*(r*r + y*y*self.tan_thv_sq + 2.0*y*self.tan_thv*cos_phi*r)) / np.power(y, 5.0 - self.k)
        return chi

    def _r_prime(self, phi, r0, y):
        cos_phi = np.cos(phi)
        return np.sqrt(r0*r0 + 2*r0*y*self.tan_thv + y*y*cos_phi*cos_phi*self.tan_thv_sq) - y*self.tan_thv*cos_phi

    def _phi_integrand(self, phi, r0, y):
        if r0 == 0.0:
            r0 += self.TINY

        return np.power(self._r_prime(phi, r0, y) / r0, 2.0)

    def integrate_phi(self, r0, y, full_output=False):
        return quad(self._phi_integrand, 0.0, 2.0*np.pi, args=(r0, y,), full_output=full_output)

    def _r0_max(self, y):
        return np.sqrt((y - np.power(y, 5.0 - self.k)) / self.ck) - y*self.tan_thv

    def _integrand(self, phi, r0, y):
        chi = self.chi(phi, r0, y)
        return r0*np.power(y, self.y_exp)*np.power(chi, self.chi_exp)*np.power((7.0 - 2.0*self.k)*chi*np.power(y, 4.0 - self.k) + 1.0, self.bg - 2.0)*self._phi_integrand(phi, r0, y)

    def _phi_lim_lo(self, r0, y):
        return 0.0

    def _phi_lim_hi(self, r0, y):
        return 2.0*np.pi

    def _r0_lim_lo(self, y):
        return self.TINY

    def _r0_lim_hi(self, y):
        r0min = self._r0_lim_lo(y)
        r0max = self._r0_max(y)
        if r0max > r0min:
            return r0max
        else:
            return r0min

    def integrate(self):
        return tplquad(self._integrand, self.TINY, 1.0, self._r0_lim_lo, self._r0_lim_hi, self._phi_lim_lo, self._phi_lim_hi)

    # @staticmethod
    def _chi_lim_lo(self, phi):
        return 1.0

    # @staticmethod
    def _chi_lim_hi(self, phi):
        return np.inf

    # @staticmethod
    def _y_lim_lo(self, chi, phi):
        return 0.0

    def _y_lim_hi(self, chi, phi):
        return np.power(chi, 1.0 / (self.k - 4.0))

    def _integrand_alt(self, y, chi, phi):
        # r_factor = (1.0 / (1.0 - (2.0*self.ck*np.power(y, self.k - 4.0)*np.cos(phi)*self.tan_thv) / np.sqrt(-self.ck*np.power(y, self.k - 9.0)*(y**4.0*chi + y**self.k*(-1 + self.ck*y*np.sin(phi)**2.0*self.tan_thv_sq)))))
        # return r_factor*np.power(y, self.y_exp)*np.power(chi, self.chi_exp)*np.power((7.0 - 2.0*self.k)*chi*np.power(y, 4.0 - self.k) + 1.0, self.bg - 2.0)
        return np.power(y, self.y_exp) * np.power(chi, self.chi_exp) * np.power(
            (7.0 - 2.0 * self.k) * chi * np.power(y, 4.0 - self.k) + 1.0, self.bg - 2.0)

    def integrate_alt(self):
        return tplquad(self._integrand_alt, 0.0, 2.0*np.pi, self._chi_lim_lo, self._chi_lim_hi, self._y_lim_lo, self._y_lim_hi)


def phi_integration_plot(scaling="log"):
    N = 100
    SIG = 2.0
    TINY = 1.0e-3
    df_list = []
    for thv in [0.0, 0.5, 1.0, 3.0]:
        for kap in [0.0, 1.0, 3.0, 10.0]:
            THETA_V = np.radians(thv*SIG)
            KAPPA = kap
            grb = GrbaInt(THETA_V, KAPPA, sig=SIG)
            for Y in [TINY, 0.1, 0.3, 0.5, 0.8, 0.9, 1.0-TINY]:
                R0_MAX = grb._r0_max(Y)
                # CHI_MAX = grb.chi(R0_MAX, Y)
                # CHI_MIN = grb.chi(TINY, Y)
                # print thv*SIG, kap, Y, R0_MAX, CHI_MAX, CHI_MIN
                r0_array = np.flip(np.linspace(R0_MAX, 0.0, num=N, endpoint=False), 0)
                int_array = np.zeros(N)
                if R0_MAX < r0_array[0]:
                    continue

                for i, r0 in enumerate(r0_array):
                    int_array[i] = grb.integrate_phi(r0, Y)[0]

                thv_array = np.repeat(thv*SIG, N)
                kap_array = np.repeat(kap, N)
                y_array = np.repeat(Y, N)
                data = {
                    "thv": thv_array,
                    "kap": kap_array,
                    "y": y_array,
                    "r0": r0_array,
                    "phi_int": int_array
                }
                df_list.append(pd.DataFrame(data=data))

    plot_df = pd.concat(df_list)
    grid = sns.FacetGrid(
        plot_df,
        hue='y', col='kap', row='thv',
        palette='Paired'
    )
    grid.map(plt.plot, 'r0', 'phi_int', lw=1)
    grid.set_titles(r"$\kappa = {col_name}$ | $\theta_V = {row_name}$")
    grid.set_axis_labels(r"$r_0'$", r"$\phi$ Integral")
    handles, labels = grid.fig.get_axes()[0].get_legend_handles_labels()
    lgd = plt.legend(
        handles, labels,
        ncol=7, labelspacing=0.,
        title=r"$y$",
        loc='upper right', bbox_to_anchor=[0.15, -0.2],
        fancybox=True, framealpha=0.5
    )
    for ax in grid.axes.flat:
        ax.set_yscale(scaling)

    # plt.show()
    plt.savefig(
        "../../plots/phi_int_r'-alt_test.pdf",
        dpi=900,
        bbox_extra_artists=(lgd,),
        bbox_inches='tight'
    )


def phi_integrand_plot(scaling="log"):
    N = 100
    SIG = 2.0
    TINY = 1.0e-3
    df_list = []
    for Y in [TINY, 0.1, 0.3, 0.5, 0.8, 0.9, 1.0 - TINY]:
        for thv in [0.0, 0.5, 1.0, 3.0]:
            for kap in [0.0, 1.0, 3.0, 10.0]:
                THETA_V = thv*SIG
                KAPPA = kap
                grb = GrbaInt(THETA_V, KAPPA, sig=SIG)
                R0_MAX = grb._r0_max(Y)
                r0_vals = np.linspace(0.0, R0_MAX, num=7)
                for R0 in r0_vals:
                    phi_array = np.linspace(0.0, 2.0*np.pi, num=N)
                    int_array = grb._phi_integrand(phi_array, R0, Y)
                    thv_array = np.repeat(thv*SIG, N)
                    kap_array = np.repeat(kap, N)
                    r0_array = np.repeat(R0, N)
                    data = {
                        "thv": thv_array,
                        "kap": kap_array,
                        "phi": phi_array / np.pi,
                        "r0": r0_array,
                        "phi_int": int_array
                    }
                    df_list.append(pd.DataFrame(data=data))

        plot_df = pd.concat(df_list)
        grid = sns.FacetGrid(
            plot_df,
            hue='r0', col='kap', row='thv',
            palette='Paired'
        )
        grid.map(plt.plot, 'phi', 'phi_int', lw=1)
        grid.set_titles(r"$\kappa = {col_name}$ | $\theta_V = {row_name}$")
        grid.set_axis_labels(r"$\phi [\pi]$", r"$\phi$ Integrand")
        handles, labels = grid.fig.get_axes()[0].get_legend_handles_labels()
        lgd = plt.legend(
            handles, labels,
            ncol=7, labelspacing=0.,
            title=r"$r_0'$",
            loc='upper right', bbox_to_anchor=[0.15, -0.2],
            fancybox=True, framealpha=0.5
        )
        for ax in grid.axes.flat:
            ax.set_yscale(scaling)

        # plt.show()
        plt.savefig(
            "../../plots/phi_integrand_r'-alt_y={}.pdf".format(str(Y)),
            dpi=900,
            bbox_extra_artists=(lgd,),
            bbox_inches='tight'
        )


def chi_test_plot(scaling="log"):
    N = 100
    SIG = 2.0
    TINY = 1.0e-3
    for Y in [TINY, 0.1, 0.3, 0.5, 0.8, 0.9, 1.0 - TINY]:
        df_list = []
        for thv in [0.0, 0.5, 1.0, 3.0]:
            for kap in [0.0, 1.0, 3.0, 10.0]:
                THETA_V = thv * SIG
                KAPPA = kap
                grb = GrbaInt(THETA_V, KAPPA, sig=SIG)
                R0_MAX = grb._r0_max(Y)
                r0_vals = np.linspace(0.0, R0_MAX, num=7)
                for R0 in r0_vals:
                    r0_array = np.repeat(round(R0, 3), N)
                    phi_array = np.linspace(0.0, 2.0 * np.pi, num=N)
                    r_vals = grb._r_prime(phi_array, R0, Y)
                    chi_array = grb.chi(r0_array, Y)
                    chi_prime_array = (Y - grb.ck*(np.power(Y*grb.tan_thv, 2.0) + 2.0*r_vals*Y*grb.tan_thv*np.cos(phi_array) + r_vals**2.0)) / np.power(Y, 5.0 - grb.k)
                    thv_array = np.repeat(thv * SIG, N)
                    kap_array = np.repeat(kap, N)
                    data = {
                        "thv": thv_array,
                        "kap": kap_array,
                        "phi": phi_array / np.pi,
                        "chi": chi_array,
                        "chi_alt": chi_prime_array,
                        "r0": r0_array
                    }
                    df_list.append(pd.DataFrame(data=data))

        plot_df = pd.concat(df_list)
        grid = sns.FacetGrid(
            plot_df,
            hue='r0', col='kap', row='thv',
            palette='Paired'
        )
        grid.map(plt.plot, 'phi', 'chi', lw=1)
        grid.map(plt.plot, 'phi', 'chi_alt', lw=1, ls='dashed')
        grid.set_titles(r"$\kappa = {col_name}$ | $\theta_V = {row_name}$")
        grid.set_axis_labels(r"$\phi [\pi]$", r"$\chi$")
        handles, labels = grid.fig.get_axes()[0].get_legend_handles_labels()
        lgd = plt.legend(
            handles, labels,
            ncol=7, labelspacing=0.,
            title=r"$r_0'$",
            loc='upper right', bbox_to_anchor=[0.15, -0.2],
            fancybox=True, framealpha=0.5
        )
        for ax in grid.axes.flat:
            ax.set_yscale(scaling)

        # plt.show()
        plt.savefig(
            "../../plots/chi_r0-r'_y={}.pdf".format(str(Y)),
            dpi=900,
            bbox_extra_artists=(lgd,),
            bbox_inches='tight'
        )
        plt.clf()


def cuba_y_test(TINY, scaling="log"):
    N = 200
    SIG = 2.0
    df_list = []
    y_vals = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for thv in [0.0, 0.5, 1.0, 3.0]:
        for kap in [0.0, 1.0, 3.0, 10.0]:
            # THETA_V = np.radians(thv * SIG)
            THETA_V = thv*SIG
            KAPPA = kap
            thv_array = np.repeat(THETA_V, N)
            kap_array = np.repeat(KAPPA, N)
            grb = GrbaIntCuba(THETA_V, KAPPA, sig=SIG)

            # for yval in y_vals:
            #     int_array = np.zeros(N)
            #     y_array = np.repeat(yval, N)
            #     # eps_array = np.logspace(-19, -1, N)
            #     eps_array = np.linspace(TINY, 1, N)
            #     for i, e in enumerate(eps_array):
            #         int_array[i] = grb.integrate_r0_y(yval, e, vectorized=True)[0][0]
            # y_vals = np.linspace(grb.TINY, 1.0-grb.TINY, num=N)
            y_vals = np.linspace(0.0, 1.0, num=N, dtype=np.float64)
            int_array = np.zeros(N)
            for i, y in enumerate(y_vals):
                int_array[i] = grb.integrate_r0_y(y, TINY, vectorized=True)[0][0]

            data = {
                "thv": thv_array,
                "kap": kap_array,
                "y": y_vals,
                "int": int_array
            }
            df_list.append(pd.DataFrame(data=data))

    plot_df = pd.concat(df_list)
    grid = sns.FacetGrid(
        plot_df,
        col='kap', row='thv',
        palette='Paired',
        sharey=False
    )
    grid.map(plt.plot, 'y', 'int', lw=1)
    grid.set_titles(r"$\kappa = {col_name}$ | $\theta_V = {row_name}$")
    grid.set_axis_labels(r"$y$", r"$y-integrand$")
    # handles, labels = grid.fig.get_axes()[0].get_legend_handles_labels()
    # lgd = plt.legend(
    #     handles, labels,
    #     ncol=len(y_vals), labelspacing=0.,
    #     title=r"$y$ ($\epsilon =${})".format(TINY),
    #     loc='upper right', bbox_to_anchor=[0.15, -0.2],
    #     fancybox=True, framealpha=0.5
    # )
    for ax in grid.axes.flat:
        ax.set_yscale(scaling)
        # ax.set_xscale(scaling)

    plt.show()
    # plt.savefig(
    #     "../../plots/y_integrand_eps={}.pdf".format(TINY),
    #     dpi=900,
    #     bbox_inches='tight'
    # )  # bbox_extra_artists=(lgd,),
    # plt.clf()


def rPrime_test(TINY, scaling="log"):
    N = 100
    SIG = 2.0
    df_list = []
    for thv in [0.0, 0.5, 1.0, 3.0]:
    # for thv in [0.0]:
        THETA_V = thv*SIG
        thv_array = np.repeat(THETA_V, N)
        grb = GrbaIntCuba(THETA_V, 0.0, sig=SIG)
        # grb = GrbaIntRP(THETA_V, 0.0, sig=SIG)
        # x_array = np.linspace(0, 1, N)
        # for phi in np.radians(np.linspace(0.0, 360.0, 9)):
        #     phi_array = np.repeat(phi/np.pi, N)
        #     r_array = -y*grb.tan_thv*np.cos(phi) + np.sqrt(x_array**2 - y**2*grb.tan_thv_sq*np.sin(phi))

        # y_vals = np.linspace(grb.TINY, 1.0-grb.TINY, num=N)
        y_vals = np.linspace(0.0, 1.0, num=N, dtype=np.float128)
        int_array = np.zeros(N)
        err_array = np.zeros(N)
        for i, y in enumerate(y_vals):
            int_val = grb.integrate_r0_y(y, TINY, vectorized=True)
            int_array[i] = int_val[0][0]
            err_array[i] = int_val[1][0]

        data = {
            "thv": thv_array,
            "y": y_vals,
            "int": int_array,
            "err": err_array
        }
        df_list.append(pd.DataFrame(data=data))



    plot_df = pd.concat(df_list)
    grid = sns.FacetGrid(
        plot_df,
        col='thv', col_wrap=2,
        sharey=False,
        sharex=True
    )
    grid = grid.map(plt.plot, 'y', 'int', lw=1)  # .add_legend(title=r"$\phi [\pi]$"))
    grid = grid.map(plt.plot, 'y', 'err', lw=1, ls='dashed')
    # grid = grid.map(plt.contour, "phi", "rp", "root")
    grid.set_titles(r"$\theta_V = {col_name}^\circ$")
    grid.set_axis_labels(r"$y$", r"$int$")
    # handles, labels = grid.fig.get_axes()[0].get_legend_handles_labels()
    # lgd = plt.legend(
    #     handles, labels,
    #     ncol=3, labelspacing=0.,
    #     title=r"$\phi [\pi]$",
    #     loc='lower right', bbox_to_anchor=[1.0, 0.0],
    #     fancybox=True, framealpha=0.5
    # )
    for ax in grid.axes.flat:
        ax.set_yscale(scaling)
        # ax.set_xscale(scaling)

    plt.show()
    # plt.savefig(
    #     "../../plots/integrand_eps={}_{}.pdf".format(TINY, scaling),
    #     dpi=900,
    #     bbox_inches='tight'
    # )  # ,bbox_extra_artists=(lgd,)


def y_roots_rp():
    def contourplot(*args, **kwargs):
        chis = kwargs['data'].chi
        data = kwargs.pop("data").pivot(args[1], args[0])[args[2]]
        X, Y = np.meshgrid(data.columns, data.index)
        ax = plt.gca()
        mappable = ax.contourf(X, Y, data,
                               *args[3:],
                               norm=colors.SymLogNorm(linthresh=1.0, vmin=chis.min(), vmax=chis.max()),
                               **kwargs)
        ax.figure.colorbar(mappable)

    N = 100
    SIG = 2.0
    df_list = []
    phi = np.linspace(0, 2, N)
    rp = np.linspace(0, 1, N)
    # rp = np.linspace(-19, 0, N)
    P, R = np.meshgrid(phi, rp)
    for thv in [0.0, 0.5, 1.0, 3.0]:
        print(thv)
        grb = GrbaIntRP(thv*SIG, 1.0)
        root = np.zeros(P.shape)
        chis = np.zeros(P.shape)
        err = np.zeros(P.shape)
        for i in range(len(P)):
            for j in range(len(P[i])):
                rp_val = np.power(10.0, R[i][j])
                root_val = grb._y_roots(P[i][j]*np.pi, rp_val, 0.1)
                root[i][j] = root_val[0][0]
                err[i][j] = int(root_val[1])
                chis[i][j] = grb.chi(P[i][j]*np.pi, rp_val, root_val[0][0])

        df = pd.DataFrame(
            dict(
                phi=P.ravel(),
                rp=R.ravel(),
                root=root.ravel(),
                chi=chis.ravel(),
                thv=np.repeat(thv*SIG, len(P.ravel()))
            )
        )
        df_list.append(df)

    plot_df = pd.concat(df_list)
    g = sns.FacetGrid(plot_df, col='thv', col_wrap=2)
    g.map_dataframe(contourplot, 'phi', 'rp', 'chi')
    g.set_titles(r"$\theta_V = {col_name}^\circ$")
    g.set_axis_labels(r"$\phi [\pi]$", r"$r'$")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)


    plt.show()


def time():
    # from scipy.integrate import quad
    grb = GrbaIntRP(6.0, 0.0)
    phi = np.linspace(0.0, np.pi, 10)
    rp = np.linspace(0.0, 1.0, 10)
    g = np.repeat(0.1, len(phi))
    roots = grb._y_roots(phi, rp, g)
    # val = grb.integrate_r0_y(0.5, vectorized=True)
    # val = quad(func, 0.0, 1.0)
    # print(roots)


def test():
    grb = GrbaIntRP(0.0, 0.0)
    print(grb.integrate())
    int_val, int_err = grb.integrate_cuba()
    print(int_val, int_err)
    phi = np.repeat(0.0, 10)
    rp = np.repeat(1.0e-19, 10)
    # phi = np.linspace(0, np.pi, 10)
    # rp = np.linspace(0, 1, 10)
    # g = np.repeat(0.1, 10)
    # roots = grb._y_roots(phi, rp, g)
    ymin = grb._y_roots(phi, rp, np.repeat(0.1, 10))[0]
    ymax = grb._y_roots(phi, rp, np.repeat(0.9, 10))[0]
    y = np.linspace(0, 1, 10)
    print(ymin)
    print(ymax)
    print(y)
    print(np.logical_or(y < ymin, y > ymax))


if __name__ == "__main__":
    # y_roots_rp()
    test()
    # phi_integrand_plot(scaling="linear")
    # chi_test_plot(scaling="log")
    # for eps in np.logspace(-39, -19, 10, endpoint=False):
    #     cuba_y_test(eps, scaling="log")
        # break

    # cuba_y_test(0.0, scaling="log")
    # rPrime_test(1.0e-19)
    # for y in [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999]:
    #     print(y)
    #     rPrime_test(y, scaling="linear")
    # import timeit
    # print(timeit.repeat('time()', setup="from __main__ import time", number=1000))
