

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import quad, tplquad, nquad


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
        if y == 0.0:
            y += self.TINY

        # eng0 = self.energy_profile(0.0, r0 / y)
        # chi = (y - self.gk*eng0*np.power(y*self.tan_thv + r0, 2.0)) / np.power(y, 5.0 - self.k)
        chi = (y - self.ck*np.power(y*self.tan_thv + r0, 2.0)) / np.power(y, 5.0 - self.k)
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
        chi = self.chi(r0, y)
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
                THETA_V = np.radians(thv*SIG)
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
                THETA_V = np.radians(thv * SIG)
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


if __name__ == "__main__":
    # phi_integrand_plot(scaling="linear")
