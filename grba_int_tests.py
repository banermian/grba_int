import sys
import six
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve, root, brentq
from scipy.integrate import quad, romberg
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from pprint import pprint
from py_grba_int import GrbaInt

# THV = 0.0
# KAP = 0.0
SIG = 2.0
K = 0.0
P = 2.2
GA = 1.0
R0MIN = 1.0e-9

def chi_func(thv, kap, r0=0.0, y=0.0):
    grb = GrbaInt(thv, kap, SIG, K, P, GA)
    c = np.vectorize(grb.chi)
    return c(r0, y)

def plot_chi_test(scaling='log'):
    pdf_list = []
    r0_vals = np.linspace(-0.75, 0.75)
    for y_val in [0.001, 0.1, 0.3, 0.5, 0.8, 0.9, 0.999]:
        pdf = plot_grid(chi_func, 'Chi', r0=r0_vals, y=y_val)
        # print pdf.describe()
        # sys.exit(0)
        pdf_list.append(pdf)

    plot_df = pd.concat(pdf_list, ignore_index=True)
    grid = sns.FacetGrid(
        plot_df,
        hue='y', col='KAP', row='THV',
        palette='Paired'
    )
    grid = grid.map(plt.plot, 'r0', 'Chi', ls='solid', lw=1.5)
    # grid = grid.map(plt.plot, 'r0', 'Chi_jac', ls='dashed', lw=1.5)
    for ax in grid.axes.flat:
        ax.set_yscale(scaling)
        # ax.set_ylim(0.0, 2.0)
        ax.set_xlim(min(r0_vals), max(r0_vals))
        ax.axhline(y=1.0, ls='dashed', c='black', lw=1)

    grid.set_titles(r"$\kappa = {col_name}$ | $\theta_V = {row_name}$")
    grid.set_axis_labels(r"$r_0'$", r"$\chi$")
    handles, labels = grid.fig.get_axes()[0].get_legend_handles_labels()
    lgd = plt.legend(
        handles, labels,
        ncol=7, labelspacing=0.,
        title=r"$y$",
        loc = 'upper right', bbox_to_anchor=[0.15, -0.2],
        fancybox=True, framealpha=0.5
    )
    plt.savefig(
        "./plots/chi-r0'_ext_{}.pdf".format(scaling),
        dpi=900,
        bbox_extra_artists=(lgd,),
        bbox_inches='tight'
        )

def r0_int_func(thv, kap, y=0.0):
    grb = GrbaInt(thv, kap, SIG, K, P, GA)
    R0MAX = grb.r0_max(y, 0.21, 1.0e-7)
    CHIMAX = grb.chi(R0MAX, y)
    if (R0MAX <= 0.0) or (CHIMAX >= 10.0) or (CHIMAX <= 0.0):
        return (np.array([0.0]), np.array([0.0]))
    
    r0 = np.linspace(1.0e-9, R0MAX)
    # f = np.vectorize(grb.r0_int)
    f = np.vectorize(grb.r0_int_phi)
    return (r0, f(r0, y))

def plot_r0Int_test(scaling='log'):
    pdf_list = []
    for y_val in [0.001, 0.1, 0.3, 0.5, 0.8, 0.9, 0.999]:
        print y_val
        pdf = plot_grid(r0_int_func, 'Integrand', y=y_val)
        # print pdf.describe()
        # sys.exit(0)
        pdf_list.append(pdf)

    plot_df = pd.concat(pdf_list, ignore_index=True)
    grid = sns.FacetGrid(
        plot_df,
        hue='y', col='KAP', row='THV',
        palette='Paired'
    )
    grid = grid.map(plt.plot, 'r0', 'r0_int', ls='solid', lw=1.5)
    # grid = grid.map(plt.plot, 'r0', 'Chi_jac', ls='dashed', lw=1.5)
    for ax in grid.axes.flat:
        ax.set_yscale(scaling)
        # ax.set_ylim(0.0, 2.0)
        # ax.set_xlim(min(r0_vals), max(r0_vals))
        # ax.axhline(y=1.0, ls='dashed', c='black', lw=1)

    grid.set_titles(r"$\kappa = {col_name}$ | $\theta_V = {row_name}$")
    grid.set_axis_labels(r"$r_0'$", r"$r_0'$ Integrand")
    handles, labels = grid.fig.get_axes()[0].get_legend_handles_labels()
    lgd = plt.legend(
        handles, labels,
        ncol=7, labelspacing=0.,
        title=r"$y$",
        loc = 'upper right', bbox_to_anchor=[0.15, -0.2],
        fancybox=True, framealpha=0.5
    )
    plt.savefig(
        "./plots/r0'_integrand_test_{}.pdf".format(scaling),
        dpi=900,
        bbox_extra_artists=(lgd,),
        bbox_inches='tight'
        )

def plot_grid(f, f_name='', jac=False, **kwargs):
    df_list = []
    for thv in [0.0, 0.5, 1.0, 3.0]:
        for kap in [0.0, 1.0, 3.0, 10.0]:
            print thv*SIG, kap
            data = {}
            THETA_V = np.radians(thv*SIG)
            KAP = kap
            r0_array, func_array = f(THETA_V, KAP, **kwargs)
            kap_array = np.repeat(KAP, len(func_array))
            thv_array = np.repeat(thv*SIG, len(func_array))
            data["THV"] = thv_array
            data["KAP"] = kap_array
            data[f_name] = func_array
            data['r0'] = r0_array

            if jac:
                jac_array = np.gradient(func_array)
                data["{}_jac".format(f_name)] = jac_array

            for k, v in six.iteritems(kwargs):
                if isinstance(v, np.ndarray):
                    data[k] = np.array(v)
                else:
                    data[k] = np.repeat(v, len(func_array))

            loc_df = pd.DataFrame(data=data)
            df_list.append(loc_df)

    df = pd.concat(df_list, ignore_index=True)
    # plot_df = pd.melt(df,
    #                   id_vars=["THETA_V", "KAP"],
    #                   value_vars=[k for k,v in six.iteritems(kwargs)]
    #                   )
    # col_names = list(df.columns.values)
    # col_names.remove(f_name)
    # plot_df = pd.melt(df, id_vars = col_names, value_vars=[f_name])
    return df  # , plot_df

def phi_int_test():
    HEADER_STRING =  "{}|{}|{}|{}|{}".format(
        str.center('THETA_V', 11),
        str.center('KAPPA', 11),
        str.center('Y', 11),
        str.center('R0', 11),
        str.center('PHI INT', 11)
        )
    print HEADER_STRING
    print "-"*len(HEADER_STRING)
    for thv in [0.0, 2.0, 6.0]:
        for kap in [0.0, 1.0, 3.0, 10.0]:
            for Y_VAL in [0.001, 0.1, 0.5, 0.9, 0.999]:
                THETA_V = np.radians(thv)
                KAP = kap
                grb = GrbaInt(THETA_V, KAP, SIG, K, P, GA)
                R0MAX = grb.r0_max(Y_VAL,  0.21, 1.0e-7)
                CHIMAX = grb.chi(R0MAX, Y_VAL)
                if (R0MAX > 0.0) and (CHIMAX > 0.0) and (CHIMAX < 10.0):
                    for R0 in np.linspace(R0MIN, R0MAX-R0MIN):
                        PHI_INT = grb.r0_int_phi(R0, Y_VAL)
                        print "{}|{}|{}|{}|{}".format(
                            str.center('{:05.2f}'.format(thv), 11),
                            str.center('{:05.2f}'.format(kap), 11),
                            str.center('{:05.3f}'.format(Y_VAL), 11),
                            str.center('{:05.3e}'.format(R0), 11),
                            str.center('{:05.3e}'.format(PHI_INT), 11)
                            )
                else:
                    print "{}|{}|{}|{}|{}".format(
                            str.center('{:05.2f}'.format(thv), 11),
                            str.center('{:05.2f}'.format(kap), 11),
                            str.center('{:05.3f}'.format(Y_VAL), 11),
                            str.center('{:05.3e}'.format(-1.0), 11),
                            str.center('{:05.3e}'.format(0.0), 11)
                            )

def phi_root_test(y, thv, kap):
    THETA_V = np.radians(thv)
    KAPPA = kap
    grb = GrbaInt(THETA_V, KAPPA, SIG, K, P, GA)
    R0_MAX = grb.r0_max(y,  0.21, 1.0e-7)
    

def test():
    HEADER_STRING =  "{}|{}|{}|{}|{}|{}|{}".format(
        str.center('THETA_V', 11),
        str.center('KAPPA', 11),
        str.center('Y', 11),
        str.center('CHI', 11),
        str.center('PHI INT', 11),
        str.center('R0 MAX', 11),
        str.center('R0 INT', 11)
        )
    print HEADER_STRING
    print "-"*len(HEADER_STRING)
    for thv in [0.0, 2.0, 6.0]:
        for kap in [0.0, 1.0, 3.0, 10.0]:
            for Y_VAL in [0.001, 0.1, 0.5, 0.9, 0.999]:
                vals = np.zeros(5)
                THETA_V = np.radians(thv)
                KAP = kap
                grb = GrbaInt(THETA_V, KAP, SIG, K, P, GA)
                R0MAX = grb.r0_max(Y_VAL, 0.21, 1.0e-7)
                # R0MAX = r0_max(Y_VAL, grb)
                # grb.integrand(vals, R0MAX-1.0e-3, Y_VAL)
                CHI_VAL = grb.chi(R0MAX, Y_VAL)
                PHI_INT = grb.r0_int_phi(R0MAX, Y_VAL)
                R0_INT = grb.r0_int(R0MAX, Y_VAL)
                # PHI_INT = vals[3]
                # R0_INT = vals[4]
                # def func(r):
                #     vals = np.zeros(5)
                #     grb.integrand(vals, r, Y_VAL)
                #     return vals[4]

                # if R0MIN < R0MAX:
                #     R0_INT = quad(func, R0MIN, R0MAX, epsabs=1.0e-5, epsrel=1.0e-5, full_output=0, points=[R0MIN, R0MAX])[0]
                    # R0_INT = romberg(np.vectorize(func), R0MIN, R0MAX, tol=1.0e-5, rtol=1.0e-5, show=True, vec_func=True)
                # else:
                #     R0_INT = 0.0

                print "{}|{}|{}|{}|{}|{}|{}".format(
                    str.center('{:05.2f}'.format(thv), 11),
                    str.center('{:05.2f}'.format(kap), 11),
                    str.center('{:05.3f}'.format(Y_VAL), 11),
                    str.center('{:05.3e}'.format(CHI_VAL), 11),
                    str.center('{:05.3e}'.format(PHI_INT), 11),
                    str.center('{:05.3e}'.format(R0MAX), 11),
                    str.center('{:05.3e}'.format(R0_INT), 11)
                    )

                    # l = []
                    # for r0 in np.linspace(R0MIN, R0_MAX, 10):
                    #     l.append((r0, func(r0)))
                    #
                    # pprint(l)

def r0_integral(y_val, grb):
    # R0MAX = r0_max(y_val, grb)
    R0MAX = grb.r0_max(y_val, 0.21, 1.0e-7)
    CHI_MAX = grb.chi(R0MAX, y_val)
    if (CHI_MAX <= 0.1) or (CHI_MAX >= 10.0):
        return 0.0

    def func(r):
        vals = np.zeros(5)
        grb.integrand(vals, r, y_val)
        return vals[4]

    # print R0MAX
    if (R0MAX > R0MIN) and (np.abs(grb.chi(R0MAX, y_val) - 1.0) <= 0.1):

        # func = np.vectorize(lambda r: g.flux(r, y_val))
        # func = lambda r: g.flux(r, y_val)
        return quad(func, R0MIN, R0MAX, epsabs=1.0e-5, epsrel=1.0e-5, full_output=0, points=[R0MIN, R0MAX])[0]
        # return romberg(func, R0MIN, R0MAX, tol=1.0e-5, rtol=1.0e-5, show=True, vec_func=False)
    else:
        return 0.0

def r0_max(y, grb):
    # Gk = (4.0 - grb.K())*grb.GA()*grb.GA()
    # def rootR0(rm):
    #     thP0 = grb.theta_prime(0.0, rm / y)
    #     # eng0 = grb.energy_profile(0.0, rm / y)
    #     rExp = -np.power(np.divide(thP0, grb.SIG()), 2.0*grb.KAP())
    #     lhs = np.divide(y - np.power(y, 5.0 - grb.K()), Gk)
    #     rhs = np.exp2(rExp)*(y*np.tan(grb.THV()) + rm)**2.0
    #     return rhs - lhs
    def rootR0(rm):
        return grb.root_func_r0(rm, y)

    rootValR0_f = fsolve(rootR0, 0.1)[0]
    return rootValR0_f

def plot_r0Int(Y):
    PLOT_MAX = []
    df_list = []
    vals_array = np.zeros(5)
    for KAP in (0.0, 1.0, 3.0, 10.0):
        for THETA_V in (0.0, 2.0, 6.0):
            THV = np.radians(THETA_V)
            grb = GrbaInt(THV, KAP, SIG, K, P, GA)
            R0MAX = r0_max(Y, grb)
            print KAP, THV, R0MAX
            data = []
            PLOT_MAX.append(R0MAX)

            if R0MAX <= 0.0:
                data = data.append(np.append(np.zeros(5) , np.array([0.0, THETA_V, KAP])))
            else:
                for R0 in np.linspace(R0MIN, R0MAX, 100)[1:-1]:
                # for R0 in np.flipud(np.linspace(R0MAX-1.0e-9, 0.0, 100, endpoint=False)):
                # for R0 in np.logspace(np.log10(R0MIN), np.log10(R0MAX), 100):
                    grb.integrand(vals_array, R0, Y)
                    vals = np.append(np.log10(vals_array), np.array([R0, THETA_V, KAP]))
                    # print vals
                    data.append(vals)
                    # root_val = grb.root_func_r0(R0, Y)
                    # jac_val = grb.root_jac_r0(R0, Y)
                    # print Y, R0, vals_array, root_val, jac_val

            loc_df = pd.DataFrame(data=data,
                                  columns=["y_part", "chi_part", "factor", "phi_int", "r0_int", "R0", "THV", "KAP"])
            df_list.append(loc_df)
    df = pd.concat(df_list, ignore_index=True)
    plot_df = pd.melt(df,
                      id_vars=["R0", "THV", "KAP"],
                      value_vars=["y_part", "chi_part", "factor", "phi_int", "r0_int"]
                      )
    grid = sns.lmplot(x='R0', y='value', data=plot_df,
                      hue='variable', col='KAP', row='THV', markers='.',
                      fit_reg=False,
                      scatter_kws={'s': 13}, line_kws={'ls':'solid', 'lw':1},
                      sharex=False
                      )  # fit_reg=True, logx=True,
    for i, ax in enumerate(grid.axes.flat):
        ax.set_xlim(0, None)  # PLOT_MAX[i]

    # print grid.axes.flat

    # plt.xlim(0.0, PLOT_MAX)
    plt.show()
    # plt.savefig('r0_integrand_y={}.pdf'.format(Y), dpi=300, orientation='landscape', bbox_inches='tight')

# grid = sns.FacetGrid(plot_df,
#                      hue='variable',
#                      col='KAP', row='THV')  # hue_kws={"marker": ['o', '^', '8', 's', 'p', '+', 'D']},
# def groupplot(y, **kwargs):
#     m = kwargs.pop('marker')
#     ax = plt.gca()
#     data = kwargs.pop("data")
#     plot_data = data.loc[data.Y == float(y), ]
#     # plot_data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
#     plt.plot(plot_data.R0, plot_data.value, marker=m, markersize=3, linewidth=1)
#
# markers = iter(['o', '^', '8', 's', 'p', '+', 'D'])
# for name, group in plot_df.groupby('Y'):
#     grid.map_dataframe(groupplot, name, marker=next(markers))
    # grid.map(groupplot, 'R0', 'value', yval=name, marker=next(markers))

# plt.legend()
# plt.show()

def main():
    TINY = 1.0e-33
    data = []
    for y_val in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    # for y_val in [0.01]:
        print "Generating plot for y = %s" % y_val
        plot_r0Int(y_val)
    #     data = []
    #     for KAP in (0.0, 1.0, 3.0, 10.0):
    #         for THETA_V in (0.0, 2.0, 6.0):
    #             THV = np.radians(THETA_V)
    #             grb = GrbaInt(THV, KAP, SIG, K, P, GA)
    #             R0MAX = r0_max(y_val, KAP, SIG, THV, GA, K, P)
    #             val = scipy.quad()
    #             print KAP, THETA_V, y_val, R0MAX
    #             if R0MAX > 0.0:
    #                 for r0 in np.logspace(-9, np.log10(R0MAX)):
    #                     data.append(np.array([np.log10(grb.phi_int(r0)), r0, y_val, THETA_V, KAP]))
    #                     # data.append(np.array([grb.phi_int(r0/y_val), r0, 'r0/y', THETA_V, KAP]))
    # df = pd.DataFrame(data=data,
    #                   columns=["phi_int", "R0", "Y", "THV", "KAP"]
    #                   )
    # grid = sns.lmplot(x='R0', y='phi_int', data=df,
    #                   hue='Y', col='KAP', row='THV', markers='.',
    #                   fit_reg=False, sharex=False,
    #                   scatter_kws={'s': 13}, line_kws={'ls':'solid', 'lw':1}
    #                   )
    # for i, ax in enumerate(grid.axes.flat):
    #     ax.set_xlim(0, None)
    #
    # print y_val
    # plt.show()

if __name__ == '__main__':
    print "running the module"
    phi_int_test()
    # print r0_integral(0.5)
    # main()
    # plot_chi_test()
    # plot_r0Int_test()