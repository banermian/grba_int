cdef extern from "grba_int.h":
    cdef struct params:
        const double THV
        const double KAP
        const double SIG
        const double K
        const double P
        const double GA

    cdef cppclass GrbaIntegrator:
        GrbaIntegrator(const double THV, const double KAP, const double SIG, const double K, const double P, const double GA)
        GrbaIntegrator(params& p)
        double ThetaPrime(double phi, double r)
        double EnergyProfile(double phi, double r)
        double Chi(double r0, double y)
        double IntegrandY(double y)
        double IntegrandChi(double r0, double y)
        double IntegrandChi(double chi)
        double IntegrandFac(double r0, double y)
        double IntegrandPhi(double r0, double y)
        double Integrand(double r0, double y)
        int IntegrandG(double *vals, double r0, const double y)
        double R0Max(double y, double g, double xacc)
        double FluxG(double r0, const double y)
        const double thv, kap, sig, k, p, ga, gk, bg, tan_thv

cdef extern from "r0_int/r0_int.h":
    cdef cppclass RootFuncR0:
        RootFuncR0(double Y, double THV, double KAP, double SIG, double K, double P, double GA)
        RootFuncR0(double Y, params& p)
        double F(double r0)
        double DF(double r0)

cdef extern from "phi_int/phi_int.h":
    cdef cppclass RootFuncPhi:
        RootFuncPhi(double R0, double THV, double KAP, double SIG, double K, double P, double GA)
        RootFuncPhi(double R0, params& p)
        double F(double r)
        double DF(double r)
        void SetPhi(double phi_val)
        double r0

    # int fcn(void *p, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag)
    # double RootPhi(RootFuncPhi& func, double g, double xacc)
    # double SimpsPhi(RootFuncPhi& func, double a, double b, double eps = 1.0e-9)
    double PhiIntegrate(double R0, double THV, double KAP, double SIG, double K, double P, double GA)
