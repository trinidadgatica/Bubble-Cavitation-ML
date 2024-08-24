from scipy.integrate import ode
import numpy as np

from main import Model


class Solver(Model):
    def runner_ode_rp(self, t, tstep):
        tau = t
        dtau = tstep
        Y0 = np.array([1, 0])
        Yint = ode(self.rayleigh_plesset_equation).set_integrator('vode', method='BDF', atol=1e-20)
        Yint.set_initial_value(Y0, tau[0])
        Ysol = np.ndarray(shape=(len(tau), 2), order='F')

        idx_int = 0
        while Yint.successful() and Yint.t < tau[-1]:
            Ysol[idx_int:] = Yint.integrate(Yint.t + dtau)
            idx_int = idx_int + 1
        Y = Ysol.real

        return np.array(Y)

    def runner_ode_km(self, t, tstep):
        tau = t
        dtau = tstep
        Y0 = np.array([1, 0])
        Yint = ode(self.keller_miksis_equation).set_integrator('vode', method='BDF', atol=1e-20)
        Yint.set_initial_value(Y0, tau[0])
        Ysol = np.ndarray(shape=(len(tau), 2), order='F')

        idx_int = 0
        while Yint.successful() and Yint.t < tau[-1]:
            Ysol[idx_int:] = Yint.integrate(Yint.t + dtau)
            idx_int = idx_int + 1
        Y = Ysol.real

        return np.array(Y)

    def runner_ode_g(self, t, tstep):
        tau = t
        dtau = tstep
        Y0 = np.array([1, 0])
        Yint = ode(self.gilmore_equation).set_integrator('vode', method='BDF', atol=1e-20)
        Yint.set_initial_value(Y0, tau[0])
        Ysol = np.ndarray(shape=(len(tau), 2), order='F')

        idx_int = 0
        while Yint.successful() and Yint.t < tau[-1]:
            Ysol[idx_int:] = Yint.integrate(Yint.t + dtau)
            idx_int = idx_int + 1
        Y = Ysol.real

        return np.array(Y)


