import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.optimize import fsolve


class Cell:
    def __init__(self, ribo_min, p_in, p_out, k_on, k_off, alpha, beta, gama, abx_env, t_start=0, t_end=-1):
        self.ribo_min = ribo_min
        self.p_in = p_in
        self.p_out = p_out
        self.k_on = k_on
        self.k_off = k_off
        self.alpha = alpha  # ribosome synthesis
        self.beta = beta  # translation
        self.gama = gama  # cell wall synthesis
        self.abx_env = abx_env
        self.t_start = t_start
        self.t_end = t_end

        self.V = np.array([])
        self.R_u = np.array([])
        self.R_b = np.array([])
        self.p = np.array([])

    def ribosome_mechanism(self, t, x, pbar, state):
        # progress bar feature edited from https://gist.github.com/thomaslima/d8e795c908f334931354da95acb97e54
        # x = [a, r_u, r_b, p, v]
        # cannot use matrix form because non-linearity

        reversible_binding = - self.k_on * x[0] * (x[1] - self.ribo_min) + self.k_off * x[2]
        dilution_by_growth = - self.gama * x[3]
        if (self.t_end == -1 and t > self.t_start) or (self.t_end != -1 and self.t_start < t < self.t_end):
            external_abx = self.abx_env
        else:
            external_abx = 0
        abx_influx = self.p_in * external_abx - self.p_out * x[0]
        ribosome_syn = self.alpha * (x[1] - self.ribo_min)
        cell_wall_syn = self.beta * (x[1] - self.ribo_min) - self.gama * x[3]
        a_dot = dilution_by_growth * x[0] + reversible_binding + abx_influx
        r_u_dot = dilution_by_growth * x[1] + reversible_binding + ribosome_syn
        r_b_dot = dilution_by_growth * x[2] - reversible_binding
        p_dot = dilution_by_growth * x[3] + cell_wall_syn
        v_dot = - dilution_by_growth * x[4]

        if pbar:
            last_t, dt = state
            n = int((t - last_t) / dt)
            pbar.update(n)

            # we need this to take into account that n is a rounded number.
            state[0] = last_t + dt * n
        return np.array([a_dot, r_u_dot, r_b_dot, p_dot, v_dot])

    def cell_growth(self, init, length, methods='RK45', t_eval=None, show_progress=True):
        t = (0.0, float(length))
        if show_progress:
            with tqdm(total=1000, unit="‰") as pbar:
                timeseries = solve_ivp(
                    self.ribosome_mechanism,
                    t,
                    init,
                    method=methods,
                    t_eval=t_eval,
                    args=[pbar, [0.0, (length - 0.0) / 1000]],
                    first_step=0.01
                )
        else:
            timeseries = solve_ivp(
                self.ribosome_mechanism,
                t,
                init,
                method=methods,
                t_eval=t_eval,
                args=[None, [0.0, (length - 0.0) / 1000]],
            )
        # if plot and isinstance(plot, int):
        #     plt.plot(time_series.t, time_series.y[plot+1])
        #     plt.xlabel("time")
        #     plt.ylabel("{}".format(label))
        return timeseries
