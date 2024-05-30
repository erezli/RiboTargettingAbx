import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.optimize import fsolve
import copy


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
        self.timeseries = None
        self.cell_size = None
        self.birth_size = None
        self.cell_diameter = None

        self.adder_constant = None

        self.report_size = False

    def ribosome_mechanism(self, t, x, pbar, state):
        # progress bar feature edited from https://gist.github.com/thomaslima/d8e795c908f334931354da95acb97e54
        # x = [a, r_u, r_b, p, v]
        # cannot use matrix form because non-linearity

        x = self.division_mechanism(t, x)
        cell_length = x[4] / (np.pi / 4 * self.cell_diameter ** 2)
        cell_surface = np.pi * cell_length * self.cell_diameter + np.pi * self.cell_diameter ** 2 / 2

        reversible_binding = - self.k_on * x[0] * (x[1] - self.ribo_min) + self.k_off * x[2]
        dilution_by_growth = - self.gama * x[3]
        if (self.t_end == -1 and t > self.t_start) or (self.t_end != -1 and self.t_start < t < self.t_end):
            external_abx = self.abx_env
            if self.report_size:
                print(f"Cell size at treatment start: {x[4]}")
                self.report_size = False
        else:
            external_abx = 0
        abx_influx = (self.p_in * external_abx - self.p_out * x[0]) * cell_surface
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

    def division_mechanism(self, t, x):
        if (self.t_end == -1 and t > self.t_start) or (self.t_end != -1 and self.t_start < t < self.t_end):
            self.birth_size = x[4]
        if x[4] - self.birth_size >= self.adder_constant:
            # print(f"divide at {t}")
            x[4] = x[4] / 2
            self.birth_size = x[4]
        return x

    def cell_growth(self, init, length, methods='RK45', t_eval=None, show_progress=True):
        t = (0.0, float(length))
        self.birth_size = init[4]
        self.adder_constant = init[4]
        self.cell_diameter = 0.86 * init[4] ** (1. / 3.)
        self.report_size = True

        if show_progress:
            with tqdm(total=1000, unit="‰") as pbar:
                timeseries = solve_ivp(
                    self.ribosome_mechanism,
                    t,
                    init,
                    method=methods,
                    t_eval=t_eval,
                    args=[pbar, [0.0, (length - 0.0) / 1000]],
                    max_step=0.01
                )
        else:
            timeseries = solve_ivp(
                self.ribosome_mechanism,
                t,
                init,
                method=methods,
                t_eval=t_eval,
                args=[None, [0.0, (length - 0.0) / 1000]],
                max_step=0.01
            )
        # if plot and isinstance(plot, int):
        #     plt.plot(time_series.t, time_series.y[plot+1])
        #     plt.xlabel("time")
        #     plt.ylabel("{}".format(label))
        self.timeseries = timeseries
        return timeseries

    def set_adder_trace(self, adder_constant=1):
        cell_size = copy.deepcopy(self.timeseries.y[4])

        birth_size = cell_size[0]
        for i in range(len(cell_size)):
            if self.t_end >= self.timeseries.t[i] >= self.t_start:
                birth_size = cell_size[i]
                continue

            if cell_size[i] - birth_size >= adder_constant:
                cell_size[i:] = cell_size[i:] / 2
                birth_size = cell_size[i]
        self.cell_size = cell_size
        return cell_size
