from main import Model
from classifiers.ode_runner import Solver


from scipy.integrate import odeint, quad
from scipy.signal import find_peaks
from scipy.optimize import root
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class ExperimentMaker(Model):
    def __int__(self, model_instance):
        super().__init__(model_instance.pa, model_instance.f, model_instance.r0,
                         model_instance.j0, model_instance.p0, model_instance.sigma,
                         model_instance.rho, model_instance.mu, model_instance.c,
                         model_instance.pv, model_instance.kappa)

    def RP_functions(self, time, solver, step):
        """
        Function that calculates inertial and pressure functions for the Rayleigh-Plesset equation.

        :param time: Variable of time.
        :type time: numpy.ndarray

        :return: A tuple containing the inertial function (numpy array), pressure function (numpy array),
        array of radius, and array of velocity.

        References:
        - H.G. Flynn (1975). Cavitation dynamics I.
        """
        if solver == "ODEINT":
            result_rayleigh_plesset = odeint(self.rayleigh_plesset_equation, [1, 0], time, tfirst=True,
                                             atol=1e-16)
        elif solver == "ODE":
            result_rayleigh_plesset = Solver.runner_ode_rp(self, time, step)

        else:
            raise ValueError("Invalid equation")

        radius = result_rayleigh_plesset[:, 0]
        velocity = result_rayleigh_plesset[:, 1]

        reynolds_number_fl = (self.initial_radius ** 2 * self.density) / (4 * self.viscosity * self.period)
        weber_number_fl = (2 * self.surface_tension * self.period ** 2) / (self.density *
                                                                           (
                                                                                   self.initial_radius ** 3))
        non_dimensional_pressure_fl = self.acoustic_pressure
        thoma_number_fl = (self.period ** 2) * non_dimensional_pressure_fl / (self.density *
                                                                              (
                                                                                      self.initial_radius ** 2))
        pressure_in_infinity_fl = self.atmospheric_pressure + \
                                  self.acoustic_pressure * np.sin(
            self.angular_frequency * time * self.period)
        non_dimensional_initial_pressure_fl = ((self.period ** 2) / (self.initial_radius ** 2)) * \
                                              (self.atmospheric_pressure + (
                                                      2 * self.surface_tension / self.initial_radius) -
                                               self.vapor_pressure) / self.density

        external_pressure_wall = - thoma_number_fl * (
                pressure_in_infinity_fl + self.vapor_pressure) / self.acoustic_pressure
        internal_pressure = non_dimensional_initial_pressure_fl / (radius ** (3 * self.adiabatic_index))
        surface_tension_effect = - weber_number_fl / radius
        viscosity_effect = - (1 / reynolds_number_fl) * (velocity / radius)

        pressure_contribution = external_pressure_wall + internal_pressure + surface_tension_effect + viscosity_effect

        IF = - (3 * (velocity ** 2) / (radius * 2))
        PF = pressure_contribution / radius

        return np.array(IF), np.array(PF), radius, velocity

    def KM_functions(self, time, solver, step=0.001):
        """
        Function that calculates inertial and pressure functions for the Keller-Miksis equation.

        :param step: Variable of step in time.
        :type step: float

        :param time: Variable of time.
        :type time: numpy.ndarray

        :return: A tuple containing the inertial function (numpy array), pressure function (numpy array),
        array of radius, and array of velocity.

        References:
        - H.G. Flynn (1975). Cavitation dynamics I.
        """

        if solver == "ODEINT":
            result_keller_miksis = odeint(self.keller_miksis_equation, [1, 0], time, tfirst=True, atol=1e-16)
        elif solver == "ODE":
            result_keller_miksis = Solver.runner_ode_km(self, time, step)
        else:
            raise ValueError("Invalid equation")

        radius = result_keller_miksis[:, 0]
        velocity = result_keller_miksis[:, 1]
        acceleration = np.diff(result_keller_miksis[:, 1]) / step
        acceleration = np.insert(acceleration, 0, 0)

        p_in = (self.atmospheric_pressure - self.vapor_pressure + 2 * (
                self.surface_tension / self.initial_radius)) * \
               (1 / radius) ** (3 * self.adiabatic_index)
        p_out = - self.atmospheric_pressure - 2 * (
                self.surface_tension / (
                radius * self.initial_radius)) - 4 * self.viscosity * (
                    velocity) / (
                        radius * self.period) - self.acoustic_pressure * np.sin(
            self.angular_frequency
            * time * self.period) + self.vapor_pressure
        pressure = p_in + p_out

        eq_1 = (self.period ** 2) / (self.initial_radius ** 2)
        eq_2 = (1 + (velocity * self.initial_radius / (
                self.period * self.sound_velocity))) * (
                       pressure / self.density)
        eq_3 = radius * self.initial_radius / (self.density * self.sound_velocity)

        eq_4 = ((self.atmospheric_pressure - self.vapor_pressure +
                 2 * self.surface_tension / self.initial_radius) *
                (- 3 * self.adiabatic_index * (radius ** (- 3 * self.adiabatic_index - 1))
                 * velocity / self.period))

        eq_5 = 2 * self.surface_tension * velocity / (
                (radius ** 2) * self.initial_radius * self.period)
        eq_6 = - 4 * self.viscosity * (acceleration / (radius * (self.period ** 2)))
        eq_7 = 4 * self.viscosity * (velocity ** 2) / ((radius ** 2) * (self.period ** 2))
        eq_8 = - self.acoustic_pressure * np.cos(
            2 * np.pi * time * self.period * self.frequency) * 2 * np.pi
        eq_9 = (3 / 2) * (velocity ** 2) * (1 - ((velocity * self.initial_radius) / (
                3 * self.sound_velocity * self.period)))

        pressure_contribution = eq_1 * (eq_2 + eq_3 * (eq_4 + eq_5 + eq_6 + eq_7 + eq_8))

        IF = - eq_9 / ((1 - ((velocity * self.initial_radius) / (self.sound_velocity * self.period))) * radius)
        PF = pressure_contribution / (
                (1 - ((velocity * self.initial_radius) / (self.sound_velocity * self.period))) * radius)

        return np.array(IF), np.array(PF), radius, velocity

    def G_functions(self, time, solver, step=0.001):
        """
        Function that calculates inertial and pressure functions for the Gilmore equation.

        :param solver:
        :param time: Variable of time.
        :type time: numpy.ndarray

        :param step: Variable of step in time.
        :type step: float

        :return: A tuple containing the inertial function (numpy array), pressure function (numpy array),
        array of radius, and array of velocity.

        References:
        - H.G. Flynn (1975). Cavitation dynamics I.
        """

        n = 7
        A = 304e6
        B = 303.9e6

        if solver == "ODEINT":
            result_gilmore = odeint(self.gilmore_equation, [1, 0], time, tfirst=True, atol=1e-16)
        elif solver == "ODE":
            result_gilmore = Solver.runner_ode_g(self, time, step)
        else:
            raise ValueError("Invalid equation")

        radius = result_gilmore[:, 0]
        velocity = result_gilmore[:, 1]
        acceleration = np.diff(result_gilmore[:, 1]) / (step * 2 * np.pi * self.frequency)
        acceleration = np.insert(acceleration, 0, 0)

        non_dimensional_enthalpy = self.enthalpy(radius, velocity, time, n, A, B)
        non_dimensional_delta_enthalpy = self.delta_enthalpy(radius, velocity, time, n, A, B)
        non_dimensional_sound_velocity = (self.sound_velocity *
                                          (self.initial_radius * self.angular_frequency) ** (-1))
        non_dimensional_speed_sound = np.sqrt(non_dimensional_sound_velocity ** 2 + (n - 1) * non_dimensional_enthalpy)
        common_term = self.common_factor_gilmore(radius, velocity, B)
        non_dimensional_viscosity = self.viscosity * self.angular_frequency / self.atmospheric_pressure
        non_dimensional_density = self.density * (
                self.initial_radius * self.angular_frequency) ** 2 / self.atmospheric_pressure

        eq_1 = (1 + (velocity / non_dimensional_speed_sound)) * non_dimensional_enthalpy
        eq_2 = (1 / non_dimensional_speed_sound) * (
                1 - (velocity / non_dimensional_speed_sound)) * radius * non_dimensional_delta_enthalpy

        eq_3 = acceleration * radius * (1 + ((A / self.atmospheric_pressure) ** (1 / n)) * (
                common_term ** (- 1 / n)) * 4 * non_dimensional_viscosity / (
                                                non_dimensional_speed_sound * radius * non_dimensional_density))
        eq_4 = (3 / 2) * (1 - (velocity / (3 * non_dimensional_speed_sound))) * (velocity ** 2)

        pressure_contribution = eq_1 + eq_2
        eq_3_1 = radius * (1 + ((A / self.atmospheric_pressure) ** (1 / n)) * (
                common_term ** (- 1 / n)) * 4 * non_dimensional_viscosity / (
                                   non_dimensional_speed_sound * radius * non_dimensional_density))
        IF = - eq_4 / eq_3_1
        PF = pressure_contribution / ((1 - (velocity / non_dimensional_speed_sound)) * radius)

        return np.array(IF), np.array(PF), radius, velocity

    def critical_radius(self, inertial_function, pressure_function, radius, equation, time):
        """
        Calculates critical radius for a given equation.

        :param radius:
        :param pressure_function:
        :param inertial_function:
        :param equation: Name of the equation.
        :type equation: str

        :param time: Value of time.
        :type time: np.ndarray

        :return: Value of the radius when the inertial function crosses with the pressure function at its minimum.

        References:
        - H.G. Flynn (1975). Cavitation dynamics II.
        """
        if equation == "Rayleigh-Plesset":
            # inertial_function, pressure_function, radius, velocity = self.RP_functions(time)
            times = time / (self.frequency)
        elif equation == "Keller-Miksis":
            # inertial_function, pressure_function, radius, velocity = self.KM_functions(time, step)
            times = time / (self.frequency)
        elif equation == "Gilmore":
            # inertial_function, pressure_function, radius, velocity = self.G_functions(time, step)
            times = time / (self.frequency * 2 * np.pi)
        else:
            raise ValueError("Invalid equation")

        radius_c = None

        cross_epsilon = 5e-1

        raw_intersection_points = np.where(abs(inertial_function - pressure_function) <= cross_epsilon)[0]

        min_value_pressure = np.min(pressure_function)
        min_index_pressure = np.argmin(pressure_function)

        if not raw_intersection_points.size:
            if min_value_pressure > inertial_function[min_index_pressure]:
                radius_c = - 1
            else:
                radius_c = np.max(radius)
        else:
            intersection_points = [raw_intersection_points[0]]
            for i in range(1, len(raw_intersection_points)):
                if raw_intersection_points[i] - intersection_points[-1] > 50:
                    intersection_points.append(raw_intersection_points[i])

            half_min_value = min_value_pressure / 2.0

            left_index = min_index_pressure
            while left_index > 0 and pressure_function[left_index] <= half_min_value:
                left_index -= 1

            right_index = min_index_pressure
            while right_index < len(pressure_function) - 1 and pressure_function[right_index] <= half_min_value:
                right_index += 1

            for intersection_index in intersection_points:
                if left_index <= intersection_index <= right_index:
                    temporal_radius = radius[intersection_index]
                    if radius_c is None or temporal_radius > radius_c:
                        radius_c = temporal_radius

            if radius_c is None:
                if min_value_pressure > inertial_function[min_index_pressure]:
                    radius_c = - 1
                else:
                    radius_c = np.max(radius)

        return radius_c

    def integrand_rayleigh_plesset(self, r, r_dot):
        constant_1 = (self.atmospheric_pressure / self.atmospheric_pressure + 2 * (
                self.surface_tension / (self.atmospheric_pressure * self.initial_radius)) -
                      (self.vapor_pressure / self.atmospheric_pressure))
        constant_2 = self.vapor_pressure / self.atmospheric_pressure

        return 4 * np.pi * r ** 2 * r_dot * (
                constant_1 * (1 / r) ** (3 * self.adiabatic_index) + constant_2)

    def integrand_keller_miksis(self, r, r_dot):
        constant_1 = (self.atmospheric_pressure / self.atmospheric_pressure + 2 * (
                self.surface_tension / (self.atmospheric_pressure * self.initial_radius)) -
                      (self.vapor_pressure / self.atmospheric_pressure)) * (
                             -3 * self.adiabatic_index * r_dot / r)
        constant_2 = self.vapor_pressure / self.atmospheric_pressure
        return 4 * np.pi * r ** 2 * r_dot * (
                constant_1 * (1 / r) ** (3 * self.adiabatic_index) + constant_2)

    def integrand_gilmore(self, r, r_dot):
        constant_1 = self.atmospheric_pressure / self.atmospheric_pressure + (
                2 * self.surface_tension / (self.atmospheric_pressure * self.initial_radius))
        constant_2 = (1 / r) ** (3 * self.adiabatic_index)
        return 4 * np.pi * r ** 2 * r_dot * constant_1 * constant_2

    def recursive_integration(self, integrand, t1, t2, tolerance, max_points=50, max_iterations=5, iteration=0):
        if iteration >= max_iterations:
            warnings.warn("Max iterations reached. Returning current result.", UserWarning)
            return quad(integrand, t1, t2, epsrel=tolerance)

        points = np.linspace(t1, t2, num=max_points)
        result, error = quad(integrand, t1, t2, points=points)

        if error > abs(result) / 2:
            mid_point = (t1 + t2) / 2.0

            result_left, _ = self.recursive_integration(integrand, t1, mid_point, tolerance, max_points, max_iterations,
                                                        iteration + 1)
            result_right, _ = self.recursive_integration(integrand, mid_point, t2, tolerance, max_points,
                                                         max_iterations, iteration + 1)

            return result_left + result_right, error

        return result, error

    def integrate(self, t1, t2, radius, velocity, time, equation_name, tolerance=1e-25):
        r_interp = np.interp
        r_dot_interp = np.interp

        if equation_name == "Rayleigh-Plesset":
            integrand_function = self.integrand_rayleigh_plesset
        elif equation_name == "Keller-Miksis":
            integrand_function = self.integrand_keller_miksis
        elif equation_name == "Gilmore":
            integrand_function = self.integrand_gilmore
        else:
            raise ValueError("Invalid equation name")

        def integrand(t):
            r = r_interp(t, time, radius)
            r_dot = r_dot_interp(t, time, velocity)
            return integrand_function(r, r_dot)

        return self.recursive_integration(integrand, t1, t2, tolerance)

    def integration_helper(self, interval_start, interval_end, radius, velocity, time, equation_name):
        ratio = 0
        peak_interval = None

        time_interval_start = time[interval_start]
        time_interval_end = time[interval_end]

        result_interval, error_interval = self.integrate(time_interval_start, time_interval_end, radius,
                                                         velocity, time, equation_name)
        interval = radius[interval_start: interval_end + 1]
        peaks, _ = find_peaks(interval)
        peaks = np.insert(peaks, 0, 0)

        if len(peaks) > 0:
            for position_max in peaks:
                # Find the next valley after the peak

                next_valley = np.argmin(interval[position_max:]) + position_max
                if next_valley < len(interval):
                    position_min = next_valley
                    result_compression, error_compression = self.integrate(time[position_max + interval_start],
                                                                           time[position_min + interval_start], radius,
                                                                           velocity,
                                                                           time, equation_name)
                    if abs(result_compression) != 0:
                        ratio_interval = abs(result_interval) / abs(result_compression)
                        if ratio_interval > ratio and (
                                peak_interval is None or interval[position_max] > peak_interval):
                            ratio = ratio_interval
                            peak_interval = interval[position_max]
                    else:
                        warnings.warn("division by zero", UserWarning)
                else:
                    warnings.warn("No valley found after the peak at position", UserWarning)
        else:
            warnings.warn("No peaks in the interval", UserWarning)
        return ratio, peak_interval

    def transition_radius(self, radius, velocity, equation_name, time):
        """
        Calculates the transition radius for a given equation.

        :param equation_name: Name of the equation.
        :type equation_name: str

        :param time: Value of time.
        :type time: np.ndarray

        :param step: Value of step in the integration scheme.
        :type step: float

        :return: Radius when the energy dissipation modulus is at its maximum.

        References:
        - H.G. Flynn (1975). Cavitation dynamics II.
        """
        if equation_name == "Rayleigh-Plesset":
            # inertial_function, pressure_function, radius, velocity = self.RP_functions(time)
            pass
        elif equation_name == "Keller-Miksis":
            pass
            # inertial_function, pressure_function, radius, velocity = self.KM_functions(time, step)
        elif equation_name == "Gilmore":
            # inertial_function, pressure_function, radius, velocity = self.G_functions(time, step)
            time = time / (2 * np.pi)
        else:
            raise ValueError("Invalid equation_name")
        length_plot = len(time)

        raw_crossings = np.where(np.diff(np.sign(radius - 1)))[0]
        if len(raw_crossings) > 0:
            crossings = [raw_crossings[0]]
            for i in range(1, len(raw_crossings)):
                if raw_crossings[i] - crossings[-1] > length_plot * 0.01:
                    crossings.append(raw_crossings[i])

            peaks, _ = find_peaks(radius)
            peaks = np.array(peaks)
            crossings = np.array(crossings)

            ratio = 0
            peak_interval = None
            if len(crossings) > 0 and len(peaks) > 0:
                if len(peaks) == 1 or len(crossings) == 1:
                    interval_start = peaks[0]
                    interval_end = len(time) - 1
                    ratio, peak_interval = self.integration_helper(interval_start, interval_end, radius, velocity, time,
                                                                   equation_name)
                else:
                    position = 0
                    while position < len(peaks) - 1:
                        interval_start = peaks[position]
                        higher_crossing = crossings[crossings > interval_start]

                        if len(higher_crossing) < 2:
                            interval_end = len(time) - 1
                            position = len(peaks)
                            temporal_ratio, temporal_peak_interval = self.integration_helper(interval_start,
                                                                                             interval_end,
                                                                                             radius, velocity,
                                                                                             time, equation_name)
                        else:
                            maximums = peaks[peaks > higher_crossing[1]]
                            if len(maximums) == 0:
                                interval_end = len(time) - 1
                                position = len(peaks)
                                temporal_ratio, temporal_peak_interval = self.integration_helper(interval_start,
                                                                                                 interval_end,
                                                                                                 radius, velocity,
                                                                                                 time, equation_name)
                            else:
                                interval_end = maximums[0]
                                position = np.where(peaks == interval_end)[0][0]
                                while time[interval_end] - time[interval_start] < 1:
                                    if len(maximums) > 1:
                                        interval_end = maximums[1]
                                        position = np.where(peaks == interval_end)[0][0]
                                        maximums = maximums[1:]
                                    else:
                                        interval_end = len(time) - 1
                                        position = len(peaks)
                                        break

                                temporal_ratio, temporal_peak_interval = self.integration_helper(interval_start,
                                                                                                 interval_end,
                                                                                                 radius, velocity,
                                                                                                 time, equation_name)

                        if temporal_ratio >= ratio and (
                                peak_interval is None or temporal_peak_interval > peak_interval):
                            ratio = temporal_ratio
                            peak_interval = temporal_peak_interval
            return peak_interval
        else:
            return None

    @staticmethod
    def dynamical_threshold(critical_radius, transition_radius):
        """
        Calculate the dynamical threshold by determining the maximum between the critical and transition radii for a specific initial radius.

        :param critical_radius: Critical radius.
        :type critical_radius: float

        :param transition_radius: Transition radius.
        :type transition_radius: float

        :return: Dynamical threshold.
        :rtype: float

        References:
        - H.G. Flynn (1975). Cavitation dynamics II.
        """
        if transition_radius is None:
            return critical_radius
        else:
            return max(critical_radius, transition_radius)

    def blake_threshold(self):
        """

        :return:
        """
        x_b = 2 * self.surface_tension / (self.atmospheric_pressure * self.initial_radius)
        eq_1 = (4 / 9) * x_b * ((3 * x_b / (4 * (1 + x_b))) ** (1 / 2))

        return self.atmospheric_pressure * (1 + eq_1)

    def natural_radius(self):
        omega_r = np.sqrt(3 * self.adiabatic_index * self.atmospheric_pressure / self.density) / self.initial_radius

        coefficients = [self.density * (omega_r ** 2), 0,
                        - 3 * self.adiabatic_index * (
                                self.atmospheric_pressure - self.vapor_pressure) - self.vapor_pressure,
                        2 * self.surface_tension * (1 - 3 * self.adiabatic_index),
                        4 * (self.viscosity ** 2) / self.density]

        def quartic_equation(x, *coeffs):
            return coeffs[0] * x ** 4 + coeffs[1] * x ** 3 + coeffs[2] * x ** 2 + coeffs[3] * x + coeffs[4]

        solution = root(quartic_equation, x0=np.zeros(4), args=tuple(coefficients))

        roots = solution.x
        return np.min(roots)

    def vokurka(self, time, radius, velocity, step, r):
        acceleration = np.diff(velocity) / step
        acceleration = np.insert(acceleration, 0, 0)
        eq_1 = (radius / r) * (acceleration * radius + (2 * (velocity ** 2))) - ((velocity ** 2) / 2) * (
                    (radius / r) ** 4)
        p_infty = (self.atmospheric_pressure + self.acoustic_pressure * np.sin(
            2 * np.pi * time)) / self.atmospheric_pressure
        non_dimensional_density = self.density * (
                self.initial_radius * self.angular_frequency) ** 2 / self.atmospheric_pressure
        eq_2 = non_dimensional_density / p_infty
        p = (eq_1 * eq_2 + 1) * p_infty

        return p - p_infty

    def critical_maximum_radius(self, equation):
        if equation == 'Rayleigh-Plesset' or 'Keller-Miksis':
            eq_2 = self.atmospheric_pressure + (2 * self.surface_tension / self.initial_radius) - self.vapor_pressure
        elif equation == 'Gilmore':
            eq_2 = self.atmospheric_pressure + (2 * self.surface_tension / self.initial_radius)
        else:
            raise ValueError('Invalid equation')

        eq_1 = 3 * self.adiabatic_index / (2 * self.surface_tension)
        eq_3 = self.initial_radius ** (3 * self.adiabatic_index)

        return ((eq_1 * eq_2 * eq_3) ** (1 / (3 * self.adiabatic_index - 1))) * 1e6