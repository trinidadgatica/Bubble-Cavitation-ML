import numpy as np

import warnings
warnings.filterwarnings("ignore")


class Model:

    def __init__(self, pa, f, r0, j0, p0, sigma, rho, mu, c, pv, kappa):
        self.acoustic_pressure = -pa
        self.frequency = f
        self.initial_radius = r0
        self.initial_velocity = j0
        self.period = 1 / self.frequency
        self.angular_frequency = 2 * np.pi * self.frequency
        self.atmospheric_pressure = p0
        self.surface_tension = sigma
        self.density = rho
        self.viscosity = mu
        self.sound_velocity = c
        self.vapor_pressure = pv
        self.adiabatic_index = kappa

        self.sigma = 0.83e-6 / 2.35
        self.t_0 = 500 * 3.37e-4

        self.enthalpy_history = list()
        self.delta_enthalpy_history = list()
        self.wave_speed_at_wall_history = list()
        self.radius_history = list()
        self.velocity_history = list()
        self.time_history = list()

        self.eq_1_history = list()
        self.eq_2_history = list()
        self.eq_4_history = list()

    def calculate_constants(self, time):
        """
        In an array of time, calculate the values of constants for the Rayleigh-Plesset equation.

        :param time: Array of time positions for the equation.
        :type time: numpy.ndarray

        :return: Tuple containing Reynolds, Weber, and Thoma constants, and initial pressure in the bubble wall.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """

        reynolds_number_fl = (self.initial_radius ** 2 * self.density) / (4 * self.viscosity * self.period)
        weber_number_fl = (2 * self.surface_tension * self.period ** 2) / (self.density * (self.initial_radius ** 3))
        non_dimensional_pressure_fl = self.acoustic_pressure
        thoma_number_fl = (self.period ** 2) * non_dimensional_pressure_fl / (self.density * (self.initial_radius ** 2))
        pressure_in_infinity_fl = self.atmospheric_pressure + \
                                  self.acoustic_pressure * np.sin(self.angular_frequency * time * self.period)
        non_dimensional_initial_pressure_fl = ((self.period ** 2) / (self.initial_radius ** 2)) * \
                                              (self.atmospheric_pressure + (
                                                      2 * self.surface_tension / self.initial_radius) -
                                               self.vapor_pressure) / self.density
        return [reynolds_number_fl, weber_number_fl, thoma_number_fl, pressure_in_infinity_fl,
                non_dimensional_initial_pressure_fl]

    def simple_rayleigh_plesset_equation(self, time, initial_variable):
        """
        Calculates a simplified version of the Rayleigh-Plesset equation. This version doesn't consider the effects of surface tension and viscosity.

        :param initial_variable: Set of two variables, one for radius and one for velocity.
        :type initial_variable: Tuple[float, float]

        :param time: Array of time values.
        :type time: numpy.ndarray

        :return: Tuple containing velocity and radius schemes for the given configuration of parameters.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]

        Note: This equation doesn't consider any type of damping for the system.
        """

        rn, jn = initial_variable
        pl = 0
        p_in = (self.atmospheric_pressure + 2 / self.initial_radius)
        p_out = - self.acoustic_pressure * np.sin(self.angular_frequency * time * self.period)
        pressure = p_in + p_out

        eq_1 = 1 / rn
        eq_2 = -(3 / 2) * (jn ** 2)
        eq_3 = ((self.period ** 2) / (self.initial_radius ** 2))
        eq_4 = (pl - self.atmospheric_pressure + pressure) / self.density
        return [jn, eq_1 * (eq_2 + eq_3 * eq_4)]

    def intermediate_rayleigh_plesset_equation(self, time, initial_variable):
        """
        Calculates a simplified version of the Rayleigh-Plesset equation. This version doesn't consider the effects of viscosity.

        :param initial_variable: Set of two variables, one for radius and one for velocity.
        :type initial_variable: Tuple[float, float]

        :param time: Array of time values.
        :type time: numpy.ndarray

        :return: Tuple containing velocity and radius schemes for the given configuration of parameters.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        rn, jn = initial_variable
        p_out = - self.acoustic_pressure * np.sin(self.angular_frequency * time * self.period)

        eq_1 = 1 / rn
        eq_2 = -(3 / 2) * (jn ** 2)
        eq_3 = (self.period ** 2) / (self.initial_radius ** 2 * self.density)
        eq_4 = (self.atmospheric_pressure + (2 * self.surface_tension / self.initial_radius) -
                self.vapor_pressure) / (rn ** (3 * self.adiabatic_index))
        eq_5 = self.vapor_pressure - self.atmospheric_pressure + p_out - 2 * self.surface_tension / \
               (rn * self.initial_radius)
        return [jn, eq_1 * (eq_2 + eq_3 * (eq_4 + eq_5))]

    def rayleigh_plesset_equation(self, time, initial_variable):
        """
        Calculates the Rayleigh-Plesset equation for a specified time interval.

        :param initial_variable: Set of two variables, one for radius and one for velocity.
        :type initial_variable: Tuple[float, float]

        :param time: Array of time values.
        :type time: numpy.ndarray

        :return: Tuple containing velocity and radius schemes for the given configuration of parameters.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]

        Note: This equation adds the viscosity of the liquid surrounding the bubble to the previous equation.
        """

        rn, jn = initial_variable
        self.time_history.append(time)
        self.radius_history.append(rn)
        [re, we, th, p_infinity, non_dimensional_p_0] = self.calculate_constants(time)
        eq_1 = 1 / rn
        eq_2 = -(3 / 2) * (jn ** 2)
        eq_3 = - th * (p_infinity + self.vapor_pressure) / self.acoustic_pressure
        eq_4 = non_dimensional_p_0 / (rn ** (3 * self.adiabatic_index))
        eq_5 = - we / rn
        eq_6 = - (1 / re) * (jn / rn)

        return [jn, eq_1 * (eq_2 + eq_3 + eq_4 + eq_5 + eq_6)]

    def keller_miksis_equation(self, time, initial_variable):
        """
        Solves the Keller-Miksis equation over a specified time interval.

        :param initial_variable: Set of two variables, one for radius and one for velocity.
        :type initial_variable: Tuple[float, float]

        :param time: Array of time values.
        :type time: numpy.ndarray

        :return: Tuple containing velocity and radius schemes for the given configuration of parameters.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        rn, jn = initial_variable
        self.radius_history.append(rn)
        self.velocity_history.append(jn)
        self.time_history.append(time)

        p_in = (self.atmospheric_pressure - self.vapor_pressure + 2 * (self.surface_tension / self.initial_radius)) * \
               (1 / rn) ** (3 * self.adiabatic_index)
        p_out = -self.atmospheric_pressure - 2 * (self.surface_tension / (rn * self.initial_radius)) - \
                4 * self.viscosity * (jn) / (rn * self.period) - self.acoustic_pressure * np.sin(self.angular_frequency
                                                                                                 * time * self.period) + self.vapor_pressure
        pressure = p_in + p_out

        eq_1 = rn * (1 - jn * self.initial_radius / (self.period * self.sound_velocity) +
                     4 * self.viscosity * rn * self.initial_radius /
                     (rn * (self.period ** 2) * self.density * self.sound_velocity))
        eq_2 = -(3 / 2) * (jn ** 2) + (1 / 2) * (jn ** 3) * self.initial_radius / (self.period * self.sound_velocity)
        eq_3 = (self.period ** 2) / (self.initial_radius ** 2)
        eq_4 = (1 + jn * self.initial_radius / (self.period * self.sound_velocity)) * (pressure / self.density)
        eq_5 = rn * self.initial_radius / (self.density * self.sound_velocity)
        eq_6 = ((self.atmospheric_pressure - self.vapor_pressure + 2 * self.surface_tension / self.initial_radius) *
                (-3 * self.adiabatic_index * (rn ** (-3 * self.adiabatic_index - 1)) * jn / self.period))
        eq_7 = 2 * self.surface_tension * jn / ((rn ** 2) * self.initial_radius * self.period)
        eq_8 = 4 * self.viscosity * (jn ** 2) / ((rn ** 2) * self.period ** 2)
        eq_9 = - self.acoustic_pressure * np.cos(2 * np.pi * time * self.period * self.frequency) * 2 * np.pi

        return [jn, (1 / eq_1) * (eq_2 + eq_3 * (eq_4 + eq_5 * (eq_6 + eq_7 + eq_8 + eq_9)))]

    def enthalpy(self, r, j, time, n, A, B):
        """
        Calculate the enthalpy in a time interval.

        :param r: Array of bubble radius values over time.
        :type r: numpy.ndarray

        :param j: Array of bubble velocity values over time.
        :type j: numpy.ndarray

        :param time: Value of time.
        :type time: numpy.ndarray

        :param n: Constant of Tait's state equation.
        :type n: float

        :param A: Constant of Tait's state equation.
        :type A: float

        :param B: Constant of Tait's state equation.
        :type B: float

        :return: Array of enthalpy values over time.
        :rtype: numpy.ndarray
        """

        non_dimensional_atmospheric_pressure = self.atmospheric_pressure / self.atmospheric_pressure
        non_dimensional_initial_radius = self.initial_radius / self.initial_radius
        non_dimensional_acoustic_pressure = ((self.acoustic_pressure / self.atmospheric_pressure)* np.sin((self.angular_frequency / self.angular_frequency) * time))
        non_dimensional_pressure_infty = non_dimensional_acoustic_pressure + non_dimensional_atmospheric_pressure
        non_dimensional_surface_tension = self.surface_tension / (self.atmospheric_pressure * self.initial_radius)
        non_dimensional_viscosity = self.viscosity * self.angular_frequency / self.atmospheric_pressure
        non_dimensional_density = self.density * (
                self.initial_radius * self.angular_frequency) ** 2 / self.atmospheric_pressure

        eq_1 = (
                       non_dimensional_atmospheric_pressure + 2 * non_dimensional_surface_tension / non_dimensional_initial_radius) \
               * (non_dimensional_initial_radius / r) ** (3 * self.adiabatic_index)
        eq_2 = eq_1 - 2 * non_dimensional_surface_tension / r - 4 * non_dimensional_viscosity * j / r

        eq_3 = n * (A / self.atmospheric_pressure) ** (1 / n) / (n - 1) / non_dimensional_density
        eq_4 = eq_3 * ((eq_2 + B / self.atmospheric_pressure) ** (1 - (1 / n)) -
                       (non_dimensional_pressure_infty + B / self.atmospheric_pressure) ** (1 - (1 / n)))

        return eq_4

    def common_factor_gilmore(self, r, j, B):
        non_dimensional_atmospheric_pressure = self.atmospheric_pressure / self.atmospheric_pressure
        non_dimensional_initial_radius = self.initial_radius / self.initial_radius
        non_dimensional_surface_tension = self.surface_tension / (self.atmospheric_pressure * self.initial_radius)
        non_dimensional_viscosity = self.viscosity * self.angular_frequency / self.atmospheric_pressure

        eq_1 = ((non_dimensional_atmospheric_pressure +
                 2 * non_dimensional_surface_tension / non_dimensional_initial_radius)
                * (non_dimensional_initial_radius / r) ** (3 * self.adiabatic_index))
        eq_2 = (eq_1 - (2 * non_dimensional_surface_tension / r) - (4 * non_dimensional_viscosity * j / r) +
                (B / self.atmospheric_pressure))
        return eq_2

    def delta_enthalpy(self, r, j, time, n, A, B):
        """
        Calculate the derivative of enthalpy in a time interval.

        :param r: Array of bubble radius values over time.
        :type r: numpy.ndarray

        :param j: Array of bubble velocity values over time.
        :type j: numpy.ndarray

        :param time: Array of time values.
        :type time: numpy.ndarray

        :param n: Constant of Tait's state equation.
        :type n: float

        :param A: Constant of Tait's state equation.
        :type A: float

        :param B: Constant of Tait's state equation.
        :type B: float

        :return: Array of the derivative of enthalpy values over time.
        :rtype: numpy.ndarray
        """
        non_dimensional_atmospheric_pressure = self.atmospheric_pressure / self.atmospheric_pressure
        non_dimensional_initial_radius = self.initial_radius / self.initial_radius
        non_dimensional_surface_tension = self.surface_tension / (self.atmospheric_pressure * self.initial_radius)
        non_dimensional_viscosity = self.viscosity * self.angular_frequency / self.atmospheric_pressure
        non_dimensional_density = self.density * (
                self.initial_radius * self.angular_frequency) ** 2 / self.atmospheric_pressure

        non_dimensional_acoustic_pressure = ((self.acoustic_pressure / self.atmospheric_pressure) * np.sin((self.angular_frequency / self.angular_frequency) * time))

        eq_1 = (
                       non_dimensional_atmospheric_pressure + 2 * non_dimensional_surface_tension / non_dimensional_initial_radius) \
               * (non_dimensional_initial_radius / r) ** (3 * self.adiabatic_index)
        eq_2 = eq_1 - 2 * non_dimensional_surface_tension / r - 4 * non_dimensional_viscosity * j / r
        eq_3 = (
                           non_dimensional_atmospheric_pressure + 2 * non_dimensional_surface_tension / non_dimensional_initial_radius) \
               * (non_dimensional_initial_radius ** (3 * self.adiabatic_index)) * (
                           j * - 3 * self.adiabatic_index * r ** (- 3 * self.adiabatic_index - 1))
        eq_3_1 = eq_3 + (non_dimensional_surface_tension * j / (r ** 2)) + (
                    4 * non_dimensional_viscosity * (j ** 2) / (r ** 2))
        eq_4 = (non_dimensional_atmospheric_pressure + non_dimensional_acoustic_pressure + (
                    B / self.atmospheric_pressure)) ** (- 1 / n)
        eq_5 = ((self.acoustic_pressure / self.atmospheric_pressure) * np.cos(
            (self.angular_frequency / self.angular_frequency) * time))
        eq_6 = (((A / self.atmospheric_pressure) ** (1 / n)) / non_dimensional_density) * (
                    (((eq_2 + (B / self.atmospheric_pressure)) ** (- 1 / n)) * eq_3_1) - (eq_4 * eq_5))

        return eq_6

    def gilmore_equation(self, time, initial_variable):
        """
        Solves the Gilmore equation over a specified time interval.

        :param initial_variable: Set of two variables, one for radius and one for velocity.
        :type initial_variable: Tuple[float, float]

        :param time: Array of time values.
        :type time: numpy.ndarray

        :return: Tuple containing velocity and radius schemes for the given configuration of parameters.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        rn, jn = initial_variable
        n = 7

        A = 304e6
        B = 303.9e6

        non_dimensional_enthalpy = self.enthalpy(rn, jn, time, n, A, B)
        non_dimensional_delta_enthalpy = self.delta_enthalpy(rn, jn, time, n, A, B)
        non_dimensional_sound_velocity = self.sound_velocity * (self.initial_radius * self.angular_frequency) ** (-1)
        non_dimensional_speed_sound = np.sqrt(non_dimensional_sound_velocity ** 2 + (n - 1) * non_dimensional_enthalpy)
        common_term = self.common_factor_gilmore(rn, jn, B)
        non_dimensional_viscosity = self.viscosity * self.angular_frequency / self.atmospheric_pressure
        non_dimensional_density = self.density * (
                self.initial_radius * self.angular_frequency) ** 2 / self.atmospheric_pressure

        self.enthalpy_history.append(non_dimensional_enthalpy)
        self.delta_enthalpy_history.append(non_dimensional_delta_enthalpy)
        self.wave_speed_at_wall_history.append(non_dimensional_speed_sound)
        self.time_history.append(time)
        self.radius_history.append(rn)
        self.velocity_history.append(jn)

        eq_1_in = (1 + (jn / non_dimensional_speed_sound)) * non_dimensional_enthalpy
        eq_2_in = (jn / non_dimensional_speed_sound) * (
                1 - (jn / non_dimensional_speed_sound)) * rn * non_dimensional_delta_enthalpy

        eq_4_in = (3 / 2) * (1 - (jn / (3 * non_dimensional_speed_sound))) * (jn ** 2)
        self.eq_1_history.append(eq_1_in)
        self.eq_2_history.append(eq_2_in)
        self.eq_4_history.append(eq_4_in)

        eq_1 = non_dimensional_enthalpy * (non_dimensional_speed_sound + jn) / (non_dimensional_speed_sound - jn)
        eq_2 = non_dimensional_delta_enthalpy * rn / non_dimensional_speed_sound
        eq_3 = 0.5 * (jn ** 2) * (3 * non_dimensional_speed_sound - jn) / (non_dimensional_speed_sound - jn)
        eq_4 = rn * (1 + ((A / self.atmospheric_pressure) ** (1 / n)) * (
                    common_term ** (- 1 / n)) * 4 * non_dimensional_viscosity / (
                                 non_dimensional_speed_sound * rn * non_dimensional_density))
        return [jn, (1 / eq_4) * (eq_1 + eq_2 - eq_3)]

    @staticmethod
    def density_generator_temperature(temperature):
        """
        Calculate density for water-like fluids based on temperature.

        :param temperature: Integer representing temperature in Celsius degrees.
        :type temperature: int

        :return: Calculated density in kg/m^3 corresponding to the given temperature.
        :rtype: float

        Reference:
        "Modeling the Physics of Bubble Nucleation in Histotripsy,"
        Matheus O. de Andrade, Seyyed Reza Haqshenas, Ki Joo Pahk, and Nader Saffari, 2021.
        """

        T_kelvin = temperature + 273.15
        critical_density = 322  # kg m^-3
        critical_temperature = 647.096  # Kelvin
        x = 1 - (T_kelvin / critical_temperature)
        b_1 = 1.99274064
        b_2 = 1.09965342
        b_3 = - 0.510839303
        b_4 = - 1.75493479
        b_5 = - 45.5170352
        b_6 = - 6.74694450e5
        density = critical_density * (1 + b_1 * x ** (1 / 3) + b_2 * x ** (2 / 3) + b_3 * x ** (5 / 3) +
                                      b_4 * x ** (16 / 3) + b_5 * x ** (43 / 3) + b_6 * x ** (110 / 3))
        return density

    @staticmethod
    def surface_tension_generator_temperature(temperature):
        """
        Calculate surface tension for water-like fluids based on temperature.

        :param temperature: Integer representing temperature in Celsius degrees.
        :type temperature: int

        :return: Calculated surface tension in N/m corresponding to the given temperature.
        :rtype: float

        Reference:
        "Modeling the Physics of Bubble Nucleation in Histotripsy,"
        Matheus O. de Andrade, Seyyed Reza Haqshenas, Ki Joo Pahk, and Nader Saffari, 2021.
        """

        T_kelvin = temperature + 273.15
        critical_temperature = 647.096  # Kelvin
        x = 1 - (T_kelvin / critical_temperature)
        d_1 = 235.8e-3  # Nm^-1
        d_2 = 1.256
        d_3 = - 0.625
        surface_tension = d_1 * (x ** d_2) * (1 + d_3 * x)
        return surface_tension

    @staticmethod
    def viscosity_generator_temperature(temperature):
        """
        Calculate viscosity for water-like fluids based on temperature.

        :param temperature: Integer representing temperature in Celsius degrees.
        :type temperature: int

        :return: Calculated viscosity with respect to temperature in kg/(m * s).
        :rtype: float

        Reference:
        "Viscosity of Liquid Water in the Range -50°C to 150°C,"
        Joseph Kestin, 2012.
        """

        critical_viscosity = 1002  # micro Pa s at 20 C
        a_1 = (20 - temperature) / (temperature + 96)
        a_2 = (1.2378 - (1.303e-3 * (20 - temperature)) + (3.06e-6 * ((20 - temperature) ** 2)) +
               (2.55e-8 * ((20 - temperature) ** 3)))

        viscosity = critical_viscosity * 10 ** (a_1 * a_2)  # this is in micro Pa s

        return viscosity / 1e6

    @staticmethod
    def sound_velocity_generator_temperature(temperature):
        """
        Calculate sound velocity for water-like fluids based on temperature.

        :param temperature: Integer representing temperature in Celsius degrees.
        :type temperature: int

        :return: Calculated sound velocity in m/s corresponding to the given temperature.
        :rtype: float

        Reference:
        "Speed of Sound in Pure Water,"
        V. A. D'i, Grosso, and C. W. Mader, 1972.
        """

        k_0 = 1.40238744e3
        k_1 = 5.03835027
        k_2 = - 5.81142290e-2
        k_3 = 3.34558776e-4
        k_4 = - 1.48150040e-6
        k_5 = 3.16081885e-9
        sound_velocity = ((k_0 * (temperature ** 0)) + (k_1 * (temperature ** 1)) +
                          (k_2 * (temperature ** 2)) + (k_3 * (temperature ** 3)) +
                          (k_4 * (temperature ** 4)) + (k_5 * (temperature ** 5)))
        return sound_velocity


if __name__ == '__main__':
    print('Model complete')
