from thresholds.experiment_maker import ExperimentMaker
from main import Model

from joblib import Parallel, delayed
from scipy.stats import kurtosis
import numpy as np
import itertools
import csv


class DataCreator(Model):

    def pipeline_cavitation_type(self, temperature, initial_radius, acoustic_pressure, frequency):
        sound_velocity = Model.sound_velocity_generator_temperature(temperature)
        surface_tension = Model.surface_tension_generator_temperature(temperature)
        density = Model.density_generator_temperature(temperature)
        viscosity = Model.viscosity_generator_temperature(temperature)

        self.initial_radius = initial_radius
        self.frequency = frequency
        self.acoustic_pressure = acoustic_pressure
        self.density = density
        self.viscosity = viscosity
        self.sound_velocity = sound_velocity
        self.surface_tension = surface_tension

        result = [self.initial_radius, self.acoustic_pressure, self.frequency, temperature, self.density,
                  self.viscosity, self.surface_tension, self.sound_velocity, self.vapor_pressure]
        exp_maker = ExperimentMaker(self.acoustic_pressure, self.frequency, self.initial_radius, 0,
                                    self.atmospheric_pressure, self.surface_tension, self.density, self.viscosity,
                                    self.sound_velocity, self.vapor_pressure, self.adiabatic_index)

        step_size = 1e-3 / self.frequency
        time = np.arange(0, 20 / self.frequency, step_size)
        if self.acoustic_pressure >= exp_maker.blake_threshold():

            equations_name = ["Rayleigh-Plesset", "Keller-Miksis", "Gilmore"]
            for equation in equations_name:
                if equation == "Rayleigh-Plesset":
                    times = time * self.frequency
                    steps = step_size * self.frequency
                    inertial_function, pressure_function, radius, velocity = exp_maker.RP_functions(time=times,
                                                                                                    solver='ODEINT',
                                                                                                    step=steps)
                    radius_rp = radius
                    velocity_rp = velocity * self.initial_radius * self.frequency / self.sound_velocity

                elif equation == "Keller-Miksis":
                    times = time * self.frequency
                    steps = step_size * self.frequency
                    inertial_function, pressure_function, radius, velocity = exp_maker.KM_functions(time=times,
                                                                                                    solver='ODEINT',
                                                                                                    step=steps)

                    radius_km = radius
                    velocity_km = velocity * self.initial_radius * self.frequency / self.sound_velocity

                elif equation == "Gilmore":
                    times = time * 2 * np.pi * self.frequency
                    steps = step_size * 2 * np.pi * self.frequency
                    inertial_function, pressure_function, radius, velocity = exp_maker.G_functions(time=times,
                                                                                                   solver='ODEINT',
                                                                                                   step=steps)
                    radius_g = radius
                    velocity_g = velocity * self.initial_radius * self.frequency / self.sound_velocity

                critical_radius = exp_maker.critical_radius(inertial_function, pressure_function, radius, equation,
                                                            times)
                transition_radius = exp_maker.transition_radius(radius, velocity, equation, times)
                if critical_radius == -1:
                    result.append(False)
                else:
                    dynamical_threshold = exp_maker.dynamical_threshold(critical_radius, transition_radius)
                    maximum_radius = np.max(radius)
                    if maximum_radius > dynamical_threshold:
                        result.append(False)
                    else:
                        result.append(True)

                midpoint_index = len(radius) // 2
                radius_second_half = radius[midpoint_index:]
                tolerance = 5e-1
                radius_near_zero = np.all(np.abs(radius_second_half) < tolerance)

                midpoint_index = len(velocity) // 2
                velocity_second_half = velocity[midpoint_index:]
                tolerance = 5e-1
                velocity_near_zero = np.all(np.abs(velocity_second_half) < tolerance)

                acoustic_emissions_raw = exp_maker.vokurka(time, radius, velocity, step_size, 10)
                if acoustic_emissions_raw is not None:
                    percentage_to_keep = 0.9
                    num_to_keep = int(len(acoustic_emissions_raw) * percentage_to_keep)
                    acoustic_emissions = acoustic_emissions_raw[-num_to_keep:]

                    kurtosis_value = kurtosis(acoustic_emissions)
                    rms = np.sqrt(sum(acoustic_emissions ** 2) / len(acoustic_emissions))
                    crest_factor = max(abs(acoustic_emissions)) / rms
                    result.append((kurtosis_value, crest_factor))

                else:
                    result.append((np.inf, np.inf))

                natural_radius = exp_maker.natural_radius()
                critical_maximum = exp_maker.critical_maximum_radius(equation)

                filtered_velocity = abs(velocity[~np.isnan(velocity)])
                mach_number = np.max(filtered_velocity) * self.initial_radius * self.frequency / self.sound_velocity
                logarithmic_mach_number = np.log(mach_number)
                if logarithmic_mach_number >= 0:
                    result.append(False)
                elif logarithmic_mach_number < 0:
                    if self.initial_radius >= natural_radius:
                        result.append(True)
                    else:
                        result.append(False)
                else:
                    result.append(None)

                expansion_radius = np.max(radius)
                result.append((expansion_radius, critical_maximum))
                result.append((radius_near_zero, velocity_near_zero))
                result.append(logarithmic_mach_number)
                result.append((critical_radius, transition_radius))


            return result
        else:
            print('Not enough acoustic pressure to oscillates')
            result.append([None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                           None, None, None, None, None, None, None, None, None])
            return result

    def parallel_creator(self, number_experiments):

        parallel_pool = Parallel(n_jobs=-1)
        parallel = delayed(DataCreator.pipeline_cavitation_type)
        temperature_grid = np.linspace(10, 60, number_experiments)
        initial_radius_grid = np.linspace(1e-6, 20e-6, number_experiments)  # 1e-6, 5e-5

        acoustic_presure_grid = np.linspace(2 * self.atmospheric_pressure, 30 * self.atmospheric_pressure,
                                            number_experiments)
        frequency_grid = np.linspace(20e3, 20e5, number_experiments)

        parameters = list(
            itertools.product(temperature_grid, initial_radius_grid, acoustic_presure_grid, frequency_grid))

        parallel_tasks = [parallel(self, temperature, initial_radius, acoustic_pressure, frequency)
                          for temperature, initial_radius, acoustic_pressure, frequency in parameters]

        parallel_results = parallel_pool(parallel_tasks)

        file1 = open(f"Data/cavitation_type_dataset.csv", "w")
        writer1 = csv.writer(file1)
        writer1.writerow(['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                          'surface_tension', 'sound_velocity', 'vapor_pressure',
                          'RP_dynamical_threshold', 'RP_acoustic_emissions', 'RP_mach_number', 'RP_expansion_radius',
                          'RP_near_zero', 'RP_max_velocity', 'RP_critical_transition',
                          'KM_dynamical_threshold', 'KM_acoustic_emissions', 'KM_mach_number', 'KM_expansion_radius',
                          'KM_near_zero', 'KM_max_velocity', 'KM_critical_transition',
                          'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius',
                          'G_near_zero', 'G_max_velocity', 'G_critical_transition'])

        for i in range(len(parallel_results)):

            row1 = [parallel_results[i][0], parallel_results[i][1], parallel_results[i][2], parallel_results[i][3],
                    parallel_results[i][4], parallel_results[i][5], parallel_results[i][6], parallel_results[i][7],
                    parallel_results[i][8], parallel_results[i][9], parallel_results[i][10], parallel_results[i][11],
                    parallel_results[i][12], parallel_results[i][13], parallel_results[i][14], parallel_results[i][15],
                    parallel_results[i][16], parallel_results[i][17], parallel_results[i][18], parallel_results[i][19],
                    parallel_results[i][20], parallel_results[i][21], parallel_results[i][22], parallel_results[i][23],
                    parallel_results[i][24], parallel_results[i][25], parallel_results[i][26], parallel_results[i][27],
                    parallel_results[i][28], parallel_results[i][29]]
            writer1.writerow(row1)
        file1.close()
