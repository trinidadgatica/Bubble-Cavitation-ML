from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np
import itertools
import os


import matplotlib.pyplot as plt
from main import Model


class ExperimentMaker():

    @staticmethod
    def generalization_dataset(dataset, number_points):
        initial_radius_values = sorted(dataset['initial_radius'].unique())
        acoustic_pressure_values = sorted(dataset['acoustic_pressure'].unique())
        frequency_values = sorted(dataset['frequency'].unique())
        temperature_values = sorted(dataset['temperature'].unique())

        initial_values_temperature = temperature_values[:number_points]
        final_values_temperature = temperature_values[-number_points:]

        initial_values_initial_radius = initial_radius_values[:number_points]
        final_values_initial_radius = initial_radius_values[-number_points:]

        initial_values_acoustic_pressure = acoustic_pressure_values[:number_points]
        final_values_acoustic_pressure = acoustic_pressure_values[-number_points:]

        initial_values_frequency = frequency_values[:number_points]
        final_values_frequency = frequency_values[-number_points:]

        filter_1 = dataset['initial_radius'].isin(initial_values_initial_radius + final_values_initial_radius)
        filter_2 = dataset['frequency'].isin(initial_values_frequency + final_values_frequency)
        filter_3 = dataset['acoustic_pressure'].isin(initial_values_acoustic_pressure + final_values_acoustic_pressure)
        filter_4 = dataset['temperature'].isin(initial_values_temperature + final_values_temperature)

        test_set = dataset[filter_1 | filter_2 | filter_3 | filter_4]
        train_set = dataset[~filter_1 & ~filter_2 & ~filter_3 & ~filter_4]
        return train_set, test_set

    @staticmethod
    def create_new_test_set(variable_to_vary, two_variable=False):
        num_points = 100
        ranges = {
            'initial_radius': np.linspace(1e-6, 20e-6, num_points),
            'frequency': np.linspace(100e3, 5e6, num_points),
            'acoustic_pressure': np.linspace(0.2e6, 3e6, num_points),
            'temperature': np.linspace(10, 60, num_points)
        }
        temperature = 20

        fixed_values = {
            'initial_radius': 4e-6,
            'frequency': 1e6,
            'acoustic_pressure': 1e6,
            'temperature': temperature,
            'density': Model.density_generator_temperature(temperature),
            'viscosity': Model.viscosity_generator_temperature(temperature),
            'surface_tension': Model.surface_tension_generator_temperature(temperature),
            'sound_velocity': Model.sound_velocity_generator_temperature(temperature),
            'vapor_pressure': 3.2718e3
        }
        new_test_set = pd.DataFrame()

        # For each point in the range of the variable to vary
        if two_variable:
            variable_1, variable_2 = variable_to_vary
            product_cartesian = list(itertools.product(ranges[variable_1], ranges[variable_2]))
            for value1, value2 in product_cartesian:
                new_row = fixed_values.copy()
                # Update the value of the variable to vary
                new_row[variable_1] = value1
                new_row[variable_2] = value2
                new_row_series = pd.Series(new_row)

                # Append the new row series to the DataFrame
                new_test_set = pd.concat([new_test_set, new_row_series.to_frame().T], ignore_index=True)
                new_test_set.to_csv(f'../Data/proportion_plot_general.csv')
        else:
            for value in ranges[variable_to_vary]:
                # Create a new row with fixed values
                new_row = fixed_values.copy()
                # Update the value of the variable to vary
                new_row[variable_to_vary] = value
                new_row_series = pd.Series(new_row)

                # Append the new row series to the DataFrame
                new_test_set = pd.concat([new_test_set, new_row_series.to_frame().T], ignore_index=True)
                new_test_set.to_csv(f'../Data/proportion_plot_{variable_to_vary}.csv')
        return new_test_set

    def ensemble_model(self, dataset, n_trees, generalization=False, predict_new=False, variable_to_vary=None, two_variables=False):
        results_df = pd.DataFrame()
        # create labels and features
        list_labels = ['RP_dynamical_threshold', 'RP_mach_number', 'RP_acoustic_emissions', 'RP_expansion_radius',
                       'KM_dynamical_threshold', 'KM_mach_number', 'KM_acoustic_emissions', 'KM_expansion_radius',
                       'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']
        list_features = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                         'surface_tension', 'sound_velocity', 'vapor_pressure', 'count']
        features = dataset[list_features]
        if generalization:
            train_set, test_set = self.generalization_dataset(dataset, 1)

        for label in list_labels:
            if generalization:
                features_train = train_set[list_features]
                labels_train = train_set[label]
                features_test = test_set[list_features]
                labels_test = test_set[label]
            else:
                # Split data into train and test sets
                labels = dataset[label]
                features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                            test_size=0.2,
                                                                                            random_state=42)
            features_train = features_train.drop('count', axis=1)
            features_test = features_test.drop('count', axis=1)

            # Train the model
            random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
            random_forest.fit(features_train, labels_train)

            feature_importances = random_forest.feature_importances_

            # Predictions on the training data
            train_predictions = random_forest.predict(features_train)

            if predict_new and variable_to_vary is not None:
                # Create a new test set based on the specified variable
                if os.path.exists('../Data/proportion_plot_general.csv'):
                    new_test_set = pd.read_csv('../Data/proportion_plot_general.csv')
                else:
                    new_test_set = self.create_new_test_set(variable_to_vary, two_variables)
                features_test = new_test_set[['initial_radius', 'acoustic_pressure', 'frequency', 'temperature',
                                                  'density', 'viscosity', 'surface_tension', 'sound_velocity',
                                                  'vapor_pressure']]
                test_predictions = random_forest.predict(features_test)
                if two_variables:
                    # esto es solo para tener un label aunque esté malo
                    labels_test = pd.Series(new_test_set['initial_radius'])
                else:
                    labels_test = pd.Series(new_test_set[variable_to_vary])
            else:
                # Predictions on the testing data
                test_predictions = random_forest.predict(features_test)


            column_label_name_test = pd.Series([label], name='Label name')
            column_label_test = pd.Series([labels_test.to_list()], name='Label')
            column_predicted_test = pd.Series([list(test_predictions)], name='Predicted')
            column_mae_test = pd.Series([mean_absolute_error(labels_test, test_predictions)], name='MAE')
            column_set_test = pd.Series(['Test'], name='Set')
            column_feature_importance_test = pd.Series([feature_importances], name='Feature importance')

            current_results_test_df = pd.concat([column_label_name_test, column_label_test, column_predicted_test,
                                                 column_mae_test, column_set_test,
                                                 column_feature_importance_test], axis=1)

            # Create DataFrame for train results
            column_label_name_train = pd.Series([label], name='Label name')
            column_label_train = pd.Series([labels_train.to_list()], name='Label')
            column_predicted_train = pd.Series([list(train_predictions)], name='Predicted')
            column_mae_train = pd.Series([mean_absolute_error(train_predictions, labels_train)], name='MAE')
            column_set_train = pd.Series(['Train'], name='Set')
            column_feature_importance_train = pd.Series([feature_importances], name='Feature importance')

            current_results_train_df = pd.concat([column_label_name_train, column_label_train, column_predicted_train,
                                                  column_mae_train, column_set_train,
                                                  column_feature_importance_train], axis=1)

            # Concatenate train and test DataFrames along the rows
            results_df = pd.concat([results_df, current_results_test_df], ignore_index=True)
            results_df = pd.concat([results_df, current_results_train_df], ignore_index=True)

        train_set = results_df[results_df['Set'] == 'Train']
        test_set = results_df[results_df['Set'] == 'Test']

        mae_train = np.mean(train_set['MAE'])
        mae_test = np.mean(test_set['MAE'])

        return results_df, mae_train, mae_test

    def multi_objective_model(self, dataset, n_trees, generalization=False, predict_new=False, variable_to_vary=None):
        # create labels and features
        list_labels = ['RP_dynamical_threshold', 'RP_mach_number', 'RP_acoustic_emissions', 'RP_expansion_radius',
                       'KM_dynamical_threshold', 'KM_mach_number', 'KM_acoustic_emissions', 'KM_expansion_radius',
                       'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']
        list_features = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                         'surface_tension', 'sound_velocity', 'vapor_pressure', 'count']

        if generalization:
            train_set, test_set = self.generalization_dataset(dataset, 1)
            features_train = train_set[list_features]
            labels_train = train_set[list_labels]

            features_test = test_set[list_features]
            labels_test = test_set[list_labels]
        else:
            features = dataset[list_features]
            labels = dataset[list_labels]

            # Split data into train and test sets
            features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                        random_state=42)

        features_train = features_train.drop('count', axis=1)
        features_test = features_test.drop('count', axis=1)

        # Train the model
        random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        random_forest.fit(features_train, labels_train)

        feature_importances = random_forest.feature_importances_

        # Predictions on the training data
        train_predictions = random_forest.predict(features_train)

        if predict_new and variable_to_vary is not None:
            # Create a new test set based on the specified variable
            new_test_set = self.create_new_test_set(variable_to_vary)
            features_test = new_test_set[['initial_radius', 'acoustic_pressure', 'frequency', 'temperature',
                                          'density', 'viscosity', 'surface_tension', 'sound_velocity',
                                          'vapor_pressure']]
            test_predictions = random_forest.predict(features_test)
            labels_test = pd.Series(new_test_set[variable_to_vary])
        else:
            # Predictions on the testing data
            test_predictions = random_forest.predict(features_test)


        return labels_train, train_predictions, labels_test, test_predictions, list_labels

    def expansion_model(self, dataset, n_trees, generalization=False, predict_new=False, variable_to_vary=None):
        columns_to_keep = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                           'surface_tension', 'sound_velocity', 'vapor_pressure']

        # List of columns to melt
        columns_to_melt = ['RP_dynamical_threshold', 'RP_acoustic_emissions', 'RP_mach_number', 'RP_expansion_radius',
                           'KM_dynamical_threshold', 'KM_acoustic_emissions', 'KM_mach_number', 'KM_expansion_radius',
                           'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']
        if generalization:
            train_set, test_set = self.generalization_dataset(dataset, 1)

            melted_train = pd.melt(train_set, id_vars=columns_to_keep, value_vars=columns_to_melt, var_name='category',
                                   value_name='value')
            melted_train['equation'] = melted_train['category'].str.split('_').str[0]
            melted_train['threshold'] = melted_train['category'].str.split('_').str[1]

            melted_test = pd.melt(test_set, id_vars=columns_to_keep, value_vars=columns_to_melt, var_name='category',
                                  value_name='value')
            melted_test['equation'] = melted_test['category'].str.split('_').str[0]
            melted_test['threshold'] = melted_test['category'].str.split('_').str[1]

            features_train = melted_train[columns_to_keep]
            labels_train = melted_train[['value']]
            features_test = melted_test[columns_to_keep]
            labels_test = melted_test[['value']]

        else:
            # Melt the dataframe
            melted_df = pd.melt(dataset, id_vars=columns_to_keep, value_vars=columns_to_melt, var_name='category',
                                value_name='value')

            # Extract equation and threshold from the melted column names
            melted_df['equation'] = melted_df['category'].str.split('_').str[0]
            melted_df['threshold'] = melted_df['category'].str.split('_').str[1]

            # define features and  labels
            features = melted_df[columns_to_keep]
            labels = melted_df[['value']]

            features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                        test_size=0.2,
                                                                                        random_state=42)
        # Train the model
        random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        random_forest.fit(features_train, labels_train)

        feature_importances = random_forest.feature_importances_

        # Predictions on the training data
        train_predictions = random_forest.predict(features_train)

        if predict_new and variable_to_vary is not None:
            # Create a new test set based on the specified variable
            new_test_set = self.create_new_test_set(variable_to_vary)
            features_test = new_test_set[['initial_radius', 'acoustic_pressure', 'frequency', 'temperature',
                                          'density', 'viscosity', 'surface_tension', 'sound_velocity',
                                          'vapor_pressure']]
            test_predictions = random_forest.predict(features_test)
            labels_test = pd.Series(new_test_set[variable_to_vary])
        else:
            # Predictions on the testing data
            test_predictions = random_forest.predict(features_test)

        column_label_name_test = pd.Series(['Value'], name='Label name')
        column_label_test = pd.Series([labels_test['value'].tolist()], name='Label')
        column_predicted_test = pd.Series([test_predictions], name='Predicted')
        column_mae_test = pd.Series([mean_absolute_error(labels_test, test_predictions)], name='MAE')
        column_set_test = pd.Series(['Test'], name='Set')
        column_feature_importance_test = pd.Series([feature_importances], name='Feature importance')

        current_results_test_df = pd.concat([column_label_name_test, column_label_test, column_predicted_test,
                                             column_mae_test, column_set_test,
                                             column_feature_importance_test], axis=1)

        # Create DataFrame for train results
        column_label_name_train = pd.Series(['Value'], name='Label name')
        column_label_train = pd.Series([labels_train['value'].tolist()], name='Label')
        column_predicted_train = pd.Series([train_predictions], name='Predicted')
        column_mae_train = pd.Series([mean_absolute_error(train_predictions, labels_train)], name='MAE')
        column_set_train = pd.Series(['Train'], name='Set')
        column_feature_importance_train = pd.Series([feature_importances], name='Feature importance')

        current_results_train_df = pd.concat([column_label_name_train, column_label_train, column_predicted_train,
                                              column_mae_train, column_set_train,
                                              column_feature_importance_train], axis=1)
        result = pd.concat([current_results_train_df, current_results_test_df])

        return result

    def probabilistic_model(self, dataset, n_trees, generalization=False, predict_new=False, variable_to_vary=None, two_variables=False):
        # define features and  labels
        features_list = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                         'surface_tension', 'sound_velocity', 'vapor_pressure']

        if generalization:
            train_set, test_set = self.generalization_dataset(dataset, 1)
            features_train = train_set[features_list]
            labels_train = train_set[['count']] / 12

            features_test = test_set[features_list]
            labels_test = test_set[['count']] / 12
        else:
            features = dataset[features_list]
            labels = dataset[['count']] / 12

            features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                        random_state=42)
        # Train the model
        random_forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        random_forest.fit(features_train, labels_train)

        feature_importances = random_forest.feature_importances_

        # Predictions on the training data
        train_predictions = random_forest.predict(features_train)

        if predict_new and variable_to_vary is not None:
            # Create a new test set based on the specified variable
            if os.path.exists('../Data/proportion_plot_general.csv'):
                new_test_set = pd.read_csv('../Data/proportion_plot_general.csv')
            else:
                new_test_set = self.create_new_test_set(variable_to_vary, two_variables)
            features_test = new_test_set[['initial_radius', 'acoustic_pressure', 'frequency', 'temperature',
                                          'density', 'viscosity', 'surface_tension', 'sound_velocity',
                                          'vapor_pressure']]
            test_predictions = random_forest.predict(features_test)
            if two_variables:
                labels_test_final = pd.Series(new_test_set['initial_radius'])
            else:
                labels_test_final = pd.Series(new_test_set[variable_to_vary])
        else:
            # Predictions on the testing data
            test_predictions = random_forest.predict(features_test)
            labels_test_final = labels_test['count'].tolist()

        column_label_name_test = pd.Series(['Value'], name='Label name')
        column_label_test = pd.Series([labels_test_final], name='Label')
        column_predicted_test = pd.Series([test_predictions], name='Predicted')
        column_mae_test = pd.Series([mean_absolute_error(labels_test_final, test_predictions)], name='MAE')
        column_set_test = pd.Series(['Test'], name='Set')
        column_feature_importance_test = pd.Series([feature_importances], name='Feature importance')

        current_results_test_df = pd.concat([column_label_name_test, column_label_test, column_predicted_test,
                                             column_mae_test, column_set_test,
                                             column_feature_importance_test], axis=1)

        # Create DataFrame for train results
        column_label_name_train = pd.Series(['Value'], name='Label name')
        column_label_train = pd.Series([labels_train['count'].tolist()], name='Label')
        column_predicted_train = pd.Series([train_predictions], name='Predicted')
        column_mae_train = pd.Series([mean_absolute_error(train_predictions, labels_train)], name='MAE')
        column_set_train = pd.Series(['Train'], name='Set')
        column_feature_importance_train = pd.Series([feature_importances], name='Feature importance')

        current_results_train_df = pd.concat([column_label_name_train, column_label_train, column_predicted_train,
                                              column_mae_train, column_set_train,
                                              column_feature_importance_train], axis=1)
        result = pd.concat([current_results_train_df, current_results_test_df])

        return result

    @staticmethod
    def ensembling_models(labels, predictions, model):
        if model == 'Ensemble':
            mean_prediction = np.mean(predictions, axis=0).tolist()
            mean_labels = np.mean(labels, axis=0).tolist()

            voting_sum = np.sum(predictions, axis=0)
            voting_labels = np.sum(labels, axis=0)

        elif model == 'Multi-objective':
            mean_prediction = np.mean(predictions, axis=1).tolist()
            mean_labels = np.mean(labels, axis=1).tolist()

            voting_sum = np.sum(predictions, axis=1)
            voting_labels = np.sum(labels, axis=1)


        else:
            raise ValueError('Model must be Ensemble or Multi-objective')

        return mean_prediction, mean_labels, voting_sum, voting_labels
