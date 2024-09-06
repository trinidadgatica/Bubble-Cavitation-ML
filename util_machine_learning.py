from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def ensemble_design_cross_validation(dataset, n_trees):
    results_df = pd.DataFrame()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Definir etiquetas y características
    list_labels = ['RP_dynamical_threshold', 'RP_mach_number', 'RP_acoustic_emissions', 'RP_expansion_radius',
                   'KM_dynamical_threshold', 'KM_mach_number', 'KM_acoustic_emissions', 'KM_expansion_radius',
                   'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']
    list_features = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                     'surface_tension', 'sound_velocity', 'vapor_pressure']

    fold_idx = 1

    for train_index, test_index in kf.split(dataset):
        # Dividir el conjunto de datos en entrenamiento y prueba
        train_data = dataset.iloc[train_index]
        test_data = dataset.iloc[test_index]

        for label in list_labels:
            labels_train = train_data[label]
            labels_test = test_data[label]
            features_train = train_data[list_features]
            features_test = test_data[list_features]

            # Entrenar el modelo
            random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
            random_forest.fit(features_train, labels_train)

            feature_importances = random_forest.feature_importances_

            # Predicciones en los datos de entrenamiento y prueba
            train_predictions = random_forest.predict(features_train)
            test_predictions = random_forest.predict(features_test)

            # Crear DataFrame para resultados de prueba
            current_results_test_df = pd.DataFrame({
                'Fold': [fold_idx],
                'Label name': [label],
                'Label': [labels_test.to_list()],
                'Predicted': [list(test_predictions)],
                'Set': ['Test'],
                'Feature importance': [feature_importances]
            })

            # Crear DataFrame para resultados de entrenamiento
            current_results_train_df = pd.DataFrame({
                'Fold': [fold_idx],
                'Label name': [label],
                'Label': [labels_train.to_list()],
                'Predicted': [list(train_predictions)],
                'Set': ['Train'],
                'Feature importance': [feature_importances]
            })

            # Concatenar DataFrames de entrenamiento y prueba
            results_df = pd.concat([results_df, current_results_test_df], ignore_index=True)
            results_df = pd.concat([results_df, current_results_train_df], ignore_index=True)

        fold_idx += 1

    return results_df


def multi_objective_design_cross_validation(dataset, n_trees):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Definir etiquetas y características
    list_labels = ['RP_dynamical_threshold', 'RP_mach_number', 'RP_acoustic_emissions', 'RP_expansion_radius',
                   'KM_dynamical_threshold', 'KM_mach_number', 'KM_acoustic_emissions', 'KM_expansion_radius',
                   'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']
    list_features = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                     'surface_tension', 'sound_velocity', 'vapor_pressure']

    fold_results = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(dataset), 1):
        # Dividir el conjunto de datos en entrenamiento y prueba para este pliegue
        train_data = dataset.iloc[train_index]
        test_data = dataset.iloc[test_index]

        features_train = train_data[list_features]
        labels_train = train_data[list_labels]
        features_test = test_data[list_features]
        labels_test = test_data[list_labels]

        # Entrenar el modelo
        random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        random_forest.fit(features_train, labels_train)

        # Predicciones en los datos de entrenamiento y prueba
        train_predictions = random_forest.predict(features_train)
        test_predictions = random_forest.predict(features_test)

        # Almacenar resultados de este pliegue
        fold_results.append({
            'Fold': fold_idx,
            'Labels_train': labels_train,
            'Train_predictions': train_predictions,
            'Labels_test': labels_test,
            'Test_predictions': test_predictions
        })

    return fold_results, list_labels


def expansion_design_cross_validation(dataset, n_trees):
    columns_to_keep = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                       'surface_tension', 'sound_velocity', 'vapor_pressure']

    # List of columns to melt
    columns_to_melt = ['RP_dynamical_threshold', 'RP_acoustic_emissions', 'RP_mach_number', 'RP_expansion_radius',
                       'KM_dynamical_threshold', 'KM_acoustic_emissions', 'KM_mach_number', 'KM_expansion_radius',
                       'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']

    # Melt the dataframe
    melted_df = pd.melt(dataset, id_vars=columns_to_keep, value_vars=columns_to_melt, var_name='category',
                        value_name='value')

    # Extract equation and threshold from the melted column names
    melted_df['equation'] = melted_df['category'].str.split('_').str[0]
    melted_df['threshold'] = melted_df['category'].str.split('_').str[1]

    # Define features and labels
    features = melted_df[columns_to_keep]
    labels = melted_df[['value']]

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(melted_df), 1):
        # Split the data into training and testing sets
        train_data = melted_df.iloc[train_index]
        test_data = melted_df.iloc[test_index]

        features_train = train_data[columns_to_keep]
        labels_train = train_data[['value']]
        features_test = test_data[columns_to_keep]
        labels_test = test_data[['value']]

        # Train the model
        random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        random_forest.fit(features_train, labels_train.values.ravel())

        feature_importances = random_forest.feature_importances_

        # Predictions on the training data
        train_predictions = random_forest.predict(features_train)
        test_predictions = random_forest.predict(features_test)

        # Create DataFrame for test results
        current_results_test_df = pd.DataFrame({
            'Fold': [fold_idx],
            'Label name': ['Value'],
            'Label': [labels_test['value'].tolist()],
            'Predicted': [test_predictions.tolist()],
            'Set': ['Test'],
            'Feature importance': [feature_importances]
        })

        # Create DataFrame for train results
        current_results_train_df = pd.DataFrame({
            'Fold': [fold_idx],
            'Label name': ['Value'],
            'Label': [labels_train['value'].tolist()],
            'Predicted': [train_predictions.tolist()],
            'Set': ['Train'],
            'Feature importance': [feature_importances]
        })

        # Concatenate train and test DataFrames along the rows
        results.append(current_results_train_df)
        results.append(current_results_test_df)

    result = pd.concat(results, ignore_index=True)

    return result


def likelihood_design_cross_validation(dataset, n_trees):
    # Define features and labels
    features_list = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                     'surface_tension', 'sound_velocity', 'vapor_pressure']

    # Features y Labels
    features = dataset[features_list]
    labels = dataset[['count']] / 12

    # Inicializar KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(dataset), 1):
        # Dividir el conjunto de datos en entrenamiento y prueba
        train_data = dataset.iloc[train_index]
        test_data = dataset.iloc[test_index]

        features_train = train_data[features_list]
        labels_train = train_data[['count']] / 12
        features_test = test_data[features_list]
        labels_test = test_data[['count']] / 12

        # Entrenar el modelo
        random_forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        random_forest.fit(features_train, labels_train.values.ravel())

        feature_importances = random_forest.feature_importances_

        # Predicciones en los datos de entrenamiento y prueba
        train_predictions = random_forest.predict(features_train)
        test_predictions = random_forest.predict(features_test)

        # Crear DataFrame para resultados de prueba
        current_results_test_df = pd.DataFrame({
            'Fold': [fold_idx],
            'Label name': ['Value'],
            'Label': [labels_test['count'].tolist()],
            'Predicted': [test_predictions.tolist()],
            'Set': ['Test'],
            'Feature importance': [feature_importances]
        })

        # Crear DataFrame para resultados de entrenamiento
        current_results_train_df = pd.DataFrame({
            'Fold': [fold_idx],
            'Label name': ['Value'],
            'Label': [labels_train['count'].tolist()],
            'Predicted': [train_predictions.tolist()],
            'Set': ['Train'],
            'Feature importance': [feature_importances]
        })

        # Almacenar resultados de entrenamiento y prueba
        results.append(current_results_train_df)
        results.append(current_results_test_df)

    # Concatenar todos los resultados
    result = pd.concat(results, ignore_index=True)

    return result


def ensemble_design_generalization(dataset, n_trees):
    results_df = pd.DataFrame()
    # create labels and features
    list_labels = ['RP_dynamical_threshold', 'RP_mach_number', 'RP_acoustic_emissions', 'RP_expansion_radius',
                   'KM_dynamical_threshold', 'KM_mach_number', 'KM_acoustic_emissions', 'KM_expansion_radius',
                   'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']
    list_features = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                     'surface_tension', 'sound_velocity', 'vapor_pressure']

    # Determine the 80% threshold for initial_radius
    threshold = dataset['initial_radius'].quantile(0.8)

    # Split the dataset based on the threshold
    train_data = dataset[dataset['initial_radius'] <= threshold]
    test_data = dataset[dataset['initial_radius'] > threshold]

    for label in list_labels:
        labels_train = train_data[label]
        labels_test = test_data[label]
        features_train = train_data[list_features]
        features_test = test_data[list_features]

        # Train the model
        random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        random_forest.fit(features_train, labels_train)

        feature_importances = random_forest.feature_importances_

        # Predictions on the training data
        train_predictions = random_forest.predict(features_train)
        test_predictions = random_forest.predict(features_test)

        # Create DataFrame for test results
        current_results_test_df = pd.DataFrame({
            'Label name': [label],
            'Label': [labels_test.to_list()],
            'Predicted': [list(test_predictions)],
            'Set': ['Test'],
            'Feature importance': [feature_importances]
        })

        # Create DataFrame for train results
        current_results_train_df = pd.DataFrame({
            'Label name': [label],
            'Label': [labels_train.to_list()],
            'Predicted': [list(train_predictions)],
            'Set': ['Train'],
            'Feature importance': [feature_importances]
        })

        # Concatenate train and test DataFrames along the rows
        results_df = pd.concat([results_df, current_results_test_df], ignore_index=True)
        results_df = pd.concat([results_df, current_results_train_df], ignore_index=True)

    return results_df


def multi_objective_design_generalization(dataset, n_trees):
    # create labels and features
    list_labels = ['RP_dynamical_threshold', 'RP_mach_number', 'RP_acoustic_emissions', 'RP_expansion_radius',
                   'KM_dynamical_threshold', 'KM_mach_number', 'KM_acoustic_emissions', 'KM_expansion_radius',
                   'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']
    list_features = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                     'surface_tension', 'sound_velocity', 'vapor_pressure']

    threshold = dataset['initial_radius'].quantile(0.8)

    # Split the dataset based on the threshold
    train_data = dataset[dataset['initial_radius'] <= threshold]
    test_data = dataset[dataset['initial_radius'] > threshold]

    features_train = train_data[list_features]
    labels_train = train_data[list_labels]

    features_test = test_data[list_features]
    labels_test = test_data[list_labels]

    # Train the model
    random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    random_forest.fit(features_train, labels_train)

    # Predictions on the training data
    train_predictions = random_forest.predict(features_train)

    test_predictions = random_forest.predict(features_test)

    return labels_train, train_predictions, labels_test, test_predictions, list_labels


def expansion_design_generalization(dataset, n_trees):
    columns_to_keep = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                       'surface_tension', 'sound_velocity', 'vapor_pressure']

    # List of columns to melt
    columns_to_melt = ['RP_dynamical_threshold', 'RP_acoustic_emissions', 'RP_mach_number', 'RP_expansion_radius',
                       'KM_dynamical_threshold', 'KM_acoustic_emissions', 'KM_mach_number', 'KM_expansion_radius',
                       'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']

    melted_df = pd.melt(dataset, id_vars=columns_to_keep, value_vars=columns_to_melt, var_name='category',
                        value_name='value')

    # Extract equation and threshold from the melted column names
    melted_df['equation'] = melted_df['category'].str.split('_').str[0]
    melted_df['threshold'] = melted_df['category'].str.split('_').str[1]

    # Define features and labels
    features = melted_df[columns_to_keep]
    labels = melted_df[['value']]

    # Determine the 80% threshold for initial_radius
    threshold = melted_df['initial_radius'].quantile(0.8)

    # Split the dataset based on the threshold
    train_data = melted_df[melted_df['initial_radius'] <= threshold]
    test_data = melted_df[melted_df['initial_radius'] > threshold]

    features_train = train_data[columns_to_keep]
    labels_train = train_data[['value']]
    features_test = test_data[columns_to_keep]
    labels_test = test_data[['value']]

    # Train the model
    random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    random_forest.fit(features_train, labels_train.values.ravel())

    feature_importances = random_forest.feature_importances_

    # Predictions on the training data
    train_predictions = random_forest.predict(features_train)
    test_predictions = random_forest.predict(features_test)

    # Create DataFrame for test results
    current_results_test_df = pd.DataFrame({
        'Label name': ['Value'],
        'Label': [labels_test['value'].tolist()],
        'Predicted': [test_predictions],
        'Set': ['Test'],
        'Feature importance': [feature_importances]
    })

    # Create DataFrame for train results
    current_results_train_df = pd.DataFrame({
        'Label name': ['Value'],
        'Label': [labels_train['value'].tolist()],
        'Predicted': [train_predictions],
        'Set': ['Train'],
        'Feature importance': [feature_importances]
    })

    # Concatenate train and test DataFrames along the rows
    result = pd.concat([current_results_train_df, current_results_test_df], ignore_index=True)

    return result


def likelihood_design_generalization(dataset, n_trees):
    # Define features and labels
    features_list = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                     'surface_tension', 'sound_velocity', 'vapor_pressure']

    features = dataset[features_list]
    labels = dataset[['count']] / 12

    # Determine the 80% threshold for initial_radius
    threshold = dataset['initial_radius'].quantile(0.8)

    # Split the dataset based on the threshold
    train_data = dataset[dataset['initial_radius'] <= threshold]
    test_data = dataset[dataset['initial_radius'] > threshold]

    features_train = train_data[features_list]
    labels_train = train_data[['count']] / 12
    features_test = test_data[features_list]
    labels_test = test_data[['count']] / 12

    # Train the model
    random_forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    random_forest.fit(features_train, labels_train.values.ravel())

    feature_importances = random_forest.feature_importances_

    # Predictions on the training data
    train_predictions = random_forest.predict(features_train)
    test_predictions = random_forest.predict(features_test)

    labels_test_final = labels_test['count'].tolist()

    # Create DataFrame for test results
    current_results_test_df = pd.DataFrame({
        'Label name': ['Value'],
        'Label': [labels_test_final],
        'Predicted': [test_predictions.tolist()],
        'Set': ['Test'],
        'Feature importance': [feature_importances]
    })

    # Create DataFrame for train results
    current_results_train_df = pd.DataFrame({
        'Label name': ['Value'],
        'Label': [labels_train['count'].tolist()],
        'Predicted': [train_predictions.tolist()],
        'Set': ['Train'],
        'Feature importance': [feature_importances]
    })

    # Concatenate train and test DataFrames along the rows
    result = pd.concat([current_results_train_df, current_results_test_df], ignore_index=True)

    return result
