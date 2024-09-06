import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast

import warnings
warnings.filterwarnings("ignore")


def critical_maximum_radius(data):
    eq_1 = 3 * 1.33 / (2 * data['surface_tension'])
    eq_2 = 1e5 + (2 * data['surface_tension'] / data['initial_radius']) - data['vapor_pressure']
    eq_3 = data['initial_radius'] ** (3 * 1.33)
    data['critical_maximum'] = ((eq_1 * eq_2 * eq_3) ** (1 / (3 * 1.33 - 1))) * 1e6
    return data


def safe_literal_eval(val):
    if ('nan' in val) or ('inf' in val):
        return (np.nan, 1)
    else:
        return ast.literal_eval(val)


def define_threshold(kurtosis_threshold, crest_factor_threshold, type_of_data):
    original_dataset_raw = pd.read_csv(f'Data/cavitation_type_dataset.csv')
    original_dataset = critical_maximum_radius(original_dataset_raw)
    original_dataset.dropna(subset=['RP_mach_number', 'KM_mach_number', 'G_mach_number'], inplace=True)

    max_val = original_dataset['RP_max_velocity'][original_dataset['RP_max_velocity'] != np.inf].max()
    min_val = original_dataset['RP_max_velocity'][original_dataset['RP_max_velocity'] != -np.inf].min()
    original_dataset['RP_max_velocity'] = original_dataset['RP_max_velocity'].replace([np.inf, -np.inf],
                                                                                      [max_val, min_val])

    max_val = original_dataset['KM_max_velocity'][original_dataset['KM_max_velocity'] != np.inf].max()
    min_val = original_dataset['KM_max_velocity'][original_dataset['KM_max_velocity'] != -np.inf].min()
    original_dataset['KM_max_velocity'] = original_dataset['KM_max_velocity'].replace([np.inf, -np.inf],
                                                                                      [max_val, min_val])

    max_val = original_dataset['G_max_velocity'][original_dataset['G_max_velocity'] != np.inf].max()
    min_val = original_dataset['G_max_velocity'][original_dataset['G_max_velocity'] != -np.inf].min()
    original_dataset['G_max_velocity'] = original_dataset['G_max_velocity'].replace([np.inf, -np.inf],
                                                                                    [max_val, min_val])

    if type_of_data == 1:
        original_dataset['RP_expansion'] = np.where(
            original_dataset['RP_expansion_radius'] > original_dataset['critical_maximum'], False, True)
        original_dataset['KM_expansion'] = np.where(
            original_dataset['KM_expansion_radius'] > original_dataset['critical_maximum'], False, True)
        original_dataset['G_expansion'] = np.where(
            original_dataset['G_expansion_radius'] > original_dataset['critical_maximum'], False, True)

    elif type_of_data == 2:
        original_dataset['RP_expansion_radius'] = original_dataset['RP_expansion_radius'].apply(safe_literal_eval)

        original_dataset['KM_expansion_radius'] = original_dataset['KM_expansion_radius'].apply(safe_literal_eval)
        original_dataset['G_expansion_radius'] = original_dataset['G_expansion_radius'].apply(safe_literal_eval)

        expansion_rp = original_dataset['RP_expansion_radius'].apply(
            lambda x: False if np.isnan(x[0]) or np.isnan(x[1]) else (x[0] > x[1]))
        original_dataset['RP_expansion'] = np.where(expansion_rp, False, True)

        expansion_km = original_dataset['KM_expansion_radius'].apply(
            lambda x: False if np.isnan(x[0]) or np.isnan(x[1]) else (x[0] > x[1]))
        original_dataset['KM_expansion'] = np.where(expansion_km, False, True)

        expansion_g = original_dataset['G_expansion_radius'].apply(
            lambda x: False if np.isnan(x[0]) or np.isnan(x[1]) else (x[0] > x[1]))
        original_dataset['G_expansion'] = np.where(expansion_g, False, True)
    else:
        raise

    # threshold for acoustic emissions
    original_dataset['RP_acoustic_emissions'] = original_dataset['RP_acoustic_emissions'].apply(safe_literal_eval)
    original_dataset['KM_acoustic_emissions'] = original_dataset['KM_acoustic_emissions'].apply(safe_literal_eval)
    original_dataset['G_acoustic_emissions'] = original_dataset['G_acoustic_emissions'].apply(safe_literal_eval)

    condition_rp = original_dataset['RP_acoustic_emissions'].apply(
        lambda x: False if np.isnan(x[0]) or np.isnan(x[1]) or np.isinf(x[0]) or np.isinf(x[1]) else (
                x[0] > kurtosis_threshold and x[1] > crest_factor_threshold))
    original_dataset['RP_kurtosis_crest_factor'] = np.where(condition_rp, False, True)

    condition_km = original_dataset['KM_acoustic_emissions'].apply(
        lambda x: False if np.isnan(x[0]) or np.isnan(x[1]) or np.isinf(x[0]) or np.isinf(x[1]) else (
                x[0] > kurtosis_threshold and x[1] > crest_factor_threshold))
    original_dataset['KM_kurtosis_crest_factor'] = np.where(condition_km, False, True)

    condition_g = original_dataset['G_acoustic_emissions'].apply(
        lambda x: False if np.isnan(x[0]) or np.isnan(x[1]) or np.isinf(x[0]) or np.isinf(x[1]) else (
                x[0] > kurtosis_threshold and x[1] > crest_factor_threshold))
    original_dataset['G_kurtosis_crest_factor'] = np.where(condition_g, False, True)

    # drop the columns with the values and conserve with the classifiers
    original_dataset.drop(['RP_expansion_radius', 'KM_expansion_radius', 'G_expansion_radius',
                           'RP_acoustic_emissions', 'KM_acoustic_emissions', 'G_acoustic_emissions'], axis=1,
                          inplace=True)

    original_dataset.rename(columns={'RP_expansion': 'RP_expansion_radius',
                                     'KM_expansion': 'KM_expansion_radius',
                                     'G_expansion': 'G_expansion_radius',
                                     'RP_kurtosis_crest_factor': 'RP_acoustic_emissions',
                                     'KM_kurtosis_crest_factor': 'KM_acoustic_emissions',
                                     'G_kurtosis_crest_factor': 'G_acoustic_emissions'}, inplace=True)

    usable_data = original_dataset[['RP_dynamical_threshold', 'RP_mach_number',
                                    'KM_dynamical_threshold', 'KM_mach_number', 'G_dynamical_threshold',
                                    'G_mach_number', 'RP_expansion_radius', 'KM_expansion_radius',
                                    'G_expansion_radius', 'RP_acoustic_emissions', 'KM_acoustic_emissions',
                                    'G_acoustic_emissions']]

    for col in usable_data.columns:
        original_dataset[col] = original_dataset[col].astype(int)

    original_dataset['count'] = usable_data.apply(lambda row: row.sum(), axis=1)

    return original_dataset


def separate_dataset(data, type):
    data['RP_acoustic_emissions'] = data['RP_acoustic_emissions'].apply(safe_literal_eval)
    data['KM_acoustic_emissions'] = data['KM_acoustic_emissions'].apply(safe_literal_eval)
    data['G_acoustic_emissions'] = data['G_acoustic_emissions'].apply(safe_literal_eval)

    # Extraer las columnas de kurtosis y crest factor para RP
    data[['RP_kurtosis', 'RP_crest_factor']] = pd.DataFrame(data['RP_acoustic_emissions'].tolist(), index=data.index)
    # Extraer las columnas de kurtosis y crest factor para KM
    data[['KM_kurtosis', 'KM_crest_factor']] = pd.DataFrame(data['KM_acoustic_emissions'].tolist(), index=data.index)
    # Extraer las columnas de kurtosis y crest factor para G
    data[['G_kurtosis', 'G_crest_factor']] = pd.DataFrame(data['G_acoustic_emissions'].tolist(), index=data.index)

    # Reemplazar inf y -inf en RP_kurtosis y RP_crest_factor
    data['RP_kurtosis'].replace([np.inf, -np.inf],
                                [data[data['RP_kurtosis'] != np.inf]['RP_kurtosis'].max(),
                                 data[data['RP_kurtosis'] != -np.inf]['RP_kurtosis'].min()],
                                inplace=True)

    data['RP_crest_factor'].replace([np.inf, -np.inf],
                                    [data[data['RP_crest_factor'] != np.inf]['RP_crest_factor'].max(),
                                     data[data['RP_crest_factor'] != -np.inf]['RP_crest_factor'].min()],
                                    inplace=True)

    # Reemplazar inf y -inf en KM_kurtosis y KM_crest_factor
    data['KM_kurtosis'].replace([np.inf, -np.inf],
                                [data[data['KM_kurtosis'] != np.inf]['KM_kurtosis'].max(),
                                 data[data['KM_kurtosis'] != -np.inf]['KM_kurtosis'].min()],
                                inplace=True)

    data['KM_crest_factor'].replace([np.inf, -np.inf],
                                    [data[data['KM_crest_factor'] != np.inf]['KM_crest_factor'].max(),
                                     data[data['KM_crest_factor'] != -np.inf]['KM_crest_factor'].min()],
                                    inplace=True)

    # Reemplazar inf y -inf en G_kurtosis y G_crest_factor
    data['G_kurtosis'].replace([np.inf, -np.inf],
                               [data[data['G_kurtosis'] != np.inf]['G_kurtosis'].max(),
                                data[data['G_kurtosis'] != -np.inf]['G_kurtosis'].min()],
                               inplace=True)

    data['G_crest_factor'].replace([np.inf, -np.inf],
                                   [data[data['G_crest_factor'] != np.inf]['G_crest_factor'].max(),
                                    data[data['G_crest_factor'] != -np.inf]['G_crest_factor'].min()],
                                   inplace=True)

    # Eliminar las columnas originales de emisiones acústicas
    data.drop(columns=['RP_acoustic_emissions', 'KM_acoustic_emissions', 'G_acoustic_emissions'], inplace=True)

    if (type == 2) or (type == 3):
        max_val = data['RP_max_velocity'][data['RP_max_velocity'] != np.inf].max()
        min_val = data['RP_max_velocity'][data['RP_max_velocity'] != -np.inf].min()
        data['RP_max_velocity'] = data['RP_max_velocity'].replace([np.inf, -np.inf],
                                                                  [max_val, min_val])

        max_val = data['KM_max_velocity'][data['KM_max_velocity'] != np.inf].max()
        min_val = data['KM_max_velocity'][data['KM_max_velocity'] != -np.inf].min()
        data['KM_max_velocity'] = data['KM_max_velocity'].replace([np.inf, -np.inf],
                                                                  [max_val, min_val])

        max_val = data['G_max_velocity'][data['G_max_velocity'] != np.inf].max()
        min_val = data['G_max_velocity'][data['G_max_velocity'] != -np.inf].min()
        data['G_max_velocity'] = data['G_max_velocity'].replace([np.inf, -np.inf],
                                                                [max_val, min_val])

        max_val = data['G_max_velocity'][data['G_max_velocity'] != np.inf].max()
        min_val = data['G_max_velocity'][data['G_max_velocity'] != -np.inf].min()
        data['G_max_velocity'] = data['G_max_velocity'].replace([np.inf, -np.inf],
                                                                [max_val, min_val])

        data['RP_expansion_radius'] = data['RP_expansion_radius'].apply(safe_literal_eval)
        data['KM_expansion_radius'] = data['KM_expansion_radius'].apply(safe_literal_eval)
        data['G_expansion_radius'] = data['G_expansion_radius'].apply(safe_literal_eval)

        data[['RP_radius', 'RP_radius_threshold']] = pd.DataFrame(data['RP_expansion_radius'].tolist(),
                                                                  index=data.index)
        data.drop(columns=['RP_expansion_radius'], inplace=True)

        data[['KM_radius', 'KM_radius_threshold']] = pd.DataFrame(data['KM_expansion_radius'].tolist(),
                                                                  index=data.index)
        data.drop(columns=['KM_expansion_radius'], inplace=True)

        data[['G_radius', 'G_radius_threshold']] = pd.DataFrame(data['G_expansion_radius'].tolist(), index=data.index)
        data.drop(columns=['G_expansion_radius'], inplace=True)
        if type == 3:
            data['RP_critical_transition'] = data['RP_critical_transition'].apply(safe_literal_eval)
            data['KM_critical_transition'] = data['KM_critical_transition'].apply(safe_literal_eval)
            data['G_critical_transition'] = data['G_critical_transition'].apply(safe_literal_eval)

            data[['RP_critical', 'RP_transition']] = pd.DataFrame(data['RP_critical_transition'].tolist(),
                                                                  index=data.index)
            data.drop(columns=['RP_critical_transition'], inplace=True)

            data[['KM_critical', 'KM_transition']] = pd.DataFrame(data['KM_critical_transition'].tolist(),
                                                                  index=data.index)
            data.drop(columns=['KM_critical_transition'], inplace=True)

            data[['G_critical', 'G_transition']] = pd.DataFrame(data['G_critical_transition'].tolist(),
                                                                index=data.index)
            data.drop(columns=['G_critical_transition'], inplace=True)

        data.rename(columns={'RP_max_velocity': 'RP_velocity',
                             'KM_max_velocity': 'KM_velocity',
                             'G_max_velocity': 'G_velocity'}, inplace=True)

        columns_to_melt = ['RP_dynamical_threshold', 'RP_kurtosis', 'RP_crest_factor', 'RP_mach_number', 'RP_radius',
                           'RP_velocity', 'KM_dynamical_threshold', 'RP_critical', 'RP_transition', 'KM_kurtosis',
                           'KM_crest_factor', 'KM_mach_number', 'KM_radius', 'KM_velocity', 'KM_critical',
                           'KM_transition',
                           'G_dynamical_threshold', 'G_kurtosis', 'G_crest_factor', 'G_mach_number', 'G_radius',
                           'G_velocity', 'G_critical', 'G_transition']
    elif type == 1:
        data.rename(columns={'RP_expansion_radius': 'RP_radius',
                             'KM_expansion_radius': 'KM_radius',
                             'G_expansion_radius': 'G_radius'}, inplace=True)

        columns_to_melt = ['RP_dynamical_threshold', 'RP_kurtosis', 'RP_crest_factor', 'RP_mach_number', 'RP_radius',
                           'KM_dynamical_threshold', 'KM_kurtosis', 'KM_crest_factor', 'KM_mach_number', 'KM_radius',
                           'G_dynamical_threshold', 'G_kurtosis', 'G_crest_factor', 'G_mach_number', 'G_radius']

    columns_to_keep = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                       'surface_tension', 'sound_velocity', 'vapor_pressure']

    melted_df = pd.melt(data, id_vars=columns_to_keep, value_vars=columns_to_melt, var_name='category',
                        value_name='value')

    melted_df['equation'] = melted_df['category'].str.split('_').str[0]
    melted_df['threshold'] = melted_df['category'].str.split('_').str[1]
    grouped_threshold = melted_df.groupby(columns_to_keep + ['threshold'])['value'].sum().reset_index()

    return grouped_threshold


def histogram_parameter(dataset, column, threshold=None):
    if column in ['expansion_radius', 'kurtosis', 'crest_factor', 'critical_radius', 'transition_radius',
                  'max_velocity']:
        dataset.rename(columns={'value': column}, inplace=True)
        sns.histplot(data=dataset, x=dataset[column], kde=False, discrete=False, log_scale=(True, False))
        if threshold is not None:
            plt.axvline(x=threshold, color='r', linestyle='--', linewidth=2)
        else:
            # si no hay threshold  calcular minimo y maximo threshold
            pass
        plt.xlabel(f"Distribution of {column.replace('_', ' ')} values")
    else:
        sns.histplot(data=dataset, x=dataset[column], kde=False, discrete=True)
        plt.xlabel('Number of experiments classified as stable cavitation')

    plt.savefig(f'figures/histogram_{column}.pdf')
    plt.close()


def pair_plot(data, dataset):
    util_data = data[['initial_radius', 'acoustic_pressure', 'frequency', 'temperature']]
    sns.pairplot(util_data, diag_kind='hist', kind='hist')
    plt.savefig(f'figures/pair_plot_density_{dataset}.pdf')
    plt.close()


def violin_plot(data, threshold):
    initial_parameters = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature']

    for parameter in initial_parameters:
        data_long = pd.melt(data, id_vars=[parameter],
                            value_vars=[f'RP_{threshold}', f'KM_{threshold}', f'G_{threshold}'],
                            var_name='Threshold',
                            value_name='Stable')
        data_long['Stable'] = data_long['Stable'].astype(bool)

        plt.figure(figsize=(10, 6))
        sns.violinplot(x=parameter, y='Threshold', hue='Stable', data=data_long, split=True, inner='stick')
        plt.title(f"Violin plot for {threshold.replace('_', ' ')} threshold")
        plt.xlabel(f"Distribution of {parameter.replace('_', ' ')}")
        plt.yticks([0, 1, 2], ['Rayleigh-Plesset', 'Keller-Miksis', 'Gilmore'])
        plt.savefig(f'../figures/violin_plot_{parameter}_{threshold}.pdf')
        plt.close()


def general_violin_plot(data):
    initial_parameters = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature']

    for parameter in initial_parameters:
        data_long = pd.melt(data, id_vars=[parameter],
                            value_vars=['RP_dynamical_threshold', 'KM_dynamical_threshold', 'G_dynamical_threshold',
                                        'RP_mach_number', 'KM_mach_number', 'G_mach_number',
                                        'RP_expansion_radius', 'KM_expansion_radius', 'G_expansion_radius',
                                        'RP_acoustic_emissions', 'KM_acoustic_emissions', 'G_acoustic_emissions'],
                            var_name='Threshold',
                            value_name='Stable')
        data_long['Stable'] = data_long['Stable'].astype(bool)

        plt.figure(figsize=(12, 24))
        sns.violinplot(x=parameter, y='Threshold', hue='Stable', data=data_long, split=True, inner='stick')
        # plt.title(f'Violin plot for {threshold} threshold in {dataset} dataset')
        plt.xlabel(f"Distribution of {parameter.replace('_', ' ')}")
        plt.yticks(rotation=45)

        # plt.yticks([0, 1, 2], ['Rayleigh-Plesset', 'Keller-Miksis', 'Gilmore'])
        plt.savefig(f'../figures/general_violin_plot_{parameter}.pdf')
        plt.close()


def distribution_data(data):
    columns_to_keep = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature', 'density', 'viscosity',
                       'surface_tension', 'sound_velocity', 'vapor_pressure']

    # List of columns to melt
    columns_to_melt = ['RP_dynamical_threshold', 'RP_acoustic_emissions', 'RP_mach_number', 'RP_expansion_radius',
                       'KM_dynamical_threshold', 'KM_acoustic_emissions', 'KM_mach_number', 'KM_expansion_radius',
                       'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']

    # Melt the dataframe
    melted_df = pd.melt(data, id_vars=columns_to_keep, value_vars=columns_to_melt, var_name='category',
                        value_name='value')

    # Extract equation and threshold from the melted column names
    melted_df['equation'] = melted_df['category'].str.split('_').str[0]
    melted_df['threshold'] = melted_df['category'].str.split('_').str[1]

    grouped_threshold = melted_df.groupby(columns_to_keep + ['threshold'])['value'].sum().reset_index()
    grouped_equation = melted_df.groupby(columns_to_keep + ['equation'])['value'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=grouped_threshold, x='value', hue='threshold', multiple='dodge', discrete=True, shrink=0.8)
    plt.xticks([0, 1, 2, 3])
    plt.xlabel('Number of equations classifying cavitation as stable', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'../figures/threshold_distribution.pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=grouped_equation, x='value', hue='equation', multiple='dodge', discrete=True, shrink=0.8)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xlabel('Number of classifiers classifying cavitation as stable', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'../figures/equation_distribution.pdf')
    plt.close()
