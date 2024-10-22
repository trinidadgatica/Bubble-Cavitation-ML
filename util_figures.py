f"""
Module Name: Utility Functions for Plotting Cavitation Data

Description:
This module contains utility functions to generate various plots related to cavitation data analysis.
These include time series plots, distribution plots, comparison plots between ensemble and multi-objective models,
likelihood plots, and heatmaps. The plots are configured with Seaborn settings to ensure consistency and clarity.

Functions:
- time_series_figure(data): Generates a time series plot for the given data.
- distribution_data(data): Creates distribution plots for thresholds and differential equations.
- comparison_ensemble_multi_objective(data): Compares ensemble and multi-objective models using bar plots.
- create_plot_ensemble(dataset, experiment_maker, variable_to_vary, number_trees, generalization): Generates ensemble model predictions.
- create_plot_multi_objective(dataset, experiment_maker, variable_to_vary, number_trees, generalization): Generates multi-objective model predictions.
- create_plot_likelihood(dataset, experiment_maker, variable_to_vary, number_trees, generalization): Generates likelihood model predictions.
- create_plot_expansion(dataset, experiment_maker, variable_to_vary, number_trees, generalization): Generates expansion model predictions.
- likelihood_figure(dataset, variable_to_vary, design): Creates a plot based on likelihood predictions.
- heat_map_figure(dataset): Generates a heatmap showing the relationship between initial radius and acoustic pressure.
"""

from machine_learning.machine_learning_maker import ExperimentMaker
from plot_information import *

from scipy.interpolate import griddata
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Set Seaborn style
sns.set(style="whitegrid")


def time_series_figure(data):
    """
    Generates a time series plot for the given data.

    Parameters:
    data (dict): Dictionary containing x and y axis data, labels, and plot settings.
    """
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    for i in range(len(data['x_axis_data'])):
        plt.plot(data['x_axis_data'][i], data['y_axis_data'][i], label=data['labels'][i],
                 linestyle='-', color=colors[i])

    # Set labels and font sizes
    plt.xlabel(data['x_label'], fontsize=X_LABEL_FONT_SIZE)
    plt.ylabel(data['y_label'], fontsize=Y_LABEL_FONT_SIZE)
    plt.xticks(fontsize=X_TICK_FONT_SIZE)
    plt.yticks(fontsize=Y_TICK_FONT_SIZE)

    # Set the legend with title, placing it outside the plot
    plt.legend(fontsize=LEGEND_FONT_SIZE, title=data['legend_title'],
               title_fontsize=LEGEND_TITLE_FONT_SIZE, loc=LEGEND_POSITION, bbox_to_anchor=LEGEND_SIZE)

    # Set the plot limits if specified
    if data.get('y_lim'):
        plt.ylim(data['y_lim'])
    if data.get('x_lim'):
        plt.xlim(data['x_lim'])

    # Save the plot if required
    if data['savefig']:
        plt.savefig(f"../figures/{data['name_fig']}", transparent=True, bbox_inches='tight')

    # Show the plot
    plt.show()


def distribution_data(data):
    """
    Creates distribution plots for thresholds and differential equations.

    Parameters:
    data (pd.DataFrame): DataFrame containing the relevant data for the plots.
    """
    # Columns to retain in the DataFrame
    columns_to_keep = ['initial_radius', 'acoustic_pressure', 'frequency', 'temperature',
                       'density', 'viscosity', 'surface_tension', 'sound_velocity', 'vapor_pressure']

    # Columns to melt for plotting
    columns_to_melt = ['RP_dynamical_threshold', 'RP_acoustic_emissions', 'RP_mach_number', 'RP_expansion_radius',
                       'KM_dynamical_threshold', 'KM_acoustic_emissions', 'KM_mach_number', 'KM_expansion_radius',
                       'G_dynamical_threshold', 'G_acoustic_emissions', 'G_mach_number', 'G_expansion_radius']

    melted_df = pd.melt(data, id_vars=columns_to_keep, value_vars=columns_to_melt,
                        var_name='category', value_name='value')

    # Split category into differential equation and classifier
    melted_df['Differential equation'] = melted_df['category'].str.split('_').str[0]
    melted_df['Classifier'] = melted_df['category'].str.split('_').str[1]

    # Map short names to full names
    equation_mapping = {'RP': 'Rayleigh-Plesset', 'KM': 'Keller-Miksis', 'G': 'Gilmore'}
    melted_df['Differential equation'] = melted_df['Differential equation'].map(equation_mapping)

    threshold_mapping = {
        'dynamical': 'Dynamical threshold',
        'acoustic': 'Acoustic emissions',
        'mach': 'Mach number',
        'expansion': 'Maximum radius'
    }
    melted_df['Classifier'] = melted_df['Classifier'].map(threshold_mapping)

    # Group by thresholds and equations
    grouped_threshold = melted_df.groupby(columns_to_keep + ['Classifier'])['value'].sum().reset_index()
    grouped_equation = melted_df.groupby(columns_to_keep + ['Differential equation'])['value'].sum().reset_index()

    # Plot distribution of thresholds
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax = sns.histplot(data=grouped_threshold, x='value', hue='Classifier',
                      multiple='dodge', discrete=True, shrink=0.8)
    plt.xticks([0, 1, 2, 3], fontsize=X_TICK_FONT_SIZE)
    plt.yticks(fontsize=Y_TICK_FONT_SIZE)
    plt.xlabel('Number of differential equations categorizing cavitation as stable', fontsize=X_LABEL_FONT_SIZE)
    plt.ylabel('Count', fontsize=Y_LABEL_FONT_SIZE)

    # Custom legend for classifiers
    legend_labels = ['Dynamical threshold', 'Acoustic emissions', 'Mach number', 'Maximum radius']
    legend_colors = [primary_color, secondary_color, tertiary_color, quaternary_color]
    legend_handles = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=legend_handles, fontsize=LEGEND_FONT_SIZE, title='Classifier',
              title_fontsize=LEGEND_TITLE_FONT_SIZE, loc=LEGEND_POSITION, bbox_to_anchor=LEGEND_SIZE, frameon=False)
    plt.savefig('../figures/threshold_distribution.pdf', transparent=True, bbox_inches='tight')
    plt.show()

    # Plot distribution of differential equations
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax = sns.histplot(data=grouped_equation, x='value', hue='Differential equation',
                      multiple='dodge', discrete=True, shrink=0.8)
    plt.xticks([0, 1, 2, 3, 4], fontsize=X_TICK_FONT_SIZE)
    plt.yticks(fontsize=Y_TICK_FONT_SIZE)
    plt.xlabel('Number of classifiers categorizing cavitation as stable', fontsize=X_LABEL_FONT_SIZE)
    plt.ylabel('Count', fontsize=Y_LABEL_FONT_SIZE)

    # Custom legend for differential equations
    legend_labels = ['Rayleigh-Plesset', 'Keller-Miksis', 'Gilmore']
    legend_colors = [primary_color, secondary_color, tertiary_color]
    legend_handles = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=legend_handles, fontsize=LEGEND_FONT_SIZE, title='Differential equation',
              title_fontsize=LEGEND_TITLE_FONT_SIZE, loc=LEGEND_POSITION, bbox_to_anchor=LEGEND_SIZE, frameon=False)

    plt.savefig('../figures/equation_distribution.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def comparison_ensemble_multi_objective(data):
    """
    Compares ensemble and multi-objective models using bar plots.

    Parameters:
    data (dict): Dictionary containing model performance data for different equations and thresholds.
    """
    # Data from the table
    equations = ['Rayleigh-Plesset', 'Keller-Miksis', 'Gilmore']
    thresholds = ['Dynamical threshold', 'Mach number', 'Expansion radius', 'Acoustic emissions']

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(1.5 * PLOT_WIDTH, 3 * PLOT_HEIGHT), sharex=True)

    x = np.arange(len(thresholds))
    width = 0.2  # Adjusted for more space between bars

    # Plot for Ensemble
    ax = axes[0]
    ensemble_means = []

    for i, equation in enumerate(equations):
        y = [data[equation][threshold][0] for threshold in thresholds]
        ensemble_means.extend(y)
        ax.bar(x + (i - 1) * width, y, width, label=equation, color=[primary_color, secondary_color, tertiary_color][i])

    # Add mean accuracy line
    mean_accuracy_ensemble = np.mean(ensemble_means)
    ax.axhline(mean_accuracy_ensemble, color='gray', linestyle='--', linewidth=1,
               label=f'Mean Accuracy: {mean_accuracy_ensemble:.2f}')
    ax.set_ylabel("Accuracy", fontsize=Y_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=Y_TICK_FONT_SIZE)
    ax.set_ylim(0, 1)
    ax.legend(title='Differential equation', fontsize=LEGEND_FONT_SIZE,
              title_fontsize=LEGEND_TITLE_FONT_SIZE, loc=LEGEND_POSITION, bbox_to_anchor=LEGEND_SIZE, markerscale=0.5)
    ax.text(-0.5, 1.3, 'a)', transform=ax.transAxes, fontsize=X_LABEL_FONT_SIZE, verticalalignment='top')
    ax.set_title('Ensemble', fontsize=X_LABEL_FONT_SIZE)

    # Plot for Multi-Objective
    ax = axes[1]
    multi_objective_means = []

    for i, equation in enumerate(equations):
        y = [data[equation][threshold][1] for threshold in thresholds]
        multi_objective_means.extend(y)
        ax.bar(x + (i - 1) * width, y, width, label=equation, color=[primary_color, secondary_color, tertiary_color][i])

    # Add mean accuracy line
    mean_accuracy_multi_objective = np.mean(multi_objective_means)
    ax.axhline(mean_accuracy_multi_objective, color='gray', linestyle='--', linewidth=1,
               label=f'Mean Accuracy: {mean_accuracy_multi_objective:.2f}')
    ax.set_ylabel("Accuracy", fontsize=Y_LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, rotation=45, ha="right", fontsize=X_TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=Y_TICK_FONT_SIZE)
    ax.set_ylim(0, 1)
    ax.legend(title='Differential equation', fontsize=LEGEND_FONT_SIZE,
              title_fontsize=LEGEND_TITLE_FONT_SIZE, loc=LEGEND_POSITION, bbox_to_anchor=LEGEND_SIZE, markerscale=0.5)
    ax.text(-0.5, 1.3, 'b)', transform=ax.transAxes, fontsize=X_LABEL_FONT_SIZE, verticalalignment='top')
    ax.set_title('Multi-Objective', fontsize=X_LABEL_FONT_SIZE)

    plt.tight_layout()
    plt.savefig('../figures/combined_accuracy.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def create_plot_ensemble(dataset, experiment_maker, variable_to_vary, number_trees, generalization):
    """
    Generates predictions using an ensemble model.

    Parameters:
    dataset (pd.DataFrame): The dataset to use for the model.
    experiment_maker (ExperimentMaker): An instance of ExperimentMaker to run the model.
    variable_to_vary (str): The variable to vary in the model.
    number_trees (int): The number of trees to use in the ensemble.
    generalization (bool): Whether to generalize the model.

    Returns:
    list: The mean prediction for the test set.
    """
    ensemble_results, _, _ = experiment_maker.ensemble_model(
        dataset=dataset,
        n_trees=number_trees,
        generalization=generalization,
        predict_new=True,
        variable_to_vary=variable_to_vary
    )

    # Filter for test set results
    data = ensemble_results[ensemble_results['Set'] == 'Test']

    # Calculate mean prediction
    predictions = np.array(data['Predicted'].tolist())
    mean_prediction = np.mean(predictions, axis=0).tolist()
    return mean_prediction


def create_plot_multi_objective(dataset, experiment_maker, variable_to_vary, number_trees, generalization):
    """
    Generates predictions using a multi-objective model.

    Parameters:
    dataset (pd.DataFrame): The dataset to use for the model.
    experiment_maker (ExperimentMaker): An instance of ExperimentMaker to run the model.
    variable_to_vary (str): The variable to vary in the model.
    number_trees (int): The number of trees to use in the multi-objective model.
    generalization (bool): Whether to generalize the model.

    Returns:
    list: The mean prediction for the test set.
    """
    _, _, _, predictions_test, _ = experiment_maker.multi_objective_model(
        dataset=dataset,
        n_trees=number_trees,
        generalization=generalization,
        predict_new=True,
        variable_to_vary=variable_to_vary
    )

    # Calculate mean prediction for the test set
    mean_prediction = np.mean(predictions_test, axis=1).tolist()
    return mean_prediction


def create_plot_likelihood(dataset, experiment_maker, variable_to_vary, number_trees, generalization):
    """
    Generates predictions using a probabilistic model.

    Parameters:
    dataset (pd.DataFrame): The dataset to use for the model.
    experiment_maker (ExperimentMaker): An instance of ExperimentMaker to run the model.
    variable_to_vary (str): The variable to vary in the model.
    number_trees (int): The number of trees to use in the probabilistic model.
    generalization (bool): Whether to generalize the model.

    Returns:
    np.ndarray: The first prediction from the test set.
    """
    probabilistic_results = experiment_maker.probabilistic_model(
        dataset=dataset,
        n_trees=number_trees,
        generalization=generalization,
        predict_new=True,
        variable_to_vary=variable_to_vary
    )

    # Filter for test set results
    data = probabilistic_results[probabilistic_results['Set'] == 'Test']

    # Return the first prediction
    predictions = np.array(data['Predicted'].tolist())
    return predictions[0]


def create_plot_expansion(dataset, experiment_maker, variable_to_vary, number_trees, generalization):
    """
    Generates predictions using an expansion model.

    Parameters:
    dataset (pd.DataFrame): The dataset to use for the model.
    experiment_maker (ExperimentMaker): An instance of ExperimentMaker to run the model.
    variable_to_vary (str): The variable to vary in the model.
    number_trees (int): The number of trees to use in the expansion model.
    generalization (bool): Whether to generalize the model.

    Returns:
    np.ndarray: The first prediction from the test set.
    """
    expansion_results = experiment_maker.expansion_model(
        dataset=dataset,
        n_trees=number_trees,
        generalization=generalization
    )

    # Filter for test set results
    data = expansion_results[expansion_results['Set'] == 'Test']

    # Return the first prediction
    predictions = np.array(data['Predicted'].tolist())
    return predictions[0]


def likelihood_figure(dataset, variable_to_vary, design):
    experiment_maker = ExperimentMaker()
    number_trees = 15
    generalization = False
    if design == 'Ensemble':
        prediction = create_plot_ensemble(dataset, experiment_maker, variable_to_vary, number_trees, generalization)
    elif design == 'Multi-objective':
        prediction = create_plot_multi_objective(dataset, experiment_maker, variable_to_vary, number_trees,
                                                 generalization)
    elif design == 'Likelihood':
        prediction = create_plot_likelihood(dataset, experiment_maker, variable_to_vary, number_trees, generalization)
    elif design == 'Expansion':
        prediction = create_plot_expansion(dataset, experiment_maker, variable_to_vary, number_trees, generalization)

    else:
        raise ValueError('Invalid design')
    sns.set(style="whitegrid")

    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    data = pd.read_csv(f'../Data/proportion_plot_{variable_to_vary}.csv')
    label_raw = np.array(sorted(list(data[variable_to_vary].unique())))

    if variable_to_vary == 'initial_radius':
        labels = label_raw * 1e6
        plt.xlabel('Initial radius ($\mu$m)', fontsize=X_LABEL_FONT_SIZE)

    elif variable_to_vary == 'acoustic_pressure':
        labels = label_raw * 1e-6
        plt.xlabel('Acoustic pressure (M Pa)', fontsize=X_LABEL_FONT_SIZE)

    elif variable_to_vary == 'frequency':
        labels = label_raw * 1e-6
        plt.xlabel('Frequency (M Hz)', fontsize=X_LABEL_FONT_SIZE)
    else:
        raise ValueError('Invalid variable to vary')
    sns.scatterplot(x=labels, y=prediction)
    plt.ylabel('Proportion prediction', fontsize=Y_LABEL_FONT_SIZE)
    plt.grid(True)
    plt.ylim([0, 1])

    plt.xticks(fontsize=X_TICK_FONT_SIZE)
    plt.yticks(np.linspace(0, 1, 5), fontsize=Y_TICK_FONT_SIZE)

    # Guardar el gráfico
    plt.savefig(f'../figures/{design}_{variable_to_vary}.pdf', transparent=True, bbox_inches='tight')

    # Mostrar el gráfico
    plt.show()


def heat_map_figure(dataset):
    """
    Generates a heatmap showing the relationship between initial radius and acoustic pressure.

    Parameters:
    dataset (pd.DataFrame): The dataset to use for generating the heatmap.
    """
    experiment_maker = ExperimentMaker()
    number_trees = 15
    generalization = False

    # Define variables to vary
    variable_to_vary = ('initial_radius', 'acoustic_pressure')
    ensemble_results, _, _ = experiment_maker.ensemble_model(
        dataset=dataset,
        n_trees=number_trees,
        generalization=generalization,
        predict_new=True,
        variable_to_vary=variable_to_vary,
        two_variables=True
    )

    # Read in the general data for plotting
    df = pd.read_csv('../Data/proportion_plot_general.csv')
    df['count'] = np.array(list(ensemble_results[ensemble_results['Set'] == 'Test']['Predicted'])).T.sum(axis=1)

    # Convert units for better readability
    df['initial_radius'] *= 1e6
    df['acoustic_pressure'] *= 1e-6

    # Create the grid for the contour plot
    x = np.linspace(df['initial_radius'].min(), df['initial_radius'].max(), 100)
    y = np.linspace(df['acoustic_pressure'].min(), df['acoustic_pressure'].max(), 100)
    X, Y = np.meshgrid(x, y)

    # Interpolate the count values on the grid
    Z = griddata((df['initial_radius'], df['acoustic_pressure']), df['count'], (X, Y))

    # Plotting the contour
    plt.figure(figsize=(PLOT_WIDTH*1.3, PLOT_HEIGHT*1.3))
    plt.gca().set_aspect(5)

    contour = plt.contourf(X, Y, Z, levels=np.arange(-0.5, 13, 1), cmap='RdYlBu')
    cbar = plt.colorbar(contour, ticks=np.arange(0, 13, 1))
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)
    cbar.set_label('Count', fontsize=X_LABEL_FONT_SIZE)

    plt.xlabel('Initial Radius (μm)', fontsize=X_LABEL_FONT_SIZE)
    plt.ylabel('Acoustic Pressure (MPa)', fontsize=Y_LABEL_FONT_SIZE)

    plt.xticks(fontsize=X_TICK_FONT_SIZE)
    plt.yticks(fontsize=Y_TICK_FONT_SIZE)

    # Save and show the heatmap
    plt.savefig('../figures/heat_map_cavitation_ensemble.pdf', transparent=True, bbox_inches='tight')
    plt.show()
