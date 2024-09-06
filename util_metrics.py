from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from tabulate import tabulate
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def calculate_metrics(labels, predictions, binary_classification=True):
    if binary_classification:
        accuracy = accuracy_score(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        rmse = mean_squared_error(labels, predictions, squared=False)

    else:
        predictions = np.array(predictions)
        correct = np.sum(np.isclose(labels, predictions, atol=0.05))
        total = len(labels)
        accuracy = correct / total

        mae = np.mean(np.abs(labels - predictions))
        rmse = np.sqrt(mean_squared_error(labels, predictions))

    return accuracy, mae, rmse


def calculate_overall_metrics(labels, predictions, binary_classification=True):
    accuracy_list = []
    mae_list = []
    rmse_list = []
    for i in range(5):
        # Calcular las métricas utilizando la función existente
        accuracy_i, mae_i, rmse_i = calculate_metrics(labels[i], predictions[i], binary_classification)
        accuracy_list.append(accuracy_i)
        mae_list.append(mae_i)
        rmse_list.append(rmse_i)
    return np.mean(accuracy_list), np.mean(mae_list), np.mean(rmse_list)


def ensembling_models(labels, predictions, model):
    if model == 'Ensemble':
        mean_prediction_raw = np.mean(predictions, axis=0).tolist()
        mean_labels_raw = np.mean(labels, axis=0).tolist()

        voting_predictions_raw = np.sum(predictions, axis=0)
        voting_labels_raw = np.sum(labels, axis=0)

    elif model == 'Multi-objective':
        mean_prediction_raw = np.mean(predictions, axis=1).tolist()
        mean_labels_raw = np.mean(labels, axis=1).tolist()

        voting_predictions_raw = np.sum(predictions, axis=1)
        voting_labels_raw = np.sum(labels, axis=1)

    else:
        raise ValueError('Model must be Ensemble or Multi-objective')

    voting_labels = [1 if vote >= 6 else 0 for vote in voting_labels_raw]
    voting_predictions = [1 if vote >= 6 else 0 for vote in voting_predictions_raw]

    return mean_prediction_raw, mean_labels_raw, voting_predictions, voting_labels


def table_metrics(result_ensemble, result_expansion, result_likelihood, results_mo):
    # Separate the test and train sets from each result dataframe
    test_ensemble = result_ensemble[result_ensemble['Set'] == 'Test']
    test_expansion = result_expansion[result_expansion['Set'] == 'Test']
    test_probabilistic = result_likelihood[result_likelihood['Set'] == 'Test']

    train_ensemble = result_ensemble[result_ensemble['Set'] == 'Train']
    train_expansion = result_expansion[result_expansion['Set'] == 'Train']
    train_probabilistic = result_likelihood[result_likelihood['Set'] == 'Train']

    # Unpack the multi-objective results
    labels_train_mo, train_predictions_mo, labels_test_mo, test_predictions_mo, list_labels_mo = results_mo

    # === TEST SET EVALUATION ===

    # Extract predictions and labels for the ensemble model
    predictions_ensemble_test = np.array(test_ensemble['Predicted'].tolist())
    labels_ensemble_test = np.array(test_ensemble['Label'].tolist())

    # Perform ensembling for mean and voting methods
    mean_pred_ensemble_test, mean_labels_ensemble_test, vote_pred_ensemble_test, vote_labels_ensemble_test = \
        ensembling_models(labels_ensemble_test, predictions_ensemble_test, 'Ensemble')

    mean_pred_mo_test, mean_labels_mo_test, vote_pred_mo_test, vote_labels_mo_test = \
        ensembling_models(labels_test_mo, test_predictions_mo, 'Multi-objective')

    # Calculate metrics for ensemble and multi-objective models
    acc_ensemble_mean_test, mae_ensemble_mean_test, rmse_ensemble_mean_test = calculate_metrics(
        mean_labels_ensemble_test, mean_pred_ensemble_test, False)
    acc_ensemble_vote_test, mae_ensemble_vote_test, rmse_ensemble_vote_test = calculate_metrics(
        vote_labels_ensemble_test, vote_pred_ensemble_test, True)

    acc_mo_mean_test, mae_mo_mean_test, rmse_mo_mean_test = calculate_metrics(mean_labels_mo_test, mean_pred_mo_test,
                                                                              False)
    acc_mo_vote_test, mae_mo_vote_test, rmse_mo_vote_test = calculate_metrics(vote_labels_mo_test, vote_pred_mo_test,
                                                                              True)

    # Calculate metrics for expansion and likelihood models
    acc_expansion_test, mae_expansion_test, rmse_expansion_test = calculate_metrics(
        np.array(test_expansion['Label'].values[0]), test_expansion['Predicted'].values[0], True)
    acc_likelihood_test, mae_likelihood_test, rmse_likelihood_test = calculate_metrics(
        np.array(test_probabilistic['Label'].values[0]), np.array(test_probabilistic['Predicted'].values[0]), False)

    # Create a list of models and their corresponding metrics
    table_data_test = [
        ['Ensemble Mean', acc_ensemble_vote_test, mae_ensemble_mean_test, rmse_ensemble_mean_test],
        ['Ensemble Voting', acc_ensemble_vote_test, mae_ensemble_vote_test, rmse_ensemble_vote_test],
        ['Multi-objective Mean', acc_mo_vote_test, mae_mo_mean_test, rmse_mo_mean_test],
        ['Multi-objective Voting', acc_mo_vote_test, mae_mo_vote_test, rmse_mo_vote_test],
        ['Expansion', acc_expansion_test, mae_expansion_test, rmse_expansion_test],
        ['Likelihood', acc_likelihood_test, mae_likelihood_test, rmse_likelihood_test]
    ]

    # Define the headers for the table
    headers = ['Model', 'Accuracy', 'MAE', 'RMSE']

    # Generate and print the table for the test set
    table_test = tabulate(table_data_test, headers=headers, tablefmt='pretty')
    print("Test Set Metrics:")
    print(table_test)

    # === TRAIN SET EVALUATION ===

    # Extract predictions and labels for the ensemble model (train set)
    predictions_ensemble_train = np.array(train_ensemble['Predicted'].tolist())
    labels_ensemble_train = np.array(train_ensemble['Label'].tolist())

    # Perform ensembling for mean and voting methods (train set)
    mean_pred_ensemble_train, mean_labels_ensemble_train, vote_pred_ensemble_train, vote_labels_ensemble_train = \
        ensembling_models(labels_ensemble_train, predictions_ensemble_train, 'Ensemble')

    mean_pred_mo_train, mean_labels_mo_train, vote_pred_mo_train, vote_labels_mo_train = \
        ensembling_models(labels_train_mo, train_predictions_mo, 'Multi-objective')

    # Calculate metrics for ensemble and multi-objective models (train set)
    acc_ensemble_mean_train, mae_ensemble_mean_train, rmse_ensemble_mean_train = calculate_metrics(
        mean_labels_ensemble_train, mean_pred_ensemble_train, False)
    acc_ensemble_vote_train, mae_ensemble_vote_train, rmse_ensemble_vote_train = calculate_metrics(
        vote_labels_ensemble_train, vote_pred_ensemble_train, True)

    acc_mo_mean_train, mae_mo_mean_train, rmse_mo_mean_train = calculate_metrics(mean_labels_mo_train,
                                                                                 mean_pred_mo_train, False)
    acc_mo_vote_train, mae_mo_vote_train, rmse_mo_vote_train = calculate_metrics(vote_labels_mo_train,
                                                                                 vote_pred_mo_train, True)

    # Calculate metrics for expansion and likelihood models (train set)
    acc_expansion_train, mae_expansion_train, rmse_expansion_train = calculate_metrics(
        np.array(train_expansion['Label'].values[0]), train_expansion['Predicted'].values[0], True)
    acc_likelihood_train, mae_likelihood_train, rmse_likelihood_train = calculate_metrics(
        np.array(train_probabilistic['Label'].values[0]), np.array(train_probabilistic['Predicted'].values[0]), False)

    # Create a list of models and their corresponding metrics for the train set
    table_data_train = [
        ['Ensemble Mean', acc_ensemble_vote_train, mae_ensemble_mean_train, rmse_ensemble_mean_train],
        ['Ensemble Voting', acc_ensemble_vote_train, mae_ensemble_vote_train, rmse_ensemble_vote_train],
        ['Multi-objective Mean', acc_mo_vote_train, mae_mo_mean_train, rmse_mo_mean_train],
        ['Multi-objective Voting', acc_mo_vote_train, mae_mo_vote_train, rmse_mo_vote_train],
        ['Expansion', acc_expansion_train, mae_expansion_train, rmse_expansion_train],
        ['Likelihood', acc_likelihood_train, mae_likelihood_train, rmse_likelihood_train]
    ]

    # Generate and print the table for the train set
    table_train = tabulate(table_data_train, headers=headers, tablefmt='pretty')
    print("Train Set Metrics:")
    print(table_train)


def ensembling(ensemble_type, raw_list):
    if ensemble_type == 'Voting':
        final_list = [1 if vote >= 6 else 0 for vote in raw_list]
    else:
        raise
    return final_list


def calculate_metrics_MO_cross_validation(result_mo_dd):
    accuracy_list_train_mean = []
    mae_list_train_mean = []
    rmse_list_train_mean = []

    accuracy_list_test_mean = []
    mae_list_test_mean = []
    rmse_list_test_mean = []

    accuracy_list_train_sum = []
    mae_list_train_sum = []
    rmse_list_train_sum = []

    accuracy_list_test_sum = []
    mae_list_test_sum = []
    rmse_list_test_sum = []

    for number_fold in range(5):
        label_train_raw = result_mo_dd[number_fold]['Labels_train']
        label_test_raw = result_mo_dd[number_fold]['Labels_test']

        prediction_train_raw = result_mo_dd[number_fold]['Train_predictions']
        prediction_test_raw = result_mo_dd[number_fold]['Test_predictions']

        # MEAN
        label_train_mean = np.mean(label_train_raw.values.tolist(), axis=1)
        label_test_mean = np.mean(label_test_raw.values.tolist(), axis=1)

        prediction_train_mean = np.mean(prediction_train_raw, axis=1)
        prediction_test_mean = np.mean(prediction_test_raw, axis=1)

        # SUM
        label_train_sum = ensembling('Voting', np.sum(label_train_raw.values.tolist(), axis=1))
        label_test_sum = ensembling('Voting', np.sum(label_test_raw.values.tolist(), axis=1))

        prediction_train_sum = ensembling('Voting', np.sum(prediction_train_raw, axis=1))
        prediction_test_sum = ensembling('Voting', np.sum(prediction_test_raw, axis=1))

        # Calculate metrics
        metrics_mean_train = calculate_metrics(label_train_mean, prediction_train_mean, False)
        metrics_mean_test = calculate_metrics(label_test_mean, prediction_test_mean, False)

        metrics_sum_train = calculate_metrics(label_train_sum, prediction_train_sum, True)
        metrics_sum_test = calculate_metrics(label_test_sum, prediction_test_sum, True)

        # Append metrics for mean
        accuracy_list_train_mean.append(metrics_mean_train[0])
        mae_list_train_mean.append(metrics_mean_train[1])
        rmse_list_train_mean.append(metrics_mean_train[2])

        accuracy_list_test_mean.append(metrics_mean_test[0])
        mae_list_test_mean.append(metrics_mean_test[1])
        rmse_list_test_mean.append(metrics_mean_test[2])

        # Append metrics for sum
        accuracy_list_train_sum.append(metrics_sum_train[0])
        mae_list_train_sum.append(metrics_sum_train[1])
        rmse_list_train_sum.append(metrics_sum_train[2])

        accuracy_list_test_sum.append(metrics_sum_test[0])
        mae_list_test_sum.append(metrics_sum_test[1])
        rmse_list_test_sum.append(metrics_sum_test[2])

    # Calculate mean values for all lists and return them
    return (
        np.mean(accuracy_list_train_mean), np.mean(mae_list_train_mean), np.mean(rmse_list_train_mean),
        np.mean(accuracy_list_test_mean), np.mean(mae_list_test_mean), np.mean(rmse_list_test_mean),
        np.mean(accuracy_list_train_sum), np.mean(mae_list_train_sum), np.mean(rmse_list_train_sum),
        np.mean(accuracy_list_test_sum), np.mean(mae_list_test_sum), np.mean(rmse_list_test_sum)
    )


def calculate_metrics_ensemble_cross_validation(result_ensemble):
    accuracy_list_train_mean = []
    mae_list_train_mean = []
    rmse_list_train_mean = []

    accuracy_list_test_mean = []
    mae_list_test_mean = []
    rmse_list_test_mean = []

    accuracy_list_train_sum = []
    mae_list_train_sum = []
    rmse_list_train_sum = []

    accuracy_list_test_sum = []
    mae_list_test_sum = []
    rmse_list_test_sum = []

    for fold_number in range(1,6):
        filter_train = (result_ensemble['Fold'] == fold_number) & (result_ensemble['Set'] == 'Train')
        filter_test = (result_ensemble['Fold'] == fold_number) & (result_ensemble['Set'] == 'Test')
        ensemble_train = result_ensemble[filter_train]
        ensemble_test = result_ensemble[filter_test]

        label_train_mean = np.mean(list(ensemble_train['Label']), axis=0)
        label_train_sum = ensembling('Voting', np.sum(list(ensemble_train['Label']), axis=0))

        label_test_mean = np.mean(list(ensemble_test['Label']), axis=0)
        label_test_sum = ensembling('Voting', np.sum(list(ensemble_test['Label']), axis=0))


        prediction_train_mean = np.mean(list(ensemble_train['Predicted']), axis=0)
        prediction_train_sum = ensembling('Voting', np.sum(list(ensemble_train['Predicted']), axis=0))

        prediction_test_mean = np.mean(list(ensemble_test['Predicted']), axis=0)
        prediction_test_sum = ensembling('Voting', np.sum(list(ensemble_test['Predicted']), axis=0))

        metrics_mean_train = calculate_metrics(label_train_mean, prediction_train_mean, False)
        metrics_mean_test = calculate_metrics(label_test_mean, prediction_test_mean, False)

        metrics_sum_train = calculate_metrics(label_train_sum, prediction_train_sum, True)
        metrics_sum_test = calculate_metrics(label_test_sum, prediction_test_sum, True)

         # Append metrics for mean
        accuracy_list_train_mean.append(metrics_mean_train[0])
        mae_list_train_mean.append(metrics_mean_train[1])
        rmse_list_train_mean.append(metrics_mean_train[2])

        accuracy_list_test_mean.append(metrics_mean_test[0])
        mae_list_test_mean.append(metrics_mean_test[1])
        rmse_list_test_mean.append(metrics_mean_test[2])

        # Append metrics for sum
        accuracy_list_train_sum.append(metrics_sum_train[0])
        mae_list_train_sum.append(metrics_sum_train[1])
        rmse_list_train_sum.append(metrics_sum_train[2])

        accuracy_list_test_sum.append(metrics_sum_test[0])
        mae_list_test_sum.append(metrics_sum_test[1])
        rmse_list_test_sum.append(metrics_sum_test[2])

        # Calculate mean values for all lists and return them
    return (
        np.mean(accuracy_list_train_mean), np.mean(mae_list_train_mean), np.mean(rmse_list_train_mean),
        np.mean(accuracy_list_test_mean), np.mean(mae_list_test_mean), np.mean(rmse_list_test_mean),
        np.mean(accuracy_list_train_sum), np.mean(mae_list_train_sum), np.mean(rmse_list_train_sum),
        np.mean(accuracy_list_test_sum), np.mean(mae_list_test_sum), np.mean(rmse_list_test_sum)
    )


def table_metrics_cross_validation(result_ensemble, result_mo, result_expansion, result_likelihood):
    # Separate train and test sets for the expansion and likelihood models
    result_expansion_train = result_expansion[result_expansion['Set'] == 'Train']
    result_expansion_test = result_expansion[result_expansion['Set'] == 'Test']

    result_likelihood_train = result_likelihood[result_likelihood['Set'] == 'Train']
    result_likelihood_test = result_likelihood[result_likelihood['Set'] == 'Test']

    result_mo_dd, list_labels_mo = result_mo

    # Calculate metrics for expansion model
    acc_expansion_train, mae_expansion_train, rmse_expansion_train = calculate_overall_metrics(
        result_expansion_train['Label'].tolist(),
        result_expansion_train['Predicted'].tolist()
    )
    acc_expansion_test, mae_expansion_test, rmse_expansion_test = calculate_overall_metrics(
        result_expansion_test['Label'].tolist(),
        result_expansion_test['Predicted'].tolist()
    )

    # Calculate metrics for likelihood model
    acc_likelihood_train, mae_likelihood_train, rmse_likelihood_train = calculate_overall_metrics(
        result_likelihood_train['Label'].tolist(),
        result_likelihood_train['Predicted'].tolist(),
        binary_classification=False
    )
    acc_likelihood_test, mae_likelihood_test, rmse_likelihood_test = calculate_overall_metrics(
        result_likelihood_test['Label'].tolist(),
        result_likelihood_test['Predicted'].tolist(),
        binary_classification=False
    )

    # Calculate metrics for multi-objective model (cross-validation)
    (acc_mo_train_mean, mae_mo_train_mean, rmse_mo_train_mean, acc_mo_test_mean, mae_mo_test_mean, rmse_mo_test_mean,
     acc_mo_train_sum, mae_mo_train_sum, rmse_mo_train_sum, acc_mo_test_sum, mae_mo_test_sum,
     rmse_mo_test_sum) = calculate_metrics_MO_cross_validation(result_mo_dd)

    # Calculate metrics for ensemble model (cross-validation)
    (acc_ensemble_train_mean, mae_ensemble_train_mean, rmse_ensemble_train_mean, acc_ensemble_test_mean,
     mae_ensemble_test_mean, rmse_ensemble_test_mean,
     acc_ensemble_train_sum, mae_ensemble_train_sum, rmse_ensemble_train_sum, acc_ensemble_test_sum,
     mae_ensemble_test_sum, rmse_ensemble_test_sum) = calculate_metrics_ensemble_cross_validation(result_ensemble)

    # Create the table for the test set
    table_data_test = [
        ['Ensemble Mean', acc_ensemble_test_sum, mae_ensemble_test_mean, rmse_ensemble_test_mean],
        ['Ensemble Voting', acc_ensemble_test_sum, mae_ensemble_test_sum, rmse_ensemble_test_sum],
        ['Multi-objective Mean', acc_mo_test_sum, mae_mo_test_mean, rmse_mo_test_mean],
        ['Multi-objective Voting', acc_mo_test_sum, mae_mo_test_sum, rmse_mo_test_sum],
        ['Expansion', acc_expansion_test, mae_expansion_test, rmse_expansion_test],
        ['Likelihood', acc_likelihood_test, mae_likelihood_test, rmse_likelihood_test]
    ]

    # Define the headers for the table
    headers = ['Model', 'Accuracy', 'MAE', 'RMSE']

    # Generate and print the table for the test set
    table_test = tabulate(table_data_test, headers=headers, tablefmt='pretty')
    print("Test Set Metrics:")
    print(table_test)

    # Create the table for the train set
    table_data_train = [
        ['Ensemble Mean', acc_ensemble_train_sum, mae_ensemble_train_mean, rmse_ensemble_train_mean],
        ['Ensemble Voting', acc_ensemble_train_sum, mae_ensemble_train_sum, rmse_ensemble_train_sum],
        ['Multi-objective Mean', acc_mo_train_sum, mae_mo_train_mean, rmse_mo_train_mean],
        ['Multi-objective Voting', acc_mo_train_sum, mae_mo_train_sum, rmse_mo_train_sum],
        ['Expansion', acc_expansion_train, mae_expansion_train, rmse_expansion_train],
        ['Likelihood', acc_likelihood_train, mae_likelihood_train, rmse_likelihood_train]
    ]

    # Generate and print the table for the train set
    table_train = tabulate(table_data_train, headers=headers, tablefmt='pretty')
    print("Train Set Metrics:")
    print(table_train)