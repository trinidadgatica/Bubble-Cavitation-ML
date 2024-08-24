from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from tabulate import tabulate
import numpy as np


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


def comparison_table(result_df, design):
    if design == 'MO':
        accuracy_list_mean = []
        mae_list_mean = []
        rmse_list_mean = []

        accuracy_list_voting = []
        mae_list_voting = []
        rmse_list_voting = []
        for i in range(5):
            labels = result_df[i]['Labels_test'].values.tolist()
            predictions = result_df[i]['Test_predictions']

            mean_prediction, mean_labels, voting_predictions, voting_labels = ensembling_models(labels, predictions,
                                                                                                'Multi-objective')

            # MEAN
            accuracy_m, mae_m, rmse_m = calculate_metrics(mean_labels, mean_prediction, False)
            accuracy_list_mean.append(accuracy_m)
            mae_list_mean.append(mae_m)
            rmse_list_mean.append(rmse_m)
            # VOTING
            accuracy_v, mae_v, rmse_v = calculate_metrics(voting_labels, voting_predictions, True)
            accuracy_list_voting.append(accuracy_v)
            mae_list_voting.append(mae_v)
            rmse_list_voting.append(rmse_v)
        print(accuracy_list_mean, mae_list_mean, rmse_list_mean)
        print('___________________________________________________')
        print(accuracy_list_voting, mae_list_voting, rmse_list_voting)
        print('-------------------------------------------------')
        print('Multi-objective', 'Mean')
        print('Accuracy', np.mean(accuracy_list_mean))
        print('MAE', np.mean(mae_list_mean))
        print('RMSE', np.mean(rmse_list_mean))
        print('Multi-objective', 'Voting')
        print('Accuracy', np.mean(accuracy_list_voting))
        print('MAE', np.mean(mae_list_voting))
        print('RMSE', np.mean(rmse_list_voting))

    elif design == 'Ensemble':
        accuracy_list_mean = []
        mae_list_mean = []
        rmse_list_mean = []

        accuracy_list_voting = []
        mae_list_voting = []
        rmse_list_voting = []
        for i in range(5):
            j = i + 1
            df = result_df[(result_df['Fold'] == j) & (result_df['Set'] == 'Test')]
            mean_prediction, mean_labels, voting_predictions, voting_labels = ensembling_models(
                np.array(df['Label'].tolist()), np.array(df['Predicted'].tolist()), 'Ensemble')
            # MEAN
            accuracy_m, mae_m, rmse_m = calculate_metrics(mean_labels, mean_prediction, False)
            accuracy_list_mean.append(accuracy_m)
            mae_list_mean.append(mae_m)
            rmse_list_mean.append(rmse_m)
            # VOTING
            accuracy_v, mae_v, rmse_v = calculate_metrics(voting_labels, voting_predictions, True)
            accuracy_list_voting.append(accuracy_v)
            mae_list_voting.append(mae_v)
            rmse_list_voting.append(rmse_v)

        print(accuracy_list_mean, mae_list_mean, rmse_list_mean)
        print('___________________________________________________')
        print(accuracy_list_voting, mae_list_voting, rmse_list_voting)
        print('-------------------------------------------------')
        print('Ensemble', 'Mean')
        print('Accuracy', np.mean(accuracy_list_mean))
        print('MAE', np.mean(mae_list_mean))
        print('RMSE', np.mean(rmse_list_mean))
        print('Ensemble', 'Voting')
        print('Accuracy', np.mean(accuracy_list_voting))
        print('MAE', np.mean(mae_list_voting))
        print('RMSE', np.mean(rmse_list_voting))

    else:
        raise ValueError('Wrong design')


def process_ensemble_multi_objective(results, model_type):

    all_labels_train = []
    all_predictions_train = []
    all_labels_test = []
    all_predictions_test = []
    if model_type == 'Multi-objective':
        for result in results:
            labels_train = result['Labels_train'].values
            predictions_train = result['Train_predictions']

            labels_test = result['Labels_test'].values
            predictions_test = result['Test_predictions']

            all_labels_train.append(labels_train)
            all_predictions_train.append(predictions_train)

            all_labels_test.append(labels_test)
            all_predictions_test.append(predictions_test)

        
        # Cálculo usando el método 'mean'
        mean_prediction_train, mean_labels_train, voting_predictions_train, voting_labels_train = ensembling_models(
            all_labels_train, all_predictions_train, model_type)
        mean_prediction_test, mean_labels_test, voting_predictions_test, voting_labels_test = ensembling_models(
            all_labels_test, all_predictions_test, model_type)

        # Calcular métricas para train
        print(f'{model_type} - Train Metrics (Mean)')
        calculate_overall_metrics(mean_labels_train, mean_prediction_train, False)
        print(f'{model_type} - Train Metrics (Voting)')
        calculate_overall_metrics(voting_labels_train, voting_predictions_train, True)

        # Calcular métricas para test
        print(f'{model_type} - Test Metrics (Mean)')
        calculate_overall_metrics(mean_labels_test, mean_prediction_test, False)
        print(f'{model_type} - Test Metrics (Voting)')
        calculate_overall_metrics(voting_labels_test, voting_predictions_test, True)

    elif model_type == 'Ensemble':
            pass
    else:
        raise ValueError('Wrong model type')


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
        np.array(test_expansion['Label'].tolist()), np.array(test_expansion['Predicted'].tolist()), True)
    acc_likelihood_test, mae_likelihood_test, rmse_likelihood_test = calculate_metrics(
        np.array(test_probabilistic['Label'].tolist()), np.array(test_probabilistic['Predicted'].tolist()), False)

    # Create a list of models and their corresponding metrics
    table_data_test = [
        ['Ensemble Mean', acc_ensemble_mean_test, mae_ensemble_mean_test, rmse_ensemble_mean_test],
        ['Ensemble Voting', acc_ensemble_vote_test, mae_ensemble_vote_test, rmse_ensemble_vote_test],
        ['Multi-objective Mean', acc_mo_mean_test, mae_mo_mean_test, rmse_mo_mean_test],
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
        np.array(train_expansion['Label'].tolist()), np.array(train_expansion['Predicted'].tolist()), True)
    acc_likelihood_train, mae_likelihood_train, rmse_likelihood_train = calculate_metrics(
        np.array(train_probabilistic['Label'].tolist()), np.array(train_probabilistic['Predicted'].tolist()), False)

    # Create a list of models and their corresponding metrics for the train set
    table_data_train = [
        ['Ensemble Mean', acc_ensemble_mean_train, mae_ensemble_mean_train, rmse_ensemble_mean_train],
        ['Ensemble Voting', acc_ensemble_vote_train, mae_ensemble_vote_train, rmse_ensemble_vote_train],
        ['Multi-objective Mean', acc_mo_mean_train, mae_mo_mean_train, rmse_mo_mean_train],
        ['Multi-objective Voting', acc_mo_vote_train, mae_mo_vote_train, rmse_mo_vote_train],
        ['Expansion', acc_expansion_train, mae_expansion_train, rmse_expansion_train],
        ['Likelihood', acc_likelihood_train, mae_likelihood_train, rmse_likelihood_train]
    ]

    # Generate and print the table for the train set
    table_train = tabulate(table_data_train, headers=headers, tablefmt='pretty')
    print("Train Set Metrics:")
    print(table_train)