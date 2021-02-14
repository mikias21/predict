import os
import threading
import uuid

from LearningModels.Regression.LinearRegressionModel import LinearRegressionModel
from LearningModels.Regression.PolynomialRegression import PolynomialRegression
from LearningModels.Regression.SupportVectorRegression import SupportVectorRegression
from LearningModels.Regression.DecisionTreeRegression import DecisionTreeRegression
from LearningModels.Regression.RandomForestRegression import RandomForestRegression
from LearningModels.Regression.RidgeRegression import RidgeRegressionModel
# Preloaded Algorithms
from LearningModels.RegressionPreloaded.LinearRegressionPreLoaded import LinearRegressionPreloadedModel
from LearningModels.RegressionPreloaded.PolynomialRegressionPreloaded import PolynomialRegressionPreloadedModel
from LearningModels.RegressionPreloaded.SupportVectorRegressionPreloaded import SupportVectorRegressionPreloaded
from LearningModels.RegressionPreloaded.DecisionTreeRegressionPreloaded import DecisionTreeRegressionPreloaded
from LearningModels.RegressionPreloaded.RandomForestRegressionPreloaded import RandomForestRegressionPreloaded
from LearningModels.RegressionPreloaded.RidgeRegressionPreloaded import RidgeRegressionModelPreloaded

def prepare_model_instance_regression(app, dataset, unwanted_cols, algorithm, testsize, dependant_var, indivisualInputs, polyregdegree, kernelopt, nestimators, alpha_ridge, max_iteration, solver):
    # File names to hold graphs
    filename1 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-training.png')
    filename2 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-testing.png')
    if algorithm == 'SR':
        linear_regression = LinearRegressionModel(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize)
        prediction, columns = linear_regression.make_regression()
        single_prediction = linear_regression.make_single_prediction()
        mae, mse, r2score, rmse, explained_score = linear_regression.get_seccess_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs = linear_regression.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = linear_regression.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Create threading for the graph to be generated
        graph_thread = threading.Thread(target=linear_regression.draw_graphs(filename1, filename2))
        graph_thread.setDaemon(True)
        graph_thread.start()
        return prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, rmse, explained_score, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'PR':
        polynomial_regression = PolynomialRegression(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize, polyregdegree)
        prediction, columns = polynomial_regression.make_regression()
        single_prediction = polynomial_regression.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = polynomial_regression.get_seccess_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs = polynomial_regression.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = polynomial_regression.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Create the thread
        graph_thread = threading.Thread(target=polynomial_regression.draw_graphs(filename1, filename2))
        graph_thread.setDaemon(True)
        graph_thread.start()
        return prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, maxerror, explained_score, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'SVR':
        support_vector_regression = SupportVectorRegression(dataset, dependant_var, indivisualInputs, kernelopt, unwanted_cols, testsize)
        prediction, columns = support_vector_regression.make_regression()
        single_prediction = support_vector_regression.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = support_vector_regression.get_seccess_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs = support_vector_regression.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = support_vector_regression.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Create the thread
        graph_thread = threading.Thread(target=support_vector_regression.draw_graphs(filename1, filename2))
        graph_thread.setDaemon(True)
        graph_thread.start()
        return prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, maxerror, explained_score, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'DTR':
        decision_tree_regression = DecisionTreeRegression(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize)
        prediction, columns = decision_tree_regression.make_regression()
        single_prediction = decision_tree_regression.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = decision_tree_regression.get_seccess_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs = decision_tree_regression.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = decision_tree_regression.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Create the thread
        graph_thread = threading.Thread(target=decision_tree_regression.draw_graphs(filename1, filename2))
        graph_thread.setDaemon(True)
        graph_thread.start()
        return prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, maxerror, explained_score, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'RFR':
        random_forest_regression = RandomForestRegression(dataset, dependant_var, indivisualInputs, nestimators, unwanted_cols, testsize)
        prediction, columns = random_forest_regression.make_regression()
        single_prediction = random_forest_regression.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = random_forest_regression.get_seccess_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs = random_forest_regression.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = random_forest_regression.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thread for main graphs 
        graph_thread = threading.Thread(target=random_forest_regression.draw_graphs(filename1, filename2))
        graph_thread.setDaemon(True)
        graph_thread.start()
        return prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, maxerror, explained_score, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'RR':
        ridge_regression = RidgeRegressionModel(dataset, dependant_var, indivisualInputs, alpha_ridge, max_iteration, solver, unwanted_cols, testsize)
        prediction, columns = ridge_regression.make_regression()
        single_prediction = ridge_regression.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = ridge_regression.get_seccess_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs = ridge_regression.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = ridge_regression.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thread for main graphs 
        graph_thread = threading.Thread(target=ridge_regression.draw_graphs(filename1, filename2))
        graph_thread.setDaemon(True)
        graph_thread.start()
        return prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, maxerror, explained_score, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs, model_description


def prpare_preloaded_dataset_regression(app, dataset, unwanted_cols, algorithm, testsize, dependant_var, indivisualInputs, polyregdegree, kernelopt, nestimators, alpha_ridge, max_iteration, solver):
    if algorithm == 'SR':
        linear_regression_preloaded = LinearRegressionPreloadedModel(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize)
        prediction, columns = linear_regression_preloaded.make_regression()
        single_prediction = linear_regression_preloaded.make_single_prediction()
        mae, mse, r2score, rmse, explained_score = linear_regression_preloaded.get_seccess_rate()
        reg_graphs, dist_graph, corr_graphs = linear_regression_preloaded.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = linear_regression_preloaded.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_prediction, mae, mse, r2score, rmse, explained_score, reg_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'PR':
        polynomial_regression_preloaded = PolynomialRegressionPreloadedModel(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize, polyregdegree)
        prediction, columns = polynomial_regression_preloaded.make_regression()
        single_prediction = polynomial_regression_preloaded.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = polynomial_regression_preloaded.get_seccess_rate()
        reg_graphs, dist_graph, corr_graphs = polynomial_regression_preloaded.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = polynomial_regression_preloaded.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_prediction, mae, mse, r2score, maxerror, explained_score, reg_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'SVR':
        support_vector_preloaded = SupportVectorRegressionPreloaded(dataset, dependant_var, indivisualInputs, kernelopt, unwanted_cols, testsize)
        prediction, columns = support_vector_preloaded.make_regression()
        single_prediction = support_vector_preloaded.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = support_vector_preloaded.get_seccess_rate()
        reg_graphs, dist_graph, corr_graphs = support_vector_preloaded.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = support_vector_preloaded.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_prediction, mae, mse, r2score, maxerror, explained_score, reg_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'DTR':
        decision_tree_preloaded = DecisionTreeRegressionPreloaded(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize)
        prediction, columns = decision_tree_preloaded.make_regression()
        single_prediction = decision_tree_preloaded.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = decision_tree_preloaded.get_seccess_rate()
        reg_graphs, dist_graph, corr_graphs = decision_tree_preloaded.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = decision_tree_preloaded.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_prediction, mae, mse, r2score, maxerror, explained_score, reg_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'RFR':
        random_forest_preloaded = RandomForestRegressionPreloaded(dataset, dependant_var, indivisualInputs, nestimators, unwanted_cols, testsize)
        prediction, columns = random_forest_preloaded.make_regression()
        single_prediction = random_forest_preloaded.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = random_forest_preloaded.get_seccess_rate()
        reg_graphs, dist_graph, corr_graphs = random_forest_preloaded.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = random_forest_preloaded.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_prediction, mae, mse, r2score, maxerror, explained_score, reg_graphs, dist_graph, corr_graphs, model_description
    elif algorithm == 'RR':
        ridge_preloaded = RidgeRegressionModelPreloaded(dataset, dependant_var, indivisualInputs, alpha_ridge, max_iteration, solver, unwanted_cols, testsize)
        prediction, columns = ridge_preloaded.make_regression()
        single_prediction = ridge_preloaded.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = ridge_preloaded.get_seccess_rate()
        reg_graphs, dist_graph, corr_graphs = ridge_preloaded.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = ridge_preloaded.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_prediction, mae, mse, r2score, maxerror, explained_score, reg_graphs, dist_graph, corr_graphs, model_description
