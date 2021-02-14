import os
import uuid 
import threading

from LearningModels.NeuralNetwork.MLPClassification import MLPClassification
from LearningModels.NeuralNetwork.MLPRegression import MLPRegression
from LearningModels.NeuralNetworkPreloaded.MLPClassificationPreloaded import MLPClassificationPreloaded
from LearningModels.NeuralNetworkPreloaded.MLPRegressionPreloaded import MLPRegressionPreloaded


def prepare_model_instance_neuralnetwork_preloaded(app, dataset, unwanted_cols, algorithm, testsize, dependant_var, indivisualInputs, mlpc_activation, mlpc_solver,
             mlpc_learning_rate, mlpc_max_iter, mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter):

    filename1 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-training.png')
    filename2 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-testing.png')
    filename3 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-confusion.png')
    
    if algorithm == 'MLPC':
        mlpc = MLPClassificationPreloaded(dataset, dependant_var, indivisualInputs, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, unwanted_cols, testsize)
        prediction, columns = mlpc.make_classification()
        single_classification = mlpc.make_single_classification()
        mae, mse, r2score, accuracy, confusion = mlpc.get_success_rate()
        reg_graphs, dist_graphs, corr_graphs = mlpc.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = mlpc.model_summary()
        # Thread for the main graphs
        graph_trade = threading.Thread(target=mlpc.draw_neural_net(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, filename1, filename2, filename2, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'MLPR':
        mlpr = MLPRegressionPreloaded(dataset, dependant_var, indivisualInputs, mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, unwanted_cols, testsize)
        prediction, columns = mlpr.make_regression()
        single_classification = mlpr.make_single_prediction()
        mae, mse, r2score, accuracy, confusion = mlpr.get_seccess_rate()
        reg_graphs, dist_graphs, corr_graphs = mlpr.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = mlpr.model_summary()
        # Thread for the main graphs
        graph_trade = threading.Thread(target=mlpr.draw_neural_net(filename1, filename2))
        graph_trade.setDaemon(True)
        graph_trade.start()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, filename1, filename2, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description



def prepare_model_instance_neuralnetwork(app, dataset, unwanted_cols, algorithm, testsize, dependant_var, indivisualInputs, mlpc_activation, mlpc_solver,
             mlpc_learning_rate, mlpc_max_iter, mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter):

    filename1 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-training.png')
    filename2 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-testing.png')
    filename3 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-confusion.png')

    if algorithm == 'MLPC':
        mlpc = MLPClassification(dataset, dependant_var, indivisualInputs, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, unwanted_cols, testsize)
        prediction, columns = mlpc.make_classification()
        single_classification = mlpc.make_single_classification()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = mlpc.get_more_relation_graphs(app)
        mae, mse, r2score, accuracy, confusion = mlpc.get_success_rate()
        dataset_description, dataset_columns, dataset_shape, dataset_memory = mlpc.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thread for the main graphs
        graph_trade = threading.Thread(target=mlpc.draw_neural_net(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'MLPR':
        mlpr = MLPRegression(dataset, dependant_var, indivisualInputs, mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, unwanted_cols, testsize)
        prediction, columns = mlpr.make_prediction()
        single_prediction = mlpr.make_single_prediction()
        mae, mse, r2score, maxerror, explained_score = mlpr.get_success_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = mlpr.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = mlpr.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thread for the main graphs
        graph_trade = threading.Thread(target=mlpr.draw_neural_net(filename1, filename2))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, maxerror, explained_score, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description


