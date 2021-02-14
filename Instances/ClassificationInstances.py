import os 
import uuid
import threading 
# Classification
from LearningModels.Classification.LogisticRegression import LogisticRegresser
from LearningModels.Classification.KNearestNeighbors import KNeighborsClassification
from LearningModels.Classification.SupportVectorClassifier import SupportVectorClassification
from LearningModels.Classification.NaiveBayes import NaiveBayes
from LearningModels.Classification.DecisionTreeClassification import DecisionTreeClassification
from LearningModels.Classification.RandomForestClassification import RandomForestClassification
from LearningModels.Classification.RidgeClassification import RidgeClassificationModel
# Predloaded
from LearningModels.ClassificationPreloaded.LogisticRegressionPreloaded import LogisticRegresserPreloaded
from LearningModels.ClassificationPreloaded.KNearestNeighboursPreloaded import KNeighborsClassificationPreloaded
from LearningModels.ClassificationPreloaded.SupportVectorClassificationPreloaded import SupportVectorClassificationPreloaded
from LearningModels.ClassificationPreloaded.NaiveBayesPreloaded import NaiveBayesPreloaded
from LearningModels.ClassificationPreloaded.DecisionTreePreloaded import DecisionTreeClassificationPreloaded
from LearningModels.ClassificationPreloaded.RandomForestClassificationPreloaded import RandomForestClassificationPreloaded
from LearningModels.ClassificationPreloaded.RidgeClassificationPreloaded import RidgeClassificationModelPreloaded


def prepare_preloaded_dataset_classification(app, dataset, unwanted_cols, algorithm, testsize, dependant_var, indivisualInputs,  n_neighbours, metric, p, kerneloptSvc, criterion
    , criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver):

    if algorithm == 'LR':
        logistic_regression = LogisticRegresserPreloaded(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize)
        prediction, columns = logistic_regression.make_classification()
        single_classification = logistic_regression.make_single_classification()
        mae, mse, r2score, accuracy, confusion = logistic_regression.get_success_rate()
        reg_graphs, dist_graphs, corr_graphs = logistic_regression.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = logistic_regression.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'KNN':
        knearest_neighbours = KNeighborsClassificationPreloaded(dataset, dependant_var, indivisualInputs,  n_neighbours, metric, p, unwanted_cols, testsize)
        prediction, columns = knearest_neighbours.make_classification()
        single_classification = knearest_neighbours.make_single_classification()
        mae, mse, r2score, accuracy, confusion = knearest_neighbours.get_success_rate()
        reg_graphs, dist_graphs, corr_graphs = knearest_neighbours.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = knearest_neighbours.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'SVC':
        support_vector_classification = SupportVectorClassificationPreloaded(dataset, dependant_var, indivisualInputs, kerneloptSvc, unwanted_cols, testsize)
        prediction, columns = support_vector_classification.make_classification()
        single_classification = support_vector_classification.make_single_classification()
        mae, mse, r2score, accuracy, confusion = support_vector_classification.get_success_rate()
        reg_graphs, dist_graphs, corr_graphs = support_vector_classification.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = support_vector_classification.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'NB':
        naive_bayes = NaiveBayesPreloaded(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize)
        prediction, columns = naive_bayes.make_classification()
        single_classification = naive_bayes.make_single_classification()
        mae, mse, r2score, accuracy, confusion = naive_bayes.get_success_rate()
        reg_graphs, dist_graphs, corr_graphs = naive_bayes.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = naive_bayes.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'DTC':
        decision_tree = DecisionTreeClassificationPreloaded(dataset, dependant_var, indivisualInputs, criterion, unwanted_cols, testsize)
        prediction, columns = decision_tree.make_classification()
        single_classification = decision_tree.make_single_classification()
        mae, mse, r2score, accuracy, confusion = decision_tree.get_success_rate()
        reg_graphs, dist_graphs, corr_graphs = decision_tree.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = decision_tree.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'RFC':
        random_forest = RandomForestClassificationPreloaded(dataset, dependant_var, indivisualInputs, criterion_rfc, nestimators_rfc, unwanted_cols, testsize)
        prediction, columns = random_forest.make_classification()
        single_classification = random_forest.make_single_classification()
        mae, mse, r2score, accuracy, confusion = random_forest.get_success_rate()
        reg_graphs, dist_graphs, corr_graphs = random_forest.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = random_forest.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'RC':
        ridge_classification = RidgeClassificationModelPreloaded(dataset, dependant_var, indivisualInputs, alpha_ridge, max_iteration, solver, unwanted_cols, testsize)
        prediction, columns = ridge_classification.make_classification()
        single_classification = ridge_classification.make_single_classification()
        mae, mse, r2score, accuracy, confusion = ridge_classification.get_success_rate()
        reg_graphs, dist_graphs, corr_graphs = ridge_classification.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = ridge_classification.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        return prediction, columns, single_classification, mae, mse, r2score, accuracy, confusion, reg_graphs, dist_graphs, corr_graphs, model_description

def prepare_model_instance_classification(app, dataset, unwanted_cols, algorithm, testsize, dependant_var, indivisualInputs,  n_neighbours, metric, p, kerneloptSvc, criterion
    , criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver):

    filename1 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-training.png')
    filename2 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-testing.png')
    filename3 = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-confusion.png')
    if algorithm == 'LR':
        logistic_regression = LogisticRegresser(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize)
        prediction, columns = logistic_regression.make_classification()
        single_classification = logistic_regression.make_single_classification()
        mae, mse, r2score, accuracy, confusion = logistic_regression.get_success_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = logistic_regression.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = logistic_regression.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thread for the main graphs
        graph_trade = threading.Thread(target=logistic_regression.draw_graph(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'KNN':
        nearest_neighbours = KNeighborsClassification(dataset, dependant_var, indivisualInputs,  n_neighbours, metric, p, unwanted_cols, testsize)
        prediction, columns = nearest_neighbours.make_classification()
        single_classification = nearest_neighbours.make_single_classification()
        mae, mse, r2score, accuracy, confusion = nearest_neighbours.get_success_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = nearest_neighbours.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = nearest_neighbours.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thread for the main graphs
        graph_trade = threading.Thread(target=nearest_neighbours.draw_graph(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'SVC':
        svc = SupportVectorClassification(dataset, dependant_var, indivisualInputs, kerneloptSvc, unwanted_cols, testsize)
        prediction, columns = svc.make_classification()
        single_classification = svc.make_single_classification()
        mae, mse, r2score, accuracy, confusion = svc.get_success_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = svc.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = svc.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thread for the main graphs 
        graph_trade = threading.Thread(target=svc.draw_graph(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'NB':
        naive = NaiveBayes(dataset, dependant_var, indivisualInputs, unwanted_cols, testsize)
        prediction, columns = naive.make_classification()
        single_classification = naive.make_single_classification()
        mae, mse, r2score, accuracy, confusion = naive.get_success_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = naive.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = naive.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thread for the main graphs
        graph_trade = threading.Thread(target=naive.draw_graph(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'DTC':
        dtc = DecisionTreeClassification(dataset, dependant_var, indivisualInputs, criterion, unwanted_cols, testsize)
        prediction, columns = dtc.make_classification()
        single_classification = dtc.make_single_classification()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = dtc.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = dtc.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        mae, mse, r2score, accuracy, confusion = dtc.get_success_rate()
        # Thread for the main graphs
        graph_trade = threading.Thread(target=dtc.draw_graph(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'RFC':
        rfc = RandomForestClassification(dataset, dependant_var, indivisualInputs, criterion_rfc, nestimators_rfc, unwanted_cols, testsize)
        prediction, columns = rfc.make_classification()
        single_classification = rfc.make_single_classification()
        mae, mse, r2score, accuracy, confusion = rfc.get_success_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = rfc.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = rfc.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thred for the main graphs 
        graph_trade = threading.Thread(target=rfc.draw_graph(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description
    elif algorithm == 'RC':
        rc = RidgeClassificationModel(dataset, dependant_var, indivisualInputs, alpha_ridge, max_iteration, solver, unwanted_cols, testsize)
        prediction, columns = rc.make_classification()
        single_classification = rc.make_single_classification()
        mae, mse, r2score, accuracy, confusion = rc.get_success_rate()
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs = rc.get_more_relation_graphs(app)
        dataset_description, dataset_columns, dataset_shape, dataset_memory = rc.model_summary()
        model_description = {"dataset_description":dataset_description, "dataset_columns": dataset_columns, "dataset_shape": dataset_shape, "dataset_memory": dataset_memory, "algorithm":algorithm, "dependant_var": dependant_var, "testsize": testsize, "unwanted_cols": unwanted_cols }
        # Thred for the main graphs 
        graph_trade = threading.Thread(target=rc.draw_graph(filename1, filename2, filename3))
        graph_trade.setDaemon(True)
        graph_trade.start()
        return prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description
