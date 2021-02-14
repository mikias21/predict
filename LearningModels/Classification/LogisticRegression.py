import pandas as pd
from sklearn.linear_model import LogisticRegression

from LearningModels.process_data import ProcessData
from LearningModels.auxilary_methods import Auxilary
from LearningModels.more_graphs import MoreGraphs
process_data = ProcessData()
auxilary = Auxilary()
more_graphs = MoreGraphs()

class LogisticRegresser(object):
    
    def __init__(self, data, dependant_var, indivisualInputs, unwanted_cols = None, testsize = None):
        self.data = pd.read_csv(data)
        self.unwanted_cols = unwanted_cols.split(",") if unwanted_cols is not None and type(unwanted_cols) is str else None 
        self.indivisual_inputs = indivisualInputs 
        self.test_size = testsize if testsize != "" else 0.25
        self.dependant_var = dependant_var
        self.classifier = LogisticRegression()
        self.scaler = None
        self.prediction = None
        self. x, self. y, self. x_train, self. x_test, self. y_train, self. y_test, self. dependant_var_str, self. indpendant_var_str, self.scaler =  process_data.process_data_classification(self.data, self.dependant_var, self.unwanted_cols, self.test_size)
    
    def make_classification(self):
        self.prediction = auxilary.make_classification(self.x, self.y, self.x_train, self.x_test, self.y_train, self.classifier)
        return self.prediction, self.data.columns
    
    def make_single_classification(self):
        # Create new data frame
        single_classification = auxilary.make_single_classification(self.indivisual_inputs, self.classifier, self.scaler)
        return single_classification
    
    def get_success_rate(self):
        mae, mse, r2score, accuracy, confusion = auxilary.get_success_rate_classification(self.y, self.y_test, self.prediction)
        return mae, mse, r2score, accuracy, confusion
    
    def draw_graph(self, filename1, filename2, filename3):
        title1 = "Logstic Regression Testing Data set"
        title2 = "Logstic Regression Testing Data set"
        auxilary.draw_graph_classification(self.x, self.y, self.x_train, self.x_test, self.y_train, self.y_test, 
        self.indpendant_var_str, self.dependant_var_str, filename1, filename2, filename3, self.classifier, title1, title2, self.prediction)
    
    def get_relational_columns(self):
        relational_cols = []
        data_cols = list(self.data.columns)
        for col in data_cols:
            if col not in self.unwanted_cols and col != self.dependant_var:
                relational_cols.append(col)
        return relational_cols 

    def get_more_relation_graphs(self, app):
        relational_cols = self.get_relational_columns()
        reg_graphs = more_graphs.draw_regression_plot(app, relational_cols, self.dependant_var, self.data)
        lmp_graphs = more_graphs.draw_lmplot(app, relational_cols, self.dependant_var, self.data)
        mean_graphs = more_graphs.draw_mean_estimated_graph(app, relational_cols, self.dependant_var, self.data)
        jitter_graphs = more_graphs.draw_jitter_plot(app, relational_cols, self.dependant_var, self.data)
        joint_graphs = more_graphs.draw_joint_plot_reg(app, relational_cols, self.dependant_var, self.data)
        dist_graphs = more_graphs.distribution_graph(app, self.data[self.dependant_var])
        # Create Corelation matrix
        correlation_matrix = self.data.corr().round(2)
        correlation_graphs = more_graphs.corelation_matrix(app, correlation_matrix)
        return reg_graphs, lmp_graphs, mean_graphs, jitter_graphs, joint_graphs, dist_graphs, correlation_graphs
    
    def model_summary(self):
        dataset_description, dataset_columns, dataset_shape, dataset_memory = auxilary.dataset_summary(self.data)
        return dataset_description, dataset_columns, dataset_shape, dataset_memory
    