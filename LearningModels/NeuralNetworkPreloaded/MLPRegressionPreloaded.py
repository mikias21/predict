import pandas as pd
import json
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from LearningModels.process_data import ProcessData
from LearningModels.auxilary_methods import Auxilary
from LearningModels.more_graphs import MoreGraphs
process_data = ProcessData()
auxilary = Auxilary()
more_graphs = MoreGraphs()

class MLPRegressionPreloaded(object):

    def __init__(self, data, dependant_var, indivisualInputs, mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, unwanted_cols = None, testsize = None):
        self.data = data
        self.unwanted_cols = unwanted_cols.split(",") if unwanted_cols is not None and type(unwanted_cols) is str else None 
        self.indivisual_inputs = indivisualInputs 
        self.testsize = testsize if testsize != "" else 0.25
        self.dependant_var = dependant_var
        self.activation = mlpr_activation if mlpr_activation != "null" else "relu"
        self.solver = mlpr_solver if mlpr_solver != "null" else "adam"
        self.learning_rate = mlpr_learning_rate if mlpr_learning_rate != "null" else "constant"
        self.max_iter = int(float(mlpr_max_iter)) if mlpr_max_iter != "null" else 200
        self.regressor = MLPRegressor(random_state=1, max_iter=self.max_iter, learning_rate=self.learning_rate, activation=self.activation, solver=self.solver)
        self.encoder = LabelEncoder()
        self.scaler = None
        self.prediction = None
        # self. x, self. y, self. x_train, self. x_test, self. y_train, self. y_test, self. dependant_var_str, self. indpendant_var_str=  process_data.process_data(self.data, self.dependant_var, self.unwanted_cols, self.test_size)
        self.process_data()

    def process_data(self):
        self.x = self.data.iloc[:, :self.data.columns.get_loc(self.dependant_var)]
        self.y = self.data[self.dependant_var]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.testsize, random_state=5)

    def make_regression(self):
        self.predictions = auxilary.make_prediction(self.x, self.y, self.x_train, self.x_test, self.y_train, self.regressor)
        return self.predictions, self.data.columns
        
    def make_single_prediction(self):
        client_input = json.loads(self.indivisual_inputs)
        client_input = list(client_input.values())
        for i in range(len(client_input)):
            client_input[i] = int(client_input[i])
        client_input = [client_input]
        single_prediction = self.regressor.predict(client_input)
        return single_prediction

    def get_seccess_rate(self):
        mae, mse, r2score, rmse, explained_score = auxilary.get_seccess_rate(self.y, self.y_test, self.predictions)
        return mae, mse, r2score, rmse, explained_score

    def draw_graphs(self, filename1, filename2):
        auxilary.draw_graph_preloaded(self.x.values, self.y, self.x_train, self.x_test, self.y_train, self.y_test, 'independant_var', 'dependant_var', self.regressor, filename1, filename2)
    
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
        # mean_graphs = more_graphs.draw_mean_estimated_graph(app, relational_cols, self.dependant_var, self.data)
        # jitter_graphs = more_graphs.draw_jitter_plot(app, relational_cols, self.dependant_var, self.data)
        # joint_graphs = more_graphs.draw_joint_plot_reg(app, relational_cols, self.dependant_var, self.data)
        # pairplot = more_graphs.pairplot(app, self.data)
        distplot = more_graphs.distribution_graph(app, self.data[self.dependant_var])
        # Create correlation matrix 
        correlation_matrix = self.data.corr().round(2)
        corr_graphs = more_graphs.corelation_matrix(app, correlation_matrix);
        return reg_graphs, distplot, corr_graphs
    
    def model_summary(self):
        dataset_description, dataset_columns, dataset_shape, dataset_memory = auxilary.dataset_summary(self.data)
        return dataset_description, dataset_columns, dataset_shape, dataset_memory
    
    def draw_neural_net(self, filename1, filename2):
        auxilary.draw_neural_net_regression(filename1, filename2, self.indivisual_inputs)