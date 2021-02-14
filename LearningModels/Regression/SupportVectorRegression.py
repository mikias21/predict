import pandas as pd
import numpy as np 
import json
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')

from LearningModels.process_data import ProcessData
from LearningModels.auxilary_methods import Auxilary
from LearningModels.more_graphs import MoreGraphs

process_data = ProcessData()
auxilary = Auxilary()
more_graphs = MoreGraphs()

class SupportVectorRegression(object):
    def __init__(self, data, dependant_var, indivisualInputs, kernelopt, unwanted_cols = None, testsize = None):
        self.data = pd.read_csv(data)
        self.testsize = testsize if testsize != "" else 0.25 
        self.kernelopt = kernelopt if kernelopt is not None else 'rbf'
        self.regressor = SVR(kernel=self.kernelopt)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.dependant_var = dependant_var
        self.unwanted_cols = unwanted_cols.split(",") if unwanted_cols is not None and type(unwanted_cols) is str else None
        self.indivisualInputs = indivisualInputs
        self.predictions = None 
        self.x, self.y, self.x_train, self.x_test, self.y_train, self.y_test, self.dependant_var_str, self.indpendant_var_str = process_data.process_data(self.data, self.dependant_var, self.unwanted_cols, self.testsize)
        if self.x_train is not None and self.y_train is not None:
                self.x_train = self.scaler_x.fit_transform(self.x_train)
                self.x_test = self.scaler_x.fit_transform(self.x_test)
                self.y_train = self.scaler_y.fit_transform(self.y_train)
                self.y_test = self.scaler_y.fit_transform(self.y_test)
        else:
            self.x = self.scaler_x.fit_transform(self.x)
            self.y = self.scaler_y.fit_transform(self.y)
    
    def make_regression(self):
        self.predictions = auxilary.make_prediction(self.x, self.y, self.x_train, self.x_test, self.y_train, self.regressor)
        return self.predictions, self.data.columns
    
    def make_single_prediction(self):
        prediction_val = []
        columns = list(self.data.columns)
        indivisualInputs = json.loads(self.indivisualInputs)
        keys = list(indivisualInputs.keys())

        # Add column data to the list
        for col in columns:
            if col in keys:
                prediction_val.insert(columns.index(col), np.float(indivisualInputs[col]))

        # Change the array to numpy array
        prediction_val = self.scaler_x.transform(np.array(prediction_val).reshape(-1, 1))
        # Make prediction
        single_prediction = self.regressor.predict(prediction_val)
        return self.scaler_y.inverse_transform(single_prediction) 
    
    def get_seccess_rate(self):
        mae, mse, r2score, rmse, explained_score = auxilary.get_seccess_rate(self.y, self.y_test, self.predictions)
        return mae, mse, r2score, rmse, explained_score
    
    def draw_graphs(self, filename1, filename2):
        auxilary.draw_graph_grid(self.x, self.y, self.x_train, self.x_test, self.y_train, self.y_test, self.indpendant_var_str, self.dependant_var_str, self.regressor, filename1, filename2)
    
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
        mean_graphs = more_graphs.draw_mean_estimated_graph(app, relational_cols, self.dependant_var, self.data)
        jitter_graphs = more_graphs.draw_jitter_plot(app, relational_cols, self.dependant_var, self.data)
        joint_graphs = more_graphs.draw_joint_plot_reg(app, relational_cols, self.dependant_var, self.data)
        pairplot = more_graphs.pairplot(app, self.data)
        distplot = more_graphs.distribution_graph(app, self.data[self.dependant_var])
        # Create correlation matrix 
        correlation_matrix = self.data.corr().round(2)
        corr_graphs = more_graphs.corelation_matrix(app, correlation_matrix)
        return reg_graphs, pairplot, mean_graphs, jitter_graphs, joint_graphs, distplot, corr_graphs
    
    def model_summary(self):
        dataset_description, dataset_columns, dataset_shape, dataset_memory = auxilary.dataset_summary(self.data)
        return dataset_description, dataset_columns, dataset_shape, dataset_memory