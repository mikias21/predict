import pandas as pd
import numpy as np 
import json
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')

from LearningModels.process_data import ProcessData
from LearningModels.auxilary_methods import Auxilary
from LearningModels.more_graphs import MoreGraphs

process_data = ProcessData()
auxilary = Auxilary()
more_graphs = MoreGraphs()

class SupportVectorRegressionPreloaded(object):
    def __init__(self, data, dependant_var, indivisualInputs, kernelopt, unwanted_cols = None, testsize = None):
        self.data = data
        self.testsize = testsize if testsize != "" else 0.25 
        self.kernelopt = kernelopt if kernelopt is not None else 'rbf'
        self.regressor = SVR(n_components=2, kernel=self.kernelopt)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.dependant_var = dependant_var
        self.unwanted_cols = unwanted_cols.split(",") if unwanted_cols is not None and type(unwanted_cols) is str else None
        self.indivisualInputs = indivisualInputs
        self.predictions = None 
        # self.x, self.y, self.x_train, self.x_test, self.y_train, self.y_test, self.dependant_var_str, self.indpendant_var_str = process_data.process_data(self.data, self.dependant_var, self.unwanted_cols, self.testsize)
        self.process_data()
    
    def process_data(self):
        self.x = self.data.iloc[:, :self.data.columns.get_loc(self.dependant_var)].values
        self.y = self.data[self.dependant_var].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.testsize, random_state=5)
        self.x_train = self.x_train.reshape(-1, 1)
        self.x_test = self.x_test.reshape(-1, 1)
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_test = self.y_test.reshape(-1, 1)
        self.x_train = self.scaler_x.fit_transform(self.x_train)
        self.x_test = self.scaler_x.fit_transform(self.x_test)
        self.y_train = self.scaler_y.fit_transform(self.y_train)
        self.y_test = self.scaler_y.fit_transform(self.y_test)
        

    def make_regression(self):
        # self.predictions = auxilary.make_prediction(self.x, self.y, self.x_train, self.x_test, self.y_train, self.regressor)
        self.regressor.fit(self.x_train, self.y_train)
        self.predictions = self.regressor.predict(self.x_test)
        return self.predictions, self.data.columns
    
    def make_single_prediction(self):
        client_input = json.loads(self.indivisualInputs)
        client_input = list(client_input.values())
        for i in range(len(client_input)):
            client_input[i] = int(client_input[i])
        client_input = [client_input]
        single_prediction = self.regressor.predict(client_input)
        return single_prediction
    
    def get_seccess_rate(self):
        mae, mse, r2score, rmse, explained_score = auxilary.get_seccess_rate(self.y, self.y_test, self.predictions)
        return mae, mse, r2score, rmse, explained_score
    
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
        distplot = more_graphs.distribution_graph(app, self.data[self.dependant_var])
        # Create correlation matrix 
        correlation_matrix = self.data.corr().round(2)
        corr_graphs = more_graphs.corelation_matrix(app, correlation_matrix);
        return reg_graphs, distplot, corr_graphs
    
    def model_summary(self):
        dataset_description, dataset_columns, dataset_shape, dataset_memory = auxilary.dataset_summary(self.data)
        return dataset_description, dataset_columns, dataset_shape, dataset_memory