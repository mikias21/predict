import numpy as np 
import pandas as pd 
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
np.seterr(divide='ignore', invalid='ignore')

from LearningModels.process_data import ProcessData
from LearningModels.auxilary_methods import Auxilary
from LearningModels.more_graphs import MoreGraphs

process_data = ProcessData()
auxilary = Auxilary()
more_graphs = MoreGraphs()

class PolynomialRegression(object):

    def __init__(self, data, dependant_var, indivisualInputs, unwanted_cols = None, testsize = None, polyregdegree = None):
        self.data = pd.read_csv(data)
        self.testsize = testsize if testsize != "" else 0.25
        # self.dependant_var = self.data.columns.get_loc(dependant_var)
        self.polyregdegree = 4
        if type(polyregdegree) is str and len(polyregdegree) != 0:
            self.polyregdegree = int(polyregdegree)
        self.dependant_var = dependant_var
        self.regressor_one = LinearRegression()
        self.regressor_two = LinearRegression()
        self.polynomial_features = PolynomialFeatures(degree=self.polyregdegree)
        self.unwanted_cols = unwanted_cols.split(",") if unwanted_cols is not None and type(unwanted_cols) is str else None
        self.indivisual_inputs = indivisualInputs
        self.x, self.y, self.x_train, self.x_test, self.y_train, self.y_test, self.dependant_var_str, self.indpendant_var_str = process_data.process_data(self.data, self.dependant_var, self.unwanted_cols, self.testsize)

    def make_regression(self):
        if self.x_train is None and self.y_train is None:
            self.x_poly = self.polynomial_features.fit_transform(self.x)
            self.regressor_one.fit(self.x, self.y)
            self.regressor_two.fit(self.x_poly, self.y)
            self.prediction_one = self.regressor_one.predict(self.x)
            self.prediction_two = self.regressor_two.predict(self.x_poly)
        else:
            self.x_train_poly = self.polynomial_features.fit_transform(self.x_train)
            self.x_test_poly = self.polynomial_features.fit_transform(self.x_test)
            self.regressor_one.fit(self.x_train, self.y_train)
            self.regressor_two.fit(self.x_train_poly, self.y_train)
            self.prediction_one = self.regressor_one.predict(self.x_test)
            self.prediction_two = self.regressor_two.predict(self.x_test_poly)
        return self.prediction_two, self.data.columns

    
    def make_single_prediction(self):
        prediction_val = []
        columns = list(self.data.columns)
        indivisualInputs = json.loads(self.indivisual_inputs)
        keys = list(indivisualInputs.keys())

        # Add column data to the list
        for col in columns:
            if col in keys:
                prediction_val.insert(columns.index(col), np.float(indivisualInputs[col]))

        prediction_val = prediction_val[0] if len(prediction_val) == 1 else prediction_val
        # Change the array to numpy array
        prediction_val = self.polynomial_features.fit_transform([[prediction_val]]) 
        # Make prediction
        single_prediction = self.regressor_two.predict(prediction_val)
        return single_prediction
    
    def get_seccess_rate(self):
        mae, mse, r2score, rmse, explained_score = auxilary.get_seccess_rate(self.y, self.y_test, self.prediction_two)
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
        print(dataset_description)
        print(dataset_columns)
        print(dataset_shape)
        print(dataset_memory)
        return dataset_description, dataset_columns, dataset_shape, dataset_memory

    def draw_graphs(self, filename1, filename2):
        # draw the graphs
        if self.x_train is not None and self.y_train is not None:
            plt.scatter(self.x_train, self.y_train, color='red')
            plt.plot(self.x_test, self.regressor_one.predict(self.x_test), color='green')
            plt.title("Linear Regression with out Polynomial Features")
            plt.xlabel(str(self.indpendant_var_str))
            plt.ylabel(str(self.dependant_var_str))
            plt.savefig(filename1)
            plt.close()

            plt.scatter(self.x_train, self.y_train, color='red')
            plt.plot(self.x_test, self.regressor_two.predict(self.polynomial_features.fit_transform(self.x_test)), color='green')
            plt.title("Linear Regression with Polynomial Features")
            plt.xlabel(str(self.indpendant_var_str))
            plt.ylabel(str(self.dependant_var_str))
            plt.savefig(filename2)
            plt.close()
        else:
            plt.scatter(self.x, self.y, color='red')
            plt.plot(self.x, self.regressor_two.predict(self.polynomial_features.fit_transform(self.x)), color='green')
            plt.title("Linear Regression with Polynomial Features")
            plt.xlabel(str(self.indpendant_var_str))
            plt.ylabel(str(self.dependant_var_str))
            plt.savefig(filename2)
            plt.close()

            plt.scatter(self.x, self.y, color='red')
            plt.plot(self.x, self.regressor_one.predict(self.x), color='green')
            plt.title("Linear Regression with out Polynomial Features")
            plt.xlabel(str(self.indpendant_var_str))
            plt.ylabel(str(self.dependant_var_str))
            plt.savefig(filename1)
            plt.close()
