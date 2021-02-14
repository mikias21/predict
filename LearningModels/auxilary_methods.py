import numpy as np
import pandas as pd
import json
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston, load_diabetes, load_iris
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
matplotlib.use('Agg')
np.seterr(divide='ignore', invalid='ignore')

class Auxilary(object):

    def make_prediction(self, x, y, x_train, x_test, y_train, regressor):
        if x_train is not None and y_train is not None:
            regressor.fit(x_train, y_train)  # Fit the model
            predictions = regressor.predict(x_test)  # Make Prediction
            return predictions 
        else:
            regressor.fit(x, y)
            predictions = regressor.predict(x)
            return predictions
    
    def make_classification(self, x, y, x_train, x_test, y_train, classifier):
        if x_train is not None and y_train is not None:
            classifier.fit(x_train, y_train)
            prediction = classifier.predict(x_test)
            return prediction
        else:
            classifier.fit(x, y)
            prediction = classifier.predict(x)
            return prediction

    def make_single_prediction(self, columns, indivisualInputs, regressor):
        prediction_val = []
        columns = list(columns)
        indivisualInputs = json.loads(indivisualInputs)
        keys = list(indivisualInputs.keys())

        # Add column data to the list
        for col in columns:
            if col in keys:
                prediction_val.insert(columns.index(col), np.float(indivisualInputs[col]))

        # Change the array to numpy array
        prediction_val = np.array(prediction_val).reshape(-1, 1) 
        # Make prediction
        single_prediction = regressor.predict(prediction_val)
        return single_prediction
    
    def make_single_classification(self, indivisualInputs, classifier, scaler):
        # Create new data frame
        indivisualInputs = json.loads(indivisualInputs)
        indivisual_values = []
        for i in list(indivisualInputs.values()):
            indivisual_values.append(int(float(i)))
        indivisual_values = np.array([indivisual_values])
        indivisual_values = scaler.transform(indivisual_values)
        single_prediction = classifier.predict(indivisual_values)
        return single_prediction
    
    def get_seccess_rate(self, y, y_test, prediction):
        if y_test is not None and prediction is not None:
            mae = mean_absolute_error(y_test, prediction)
            mse = mean_squared_error(y_test, prediction)
            r2score = r2_score(y_test, prediction)
            rmse = math.sqrt(mse)
            # explained_score = explained_variance_score(y_test, prediction)
            return mae, mse, r2score, rmse, 'Not Avaliable'
        elif prediction is not None:
            mae = mean_absolute_error(y, prediction)
            mse = mean_squared_error(y, prediction)
            r2score = r2_score(y, prediction)
            rmse = math.sqrt(mse)
            # explained_score = explained_variance_score(y_test, prediction)
            return mae, mse, r2score, rmse, 'Not Avaliable'
    
    def get_success_rate_classification(self, y, y_test, prediction):
        if y_test is not None and prediction is not None:
            mae = mean_absolute_error(y_test, prediction)
            mse = mean_squared_error(y_test, prediction)
            r2score = r2_score(y_test, prediction)
            accuracy = accuracy_score(y_test, prediction)
            confusion = confusion_matrix(y_test, prediction)
            return mae, mse, r2score, accuracy, confusion
        elif prediction is not None:
            mae = mean_absolute_error(y, prediction)
            mse = mean_squared_error(y, prediction)
            r2score = r2_score(y, prediction)
            accuracy = accuracy_score(y, prediction)
            confusion = confusion_matrix(y, prediction)
            return mae, mse, r2score, accuracy, confusion
    
    def dataset_summary(self, dataset):
        dataset_description = {"count":str(dataset.count().to_dict()), "max":str(dataset.max().to_dict()), "min":str(dataset.max().to_dict()),
                                 "std": str(dataset.std().to_dict()), "mean": str(dataset.mean().to_dict())}
        dataset_columns = dataset.columns.to_list()
        dataset_shape = dataset.shape
        dataset_memory = dataset.memory_usage().to_dict()
        return dataset_description, dataset_columns, dataset_shape, dataset_memory

    def get_preloaded_dataset(self, datatitle):
        if datatitle == "boston":
            boston_dataset = load_boston()
            dataset = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)  
            dataset['MEDV'] = boston_dataset.target
            return dataset
        elif datatitle == "diabetes":
            diabetes_dataset = load_diabetes()
            dataset = pd.DataFrame(data=diabetes_dataset.data, columns=diabetes_dataset.feature_names)  
            return dataset
        elif datatitle == "iris":
            iris_dataset = load_iris()
            dataset = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
            return dataset

    
    def draw_graph(self, x, y, x_train, x_test, y_train, y_test, indpendant_var_str, dependant_var_str, regressor, filename1, filename2):
        
        if x_train is not None or y_train is not None:
            # x = np.array(x, dtype='float')
            # y = np.array(y, dtype='float')
            # x_train = np.array(x_train, dtype='float')
            # x_test = np.array(x_test, dtype='float')
            # y_train = np.array(x_test, dtype='float')
            # y_test = np.array(y_test, dtype='float')

            # np.isfinite(x).all()
            # np.isfinite(y).all()
            # np.isfinite(x_train).all()
            # np.isfinite(x_test).all()
            # np.isfinite(y_train).all()
            # np.isfinite(y_test).all()

            # Graph for the training data set
            plt.scatter(x_train, y_train, color='red')
            plt.plot(x_train, regressor.predict(x_train), color='green')
            plt.title("Simple Linear Regression Training Data Set")
            plt.xlabel(str(indpendant_var_str))
            plt.ylabel(str(dependant_var_str))
            plt.savefig(filename1)
            plt.close()

            # Graph for the testing data set
            plt.scatter(x_test, y_test, color='red')
            plt.plot(x_test, regressor.predict(x_test), color='green')
            plt.title("Simple Linear Regression Testing Data set")
            plt.xlabel(str(indpendant_var_str))
            plt.ylabel(str(dependant_var_str))
            plt.savefig(filename2)
            plt.close()
        else:
            # Graph for the training data set
            plt.scatter(x, y, color='red')
            plt.plot(x, regressor.predict(x), color='green')
            plt.title("Simple Linear Regression Training Data Set")
            plt.xlabel(str(indpendant_var_str))
            plt.ylabel(str(dependant_var_str))
            plt.savefig(filename1)
            plt.close()

            # Graph for the testing data set
            plt.scatter(x, y, color='red')
            plt.plot(x, regressor.predict(x), color='green')
            plt.title("Simple Linear Regression Testing Data set")
            plt.xlabel(str(indpendant_var_str))
            plt.ylabel(str(dependant_var_str))
            plt.savefig(filename2)
            plt.close()
    
    def draw_graph_grid(self, x, y, x_train, x_test, y_train, y_test, indpendant_var_str, dependant_var_str, regressor, filename1, filename2):

        

        if x_train is not None or y_train is not None:

            # Change values to float
            # x = np.array(x, dtype='float')
            # y = np.array(y, dtype='float')
            # x_train = np.array(x_train, dtype='float')
            # x_test = np.array(x_test, dtype='float')
            # y_train = np.array(x_test, dtype='float')
            # y_test = np.array(y_test, dtype='float')

            # np.isfinite(x).all()
            # np.isfinite(y).all()
            # np.isfinite(x_train).all()
            # np.isfinite(x_test).all()
            # np.isfinite(y_train).all()
            # np.isfinite(y_test).all()

            # Graph for the training data set
            grid = np.arange(min(x_train), max(x_train), 0.01)
            grid = grid.reshape((len(grid), 1)) 
            plt.scatter(x_train, y_train, color='red')
            plt.plot(x_train, regressor.predict(x_train), color='green')
            plt.title("Simple Linear Regression Training Data Set")
            plt.xlabel(str(indpendant_var_str))
            plt.ylabel(str(dependant_var_str))
            plt.savefig(filename1)
            plt.close()

            # Graph for the testing data set
            grid = np.arange(min(x_test), max(x_test), 0.01)
            grid = grid.reshape((len(grid), 1))
            plt.scatter(x_test, y_test, color='red')
            plt.plot(x_test, regressor.predict(grid), color='green')
            plt.title("Simple Linear Regression Testing Data set")
            plt.xlabel(str(indpendant_var_str))
            plt.ylabel(str(dependant_var_str))
            plt.savefig(filename2)
            plt.close()
        else:
            # Graph for the training data set
            grid = np.arange(min(x), max(x), 0.01)
            grid = grid.reshape((len(grid), 1))
            plt.scatter(x, y, color='red')
            plt.plot(grid, regressor.predict(grid), color='green')
            plt.title("Simple Linear Regression Training Data Set")
            plt.xlabel(str(indpendant_var_str))
            plt.ylabel(str(dependant_var_str))
            plt.savefig(filename1)
            plt.close()

            # # Graph for the testing data set
            plt.scatter(x, y, color='red')
            plt.plot(grid, regressor.predict(grid), color='green')
            plt.title("Simple Linear Regression Testing Data set")
            plt.xlabel(str(indpendant_var_str))
            plt.ylabel(str(dependant_var_str))
            plt.savefig(filename2)
            plt.close()
        
    def draw_graph_classification(self, x, y, x_train, x_test, y_train, y_test, indpendant_var_str, dependant_var_str, filename1, filename2, filename3,
        classifier, title1, title2, prediction):
            
        np.isfinite(x).all()
        np.isfinite(y).all()
        np.isfinite(x_train).all()
        np.isfinite(x_test).all()
        np.isfinite(y_train).all()
        np.isfinite(y_test).all()
        
        if x_train is not None and y_train is not None:
            x_set, y_set = x_train, y_train
            x1, x2 = np.meshgrid(
                np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
                np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01)
            )
            plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), cmap=ListedColormap(('red', 'green')), alpha=0.75)
            plt.xlim(x1.min(), x1.max())
            plt.ylim(x2.min(), x2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], cmap=ListedColormap(('red', 'green'))(j), label=i)
            plt.title(title1)
            plt.xlabel(indpendant_var_str)
            plt.ylabel(dependant_var_str)
            plt.legend()
            plt.savefig(filename1)
            plt.close()

            x_set, y_set = x_test, y_test
            x1, x2 = np.meshgrid(
                np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
                np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01)
            )
            plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), cmap=ListedColormap(('red', 'green')), alpha=0.75)
            plt.xlim(x1.min(), x1.max())
            plt.ylim(x2.min(), x2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], cmap=ListedColormap(('red', 'green'))(j), label=j)
            plt.title(title2)
            plt.xlabel(indpendant_var_str)
            plt.ylabel(dependant_var_str)
            plt.legend()
            plt.savefig(filename2)
            plt.close()

            # Draw the seaborn graph for confusion matrix
            score = classifier.score(x_test, y_test)
            cm = confusion_matrix(y_test, prediction)
            plt.figure(figsize=(7, 7))
            sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            all_sample_title = 'Accuracy Score: {0}'.format(score)
            plt.title(all_sample_title, size=15)
            plt.savefig(filename3)
            plt.close()

        else:
            x_set, y_set = x, y
            x1, x2 = np.meshgrid(
                np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
                np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01)
            )
            plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), cmap=ListedColormap(('red', 'green')), alpha=0.75)
            plt.xlim(x1.min(), x1.max())
            plt.ylim(x2.min(), x2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], cmap=ListedColormap(('red', 'green'))(j), label=i)
            plt.title(title1)
            plt.xlabel(indpendant_var_str)
            plt.ylabel(dependant_var_str)
            plt.legend()
            plt.savefig(filename1)
            plt.close()

            x_set, y_set = x, y
            x1, x2 = np.meshgrid(
                np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
                np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01)
            )
            plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), cmap=ListedColormap(('red', 'green')), alpha=0.75)
            plt.xlim(x1.min(), x1.max())
            plt.ylim(x2.min(), x2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], cmap=ListedColormap(('red', 'green'))(j), label=i)
            plt.title(title2)
            plt.xlabel(indpendant_var_str)
            plt.ylabel(dependant_var_str)
            plt.legend()
            plt.savefig(filename2)
            plt.close()
    

    def draw_neural_net_classification(self, filename1, filename2, filename3, indivisual_inputs, x_test, y_test, prediction, classifier):

        np.isfinite(x_test).all()
        np.isfinite(y_test).all()
    
        layer1 = int(len(json.loads(indivisual_inputs)))
        layer_sizes = [layer1, 10, 1]
        left, right, bottom, top = .1, .9, .1, .9
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis("off")
        n_layers = len(layer_sizes)
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)
        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                    [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                    ax.add_artist(line)

        # Get draw the graph for the confusion matrix

        # if x_test is not None and y_test is not None:
        #     score = classifier.score(x_test, y_test)
        #     cm = confusion_matrix(y_test, prediction)
        #     plt.figure(figsize=(7, 7))
        #     sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
        #     plt.ylabel('Actual label')
        #     plt.xlabel('Predicted label')
        #     all_sample_title = 'Accuracy Score: {0}'.format(score)
        #     plt.title(all_sample_title, size=15)
        #     plt.savefig(filename3)
        #     plt.close() 

        plt.savefig(filename1)
        plt.savefig(filename2)
        plt.savefig(filename3)

    def draw_neural_net_regression(self, filename1, filename2, indivisual_inputs):
        
        layer1 = int(len(json.loads(indivisual_inputs)))
        layer_sizes = [layer1, 10, 1]
        left, right, bottom, top = .1, .9, .1, .9
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis("off")
        n_layers = len(layer_sizes)
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)
        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                    [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                    ax.add_artist(line)

        plt.savefig(filename1)
        plt.savefig(filename2)
    
    def draw_graph_preloaded(self, x, y, x_train, x_test, y_train, y_test, indpendant_var_str, dependant_var_str, regressor, filename1, filename2):
        if x_train is None or y_train is None:
            plt.scatter(x, y)
            plt.xlabel(indpendant_var_str)
            plt.ylabel(dependant_var_str)
            plt.title("Plotted graph")
            plt.savefig(filename1)
            plt.close()
        else:
            plt.scatter(x_train, y_train)
            plt.xlabel(indpendant_var_str)
            plt.ylabel(dependant_var_str)
            plt.title("Plotted graph")
            plt.savefig(filename1)
            plt.close()