import re
import os
import json

class InputDataController(object):

    def __init__(self):
        self.ml_types = ['REG', 'CLASS', 'NEURAL']

        self.algos = [
            'SR', 'MR', 'PR', 'SVR', 'DTR', 'RFR', 'RR',
            'LR', 'KNN', 'SVC', 'KSC', 'NB', 'DTC', 'RFC', 'RC',
            'MLPC', 'MLPR'
        ]

        self.kernel = [
            'linear', 'poly', 'sigmoid', 'precomputed', 'rbf'
        ]

        self.metric = [
            'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'wminkowski', 'seuclidean', 'mahalanobis'
        ]

        self.criterion = [
            'gini', 'entropy'
        ]

        self.solver = [
            'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
        ]

        self.mlpc_activation = [
            'identity', 'logistic', 'tanh', 'relu'
        ]

        self.mlpc_solver = [
            'lbfgs', 'sgd', 'adam'
        ]

        self.mlpc_learning_scale = [
            'constant', 'invscaling', 'adaptive'
        ]

    def setIndivisualInputs(self, indivisualInputs):
        indivisualInputs = json.loads(indivisualInputs)
        keys = list(indivisualInputs.keys())
        values = list(indivisualInputs.values())

        if(len(keys) != len(values)):
            raise Exception("something is wrong try agin")
            return False
        for i in keys:
            if len(i) == 0:
                raise Exception("Something is wrong with setting up prediction values")
                return False 
        for val in values:
            if len(val) == 0 or len(val) > 10:
                raise Exception("Specifiy the values for indivisual values and should not be more than 10")
                return False 
            elif not re.match("^[A-Za-z0-9_-]*$", val):
                raise Exception("numbers and letters allowed only")
                return False 
        return True

    def setDependantVar(self, data):
        if data == "undefined":
            return True
        if data is None or len(data) < 1 or len(data) > 15:
            raise Exception("Dependant variable can not be empty or more than 15 letters long")
            return False
        elif not re.match("^[A-Za-z0-9_-]*$", data):
            raise Exception("Dependant variable is invalid on letters, numbers and undescores allowed")
            return False
        else:
            return True
    
    def setTestSize(self, testsize):
        if testsize == "undefined":
            return True
        elif testsize != "":
            if not re.match("^[+]?([0-9]+(?:[\.][0-9]*)?|\.[0-9]+)$", testsize):
                raise Exception("Test size should be integer or decimal only")
                return False
            elif int(float(testsize)) > 10:
                raise Exception("Test size in invalid can not be more then 10")
                return False
            else:
                return True
        else:
            return True


    def setMlType(self, type):
        if type not in self.ml_types:
            raise Exception("Invalid Machine Learning Type selected")
            return False
        else:
            return True
    
    def setAlgorithm(self, algorithm):
        if algorithm not in self.algos:
            raise Exception("Invalid Algorithm selected, please choose from thoes provided")
            return False
        else:
            return True
    
    def setCSVFile(self, file):
        if file is None:
            raise Exception("You must upload a file .CSV format")
            return False 
        else:
            extension = file.filename.split('.')[1]
            defaultsize = 10 * 1024 * 1024
            file.seek(0, 2)
            size = file.tell()
            file.seek(0, 0)
            if extension != 'csv':
                raise Exception("File must be .CSV format")
                return False
            # elif  len(file.stream.read()) > defaultsize:
            #     raise Exception("File must be .CSV format")
            #     return False 
            else:
                return True
    
    def set_polyreg_degree(self, polyregdegree):
        if polyregdegree == "undefined":
            return True
        elif type(polyregdegree) is str and len(polyregdegree) == 0:
            return True
        elif polyregdegree is None:
            return True
        else:
            polyregdegree = int(float(polyregdegree))
            if(type(polyregdegree) is int):
                return True
            else:
                raise Exception("Degree should be an Integer value / Number with out decimals")
                return False
    
    def set_kernel_type(self, kernelopt):
        if kernelopt == "undefined":
            return True
        elif type(kernelopt) is str and len(kernelopt) == 0:
            return True 
        elif kernelopt is None:
            return True 
        else:
            if not kernelopt in self.kernel:
                raise Exception("kernel type is unknown, select from the specified options")
                return False 
            return True
    
    def set_nestimators(self, nestimators):
        if nestimators == "undefined":
            return True
        elif type(nestimators) is str or len(nestimators) == 0 and nestimators == 'null':
            return True
        elif nestimators is None:
            return True
        elif not nestimators.isnumeric():
            raise Exception("nestimators is allowed to be integer only")
            return False
        else:
            return True 

    def set_numberof_neighbours(self, n_neighbours):
        if n_neighbours == 'null' or n_neighbours == 'undefined':
            return True
        elif n_neighbours is None or n_neighbours == 'undefined':
            return True
        elif not n_neighbours.isnumeric():
            raise Exception("number of neighbours is allowe to be integer only")
            return False
        else:
            return True

    def set_metric(self, metric):
        if metric == 'null' or metric == 'undefined':
            return True 
        elif metric is None or metric == 'undefined':
            return True 
        else:
            if not metric in self.metric:
                raise Exception("metric type is unknown, select from the specified options")
                return False 
            return True
    
    def set_p(self, p):
        if p == "undefined":
            return True
        elif p == 'null' or len(p) == 0:
            return True
        elif not p.isnumeric():
            raise Exception("p is allowed to be integer only")
            return False
        else:
            return True
    
    def set_criterion(self, criterion):
        if criterion == 'null' or criterion == 'undefined':
            return True 
        else:
            if not criterion in self.criterion:
                raise Exception("criterion type is unknown, select from the specified options")
                return False 
            return True
    
    def set_alpha_ridge(self, alpha):
        if alpha == 'null' or alpha == 'undefined':
            return True 
        elif not re.match("^\d*\.?\d*$", alpha):
            raise Exception("Auto must be a decimal Number only")
            return False 
        else: 
            return True

    def set_max_iteration(self, max_iteration):
        if max_iteration == 'null' or max_iteration == 'undefined':
            return True 
        elif not re.match("^\d*\.?\d*$", max_iteration):
            raise Exception("Max Iteration must be a number only")
            return False 
        else: 
            return True

    def set_solver(self, solver):
        if solver == 'null' or solver == 'undefined':
            return True 
        elif solver not in self.solver:
            raise Exception("solver type is unknown, select from the specified options")
        else:
            return True 
    
    def set_mlpc_activation(self, activation):
        if activation == 'null' or activation == 'undefined':
            return True 
        elif activation not in self.mlpc_activation:
            raise Exception("activation type is unknown, select from the specified options")
        else:
            return True 

    def set_mlpc_solver(self, solver):
        if solver == 'null' or solver == 'undefined':
            return True 
        elif solver not in self.mlpc_solver:
            raise Exception("solver type is unknown, select from the specified options")
        else:
            return True 

    def set_mlpc_learning_rate(self, learning_rate):
        if learning_rate == 'null' or learning_rate == 'undefined':
            return True 
        elif learning_rate not in self.mlpc_learning_scale:
            raise Exception("learning rate type is unknown, select from the specified options")
        else:
            return True 