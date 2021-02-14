import sqlite3

from typing_extensions import final 
from Model.MainModelConfig import MainModelConfig
import threading

lock = threading.Lock()

class UserDataModel(MainModelConfig):

    def __init__(self):
        super().__init__()
    
    def insert_user_config_inputs(self, filename, dependant_var, unwanted_cols, ml_type, algorithm, testsize, \
        indivisual_inputs, polyregdegree, kernelopt, nestimators, n_neighbours, metic, p, kernelopt_svc, criterion, \
        criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
        mlpr_activation, mlpr_solver, mlpr_learning_rate,  mlpr_max_iter, model_description, userkey, model_id
        ):
        
        if self.cursor.execute("INSERT INTO user_actions_history VALUES \
        (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL) \
        ", (filename, dependant_var, unwanted_cols, ml_type, algorithm, testsize, indivisual_inputs, polyregdegree, kernelopt, \
           nestimators, n_neighbours, metic, p, kernelopt_svc, criterion, criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration,
           solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, mlpr_activation, mlpr_solver, mlpr_learning_rate,  mlpr_max_iter, \
               model_description, userkey, model_id # 29
            )) :
        
            self.con.commit()
            return True
        else:
            return False
    
    def insert_regression_result(self, prediction, columns, single_prediction, filename1, filename2, mae, mse, \
        r2score, maxerror, explained_score, reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, \
        dist_graph, corr_graphs, model_description, userkey, model_id):

        if self.cursor.execute("INSERT INTO regression_result VALUES(NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)", \
            (prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, maxerror, explained_score, reg_graphs, \
            jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs, model_description, userkey, model_id)):
            self.con.commit()
            return True 
        else:
            return False 
    
    def insert_classification_result(self, prediction, columns, single_classification, filename1, filename2, mae, mse, r2score, accuracy, confusion, \
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description, userkey, model_id):
        if self.cursor.execute("INSERT INTO classification_result VALUES(NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)", \
        (prediction, columns, single_classification, filename1, filename2, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, \
        lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description, userkey, model_id)):
            self.con.commit()
            return True 
        else:
            return False
    
    def insert_neural_result(self, prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion,\
        reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description, userkey, model_id):
        
        if self.cursor.execute("INSERT INTO neural_result VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)", \
            (prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, \
            reg_graphs, jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description, userkey, model_id)):
            self.con.commit()
            return True 
        else:
            return False
    
    def select_dataset(self, userkey):
        self.cursor.execute("SELECT filename, model_description, model_id FROM user_actions_history WHERE userkey = ? ORDER BY timestamp DESC", (userkey,))
        data = self.cursor.fetchall()
        return data 
    
    def get_dataset_info(self, model_id, user_id):
        self.cursor.execute("SELECT model_description FROM user_actions_history WHERE model_id = ? AND userkey = ? ORDER BY timestamp DESC", (model_id, user_id))
        data = self.cursor.fetchall()
        return data
    
    def select_user_model_history(self, userkey):
        self.cursor.execute("SELECT * FROM user_actions_history WHERE userkey = ? ORDER BY timestamp DESC", (userkey,))
        data = self.cursor.fetchall()
        return data 

    def get_specific_model(self, id, userkey):
        self.cursor.execute("SELECT * FROM user_actions_history WHERE id = ? AND userkey = ?", (id, userkey))
        data = self.cursor.fetchall()
        return data 
    
    def delete_model(self, id, userkey):
        if self.cursor.execute("DELETE FROM user_actions_history WHERE id = ? AND userkey = ?", (id, userkey)):
            self.con.commit()
            return True 
        return False 

    def delete_regression(self, model_id, userkey):
        if self.cursor.execute("DELETE FROM regression_result WHERE model_id = ? AND userkey = ?", (model_id, userkey)):
            self.con.commit()
            return True 
        return False 
    
    def delete_classification(self, model_id, userkey):
        if self.cursor.execute("DELETE FROM classification_result WHERE model_id = ? AND userkey = ?", (model_id, userkey)):
            self.con.commit()
            return True 
        return False 
    
    def delete_neural(self, model_id, userkey):
        if self.cursor.execute("DELETE FROM neural_result WHERE model_id = ? AND userkey = ?", (model_id, userkey)):
            self.con.commit()
            return True 
        return False 
    
    def get_regression_result(self, model_id, userkey):
        self.cursor.execute("SELECT * FROM regression_result WHERE model_id = ? AND userkey = ?", (model_id, userkey))
        data = self.cursor.fetchall()
        return data 
    
    def get_classification_result(self, model_id, userkey):
        self.cursor.execute("SELECT * FROM classification_result WHERE model_id = ? AND userkey = ?", (model_id, userkey))
        data = self.cursor.fetchall()
        return data 
    
    def get_neural_result(self, model_id, userkey):
        self.cursor.execute("SELECT * FROM neural_result WHERE model_id = ? AND userkey = ?", (model_id, userkey))
        data = self.cursor.fetchall()
        return data 
    
    def get_dataset_url(self, model_id, userkey):
        self.cursor.execute("SELECT filename FROM user_actions_history WHERE model_id = ? AND userkey = ?", (model_id, userkey))
        data = self.cursor.fetchall()
        return data
    
    def delete_dataset_record(self, model_id, userkey):
        if self.cursor.execute("DELETE FROM user_actions_history WHERE model_id = ? AND userkey = ?", (model_id, userkey)):
            self.con.commit()
            return True
        return False

    def get_total_models_model(self, userkey):
        try:
            lock.acquire(True)
            self.cursor.execute("SELECT COUNT(*) FROM user_actions_history WHERE userkey = ?", (userkey,))
            rowcount = self.cursor.fetchall()
            return rowcount
        finally:
            lock.release() 
    
    def get_total_regression_model(self, userkey):
        try:
            lock.acquire(True)
            self.cursor.execute("SELECT COUNT(*) FROM regression_result WHERE userkey = ?", (userkey,))
            rowcount = self.cursor.fetchall()
            return rowcount
        finally:
            lock.release() 

    def get_total_classification_model(self, userkey):
        try:
            lock.acquire(True)
            self.cursor.execute("SELECT COUNT(*) FROM classification_result WHERE userkey = ?", (userkey,))
            rowcount = self.cursor.fetchall()
            return rowcount
        finally:
            lock.release() 
    
    def get_total_neural_model(self, userkey):
        try:
            lock.acquire(True)
            self.cursor.execute("SELECT COUNT(*) FROM neural_result WHERE userkey = ?", (userkey,))
            rowcount = self.cursor.fetchall()
            return rowcount
        finally:
            lock.release() 
