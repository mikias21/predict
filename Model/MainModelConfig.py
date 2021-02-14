import sqlite3

class MainModelConfig:

    def __init__(self):
        self.db = "./DB/predict_main.db"
        self.con = sqlite3.connect(self.db, check_same_thread=False)
        self.cursor = self.con.cursor()
        

    