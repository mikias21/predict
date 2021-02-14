from Model.MainModelConfig import MainModelConfig

class UserMainModel(MainModelConfig):
    def __init__(self):
        super().__init__()

    def insert_signup_user(self, name, email, password, userkey): 
        if self.cursor.execute("INSERT INTO signup_users VALUES(NULL, ?, ?, ?, ?, NULL, NULL)", (name, email, password, userkey)):
            self.con.commit()
            return True 
        return False 
    
    def check_email(self, email):
        self.cursor.execute("SELECT * FROM signup_users WHERE user_email = ?", (email,))
        data = self.cursor.fetchall()
        return data
    
    def insert_login_user(self, name, email, userkey):
        if self.cursor.execute("INSERT INTO login_users VALUES(NULL, ?, ?, ?, NULL)", (name, email, userkey)):
            self.con.commit()
            return True 
        return False 
    
    def change_password(self, email, password):
        if self.cursor.execute("UPDATE signup_users SET user_password = ? WHERE user_email = ? ", (password, email)):
            self.con.commit()
            return True 
        return False

    def get_signup_date(self, email):
        self.cursor.execute("SELECT user_signup_date FROM signup_users WHERE user_email = ?", (email,))
        data = self.cursor.fetchall()
        return data
    
    def insert_user_picture(self, email, path):
        if self.cursor.execute("UPDATE signup_users SET user_picture = ? WHERE user_email = ?", (path, email)):
            self.con.commit()
            return True 
        return False
    
    def inser_google_user(self, username, useremail, userpicture, userkey):
        if self.cursor.execute("INSERT INTO google_users_signin VALUES(NULL, ?, ?, ?, ?, NULL)", (username, useremail, userpicture, userkey)):
            self.con.commit()
            return True 
        return False

    def check_google_user(self, email):
        self.cursor.execute("SELECT * FROM google_users_signin WHERE user_email = ?", (email,))
        data = self.cursor.fetchall()
        return data 
    
    def get_google_signup_date(self, email):
        self.cursor.execute("SELECT signup_date FROM google_users_signin WHERE user_email = ?", (email,))
        data = self.cursor.fetchall()
        return data
    
    def insert_issue_db(self, email, issue):
        if self.cursor.execute("INSERT INTO user_issue VALUES(NULL, ?, ?, NULL)", (email, issue)):
            self.con.commit()
            return True 
        return False  
    
    def get_user_key(self, email):
        self.cursor.execute("SELECT user_key FROM signup_users WHERE user_email = ?", (email,))
        data = self.cursor.fetchall()
        return data