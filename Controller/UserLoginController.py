import re 
import random
from Model.UserMainModel import UserMainModel

class UserLoginController(object):
    
    def __init__(self):
        self.email_regex = "^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$"

    def set_email(self, email): 
        if len(email) == 0 or len(email) > 40:
            raise Exception("Email can not be empty or more than 40 characthers")
            return False 
        elif not re.match(self.email_regex, email):
            raise Exception("Email is invalid. use valid email address")
            return False 
        return True 
    
    def set_password(self, password):
        if len(password) < 8 or len(password) > 20:
            raise Exception("Email and password combination is invalid")
            return False 
        return True 

class UserSignupController(UserLoginController):
    def __init__(self):
        super().__init__()
        self.name_regex = "^[A-Za-z.\s_-]+$"
        self.issue_regex = "^[\.a-zA-Z0-9,? ]*$"
        self.allowed_extensions = ['jpg', 'jpeg', 'png']
        self.model = UserMainModel()


    def set_name(self, name):
        if len(name) < 1 or len(name) > 40:
            raise Exception("Name can not be empty or more than 40 letters")
            return False 
        elif not re.match(self.name_regex, name):
            raise Exception("Only Letters and spaces are allowed for name")
            return False 
        else:
            return True 
    
    def set_email(self, email):
        if len(email) == 0 or len(email) > 40:
            raise Exception("Email can not be empty or more than 40 characthers")
            return False 
        elif not re.match(self.email_regex, email):
            raise Exception("Email is invalid. use valid email address")
            return False
        elif self.model.check_email(email):
            raise Exception("Email is already used, either Login or use another email")
            return False
        else: return True

    def set_password(self, password, cpass):
        if len(password) < 8 or len(password) > 20:
            raise Exception("password can not be less than 8 or more than 20 characters")
            return False 
        elif password != cpass:
            raise Exception("Please confirm password")
            return False 
        else: return True 
    
    def get_user_key(self):
        userhash = random.getrandbits(128)
        return str(userhash)
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def set_user_profile_picture(self, image):
        if image is None:
            raise Exception("You must upload a file .CSV format")
            return False 
        return True

    def set_issue(self, issue):
        if len(issue) <= 0:
            raise Exception("tell's us your issue")
            return False 
        elif not re.match(self.issue_regex, issue):
            raise Exception("letters, numbers ? , . allowed")
            return False 
        return True 
    
    def set_email_issue(self, email):
        if len(email) == 0 or len(email) > 40:
            raise Exception("Email can not be empty or more than 40 characthers")
            return False 
        elif not re.match(self.email_regex, email):
            raise Exception("Email is invalid. use valid email address")
            return False
        else: return True
    

    