from math import e
from threading import Thread
from flask import Flask, flash, render_template, request, session, jsonify, redirect, url_for
import flask
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import os
import uuid
import threading
import numpy as np
import json
import shutil

# Models Instances
from Instances.RegressionInstances import *
from Instances.ClassificationInstances import *
from Instances.NeuralInstances import *
# Custom Imports
from Controller.InputDataController import InputDataController
# Web Scrapping 
from Controller.DataScrapper import DataScrapper
# Dataset Markup
from Controller.DataMarkupHolder import DataMarkupHolder
# Auxilary methods
from LearningModels.auxilary_methods import Auxilary
# UserLoginSignup Controller
from Controller.UserLoginController import UserLoginController, UserSignupController
# DataBase Model
from Model.UserMainModel import UserMainModel
from Model.UserDataModel import UserDataModel
# Email Controller
from Controller.EmailSenderController import *

# local routes
# from Routes.url_routes import *

# Init flask app
app = Flask(__name__)
bcrypt = Bcrypt(app)
app.secret_key = os.urandom(24)
# Custom inits
copy_of_csv = None

# APP configs
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 
app.config['UPLOAD_FOLDER'] = "UserUploads"

# Objects 
input_controller = InputDataController()
data_scrapper = DataScrapper()
dataset_markup = DataMarkupHolder()
auxilary = Auxilary()
userlogin = UserLoginController()
usersignup = UserSignupController()
model = UserMainModel()
datamodel = UserDataModel()


# ---------------- Routes -------------- #

@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/datasets', methods=['GET', 'POST'])
def resutls():
    if request.method == 'GET':
        return render_template('datasets.html')

@app.route('/datasetHistory', methods=['GET'])
def dataset_history():
    if request.method == 'GET':
        return render_template('dataset_history.html')

@app.route('/models', methods=['GET'])
def models():
    if request.method == 'GET':
        return render_template('models.html')

@app.route('/settings', methods=['GET'])
def settings():
    if request.method == 'GET':
        return render_template('settings.html')

@app.route('/about', methods=['GET'])
def about():
    if request.method == 'GET':
        return render_template('about.html')

@app.route("/help", methods=['GET'])
def help():
    if request.method == 'GET':
        return render_template('help.html')


@app.route('/forgot', methods=['GET'])
def forgot():
    if request.method == 'GET':
        return render_template('forgot.html')

@app.route('/recover', methods=['GET'])
def recover():
    if request.method == 'GET':
        return render_template('recover.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['pass']
    try:
        if(userlogin.set_email(email) and userlogin.set_password(password)):
            userdata = model.check_email(email)
            if userdata:
                username = userdata[0][1]
                useremail = userdata[0][2]
                userpassword = userdata[0][3]
                userkey = userdata[0][4]
                userpicture = userdata[0][5]
                if(bcrypt.check_password_hash(userpassword, password)):
                    if model.insert_login_user(username, useremail, userkey):
                        session['username'] = username
                        session['useremail'] = email 
                        session['userkey'] = userkey
                        session['userpicture'] = userpicture
                        # Check for user directory
                        useruploads = os.path.join(os.path.dirname(app.instance_path), "UserUploads", useremail)
                        if not os.path.exists(useruploads):
                            if os.mkdir(useruploads):
                                session['userdir'] = useruploads
                            else:
                                raise Exception("Unable to Login try again")
                        session['userdir'] = useruploads
                        return url_for('index')
                    else: 
                        raise Exception("Email and password combination is invalid")
                else:
                    raise Exception("Email and password combination is invalid")
            else:
                raise Exception("Email and password combination is invalid")
    except Exception as e:
        return str(e)

@app.route('/google_signin', methods=['POST'])
def google_signin():
    if request.method == 'POST':
        username = request.form['username']
        useremail = request.form['useremail']
        userpicture = request.form['userpicture']
        userhash = usersignup.get_user_key()
        if model.check_google_user(useremail):
            session['username'] = username
            session['useremail'] = useremail 
            session['userkey'] = userhash 
            session['userpicture'] = userpicture
            useruploads = os.path.join(os.path.dirname(app.instance_path), "UserUploads", useremail)
            if not os.path.exists(useruploads):
                if os.mkdir(useruploads):
                    session['userdir'] = useruploads
                else:
                    raise Exception("Unable to Login try again")
            session['userdir'] = useruploads
            return url_for('index')
        else:
            if model.inser_google_user(username, useremail, userpicture, userhash):
                session['username'] = username
                session['useremail'] = useremail 
                session['userkey'] = userhash 
                session['userpicture'] = userpicture
                useruploads = os.path.join(os.path.dirname(app.instance_path), "UserUploads", useremail)
                if not os.path.exists(useruploads):
                    if os.mkdir(useruploads):
                        session['userdir'] = useruploads
                    else:
                        raise Exception("Unable to Login try again")
                session['userdir'] = useruploads
                return url_for('index')
            else:
                return '0' 

@app.route('/logout', methods=['GET'])
def logout():
    if request.method == 'GET':
        session.clear()
        app.secret_key = os.urandom(24)
        return render_template('index.html')

@app.route('/logout_google', methods=['GET'])
def logout_google():
    if request.method == 'GET':
        session.clear()
        app.secret_key = os.urandom(24)
        return url_for('login')

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['pass']
        cpass = request.form['cpass']

        try:
            if(usersignup.set_name(name) and usersignup.set_email(email) and usersignup.set_password(password, cpass)):
                # Generate Unique user hash
                userhash = usersignup.get_user_key()
                hashed = bcrypt.generate_password_hash(password)
                if model.insert_signup_user(name, email, hashed, userhash): 
                    # Create user folder
                    useruploads = os.path.join(os.path.dirname(app.instance_path), "UserUploads", email)
                    if not os.path.exists(useruploads):
                        os.mkdir(useruploads)
                    else:
                        raise Exception("Unable to Login try again")
                    return '1'
                else: return '0' 
        except Exception as e:
            return str(e)

@app.route('/isloggedin', methods=['GET'])
def is_logged_in():
    if request.method == "GET":
        if 'username' in session and 'useremail' in session and 'userkey' in session:
            return '1'
        else:
            return '0'

@app.route("/change_password", methods=['POST'])
def change_password():
    if request.method == 'POST':
        password = request.form["password"]
        cpassword = request.form["cpassword"]
        try:
            if usersignup.set_password(password, cpassword):
                hashed = bcrypt.generate_password_hash(password)
                if model.change_password(session['useremail'], hashed):
                    return '1'
                else:
                    return '0'
        except Exception as e:
            return str(e)

@app.route('/change_profile_picture', methods=['GET', 'POST'])
def change_profile_picture():
    if request.method == 'POST':
        profile_picture = request.files["profilePicture"]
        try:
            if usersignup.set_user_profile_picture(profile_picture) and usersignup.allowed_file(profile_picture.filename):
                dir = os.path.join(os.path.dirname(app.instance_path), 'static/userprofilepictures', str(uuid.uuid4()) + str(uuid.uuid4()) + secure_filename(profile_picture.filename))
                profile_picture.save(dir)
                if model.insert_user_picture(session['useremail'], dir):
                    session['userpicture'] = str(dir) 
                    return jsonify({"pic": str(dir)})
                else: 
                    return '0'
        except Exception as e:
            return str(e)

@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        try:
            if usersignup.set_email_issue(email):
                if model.check_email(email):
                    # get user key
                    userkey = model.get_user_key(email)[0][0]
                    if send_password_recovery_link(email, userkey):
                        return '1'
                    else:
                        return '0'
                else:
                    raise Exception("Email is not found, use your predict account email")
        except Exception as e:
            return str(e)

@app.route("/change_password_recovery", methods=['POST'])
def change_password_recovery():
    if request.method == 'POST':
        password = request.form['pass']
        cpassword = request.form['cpass']
        email = request.form['email']
        # check password
        try:
            if usersignup.set_password(password, cpassword):
                hashed = bcrypt.generate_password_hash(password)
                if model.change_password(email, hashed):
                    return '1'
                else:
                    return '0'
        except Exception as e:
            return str(e)

@app.route('/get_personal_info', methods=['GET'])
def get_personal_info():
    if request.method == 'GET':
        signup_date = model.get_signup_date(session['useremail'])
        if not signup_date:
            signup_date = model.get_google_signup_date(session['useremail'])
        return jsonify(session['username'], session['useremail'], signup_date[0][0], session['userpicture'])


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'GET':
        predictions = request.args.get('predictions')
        single_prediction = request.args.get('single_prediction')
        indivisual_inputs = request.args.get('indivisual_inputs')
        training = request.args.get('training')
        testing = request.args.get('testing')
        if predictions is None or single_prediction is None or indivisual_inputs is None or training is None or testing is None:
            return redirect(url_for('index'))
        else:
            return render_template('result.html')

@app.route('/result_preloaded', methods=['GET'])
def result_preloaded():
    if request.method == 'GET':
        predictions = request.args.get('predictions')
        single_prediction = request.args.get('single_prediction')
        indivisual_inputs = request.args.get('indivisual_inputs')
        if predictions is None or single_prediction is None or indivisual_inputs is None:
            return redirect(url_for('index'))
        else:
            return render_template('result_preloaded.html') 


@app.route('/get_dataset_table', methods=['GET'])
def get_dataset_table():
    if request.method == 'GET':
        title = request.args.get('title')
        markup = dataset_markup.select_dataset(title)
        return markup

@app.route('/getdatasets', methods=['GET'])
def get_dataset_names():
    if request.method == 'GET':
        data = datamodel.select_dataset(session['userkey'])
        for d in data:
            d = list(d)
            if d[1] != None:
                d[1] = json.loads(d[1])
        return jsonify(data)

@app.route('/model_info', methods=['GET', 'POST'])
def get_model_info():
    if request.method == 'GET':
        data = datamodel.select_user_model_history(session['userkey'])
        return jsonify(data)


@app.route('/get_specific_model', methods=['GET'])
def get_specific_model():
    if request.method == 'GET':
        id = request.args.get('id')
        data = datamodel.get_specific_model(id, session['userkey'])
        return jsonify(data)

@app.route('/delete_model', methods=['GET'])
def delete_model():
    if request.method == 'GET':
        id = request.args.get("id")
        modelid = request.args.get("modelid")
        mltype = request.args.get("mltype")
        if datamodel.delete_model(id, session['userkey']):
            if mltype == 'REG': 
                if datamodel.delete_regression(modelid, session['userkey']):
                    return '1'
                else:
                    return '0'
            if mltype == 'CLASS': 
                if datamodel.delete_classification(modelid, session['userkey']):
                    return '1'
                else:
                    return '0'
            if mltype == 'NEURAL': 
                if datamodel.delete_neural(modelid, session['userkey']):
                    return '1'
                else:
                    return '0'
        return '0'

@app.route('/get_model_result', methods=['GET'])
def get_model_result():
    if request.method == 'GET':
        modelid = request.args.get('model_id')
        mltype = request.args.get('mltype')
        if mltype == 'REG':
            data = datamodel.get_regression_result(modelid, session['userkey'])
            return jsonify(data)
        elif mltype == 'CLASS':
            data = {"data": datamodel.get_classification_result(modelid, session['userkey'])}
            return json.dumps(data)
        elif mltype == 'NEURAL':
            data = {"data": datamodel.get_neural_result(modelid, session['userkey'])}
            return json.dumps(data)
        

@app.route('/getMoreDatasetInfo', methods=['GET'])
def getMoreDatasetInfo():
    if request.method == 'GET':
        id = request.args.get("id")
        data = datamodel.get_dataset_info(id, session['userkey'])
        return jsonify(data)

@app.route('/getDatasetURL', methods=["GET"])
def getDatasetURL():
    if request.method == 'GET':
        id = request.args.get("id")
        data = datamodel.get_dataset_url(id, session['userkey'])[0][0].split("UserUploads")[-1]
        if data:
            markup = dataset_markup.load_local_dataset(app, data)
        else:
            markup = ""
        return markup

@app.route('/deleteDatasetRecord', methods=['GET'])
def deleteDatasetRecord():
    if request.method == 'GET':
        id = request.args.get("id")
        if datamodel.delete_dataset_record(id, session['userkey']):
            return '1'
        return '0'

@app.route('/get_total_models', methods=['GET'])
def get_total_models():
    if request.method == 'GET':
        data = datamodel.get_total_models_model(session['userkey'])
        return str(data[0][0])

@app.route('/get_total_regression', methods=['GET'])
def get_total_regression():
    if request.method == 'GET':
        data = datamodel.get_total_regression_model(session['userkey'])
        return str(data[0][0])

@app.route('/get_total_classification', methods=['GET'])
def get_total_classification():
    if request.method == 'GET':
        data = datamodel.get_total_classification_model(session['userkey'])
        return str(data[0][0])

@app.route('/get_total_neural', methods=['GET'])
def get_total_neural():
    if request.method == 'GET':
        data = datamodel.get_total_neural_model(session['userkey'])
        return str(data[0][0])

@app.route('/file_issue', methods=['POST'])
def file_issue():
    if request.method == 'POST':
        email = request.form['email']
        issue = request.form['issue']
        try:
            if usersignup.set_email_issue(email) and usersignup.set_issue(issue):
                if model.insert_issue_db(email, issue):
                    if send_issue(email, issue):
                        return '1'
                    return '0'
                return '0'
        except Exception as e:
            return str(e)


# None Route functions

def is_logged_in_non_route():
    if request.method == "GET":
        if 'username' in session and 'useremail' in session and 'userkey' in session:
            return True
        else:
            return False

def change_to_string(val):
    val = val.tostring()
    val = np.fromstring(val, dtype=int)
    return val

def move_single_file(source, destination):
    shutil.copy(source, destination)

def move_list_files(files, destination):
    for f in files:
        shutil.copy(f, destination)

@app.route('/MainFormData', methods=['POST'])
def main_form_data():
    if request.method == 'POST':
        dependant_var = request.form['dependant_var']
        unwanted_cols = request.form['unwanted_cols']
        ml_type = request.form['ml_type']
        algorithms = request.form['algorithms']
        testsize = request.form['testsize']
        csvFile = request.files['csvFile']
        indivisualInputs = request.form['indivisualInputs']
        polyregdegree = request.form['polyregdegree']
        kernelopt = request.form['kernelopt']
        nestimators = request.form['nestimators']
        n_neighbours = request.form['n_neighbours']
        metric = request.form['metric']
        p = request.form['p']
        kerneloptSvc = request.form['kerneloptSvc']
        criterion = request.form['criterion']
        criterion_rfc = request.form['criterion_rfc']
        nestimators_rfc = request.form['nestimators_rfc']
        alpha_ridge = request.form['alpha_ridge']
        max_iteration = request.form['max_iteration']
        solver = request.form['solver']
        mlpc_activation = request.form['mlpc_activation']
        mlpc_solver = request.form['mlpc_solver']
        mlpc_learning_rate = request.form['mlpc_learning_rate']
        mlpc_max_iter = request.form['mlpc_max_iter']
        mlpr_activation = request.form['mlpr_activation']
        mlpr_solver = request.form['mlpr_solver']
        mlpr_learning_rate = request.form['mlpr_learning_rate']
        mlpr_max_iter = request.form['mlpr_max_iter']

        try:
            # Perform validation
            if input_controller.setDependantVar(dependant_var) and input_controller.setMlType(
                ml_type) and input_controller.setAlgorithm(algorithms) and input_controller.setTestSize(testsize) and input_controller.setCSVFile(csvFile) and input_controller.setIndivisualInputs(
                indivisualInputs) and input_controller.set_polyreg_degree(polyregdegree) and input_controller.set_kernel_type(kernelopt) and input_controller.set_nestimators(nestimators) and input_controller.set_numberof_neighbours(
                n_neighbours) and input_controller.set_metric(metric) and input_controller.set_p(p) and input_controller.set_kernel_type(kerneloptSvc
                ) and input_controller.set_criterion(criterion) and input_controller.set_nestimators(nestimators_rfc) and input_controller.set_criterion(criterion_rfc
                ) and input_controller.set_alpha_ridge(alpha_ridge) and input_controller.set_max_iteration(max_iteration) and input_controller.set_solver(solver
                ) and input_controller.set_mlpc_activation(mlpc_activation) and input_controller.set_mlpc_solver(mlpc_solver) and input_controller.set_mlpc_learning_rate(mlpc_learning_rate
                ) and input_controller.set_max_iteration(mlpc_max_iter) and input_controller.set_mlpc_activation(mlpr_activation) and input_controller.set_mlpc_solver(mlpr_solver
                ) and input_controller.set_mlpc_learning_rate(mlpr_learning_rate) and input_controller.set_max_iteration(mlpr_max_iter):


                if ml_type == 'REG':
                    # Call function to create ML model instance 
                    prediction, columns, single_prediction, filename1, filename2, mae, mse, r2score, rmse, explained_score, reg_graphs, \
                    jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graph, corr_graphs, model_description = \
                    prepare_model_instance_regression(app, csvFile, unwanted_cols, algorithms, testsize, dependant_var,
                    indivisualInputs, polyregdegree, kernelopt, nestimators, alpha_ridge, max_iteration, solver)

                    # upload user models and dataset
                    if 'username' in session and 'userdir' in session and 'useremail' in session and 'userkey' in session:
                        stored_filename = secure_filename(csvFile.filename) 
                        upload_folder = session['userdir']  
                        stored_filename = os.path.join(app.config['UPLOAD_FOLDER'], session['useremail'], stored_filename)
                        if not os.path.exists(stored_filename) :
                            # Save files
                            # move_single_file(csvFile.filename, upload_folder)
                            csvFile.seek(0)
                            csvFile.save(stored_filename)
                            
                        # Save Images
                        move_single_file(filename1, upload_folder)
                        move_single_file(filename2, upload_folder)
                        move_list_files(reg_graphs, upload_folder)
                        move_list_files(jitter_graphs, upload_folder)
                        move_list_files(lmplot_graphs, upload_folder)
                        move_list_files(mean_graphs, upload_folder)
                        move_list_files(joint_graphs, upload_folder)
                        move_list_files(dist_graph, upload_folder)
                        move_list_files(corr_graphs, upload_folder)

                        # model_id 
                        model_id = usersignup.get_user_key()

                        # Insert user Input to a database
                        if(datamodel.insert_user_config_inputs(stored_filename, dependant_var, unwanted_cols, ml_type, algorithms, testsize, \
                            indivisualInputs, polyregdegree, kernelopt, nestimators, n_neighbours, metric, p, kerneloptSvc, criterion, \
                            criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
                            mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, json.dumps(model_description), session['userkey'], model_id)):
                            
                            # Change to str
                            prediction_str = str(prediction.tolist())
                            single_prediction_str = str(single_prediction.tolist())
                            cols_str = str(columns.tolist())
                            # Insert Regression Result 
                            if not datamodel.insert_regression_result(prediction_str, cols_str, single_prediction_str, filename1, filename2, mae, mse, r2score, rmse,\
                                explained_score, str(reg_graphs), str(jitter_graphs), str(lmplot_graphs), str(mean_graphs), str(joint_graphs), \
                                str(dist_graph), str(corr_graphs), json.dumps(model_description), session['userkey'], model_id):
                                raise Exception("try again unable to save your work")
                        else:
                            raise Exception("try again unable to save your work")

                    return jsonify({'prediction': prediction.tolist(), 'single_prediction': single_prediction.tolist(), 'redirect': url_for('result'),
                        'indivisualInputs':indivisualInputs, 'training': filename1, 'testing': filename2, 'mae': mae.tolist(), 'mse': mse, 'r2score': r2score,
                        'rmse': rmse, 'confusion': explained_score, 'mltype': ml_type, 'reg_graphs':reg_graphs, 'jitter_graphs':jitter_graphs, 'lmplot_graphs':lmplot_graphs,
                        'mean_graphs':mean_graphs, 'joint_graphs':joint_graphs, 'dist_graph': dist_graph, 'corr_graphs':corr_graphs, 'algorithm':algorithms, \
                        'model_description': model_description})
                
                elif ml_type == 'CLASS':
                    prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs,\
                    lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description  = \
                    prepare_model_instance_classification(app, csvFile, unwanted_cols, algorithms, testsize, dependant_var, indivisualInputs,  n_neighbours, metric, p, 
                    kerneloptSvc, criterion, criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver)

                    # Change to str
                    prediction_str = str(prediction.tolist())
                    single_classification = str(single_classification.tolist())
                    cols_str = str(columns.tolist())

                    # upload user models and dataset
                    if 'username' in session and 'userdir' in session and 'useremail' in session and 'userkey' in session:
                        stored_filename = secure_filename(csvFile.filename) 
                        upload_folder = session['userdir']
                        stored_filename = os.path.join(upload_folder, stored_filename)  
                        if not os.path.exists(stored_filename):
                            # Save files 
                            csvFile.save(stored_filename)

                        # Save Images
                        move_single_file(filename1, upload_folder)
                        move_single_file(filename2, upload_folder)
                        move_list_files(reg_graphs, upload_folder)
                        move_list_files(jitter_graphs, upload_folder)
                        move_list_files(lmplot_graphs, upload_folder)
                        move_list_files(mean_graphs, upload_folder)
                        move_list_files(joint_graphs, upload_folder)
                        move_list_files(dist_graphs, upload_folder)
                        move_list_files(corr_graphs, upload_folder)
                            
                        # model_id 
                        model_id = usersignup.get_user_key()

                        # Insert user Input to a database
                        if(datamodel.insert_user_config_inputs(stored_filename, dependant_var, unwanted_cols, ml_type, algorithms, testsize, \
                            indivisualInputs, polyregdegree, kernelopt, nestimators, n_neighbours, metric, p, kerneloptSvc, criterion, \
                            criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
                            mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, json.dumps(model_description), session['userkey'], model_id)):
                            
                            tmp_desc = json.dumps(model_description).replace("'", "\"")
                            # Insert Classification Result 
                            if not datamodel.insert_classification_result(prediction_str, cols_str, single_classification, filename1, filename2, mae, mse, r2score, accuracy,\
                                confusion, str(reg_graphs), str(jitter_graphs), str(lmplot_graphs), str(mean_graphs), str(joint_graphs), \
                                str(dist_graphs), str(corr_graphs), tmp_desc, session['userkey'], model_id):
                                raise Exception("try again unable to save your work")
                        else:
                            raise Exception("try again unable to save your work")

                    return jsonify({'prediction': prediction_str, 'single_prediction': single_classification, 'redirect': url_for('result'),
                        'indivisualInputs':indivisualInputs, 'training': filename1, 'testing': filename2, 'confusion_img':filename3, 'mae': mae.tolist(), 
                        'mse': mse, 'r2score': r2score, 'accuracy': accuracy, 'confusion': confusion.tolist(), 'mltype': ml_type, 'reg_graphs':reg_graphs,\
                        'jitter_graphs':jitter_graphs, 'lmplot_graphs':lmplot_graphs, 'mean_graphs':mean_graphs, 'joint_graphs':joint_graphs, \
                        'dist_graph':dist_graphs, 'corr_graphs': corr_graphs, 'algorithm':algorithms, 'model_description': model_description})

                elif ml_type == 'NEURAL' and algorithms == 'MLPC':
                    prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion, reg_graphs,\
                    jitter_graphs, lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description = \
                    prepare_model_instance_neuralnetwork(app, csvFile, unwanted_cols, algorithms, testsize, dependant_var, indivisualInputs, \
                    mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter)

                    # Change to str
                    prediction_str = str(prediction.tolist())
                    single_classification = str(single_classification.tolist())
                    cols_str = str(columns.tolist())

                    # upload user models and dataset
                    if 'username' in session and 'userdir' in session and 'useremail' in session and 'userkey' in session:
                        stored_filename = secure_filename(csvFile.filename) 
                        upload_folder = session['userdir']
                        stored_filename = os.path.join(upload_folder, stored_filename)  
                        if not os.path.exists(stored_filename):
                            csvFile.save(stored_filename)
                        
                        # Save Images
                        move_single_file(filename1, upload_folder)
                        move_single_file(filename2, upload_folder)
                        move_list_files(reg_graphs, upload_folder)
                        move_list_files(jitter_graphs, upload_folder)
                        move_list_files(lmplot_graphs, upload_folder)
                        move_list_files(mean_graphs, upload_folder)
                        move_list_files(joint_graphs, upload_folder)
                        move_list_files(dist_graphs, upload_folder)
                        move_list_files(corr_graphs, upload_folder)
                        
                        # model_id 
                        model_id = usersignup.get_user_key()

                        # Insert user Input to a database
                        if(datamodel.insert_user_config_inputs(stored_filename, dependant_var, unwanted_cols, ml_type, algorithms, testsize, \
                            indivisualInputs, polyregdegree, kernelopt, nestimators, n_neighbours, metric, p, kerneloptSvc, criterion, \
                            criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
                            mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, json.dumps(model_description), session['userkey'], model_id)):

                            tmp_desc = json.dumps(model_description).replace("'", "\"")

                            # Insert Neural Result 
                            if not datamodel.insert_neural_result(prediction_str, cols_str, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, confusion,\
                                str(reg_graphs), str(jitter_graphs), str(lmplot_graphs), str(mean_graphs), str(joint_graphs), \
                                str(dist_graphs), str(corr_graphs), tmp_desc, session['userkey'], model_id):
                                raise Exception("try again unable to save your work")
                        else:
                            raise Exception("try again unable to save your work")

                    return jsonify({'prediction': prediction_str, 'single_prediction': single_classification, 'redirect': url_for('result'),
                        'indivisualInputs':indivisualInputs, 'training': filename1, 'testing': filename2, 'confusion_img':filename3, 'mae': mae.tolist(), \
                        'mse': mse, 'r2score': r2score, 'accuracy': accuracy, 'confusion': confusion.tolist(), 'mltype': ml_type, 'reg_graphs':reg_graphs, \
                        'jitter_graphs':jitter_graphs, 'lmplot_graphs':lmplot_graphs, 'mean_graphs':mean_graphs, 'joint_graphs':joint_graphs, 'dist_graph':dist_graphs,\
                        'corr_graphs':corr_graphs, 'algorithm':algorithms, 'model_description': model_description})

                elif ml_type == 'NEURAL' and algorithms == 'MLPR':
                    prediction, columns, single_classification, filename1, filename2, mae, mse, r2score, accuracy, confusion, reg_graphs, jitter_graphs, \
                    lmplot_graphs, mean_graphs, joint_graphs, dist_graphs, corr_graphs, model_description =\
                    prepare_model_instance_neuralnetwork( app, csvFile, unwanted_cols, algorithms, testsize, dependant_var, indivisualInputs, mlpc_activation,\
                    mlpc_solver, mlpc_learning_rate, mlpc_max_iter, mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter)

                    # Change to str
                    prediction_str = str(prediction)
                    single_classification = str(single_classification)
                    cols_str = str(columns.tolist())

                    # upload user models and dataset
                    if 'username' in session and 'userdir' in session and 'useremail' in session and 'userkey' in session:
                        stored_filename = secure_filename(csvFile.filename) 
                        upload_folder = session['userdir']
                        stored_filename = os.path.join(upload_folder, stored_filename)  
                        if not os.path.exists(stored_filename):
                            csvFile.save(stored_filename)
                        
                        # Save Images
                        move_single_file(filename1, upload_folder)
                        move_single_file(filename2, upload_folder)
                        move_list_files(reg_graphs, upload_folder)
                        move_list_files(jitter_graphs, upload_folder)
                        move_list_files(lmplot_graphs, upload_folder)
                        move_list_files(mean_graphs, upload_folder)
                        move_list_files(joint_graphs, upload_folder)
                        move_list_files(dist_graphs, upload_folder)
                        move_list_files(corr_graphs, upload_folder)
                        
                        # model_id 
                        model_id = usersignup.get_user_key()

                        # Insert user Input to a database
                        if(datamodel.insert_user_config_inputs(stored_filename, dependant_var, unwanted_cols, ml_type, algorithms, testsize, \
                            indivisualInputs, polyregdegree, kernelopt, nestimators, n_neighbours, metric, p, kerneloptSvc, criterion, \
                            criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
                            mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, json.dumps(model_description), session['userkey'], model_id)):

                            tmp_desc = json.dumps(model_description).replace("'", "\"")

                            # Insert nueral  Result 
                            if not datamodel.insert_neural_result(prediction_str, cols_str, single_classification, filename1, filename2, filename2, mae, mse, r2score, accuracy, confusion,\
                                str(reg_graphs), str(jitter_graphs), str(lmplot_graphs), str(mean_graphs), str(joint_graphs), \
                                str(dist_graphs), str(corr_graphs), tmp_desc, session['userkey'], model_id):
                                raise Exception("try again unable to save your work")
                        else:
                            raise Exception("try again unable to save your work")
                    
                    return jsonify({'prediction': prediction, 'single_prediction': single_classification, 'redirect': url_for('result'),
                    'indivisualInputs':indivisualInputs, 'training': filename1, 'testing': filename2, 'mae': mae, 'mse': mse, 'r2score': r2score,
                    'accuracy': accuracy, 'confusion': confusion, 'mltype': ml_type, 'reg_graphs':reg_graphs, 'jitter_graphs':jitter_graphs, 'lmplot_graphs':lmplot_graphs,
                    'mean_graphs':mean_graphs, 'joint_graphs':joint_graphs, 'dist_graph':dist_graphs, 'corr_graphs':corr_graphs, 'algorithm':algorithms, \
                    'model_description': model_description})
        except Exception as e:
            return jsonify({"error": str(e)})




@app.route('/sklearnPreloaded', methods=['POST'])
def sklear_preloaded():
    if request.method == 'POST':
        dependant_var = request.form['dependant_var']
        unwanted_cols = request.form['unwanted_cols']
        ml_type = request.form['ml_type']
        algorithms = request.form['algorithms']
        testsize = request.form['testsize']
        indivisualInputs = request.form['indivisualInputs']
        polyregdegree = request.form['polyregdegree']
        kernelopt = request.form['kernelopt']
        nestimators = request.form['nestimators']
        n_neighbours = request.form['n_neighbours']
        metric = request.form['metric']
        p = request.form['p']
        kerneloptSvc = request.form['kerneloptSvc']
        criterion = request.form['criterion']
        criterion_rfc = request.form['criterion_rfc']
        nestimators_rfc = request.form['nestimators_rfc']
        alpha_ridge = request.form['alpha_ridge']
        max_iteration = request.form['max_iteration']
        solver = request.form['solver']
        mlpc_activation = request.form['mlpc_activation']
        mlpc_solver = request.form['mlpc_solver']
        mlpc_learning_rate = request.form['mlpc_learning_rate']
        mlpc_max_iter = request.form['mlpc_max_iter']
        mlpr_activation = request.form['mlpr_activation']
        mlpr_solver = request.form['mlpr_solver']
        mlpr_learning_rate = request.form['mlpr_learning_rate']
        mlpr_max_iter = request.form['mlpr_max_iter']
        dataset_preloaded = auxilary.get_preloaded_dataset(request.form['data_title'])

        try:

            if input_controller.setDependantVar(dependant_var) and input_controller.setMlType(
                    ml_type) and input_controller.setAlgorithm(algorithms) and input_controller.setTestSize(testsize) and \
                    input_controller.setIndivisualInputs(indivisualInputs) and input_controller.set_polyreg_degree(polyregdegree)\
                    and input_controller.set_kernel_type(kernelopt) and input_controller.set_nestimators(nestimators) and \
                    input_controller.set_numberof_neighbours(n_neighbours) and input_controller.set_metric(metric) and\
                    input_controller.set_p(p) and input_controller.set_kernel_type(kerneloptSvc) and input_controller.set_criterion(criterion)\
                    and input_controller.set_nestimators(nestimators_rfc) and input_controller.set_criterion(criterion_rfc) and\
                    input_controller.set_alpha_ridge(alpha_ridge) and input_controller.set_max_iteration(max_iteration) and\
                    input_controller.set_solver(solver) and input_controller.set_mlpc_activation(mlpc_activation) and\
                    input_controller.set_mlpc_solver(mlpc_solver) and input_controller.set_mlpc_learning_rate(mlpc_learning_rate)\
                    and input_controller.set_max_iteration(mlpc_max_iter) and input_controller.set_mlpc_activation(mlpr_activation)\
                    and input_controller.set_mlpc_solver(mlpr_solver) and input_controller.set_mlpc_learning_rate(mlpr_learning_rate)\
                    and input_controller.set_max_iteration(mlpr_max_iter):
                    

                    if ml_type == 'REG':
                        # Call function to create ML model instance 
                        prediction, columns, single_prediction, mae, mse, r2score, rmse, explained_score, reg_graphs,\
                        dist_graph, corr_graphs, model_description = prpare_preloaded_dataset_regression(app, dataset_preloaded,\
                        unwanted_cols, algorithms, testsize, dependant_var,indivisualInputs, polyregdegree, kernelopt,\
                        nestimators, alpha_ridge, max_iteration, solver)

                        # upload user models and dataset
                        if 'username' in session and 'userdir' in session and 'useremail' in session and 'userkey' in session: 
                            upload_folder = session['userdir']  
                            
                            # Save Images
                            move_list_files(reg_graphs, upload_folder)
                            move_list_files(dist_graph, upload_folder)
                            move_list_files(corr_graphs, upload_folder)

                            # model_id 
                            model_id = usersignup.get_user_key()

                            # Insert user Input to a database
                            if(datamodel.insert_user_config_inputs("dataset_preloaded", dependant_var, unwanted_cols, ml_type, algorithms, testsize, \
                                indivisualInputs, polyregdegree, kernelopt, nestimators, n_neighbours, metric, p, kerneloptSvc, criterion, \
                                criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
                                mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, json.dumps(model_description), session['userkey'], model_id)):
                                
                                # Change to str
                                prediction_str = str(prediction.tolist())
                                single_prediction_str = str(single_prediction.tolist())
                                cols_str = str(columns.tolist())
                                # Insert Regression Result 
                                if not datamodel.insert_regression_result(prediction_str, cols_str, single_prediction_str, "", "", mae, mse, r2score, rmse,\
                                    explained_score, str(reg_graphs), "", "", "", "", \
                                    str(dist_graph), str(corr_graphs), json.dumps(model_description), session['userkey'], model_id):
                                    raise Exception("try again unable to save your work")
                            else:
                                raise Exception("try again unable to save your work")

                        return jsonify({'prediction': prediction.tolist(), 'single_prediction': single_prediction.tolist(), \
                            'redirect': url_for('result_preloaded'), 'indivisualInputs':indivisualInputs, 'mae': mae.tolist(),\
                            'mse': mse, 'r2score': r2score, 'rmse': rmse, 'confusion': explained_score, 'mltype': ml_type,\
                            'reg_graphs':reg_graphs, 'dist_graph': dist_graph, 'corr_graphs':corr_graphs, 'algorithm':algorithms,\
                            'model_description': model_description})
                        
                    elif ml_type == 'CLASS':
                        prediction, columns, single_classification, mae, mse, r2score, accuracy, confusion, reg_graphs,\
                        dist_graphs, corr_graphs,  model_description = prepare_preloaded_dataset_classification(app, dataset_preloaded,\
                        unwanted_cols, algorithms, testsize, dependant_var, indivisualInputs,  n_neighbours, metric, p, kerneloptSvc, \
                        criterion, criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver)
                        
                        # upload user models and dataset
                        if 'username' in session and 'userdir' in session and 'useremail' in session and 'userkey' in session: 
                            upload_folder = session['userdir']  
                            
                            # Save Images
                            move_list_files(reg_graphs, upload_folder)
                            move_list_files(dist_graphs, upload_folder)
                            move_list_files(corr_graphs, upload_folder)

                            # model_id 
                            model_id = usersignup.get_user_key()

                            # Insert user Input to a database
                            if(datamodel.insert_user_config_inputs("dataset_preloaded", dependant_var, unwanted_cols, ml_type, algorithms, testsize, \
                                indivisualInputs, polyregdegree, kernelopt, nestimators, n_neighbours, metric, p, kerneloptSvc, criterion, \
                                criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
                                mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, json.dumps(model_description), session['userkey'], model_id)):
                                
                                # Change to str
                                prediction_str = str(prediction.tolist())
                                single_prediction_str = str(single_classification.tolist())
                                cols_str = str(columns.tolist())
                                # Insert Regression Result 
                                if not datamodel.insert_regression_result(prediction_str, cols_str, single_prediction_str, "", "", mae, mse, r2score, accuracy,\
                                    confusion, str(reg_graphs), "", "", "", "", \
                                    str(dist_graphs), str(corr_graphs), json.dumps(model_description), session['userkey'], model_id):
                                    raise Exception("try again unable to save your work")
                            else:
                                raise Exception("try again unable to save your work")
                        
                        return jsonify({'prediction': prediction.tolist(), 'single_prediction': single_classification.tolist(), 
                        'redirect': url_for('result_preloaded'), 'indivisualInputs':indivisualInputs, 'mae': mae.tolist(), 'mse': mse,
                        'r2score': r2score, 'accuracy': accuracy, 'confusion': confusion, 'mltype': ml_type, 'reg_graphs':reg_graphs,
                        'dist_graph':dist_graphs, 'corr_graphs':corr_graphs, 'algorithm':algorithms, 'model_description': model_description})
                    
                    elif ml_type == 'NEURAL' and algorithms == 'MLPC':
                        prediction, columns, single_classification, filename1, filename2, filename3, mae, mse, r2score, accuracy, \
                        confusion, reg_graphs, dist_graphs, corr_graphs, model_description= prepare_model_instance_neuralnetwork_preloaded(app, \
                        dataset_preloaded, unwanted_cols, algorithms, testsize, dependant_var, indivisualInputs, mlpc_activation, mlpc_solver, \
                        mlpc_learning_rate, mlpc_max_iter,mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter)

                        # upload user models and dataset
                        if 'username' in session and 'userdir' in session and 'useremail' in session and 'userkey' in session: 
                            upload_folder = session['userdir']  
                            
                            # Save Images
                            move_list_files(reg_graphs, upload_folder)
                            move_list_files(dist_graphs, upload_folder)
                            move_list_files(corr_graphs, upload_folder)

                            # model_id 
                            model_id = usersignup.get_user_key()

                            # Insert user Input to a database
                            if(datamodel.insert_user_config_inputs("dataset_preloaded", dependant_var, unwanted_cols, ml_type, algorithms, testsize, \
                                indivisualInputs, polyregdegree, kernelopt, nestimators, n_neighbours, metric, p, kerneloptSvc, criterion, \
                                criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
                                mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, json.dumps(model_description), session['userkey'], model_id)):
                                
                                # Change to str
                                prediction_str = str(prediction.tolist())
                                single_prediction_str = str(single_classification.tolist())
                                cols_str = str(columns.tolist())
                                # Insert Regression Result 
                                if not datamodel.insert_regression_result(prediction_str, cols_str, single_prediction_str, "", "", mae, mse, r2score, accuracy,\
                                    confusion, str(reg_graphs), "", "", "", "", \
                                    str(dist_graphs), str(corr_graphs), json.dumps(model_description), session['userkey'], model_id):
                                    raise Exception("try again unable to save your work")
                            else:
                                raise Exception("try again unable to save your work")

                        return jsonify({'prediction': prediction.tolist(), 'single_prediction': single_classification.tolist(), \
                        'redirect': url_for('result_preloaded'), 'indivisualInputs':indivisualInputs, 'training': filename1, \
                        'testing': filename2, 'confusion_img':filename3, 'mae': mae.tolist(), 'mse': mse, 'r2score': r2score, 'accuracy': accuracy,\
                        'confusion': confusion, 'mltype': ml_type, 'reg_graphs':reg_graphs, 'dist_graph':dist_graphs, 'corr_graphs':corr_graphs,\
                        'algorithm':algorithms, 'model_description': model_description})
                    
                    elif ml_type == 'NEURAL' and algorithms == 'MLPR':
                        prediction, columns, single_classification, filename1, filename2, mae, mse, r2score, accuracy, confusion,\
                        reg_graphs, dist_graphs, corr_graphs, model_description = prepare_model_instance_neuralnetwork_preloaded(app, \
                        dataset_preloaded, unwanted_cols, algorithms, testsize, dependant_var, indivisualInputs, mlpc_activation,\
                        mlpc_solver, mlpc_learning_rate, mlpc_max_iter, mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter)
                        
                        # upload user models and dataset
                        if 'username' in session and 'userdir' in session and 'useremail' in session and 'userkey' in session: 
                            upload_folder = session['userdir']  
                            
                            # Save Images
                            move_list_files(reg_graphs, upload_folder)
                            move_list_files(dist_graphs, upload_folder)
                            move_list_files(corr_graphs, upload_folder)

                            # model_id 
                            model_id = usersignup.get_user_key()

                            # Insert user Input to a database
                            if(datamodel.insert_user_config_inputs("dataset_preloaded", dependant_var, unwanted_cols, ml_type, algorithms, testsize, \
                                indivisualInputs, polyregdegree, kernelopt, nestimators, n_neighbours, metric, p, kerneloptSvc, criterion, \
                                criterion_rfc, nestimators_rfc, alpha_ridge, max_iteration, solver, mlpc_activation, mlpc_solver, mlpc_learning_rate, mlpc_max_iter, \
                                mlpr_activation, mlpr_solver, mlpr_learning_rate, mlpr_max_iter, json.dumps(model_description), session['userkey'], model_id)):
                                
                                # Change to str
                                prediction_str = str(prediction.tolist())
                                single_prediction_str = str(single_classification.tolist())
                                cols_str = str(columns.tolist())
                                # Insert Regression Result 
                                if not datamodel.insert_regression_result(prediction_str, cols_str, single_prediction_str, "", "", mae, mse, r2score, accuracy,\
                                    confusion, str(reg_graphs), "", "", "", "", \
                                    str(dist_graphs), str(corr_graphs), json.dumps(model_description), session['userkey'], model_id):
                                    raise Exception("try again unable to save your work")
                            else:
                                raise Exception("try again unable to save your work")

                        return jsonify({'prediction': prediction.tolist(), 'single_prediction': single_classification.tolist(), \
                        'redirect': url_for('result_preloaded'), 'indivisualInputs':indivisualInputs, 'training': filename1, \
                        'testing': filename2, 'mae': mae.tolist(), 'mse': mse, 'r2score': r2score, 'accuracy': accuracy, 'confusion': confusion,\
                        'mltype': ml_type, 'reg_graphs':reg_graphs, 'dist_graph':dist_graphs, 'corr_graphs':corr_graphs,\
                        'algorithm':algorithms, 'model_description': model_description})
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)})

        



# ---------------- Routes -------------- #




# Run app
if __name__ == '__main__':
    app.debug = True
    app.static_folder = "static"
    app.config["CACHE_TYPE"] = "null"
    app.config['SESSION_TYPE'] = 'filesystem'
    session.init_app(app)
    app.run()
