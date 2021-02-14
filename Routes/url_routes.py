from flask import Blueprint, request, render_template

urls_blueprint = Blueprint("urls", __name__,)

# ------------ Routes ------------- #


@urls_blueprint.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

@urls_blueprint.route('/datasets', methods=['GET', 'POST'])
def resutls():
    if request.method == 'GET':
        return render_template('datasets.html')

@urls_blueprint.route('/datasetHistory', methods=['GET'])
def dataset_history():
    if request.method == 'GET':
        return render_template('dataset_history.html')

@urls_blueprint.route('/models', methods=['GET'])
def models():
    if request.method == 'GET':
        return render_template('models.html')

@urls_blueprint.route('/settings', methods=['GET'])
def settings():
    if request.method == 'GET':
        return render_template('settings.html')

@urls_blueprint.route('/about', methods=['GET'])
def about():
    if request.method == 'GET':
        return render_template('about.html')

@urls_blueprint.route("/help", methods=['GET'])
def help():
    if request.method == 'GET':
        return render_template('help.html')


@urls_blueprint.route('/forgot', methods=['GET'])
def forgot():
    if request.method == 'GET':
        return render_template('forgot.html')

@urls_blueprint.route('/recover', methods=['GET'])
def recover():
    if request.method == 'GET':
        return render_template('recover.html')