import os
import pandas as pd
from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits

class DataMarkupHolder(object):

    def select_dataset(self, title):
        if title == "load_boston":
            dataset_boston = load_boston()
            dataset = pd.DataFrame(data=dataset_boston.data, columns=dataset_boston.feature_names)
            dataset.name = "Boston Dataset"
            dataset['MEDV'] = dataset_boston.target
            html = dataset.to_html()
            return html
        elif title == "load_iris":
            dataset_iris = load_iris()
            dataset = pd.DataFrame(data=dataset_iris.data, columns=dataset_iris.feature_names)
            dataset.name = "Iris Dataset"
            html = dataset.to_html()
            return html
        elif title == "load_diabetes":
            dataset_diabetes = load_diabetes()
            dataset = pd.DataFrame(data=dataset_diabetes.data, columns=dataset_diabetes.feature_names)
            dataset.name = "Diabetes Dataset"
            # dataset['MEDV'] = dataset_diabetes.target
            html = dataset.to_html()
            return html
        elif title == "load_digits":
            dataset_digits = load_digits()
            dataset = pd.DataFrame(data=dataset_digits.data, columns=dataset_digits.feature_names)
            dataset.name = "Digits Dataset"
            html = dataset.to_html()
            return html
        else:
            return title
    
    def load_local_dataset(self, app, path): 
        # os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-training.png')
        if path:
            path = path.split("\\")
            if len(path) >= 3:
                directory, filename = path[1], path[2]
                file = os.path.join(os.path.dirname(app.instance_path), 'UserUploads\\'+directory, filename)
                dataset = pd.read_csv(file)
                dataset = pd.DataFrame(dataset)
                return dataset.to_html()
            return ""

