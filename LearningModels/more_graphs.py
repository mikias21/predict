import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import os
import uuid 
from itertools import cycle
np.seterr(divide='ignore', invalid='ignore')

class MoreGraphs(object):

    def draw_regression_plot(self, app, relational_cols, dependant_var, dataset):
        graphs = []
        for col in relational_cols:
            filename = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-regression-plot.png')
            sns.regplot(x = str(col), y=str(dependant_var), data=dataset)
            plt.savefig(filename)
            plt.close()
            graphs.append(filename)
        return graphs
    
    def draw_lmplot(self, app, relational_cols, dependant_var, dataset):
        graphs = []
        for col in relational_cols:
            filename = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-lmplot.png')
            sns.lmplot(x = str(col), y=str(dependant_var), data=dataset)
            plt.savefig(filename)
            plt.close()
            graphs.append(filename)
        return graphs

    def draw_mean_estimated_graph(self, app, relational_cols, dependant_var, dataset):
        graphs = []
        for col in relational_cols:
            filename = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-mean-estimated.png')
            sns.lmplot(x = str(col), y=str(dependant_var), data=dataset, x_estimator=np.mean)
            plt.savefig(filename)
            plt.close()
            graphs.append(filename)
        return graphs

    def draw_jitter_plot(self, app, relational_cols, dependant_var, dataset):
        graphs = []
        for col in relational_cols:
            filename = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-jitter.png')
            sns.lmplot(x = str(col), y=str(dependant_var), data=dataset, logistic=True, y_jitter=.03)
            plt.savefig(filename)
            plt.close()
            graphs.append(filename)
        return graphs
    
    def draw_joint_plot_reg(self, app, relational_cols, dependant_var, dataset):
        graphs = []
        for col in relational_cols:
            filename = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-joint.png')
            sns.jointplot(x = str(col), y=str(dependant_var), data=dataset, kind="reg")
            plt.savefig(filename)
            plt.close()
            graphs.append(filename)
        return graphs

    def pairplot(self, app, dataset):
        filename = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-pairplot.png')
        sns.pairplot(dataset, hue=None, hue_order=None, palette='gist_heat', vars=None, x_vars=None, y_vars=None,
             kind='scatter', diag_kind='hist', markers=None, height=2.5, aspect=1, dropna=True,
             plot_kws=None, diag_kws=None, grid_kws=None)
        plt.savefig(filename)
        plt.close()
        return [filename]

    def distribution_graph(self, app, dataset):
        filename = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-ditplot.png')
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.displot(dataset, bins=30)
        plt.savefig(filename)
        plt.close()
        return [filename]
        
    def corelation_matrix(self, app, dataset):
        filename = os.path.join(os.path.dirname(app.instance_path), 'static/tmp', str(uuid.uuid4()) + '-corelation.png')
        sns.heatmap(data=dataset, annot=True)
        plt.savefig(filename)
        plt.close()
        return [filename]