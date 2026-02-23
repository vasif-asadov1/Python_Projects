import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc 
import math 


class ModelResults:
    def __init__(self):
        
        self.results = {} 
        self.results_for_plot = {}

    
    def ClassificationModels(self,X,y,modelname, model):
        """ 
        This function will buid models, and return evaluation metrics and curve results (fpr,tpr) as dictionaries.
        Model metrics will be stored in the results dictionary with corresponding model names as keys.

        Parameters: 
        -----------

        X : features
        y : target column 
        modelname : name of the model as string ("RandomForest", "SVM' ...)
        model : model object (SVM(),  RandomForestClassifier() ...)

        
        Returns
        -------
        self.results = {modelname: [accuracy, precision, recall, roc_auc, gini]}
        self.results_for_plot = {modelname: [false_positive_rates, true_positive_rates]}

        Example
        -------
        >>> from BuildModel import ModelResults
        >>> modeller = ModelResults()
        >>> results, results_for_plot = modeller.ClassificationModels(X,y, "RandomForest", RandomForestClassifier())

        -------
        To visualize the obtained results as readable Dataframes, use VisualizeMetrics() function
    
        """
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,
                                                            random_state=111,shuffle=True)
        

        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        gini = 2 * roc_auc - 1

        self.results[modelname] = [accuracy, precision, recall, roc_auc, gini]

        # results for plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        self.results_for_plot[modelname] = [fpr, tpr]

        return self.results, self.results_for_plot




    def MetricsDf(self):
        """
        Takes the stored model metrics as dictionaries from the class object and returns
        them as pandas dataframe having index = ['accuracy','precision', 'recall', 'roc_auc','gini'] evaluation
        metrics and columns as model names. 
    
        Example:
        --------
        >>> results_df = modeller.MetricsDf()
        >>> results_df
   
        """
    
        if len(self.results)>0:
            metrics_df = pd.DataFrame(self.results, index=['accuracy','precision', 'recall', 'roc_auc','gini'])
            return metrics_df
    
        else: 
            raise ValueError("results must be dictionary with 5 rows ['accuracy','precision', 'recall', 'ROC&AUC','gini']")




    def VisualizeMetrics(self, figwidth=10, figheight=7):
        """
        Plot ROC curves against thresholds using roc_curve metrics results as dictionary.
        It also shows the closest score in the curve to the ideal point.
        
        Parameters: 
        ----------
        figwidth : width of figure. Default is 10.
        figheight : height of figure. Default is 7.


        Example: 
        --------
        >>> modeller.VisualizeMetrics()
        
        
        """

        ncol=3
        nrow=math.ceil(len(self.results_for_plot)/ncol)
        if len(self.results_for_plot) > 0: 
            fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(figwidth*ncol, figheight*nrow))
            axes = axes.flatten()
            for i,(model,(fpr,tpr)) in enumerate(self.results_for_plot.items()):
                # Compute Youden’s J
                J = tpr - fpr
                ix = np.argmax(J)  # index of optimum point
                best_fpr, best_tpr = fpr[ix], tpr[ix]


                axes[i].plot(fpr,tpr, label=f'{model}')
                axes[i].plot([0,1],[0,1], 'k--')
                axes[i].plot(0, 1, marker='*', color='green', markersize=15, label='Ideal Point (0,1)')
                axes[i].annotate('Perfect Point (0,1)',
                    xy=(0,1),
                    xytext=(0.1,0.9),
                    arrowprops=dict(facecolor='green', arrowstyle='->'),
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'))
                axes[i].plot(best_fpr, best_tpr, 'ro')  # red dot
                axes[i].annotate(f'Optimum\n(FPR={best_fpr:.2f}, TPR={best_tpr:.2f})',
                    xy=(best_fpr, best_tpr), 
                    xytext=(best_fpr+0.1, best_tpr-0.1),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'))
                axes[i].set_title(f'ROC Curve of {model}')
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].legend(fontsize=14)

            for j in range(len(self.results_for_plot),len(axes)):
                fig.delaxes(axes[j])


            plt.tight_layout()
            plt.show()                
        
































