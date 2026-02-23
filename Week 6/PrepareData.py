import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import re 
import math 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from scipy.stats import f_oneway, chi2_contingency


class DataPreparation: 
    def __init__(self,df):
        self.df = df.copy()
        self.original = df.copy()

# ---------BACKUP ORIGINAL DATA---------------------------
    def original_data(self):
        """
        Returns the original data        
        """
        return self.original
    

# ---------CLEAN COLUMNS OF THE DATA---------------------------
    def clean_column_names(self):
        """
        Steps:
        1. Adds underscore between the lowercase and capitalcase letters: YearlyIncome -> Yearly_Income
        2. Lowercase all letters: Yearly_Income -> yearly_income
        2. Replace non-numeric and non-alphabetic characters with underscore: Capital&city >- capital_city
        """
        columns = self.df.columns 
        columns = [re.sub(r'([a-z])([A-Z])', r'\1_\2', col).lower() for col in columns]
        columns = [re.sub(r'[^\w+]', '_', col) for col in columns]
        self.df.columns = columns

        return self.df 
    
    
# ---------CLEAN VALUES IN EACH COLUMNS OF DATA---------------------------

    def clean_values(self,cols=None):
        """
        Steps: 
        1. Removes the spaces before and after the values: ' male' -> 'male'
        2. Replaces more than 1 space with 1 space 'low   income' -> 'low income'        
        """
        if cols is None:
            cols = self.df.select_dtypes(include='object').columns 
            
        for col in cols:
            
            self.df[col] = self.df[col].astype(str).str.strip()
            self.df[col] = self.df[col].str.replace(r'\s+',' ',regex=True)
        
        return self.df 



# ---------VISUALIZE THE NUMERICAL DATA---------------------------

    def visualize_distributions(self, fig_width=15, fig_height=10):
        """
        Visualizes the distributions using sns.histplot() function
        
        """
        numeric_cols = self.df.select_dtypes(include='number').columns
        cols = 3 
        rows = math.ceil(len(numeric_cols)/cols)
        fig,axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width,fig_height))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(data=self.df, x=col, ax = axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(f'{col}')
            
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])


        plt.tight_layout()
        plt.show()

    
# ---------SCALE THE NUMERICAL DATA---------------------------

    def scale_data(self, cols, method):
        """
        Parameters:

        cols : columns to scale
        method : 3 methods are available: 'StandardScaler', 'RobustScaler' and 'MinMaxScaler'. 
        Method names are case-insensitive.        
        
        """

        if method.lower()=='standardscaler': 
            scaler = StandardScaler()
            self.df[cols] = scaler.fit_transform(self.df[cols])
        elif method.lower() == 'robustscaler': 
            scaler = RobustScaler()
            self.df[cols] = scaler.fit_transform(self.df[cols])
        elif method.lower() == 'minmaxscaler':
            scaler = MinMaxScaler()
            self.df[cols] = scaler.fit_transform(self.df[cols])
        else: 
            raise ValueError("method can be 'StandardScaler', 'RobustScaler' or 'MinMaxScaler'" )
        
        return self.df 



# ---------VISUALIZE THE OUTLIERS IN NUMERICAL COLUMNS---------------------------

    def visualize_outliers(self):
        """ 
        Visualize the outliers using seaborn poxplots.
        """
        numeric_cols = self.df.select_dtypes(include='number').columns
        cols = 3 
        rows = math.ceil(len(numeric_cols)/cols)
        fig,axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,10))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=self.df, x=col, ax = axes[i])
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_xlabel(f'{col}')
            
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])


        plt.tight_layout()
        plt.show()


# ---------REMOVE OUTLIERS---------------------------

    def remove_outliers(self,cols, action='clip'):
        """
        Remove the outliers using two methods: 'clip' and 'drop'.

        Parameters: 

        cols : array of columns which will be used.
        action : 'clip' coerce the outliers, make them equal to the min/max values without dropping them;
        'drop' removes the outliers.
        """
    
        for col in cols: 
            Q1 = np.percentile(self.df[col].dropna(), 25)
            Q3 = np.percentile(self.df[col].dropna(), 75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5*IQR 
            lower_bound = Q1 - 1.5*IQR 

            if action == 'clip':
                self.df[col] = np.clip(a=self.df[col], a_max=upper_bound, a_min=lower_bound)
            elif action == 'drop':
                self.df = self.df[~ ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))]
            else:
                raise ValueError("action must be 'drop' or 'clip'")
        return self.df 
    

    
# ---------VISUALIZE OUTLIERS USING MATPLOTLIB---------------------------

    def visualize_outliers_plt(self, fig_height=15, fig_width=10): 
        """
        Visualize the outliers using matplotlib boxplots.
        """
        numeric_cols = self.df.select_dtypes(include='number').columns
        cols = 3 
        rows = math.ceil(len(numeric_cols)/cols)
        fig,axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_height,fig_width))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].boxplot(self.df[col].dropna())
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_xlabel(f'{col}')
            
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()


# ---------VISUALIZE MISSING VALUES---------------------------

    def visualize_missing(self,visualize = False,fig_height=10, fig_width=8):
        """
        It will return the missing values and the total percentage of missing values for each column.

        Parameters: 

        visualize=False : return only the missing values with percentages (as a DataFrame)

        visualize=True: return missing values with percentages and visualizes them using heatmap

        fig_height : plt figure height (default value is 10)

        fig_width : plt figure width (default value is 8)
        """


        missing_df = np.zeros((len(self.df.columns),2))
        missing_df = pd.DataFrame(data=missing_df, index=self.df.columns, columns=['num_of_missing','%_of_total'])
        
        for idx in missing_df.index:
            missing_df.loc[idx] = [
                self.df[idx].isnull().sum(),
                round((self.df[idx].isnull().sum()/self.df.shape[0])*100,2)
            ]

        if visualize:
            fig = plt.figure(figsize=(fig_height, fig_width))
            sns.heatmap(self.df.isnull(), cbar=False)
            plt.yticks([])
            plt.xticks(rotation=45,fontsize=12)
            plt.show()
        
        return missing_df



# ---------HANDLE MISSING VALUES---------------------------

    def impute_missing(self,cols,dtype_,by_):
        """ 
        Parameters: 

        cols : the columns to impute the missing values

        dype_ : data type of the columns (numeric or object). It an be numeric (number) or categorical (object)

        by_ : method to impute the columns:
         
            numeric:  'mean' - impute with mean value, 'mode' - impute with most frequent value,
            'median' - impute with the median value, if it is constant number then impute with this constant. 

            object: 'mode' - impute with most frequent category value, 'none' - impute with 'none' string,
          
        Returns dataframe with missing values imputed.
        """
        if dtype_.lower() == 'numeric':
            for col in cols:
                if by_.lower()== 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif by_.lower() == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif by_.lower() == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif np.issubdtype(type(by_), np.number): 
                    self.df[col] = self.df[col].fillna(by_)
                else:
                    print('wrong arguments')
        elif dtype_.lower() =='object':
            for col in cols:
                if by_.lower() == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif by_.lower() == 'none':
                    self.df[col] = self.df[col].fillna(by_)
                else: 
                    print('wrong arguments')
        else:
            print('wrong arguments')
        
        return self.df 


# ---------CHECK DUPLICATE ROWS---------------------------

    def check_duplicates(self,delete=False):
        if delete == False: 
            print("Numer of duplicated values: ", self.df.duplicated().sum())
            return self.df[self.df.duplicated(keep=False)].reset_index(drop=True)
        elif delete == True: 
            self.df = self.df.drop_duplicates(keep='first')
            return self.df
        else: 
            raise ValueError("Argument 'delete' must be True or False")
        
    

# ---------CORRELATIONS & RELATIONSHIPS---------------------------

# ---------NUMERICAL VS NUMERICAL---------------------------

    def correlation_num_num(self, cols, visualize=None,all=False,target=None, fig_height = 15, fig_width = 10):
        """
        Compute correlations between numerical columns and optionally visualize them.

        Parameters
        ----------
        cols : list
            List of numerical columns to include in correlation.
        visualize : str or None
            'heatmap', 'pairplot', or None. Determines whether to plot or just return table.
        all : bool
            If True, use all columns in cols. If False, show correlations with target only.
        target : str
            Required if all=False. Column to correlate with remaining columns.
        fig_height, fig_width : int
            Figure size for visualization.

        Returns
        -------
        pd.DataFrame if visualize is None, otherwise shows plot.
        """
        corr_table = self.df[cols].fillna(0).corr()

        if all:
            if visualize.lower() == 'heatmap':
                fig = plt.figure(figsize=(fig_height,fig_width))
                sns.heatmap(corr_table, cbar=False, 
                    #cmap = ['#baf7dd', '#9fedcc', '#83e6bc','#6de8b4','#4ae0a1', '#2fd690',
                    #          '#15cf81','#07b36b','#078f56','#117d50','#207350', '#235c44','#0e422c','#04331f'],
                    cmap='coolwarm',
                    annot=True
                    )
                plt.title('Correlation heatmap')
                plt.show()
        
            elif visualize.lower() == 'pairplot':
                fig = plt.figure(figsize=(fig_height,fig_width))
                sns.pairplot(data=self.df[cols])
                plt.title("Correlation plot")

            else:
                return corr_table
            
        elif all==False:
            if target is None: 
                raise AttributeError("you must specify target column")

            fig = plt.figure(figsize=(fig_height,fig_width))
            target_corr = corr_table[[target]].sort_values(by=target, ascending=False)
            sns.heatmap(target_corr.T, cbar=False, 
                #cmap = ['#baf7dd', '#9fedcc', '#83e6bc','#6de8b4','#4ae0a1', '#2fd690',
                #          '#15cf81','#07b36b','#078f56','#117d50','#207350', '#235c44','#0e422c','#04331f'],
                cmap='coolwarm',
                annot=True,
                square=True
                )
            plt.title('Correlation heatmap of target')
            plt.show()
            


# ---------NUMERICAL VS CATEGORIC TARGET---------------------------

    def correlation_num_cat(self, cols, target, visualize=None, fig_height=8, fig_width=6):
        """
        Compute ANOVA F-statistics and p-values for numeric columns against a categorical target.

        For each numeric column, the function performs a one-way ANOVA test to determine 
        if the means of the numeric values differ across the groups defined by the target.

        Parameters
        ----------
        cols : list of str
            List of numeric column names to test against the target.
        target : str
            Name of the categorical target column.
        visualize : str, optional
            'heatmap' to display a heatmap of the results. Default is None (no plot).
        fig_height : int, optional
            Height of the figure for heatmap visualization. Default is 8.
        fig_width : int, optional
            Width of the figure for heatmap visualization. Default is 6.

        Returns
        -------
        pandas.DataFrame
            DataFrame with two rows: 
                'f_stat' - F-statistics of the ANOVA tests,
                'p_value' - corresponding p-values,
            and columns corresponding to each numeric column in `cols`.

        Notes
        -----
        - If a group has less than 2 unique values, both F-statistic and p-value are set to NaN.
        - Works only with numeric columns; non-numeric columns may cause errors.
        
        Example
        -------
        >>> numeric_cols = ['age', 'income', 'loan_amount']
        >>> df = pd.DataFrame({
        ...     'age': [25, 35, 45, 25, 35],
        ...     'income': [50000, 60000, 70000, 52000, 61000],
        ...     'loan_amount': [200, 300, 400, 220, 310],
        ...     'loan_status': ['approved','denied','approved','denied','approved']
        ... })
        >>> prep = DataPreparation(df)
        >>> prep.correlation_num_cat(numeric_cols, target='loan_status', visualize='heatmap')
        """

        target_values = self.df[target].dropna().unique()
        num_of_groups = len(target_values)


        # create groups
        anova_results = {}
        for col in cols:
            groups=[]
            for i in range(num_of_groups):
                groups.append(self.df[self.df[target] == target_values[i]][col].dropna())

            if any(len(np.unique(g)) < 2 for g in groups):
                f_stat, p_value = np.nan, np.nan
            
            else: 
                f_stat, p_value = f_oneway(*groups)
            
            anova_results[col] = [f_stat, p_value]
        
        anova_df = pd.DataFrame(anova_results, index = ['f_stat', 'p_value'])

        if visualize is not None and visualize.lower() == 'heatmap':
            fig = plt.figure(figsize=(fig_height,fig_width))
            sns.heatmap(anova_df.T,cbar=False,cmap='viridis',annot=True, fmt='.2f')
            plt.title('Heatmap of ANOVA test')
            plt.show()
            
        return anova_df
      



# ---------CATEGORICAL VARIABLES ---------------------------

    def correlation_cat_cat(self, cols, target, visualize=None, fig_height=10, fig_width = 8):
        """
        Compute Chi-square statistics and Cramér's V for multiple categorical variables 
        against a categorical target variable.

        For each column in `cols`, the function performs a Chi-square test of independence 
        against the `target` column and computes Cramér's V as a measure of association strength.

        Parameters
        ----------
        cols : list of str
            List of categorical column names to test against the target.
        target : str
            Name of the categorical target column.
        visualize : str or None, optional
            If 'heatmap', displays a heatmap of Cramér's V and p-values.
            Default is None (no visualization).
        fig_height : int, optional
            Height of the heatmap figure. Default is 10.
        fig_width : int, optional
            Width of the heatmap figure. Default is 8.

        Returns
        -------
        pandas.DataFrame
            DataFrame with index as column names in `cols` and columns:
                'chi2'      - Chi-square statistic,
                'cramers_v' - Cramér's V (association strength),
                'p_value'   - p-value of the test.
            The DataFrame is sorted by ascending p-values.

        Notes
        -----
        - Cramér's V ranges from 0 (no association) to 1 (perfect association).
        - Requires categorical columns; non-categorical columns may cause errors.
        - Useful for exploring relationships between categorical features and target in datasets.

        Example
        -------
        >>> df = pd.DataFrame({
        ...     'gender': ['Male','Female','Male','Female'],
        ...     'marital_status': ['Single','Married','Married','Single'],
        ...     'loan_status': ['approved','denied','approved','denied']
        ... })
        >>> prep = DataPreparation(df)
        >>> prep.correlation_cat_cat(cols=['gender','marital_status'], target='loan_status', visualize='heatmap')
        """


        chi_results = {}
        for col in cols: 
            table = pd.crosstab(self.df[col],self.df[target])
            chi2,p_value, dof, _ = chi2_contingency(table)

            n = table.sum().sum()
            k = min(table.shape)-1
            cramer_result = np.sqrt(chi2/(n*k))

            chi_results[col] = [chi2, cramer_result, p_value]

        chi2_df = pd.DataFrame(chi_results,index = ['chi2', 'cramers_v','p_value'])
        chi2_df = chi2_df.T.sort_values(by='p_value', ascending=True)
        if visualize is not None and visualize.lower() == 'heatmap':
            fig = plt.figure(figsize=(fig_height,fig_width))
            sns.heatmap(chi2_df[['cramers_v','p_value']],cbar=False,cmap='viridis',annot=True, fmt='.2f')
            plt.title('Heatmap of Chi2 test')
            plt.show()
        
        return chi2_df 
    


# ---------ENCODE CATEGORICAL COLUMNS---------------------------

    def encoder(self, cols, method, categories_order = None):
        """
        Method = OneHotEncoder: it will make all unique values inside the columns equal to 1 and 0.
        Method = Dummy: same with one hot encoder, but the first option will be dropped.
        Method = Ordinal: it will encode categories according to orders. You must specify the order you want to encode

        Example
    -------
    >>> import pandas as pd
    >>> from PrepareData import DataPreparation
    >>> df = pd.DataFrame({'Size': ['Small','Medium','Large']})
    >>> cleaner = DataPreparation(df)
    >>> encoded_df = cleaner.encoder(df, cols=['Size_encoded'], categories_order=[['Small','Medium','Large']])
       Size_encoded
    0   0.0
    1   1.0
    2   2.0
        """
        if method.lower() == 'onehotencoder':
            coder = OneHotEncoder(sparse=False, drop=None)
            encoded = coder.fit_transform(self.df[cols])
            encoded_df = pd.DataFrame(encoded, columns=coder.get_feature_names_out(self.df[cols]),
                                      index=self.df.index)
            
            self.df = pd.concat([self.df, encoded_df], axis=1)

        elif method.lower() == 'dummy':
            dummies = pd.get_dummies(self.df[cols],drop_first=True)
            self.df = pd.concat([self.df, dummies],axis=1)

        elif method.lower() == 'ordinal':
            if categories_order == None: 
                raise AttributeError("You must specify categories_order")
            coder = OrdinalEncoder(categories=categories_order)

            encoded = coder.fit_transform(self.df[cols])

            for i, col in enumerate(cols):
                self.df[f"encoded_{col}"] = encoded[:,i]
        
        else:
            raise AttributeError("You must specify correct method")
        
        return self.df 

















         

