import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA():
    """
      Initialize a new instance of the EDA class

      Parameters:
        data (pd.DataFrame): The dataset that EDA will be performed on
        autologger (boolean, optional, defaults to True): If True, then important information regarding the dataset will automatically be recorded as certain methods are used

      """
    def __init__(self, data, autologger=True):
        # Ensure that the input 'data' is a pandas DataFrame
        assert isinstance(data, pd.DataFrame), 'Data must be a dataframe object'

        # Initialize the class with the given DataFrame and store a copy of the original data
        self.data = data
        self.data_original = data

        # Extract column names, numeric, and categorical columns
        self.columns = data.columns
        self.numeric = data.select_dtypes(include=np.number)
        self.categorical = data.select_dtypes(include='object')

        # Create a logger dictionary to track the number of rows, columns, and other useful information
        self.logger = {'Number of Rows': data.shape[0], 'Number of Columns:': data.shape[1]}
        self.autologger = autologger

        # Calculate summary statistics for data
        self.summary = data.describe()

    def hist(self, all=False, variables=None, num_histograms=None, n_rows=None, n_columns=None, size=12):
      """
        Generates historgrams based on the features in the dataset

        Args:
          all (boolean, optional, defaults to False): If True, then a histogram of every single value will be displayed
          variables (list, optional, defaults to None): If not none, then a histogram of any column/feature in variables(and the data) will be generated; num_histograms must be null if variables is not
          num_histogram (int, optional, defaults to None): If not none, then a histogram of the first n number of columns/features in the data will be generated; variables must be null if num_histogram is not
          n_rows (int, optional, defaults to None): If not none, then it is the number of rows that the display of the histograms will take up;
          n_cols (int, optional, defaults to None): If not none, then it is the number of columns that the display of the histograms will take up; n_rows * n_columns should be greater than or equal to the number of histograms desired
          size (int, optional, defaults to 12): Controls how big or small the display of the historgrams will be

      """
        # Plot histograms for all numeric columns
      if all:
        return self.data.hist(figsize=(size, size))

        if variables is not None and num_histograms is not None:
          return "You provided values for both variables and num_histograms. Note that num_histograms should only be provided if you want the first n histograms in the dataframe to be created. If you did intend this effect, then no value for the variable parameter should be passed. Similarly, if you wanted to to create a histogram of specific variables only, then you do not need to give a value for the num_histograms parameter."
        columns = self.numeric.columns

        if self.autologger:
          logged_columns = []

        if variables is not None:
          i = 1
          num_box = len(variables) if variables is None else len(variables)

          if n_rows is None or n_cols is None:
            n_rows = int(num_box ** 0.5)
            n_cols = int(num_box / n_rows) + 1

          for var in variables:
            if var not in self.numeric.columns:
              print(f'{var} is not a part of the dataset or is a categorical variable, so no histogram will be created for it')
            else:
              plt.subplot(n_rows, len(variables), i)
              plt.hist(self.data[var])
              plt.xlabel(columns[i])
              if self.autologger:
                logged_columns.append(var)
              i += 1

          plt.tight_layout()

    def density(self, all=False, variables=None, num_dense=None, n_rows=None, n_columns=None, size=12):
        """
          Generates historgrams based on the features in the dataset

          Args:
            all (boolean, optional, defaults to False): If True, then a density plot of every single value will be displayed
            variables (list, optional, defaults to None): If not none, then a density plot of any column/feature in variables(and the data) will be generated; num_dense must be null if variables is not
            num_dense (int, optional, defaults to None): If not none, then a density plot of the first n number of columns/features in the data will be generated; variables must be null if num_dense is not
            n_rows (int, optional, defaults to None): If not none, then it is the number of rows that the display of the density plots will take up;
            n_cols (int, optional, defaults to None): If not none, then it is the number of columns that the display of the density plots will take up; n_rows * n_columns should be greater than or equal to the number of density plots desired
            size (int, optional, defaults to 12): Controls how big or small the display of the density plots will be

        """

        if all:
          return self.data.plot.density(figsize=(size, size))

        if variables is not None and num_dense is not None:
          return "You provided values for both variables and num_dense. Note that num_dense should only be provided if you want the first n density plots in the dataframe to be created. If you did intend this effect, then no value for the variable parameter should be passed. Similarly, if you wanted to to create a density plot of specific variables only, then you do not need to give a value for the num_dense parameter."
        columns = self.numeric.columns

        if self.autologger:
          logged_columns = []

        if variables is not None:
          i = 1
          num_box = len(variables) if variables is None else len(variables)

          if n_rows is None or n_cols is None:
            n_rows = int(num_box ** 0.5)
            n_cols = int(num_box / n_rows) + 1

          for var in variables:
            if var not in self.numeric.columns:
              print(f'{var} is not a part of the dataset or is a categorical variable, so no histogram will be created for it')
            else:
              plt.subplot(n_rows, len(variables), i)
              sns.kdeplot(self.data[var])
              plt.xlabel(columns[i])
              if self.autologger:
                logged_columns.append(var)
              i += 1

          plt.tight_layout()

    def heatmap(self, cmap='coolwarm'):
        """
          Create a heatmap of the correlation matrix for numeric columns using seaborn
        """
        return sns.heatmap(self.numeric.corr(), annot=True, cmap=cmap)

    def scatter(self, x, y, n_rows=None, n_cols=None):
      """
        Generates historgrams based on the features in the dataset

        Args:
          all (boolean, optional, defaults to False): If True, then a density plot of every single value will be displayed
          variables (list, optional, defaults to None): If not none, then a density plot of any column/feature in variables(and the data) will be generated; num_dense must be null if variables is not
          num_dense (int, optional, defaults to None): If not none, then a density plot of the first n number of columns/features in the data will be generated; variables must be null if num_dense is not
          n_rows (int, optional, defaults to None): If not none, then it is the number of rows that the display of the density plots will take up;
          n_cols (int, optional, defaults to None): If not none, then it is the number of columns that the display of the density plots will take up; n_rows * n_columns should be greater than or equal to the number of density plots desired
          size (int, optional, defaults to 12): Controls how big or small the display of the density plots will be

      """

      assert all(var in self.columns for var in x) or all(var in self.columns for var in y) in self.columns, "One or more column provided is not in the data"
      assert len(x) == len(y), "The lengths of independent_variables(x) list and dependent_variables(y) list must be the same"

      if n_rows is None or n_cols is None:
            n_rows = int(len(x) ** 0.5)
            n_cols = int(len(y) / n_rows) + 1

      for i, (var1, var2) in enumerate(zip(x, y), 1):
        plt.subplot(n_rows, n_cols, i)
        sns.scatterplot(x=self.data[var1], y=self.data[var2])
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.title(f'{y} vs {x}')

        if self.autologger:
          self.change_log(f"Correlation between {x} and {y}:", self.data[var1].corr(self.data[var2]))

    def boxplot(self, variables=None, num_boxplots=None, n_rows=None, n_cols=None):
      """
        Generates boxplots based on the features in the dataset

        Args:
          all (boolean, optional, defaults to False): If True, then a boxplot of every single value will be displayed
          variables (list, optional, defaults to None): If not none, then a boxplot of any column/feature in variables(and the data) will be generated; num_boxplots must be null if variables is not
          num_boxplot (int, optional, defaults to None): If not none, then a boxplot of the first n number of columns/features in the data will be generated; variables must be null if num_boxplots is not
          n_rows (int, optional, defaults to None): If not none, then it is the number of rows that the display of the boxplots will take up;
          n_cols (int, optional, defaults to None): If not none, then it is the number of columns that the display of the boxplots will take up; n_rows * n_columns should be greater than or equal to the number of boxplots desired
          size (int, optional, defaults to 12): Controls how big or small the display of the boxplots will be

      """

      if variables is not None and num_boxplots is not None:
        return "You provided values for both variables and num_boxplots. Note that num_boxplots should only be provided if you want the first n boxplots in the dataframe to be created. If you did intend this effect, then no value for the variable parameter should be passed. Similarly, if you wanted to to create a boxplot of specific variables only, then you do not need to give a value for the num_boxplots parameter."
      columns = self.numeric.columns

      if self.autologger:
        logged_columns = []

      if variables is not None:
        i = 1
        num_box = len(variables) if variables is None else len(variables)

        if n_rows is None or n_cols is None:
          n_rows = int(num_box ** 0.5)
          n_cols = int(num_box / n_rows) + 1

        for var in variables:
          if var not in self.numeric.columns:
            print(f'{var} is not a part of the dataset or is a categorical variable, so no boxplot will be created for it')
          else:
            plt.subplot(n_rows, len(variables), i)
            plt.boxplot(self.data[var])
            plt.xlabel(columns[i])
            if self.autologger:
              logged_columns.append(var)
            i += 1

      # Plot boxplots for numeric columns
      if num_boxplots is not None:
        num_box = len(self.numeric.columns) if num_boxplots is None or num_boxplots > len(self.numeric.columns) else num_boxplots

        if n_rows is None or n_cols is None:
            n_rows = int(num_box ** 0.5)
            n_cols = int(num_box / n_rows) + 1

        for i in range(1, num_box + 1):
            plt.subplot(n_rows, n_cols, i)
            plt.boxplot(self.data[columns[i]])
            plt.xlabel(columns[i])
            if self.autologger:
              logged_columns.append(columns[i])

      plt.tight_layout()

      if self.autologger:
        for col in logged_columns:
          Q1 = self.data[col].quantile(0.25)
          Q3 = self.data[col].quantile(0.75)
          IQR = Q3 - Q1
          outliers = self.data[(self.data[col] > Q3 + (IQR * 1.5)) | (self.data[col] < Q1 - (IQR * 1.5))]
          self.change_log(f"{col} number of outliers: ", outliers.count())
          self.change_log(f"{col} outliers", outliers)

    def bar_graph(self, cat_columns=['all'], num_var='count', n_rows=None, n_cols=None, size=12):
      """
        Generates bar_graphs based on the categorical features of the dataset

        Args:
          cat_columns (list, optional, defaults to '['all']): List of all categorical columns for bar graphs to be generated. If ['all'], then a bar graph for every single categorical column in the dataset will be displayed
          num_var (str, optional, defaults to 'count'): The numerical value that the categorical columns will be graphed on
          n_rows (int, optional, defaults to None): If not none, then it is the number of rows that the display of the bar graphs will take up;
          n_cols (int, optional, defaults to None): If not none, then it is the number of columns that the display of the bar graphs will take up; n_rows * n_columns should be greater than or equal to the number of bar graphs desired
          size (int, optional, defaults to 12): Controls how big or small the display of the bar graphs will be

      """
      if self.logger:
        logged_columns = []

      if cat_columns == ['all']:
        columns = self.categorical.columns
      else:
        columns = cat_columns

      if n_rows is None or n_cols is None:
          n_rows = int(len(columns) ** 0.5)
          n_cols = int(len(columns) / n_rows) + 1

      plt.figure(figsize=(size, size))
      for i, column in enumerate(columns):
        plt.subplot(n_rows, n_cols, i + 1)
        if num_var == 'count':
          sns.countplot(x=column, data=self.data)
        else:
          assert num_var in self.numeric.columns, 'num_var must be the name of a numeric column that is in the dataset or equal to count'
          plt.bar(self.data[column], self.data[num_var])
        plt.xlabel(column)
        plt.ylabel(num_var)

      plt.tight_layout()

      if self.autologger:
        for col in logged_columns:
          self.change_log(f'{col} records:', [{val: val.count()} for val in self.data[col].unique()])


    def change_log(self, key, value):
      """
      Adds or edits the log

      Args:
        key (str, int, float, or boolean): description of what the new information being added or changed is
        value (str, int, float, boolean, or list): information being added

      """
        # Update the logger dictionary with a new key-value pair
      self.logger[key] = value

    def remove_log(self, key):
        """
        Removes a piece of information from the log

        Args:
          key (str, int, float, or boolean): the description of the value that will be removed

        """
        self.logger.pop(key)

    def show_log(self):
        """
        Display the contents of the logs in a nicer format
        """
        for key, value in self.logger.items():
            print(f'{key}: \t{value}')

    def update_summary(self):
        """
          Recalculate summary statistics for the DataFrame
        """
        self.summary = self.data.describe()

    def remove_null(self, handle_num_nulls='median', handle_cat_nulls='mode', remove_null_under_5=False, remove_column_under_60=False):
        """
        Handles null values in the dataset

        Args:
          handle_num_nulls (str, optional, defaults to median): the value that will replace any null values found in any numerical column
          handle_cat_nulls (str, optional, defaults to mode): the value that will replace any null values found in any categorical column
          remove_null_under_5 (boolean, optional, defaults to False): if True, then any column where null values contribute to less than or equal to 5% of the dataset will be completely removed
          remove_column_under_60 (boolean, optional, defaults to False): if True, then any column where null values contribute to more than or equal to 60% of the dataset wll be completely removed

        """
        # Check for null values in the DataFrame
        null_check = self.check_null()

        if null_check == "No null values":
            return "No null values to be removed"

        null_percentages = null_check

        for column, percentage in null_percentages.items():
            if percentage <= 5.00:
                if remove_null_under_5:
                    # Remove rows with null values in columns with less than 5% nulls
                    self.data = self.data[self.data[column].notna()]
            elif percentage >= 60.00:
                if remove_column_under_60:
                    # Remove columns with more than 60% nulls
                    self.data.drop(columns=[column], inplace=True)
            else:
                # Handle null values based on the specified method (median or mode)
                self._handle_null(column, self.data[column].dtype, handle_num_nulls if self.data[column].dtype != 'object' else handle_cat_nulls)

    def _handle_null(self, column, datatype, value):
        """
          Helper method to handle null values in a column based on its datatype
        """
        if datatype == 'object':
            if value == 'mode':
                self.data[column].fillna(self.data[column].mode().iloc[0], inplace=True)
            else:
                self.data[column].fillna(value, inplace=True)
        else:
            if value == 'mode':
                self.data[column].fillna(self.data[column].mode().iloc[0], inplace=True)
            elif value == 'median':
                self.data[column].fillna(self.data[column].median(), inplace=True)
            elif value == 'mean':
                self.data[column].fillna(self.data[column].mean(), inplace=True)
            else:
                self.data[column].fillna(value, inplace=True)

    def check_null(self):
        """
          Check for null values in the DataFrame and return a dictionary with column-wise null percentages
        """
        if self.data.isna().sum().sum() == 0:
            return "No null values"

        null_percentages = {}
        for column in self.columns:
            percentage = round(100 * self.data[column].isna().sum() / len(self.data[column]), 2)
            null_percentages[column] = f'{percentage}%'

        return null_percentages

    def encode(self):
        """
        Encode categorical columns with integer labels
        """

        mapping = {}
        for column in self.categorical:
            for i, value in enumerate(self.data[column].unique()):
                mapping[value] = i
            self.data[column] = self.data[column].map(lambda x: mapping[x])

        # Update the categorical and numeric attributes after encoding
        self.categorical = pd.DataFrame()
        self.numeric = self.data

    def scaling(self, scaler):
        """
        Scale numeric columns using either StandardScaler or MinMaxScaler

        Args:
          Scaler (str): The type of scaling used

        """

        if scaler not in ('StandardScaler', 'MinMaxScaler'):
            return "Numeric columns were not scaled because an invalid scaling option was passed. Pass in either StandardScaler or MinMaxScaler"

        if scaler == 'StandardScaler':
            for column in self.numeric.columns:
                mean = self.data[column].mean()
                std = self.data[column].std()
                self.data[column] = (self.data[column] - mean) / std

        if scaler == 'MinMaxScaler':
            for column in self.numeric.columns:
                min_value = self.data[column].min()
                max_value = self.data[column].max()
                self.data[column] = (self.data[column] - min_value) / (max_value - min_value)

        return "Numeric columns successfully scaled"