import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Exploratory_Data_Analysis():
    def __init__(self, data):
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

        # Calculate summary statistics for data
        self.summary = data.describe()

    def hist(self, size=20):
        # Plot histograms for all numeric columns
        return self.data.hist(figsize=(size, size))

    def density(self, size=12):
        # Plot kernel density estimation for all numeric columns
        return self.data.plot.density(figsize=(size, size))

    def heatmap(self, cmap='coolwarm'):
        # Create a heatmap of the correlation matrix for numeric columns using seaborn
        return sns.heatmap(self.numeric.corr(), annot=True, cmap=cmap)

    def boxplot(self, num_boxplots=None, n_rows=None, n_cols=None):
        # Plot boxplots for numeric columns
        columns = self.numeric.columns
        num_box = len(self.numeric.columns) if num_boxplots is None or num_boxplots > len(self.numeric.columns) else num_boxplots

        if n_rows is None or n_cols is None:
            n_rows = int(num_box ** 0.5)
            n_cols = int(num_box / n_rows) + 1

        for i in range(1, num_box + 1):
            plt.subplot(n_rows, n_cols, i)
            plt.boxplot(self.data[columns[i]])
            plt.xlabel(columns[i])

        plt.tight_layout()
        # Return the current figure
        return plt.gcf()

    def change_log(self, key, value):
        # Update the logger dictionary with a new key-value pair
        self.logger[key] = value

    def remove_log(self, key):
        # Remove a key-value pair from the logger dictionary
        self.logger.pop(key)

    def show_log(self):
        # Display the contents of the logger dictionary
        for key, value in self.logger.items():
            print(f'{key}: {value}')

    def update_summary(self):
        # Recalculate summary statistics for the DataFrame
        self.summary = self.data.describe()

    def remove_null(self, handle_num_nulls='median', handle_cat_nulls='mode', remove_null_under_5=True, remove_column_under_60=True):
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
        # Helper method to handle null values in a column based on its datatype
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
        # Check for null values in the DataFrame and return a dictionary with column-wise null percentages
        if self.data.isna().sum().sum() == 0:
            return "No null values"

        null_percentages = {}
        for column in self.columns:
            percentage = round(100 * self.data[column].isna().sum() / len(self.data[column]), 2)
            null_percentages[column] = f'{percentage}%'

        return null_percentages

    def encode(self):
        # Encode categorical columns with integer labels
        mapping = {}
        for column in self.categorical:
            for i, value in enumerate(self.data[column].unique()):
                mapping[value] = i
            self.data[column] = self.data[column].map(lambda x: mapping[x])

        # Update the categorical and numeric attributes after encoding
        self.categorical = pd.DataFrame()
        self.numeric = self.data

    def scaling(self, scaler):
        # Scale numeric columns using either StandardScaler or MinMaxScaler
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

