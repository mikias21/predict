from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class ProcessData(object):

    def _covert_column(self, data, dependant_var, x, y):
        cols_to_convert = [] # list to store columns to be converted
        for col in data.columns:   # loop through dataframe columns
            if not is_numeric_dtype(data[col].dtype) and is_string_dtype(data[col].dtype):    # check datatypes
                cols_to_convert.append(data.columns.get_loc(col))  
        for col in cols_to_convert: # Do actual conversion
            if col < dependant_var:
                encoder = LabelEncoder()
                x[:, col] = encoder.fit_transform(x[:, col])
                composer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [col])], remainder='passthrough')
                x = composer.fit_transform(x)
                x = x[:, 1:]
            else:
                encoder = LabelEncoder()
                y[:, col] = encoder.fit_transform(y[:, col])
                composer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [col])], remainder='passthrough')
                y = composer.fit_transform(x)
                y = y[:, 1:]

    def process_data(self, data, dependant_var, unwanted_cols, testsize):
        """
            This method will process the input data for the learning model
            this will get the dependant and independent variables
            split the data for training and testing the model 
        """
        indpendant_var_str = "[ "
        dependant_var_str = "[ "
        x , y , x_train , x_test , y_train , y_test = None, None, None, None, None, None

        # Move the dependant variable to the end of the data frame
        cols_at_end = [dependant_var]
        data = data[ [col for col in data if col not in cols_at_end] + [ c for c in cols_at_end if c in data] ]

        # Remove unwanted cols
        for col in unwanted_cols:
            if col != '':
                del data[col]
        
        # Set dependant variable
        dependant_var = data.columns.get_loc(dependant_var)

        # Get the independant values and dependant values
        x = data.iloc[:, :dependant_var].values
        y = data.iloc[:, -1].values

        # Do conversion if the dataset contains non int or float values
        self._covert_column(data, dependant_var, x, y)   # call the convert column function to do the conversion

        # If the row of data is less than 10 don't split data 
        # This is because there won't be enough data for spliting
        # Just train the data with the provided amount without 
        # spliting
        if len(data.index) > 10:
            # Get the training and testing dataset
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=0)

            # Reshape the arrays
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
        else:
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
        
        # Set dependant var string value and indpendant vars
        dependant_var_str += data.columns[dependant_var] + " ]" 
        for col in data.columns:
            if col != data.columns[dependant_var]:
                indpendant_var_str += col + ", "
        indpendant_var_str += " ]"

        # return values 
        return x, y, x_train, x_test, y_train, y_test, dependant_var_str, indpendant_var_str
    
    def process_data_classification(self, data, dependant_var, unwanted_cols, testsize):
    
        indpendant_var_str =  "[ "
        dependant_var_str = "[ "
        x , y , x_train , x_test , y_train , y_test = None, None, None, None, None, None
        scaler = StandardScaler()

        cols_at_end = [dependant_var]
        data = data[ [col for col in data if col not in cols_at_end] + [c for c in cols_at_end if c in data]]

        for col in unwanted_cols:
            if col != '':
                del data[col]
        
        dependant_var = data.columns.get_loc(dependant_var)

        x = data.iloc[:, :dependant_var].values
        y = data.iloc[:, -1].values

        self._covert_column(data, dependant_var, x, y)

        if len(data.index) > 10:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=0)
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test) 
        x = scaler.fit_transform(x)

        dependant_var_str += data.columns[dependant_var] + " ]"
        for col in data.columns:
            if col != data.columns[dependant_var]:
                indpendant_var_str += col + " , "
        indpendant_var_str += " ]"

        return x, y, x_train, x_test, y_train, y_test, dependant_var_str, indpendant_var_str, scaler