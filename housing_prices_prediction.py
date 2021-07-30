from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split

# Reading the data
X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Removing rows with missing target, separates target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Breaking off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = \
    train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)


# Selecting categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Selecting numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                  X_train_full[cname].dtype in ['int64', 'float64']]

# Keeping selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Defining the model
forrest_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', forrest_model)
                                 ])

# Preprocessing of training data, fit model
model_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, getting predictions
price_predictions = model_pipeline.predict(X_valid)

# Evaluating the model
evaluation_score = mean_absolute_error(y_valid, price_predictions)
print('Mean Absolute Error (in US$):', evaluation_score)

# Preprocessing of the test data, fit model
preds_test = model_pipeline.predict(X_test)

# Saving the test predictions to a file
forrest_output = pd.DataFrame({'Id': X_test.index,
                               'SalePrice': preds_test})
forrest_output.to_csv('results.csv', index=False)