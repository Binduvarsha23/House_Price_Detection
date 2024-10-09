# packages need to be downloaded pip install numpy scipy pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor



# Load the dataset
housing = pd.read_csv("data.csv")

# Stratified split to maintain 'CHAS' category proportion
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Create a copy of the training data for exploration
housing = strat_train_set.copy()

# Feature engineering: add new feature TAXRM
housing["TAXRM"] = housing['TAX'] / housing['RM']

# Impute missing values for 'RM' (median strategy)
median = housing['RM'].median()
housing['RM'].fillna(median, inplace=True)

# Check correlation
corr_matrix = housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))

# Split the dataset into features and labels
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Define a pipeline to handle missing values and scaling
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# Prepare the training data using the pipeline
housing_num_tr = my_pipeline.fit_transform(housing)

# Model selection (Random Forest)
model = RandomForestRegressor()
# model = LinearRegression()
# model = DecisionTreeRegressor()
model.fit(housing_num_tr, housing_labels)

# Evaluate the model on the test set
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)

final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(f"RMSE: {final_rmse}")

# Save the model
dump(model, 'Dragon.joblib')

# Plot predicted vs actual values
plt.scatter(Y_test, final_predictions)
plt.xlabel("Actual Values (MEDV)")
plt.ylabel("Predicted Values (MEDV)")
plt.title("Actual vs Predicted")
plt.show()

# Plot residuals (errors)
residuals = Y_test - final_predictions
plt.hist(residuals, bins=50)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
plt.show()