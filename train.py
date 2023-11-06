from random import randint, uniform
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from data import get_train_data
import numpy as np


df = get_train_data()
target_column = "Transported"

# Split the data into features (X) and target variable (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

categorical_columns = df.select_dtypes(include=['object', 'category']).columns

ordinal_encoder = make_column_transformer(
    (
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan),
        make_column_selector(dtype_include="object"),
    ),
    remainder="passthrough",
    # Use short feature names to make it easier to specify the categorical
    # variables in the HistGradientBoostingRegressor in the next step
    # of the pipeline.
    verbose_feature_names_out=False,
)

hist_native = make_pipeline(
    ordinal_encoder,
    HistGradientBoostingClassifier(
        random_state=42,
        categorical_features=categorical_columns,
    ),
).set_output(transform="pandas")

# Define a hyperparameter grid for RandomizedSearchCV
param_dist = {
    'histgradientboostingclassifier__learning_rate': np.random.uniform(0.01, 1.0, 10),
    'histgradientboostingclassifier__max_iter': np.random.randint(10, 1000, 10),
    'histgradientboostingclassifier__max_leaf_nodes': np.random.randint(15, 128, 10),
    'histgradientboostingclassifier__min_samples_leaf': np.random.randint(1, 9, 10),
    'histgradientboostingclassifier__l2_regularization': np.random.uniform(0.0, 0.3, 10),
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    hist_native,
    param_distributions=param_dist,
    n_iter=5,  # Number of random parameter combinations to try
    cv=5,  # Number of cross-validation folds
    scoring='accuracy',
    random_state=42
)

# Perform random search
random_search.fit(X, y)

# Get the best hyperparameters
best_params = random_search.best_params_
best_model = random_search.best_estimator_
print(best_params)

# Fit the best model on the entire training dataset
best_model.fit(X, y)
y_pred_best = best_model.predict(X)
accuracy_best = accuracy_score(y, y_pred_best)
print(accuracy_best)
