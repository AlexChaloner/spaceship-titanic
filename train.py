from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from data import get_train_data


df = get_train_data()
target_column = "Transported"

# Split the data into features (X) and target variable (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

clf = RandomForestClassifier(n_estimators=100, random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)

print(scores)