import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.pipeline import Pipeline


ds=pd.read_csv("C:\\Users\\visma\\Downloads\\Titanic-Dataset.csv")
#print(ds.info())
#print(ds.describe())

#ds['Age'].hist()
#plt.show()
ds['Age'].fillna(ds['Age'].mean(), inplace=True)
ds = ds.drop('Cabin', axis=1)
ds['Embarked'].fillna(ds['Embarked'].mode()[0], inplace=True)
print(ds.isnull().sum())
ds = pd.get_dummies(ds, columns=['Sex', 'Embarked'], drop_first=True)
ds = ds.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
X = ds.drop('Survived', axis=1)
y = ds['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
test_size=0.2
random_state=42


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, model.predict(X_test_scaled)))
scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5
)

print(scores)
print(scores.mean())

pipeline = Pipeline([
 ('scaler', scaler),
 ('model', model)
])


joblib.dump(model, "titanic_model.joblib")
joblib.dump(scaler, "titanic_scaler.joblib")
joblib.dump(pipeline,"pipeline.joblib")

print("Model saved successfully")


