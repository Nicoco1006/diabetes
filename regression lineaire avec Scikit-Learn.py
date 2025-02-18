# regression lineaire avec Scikit-Learn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"R² Entrainement : {train_score:.3f}")
print(f"R² Test : {test_score:.3f}")

if test_score < 0.5:
    print("underfitting")
elif test_score < (train_score -0.1):
    print("surement overfitting")
else:
    print("Bon modèle")




diabetes = load_diabetes()
print(diabetes.target)
#print(diabetes.DESCR)
