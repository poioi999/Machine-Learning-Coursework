import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.svm import SVC

training_data = np.loadtxt("classification_train.txt")

X = training_data[:,:-1]
y = [int(z) for z in training_data[:,-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, solver='liblinear'))

param_grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 100], "logisticregression__penalty": ['l1', 'l2']}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print(f"Best 5-fold CV accuracy: {grid.best_score_:.4f}")

score = grid.score(X_test, y_test)
print(f"Test set accuracy: {score:.4f}")
'''

'''
clf = LogisticRegression(C=0.1, penalty='l1', random_state=0).fit(X_train_scaled, y_train)

y_test_pred = clf.predict(X_test_scaled)
score = clf.score(X_test_scaled, y_test)
print(score)
'''

'''
fig = plt.figure(figsize=(5, 5))

ax2 = fig.add_subplot(122)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title("Confusion Matrix")

plt.tight_layout()
plt.show()
'''
'''
model = DecisionTreeClassifier(ccp_alpha=0.0, criterion='gini', max_depth=1, min_samples_split=5).fit(X_train, y_train)

print("Training set accuracy: {}".format(model.score(X_train, y_train)))
print("Test set accuracy: {}".format(model.score(X_test, y_test)))

print(export_text(model, feature_names=["x1", "x2", "x3"]))
'''

'''
param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [1, 5, 10, 20, 30],
              'ccp_alpha': [0.0, 0.005, 0.01, 0.02, 0.05],
              'criterion': ['gini', 'entropy']}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best DT params:", grid_search.best_params_)
print(f"Best 5-fold CV accuracy: {grid_search.best_score_:.4f}")
print(f"Test set accuracy: {grid_search.score(X_test, y_test):.4f}")
'''

svm = SVC(C=10, kernel='rbf', gamma='scale', random_state=0)
svm.fit(X_train_scaled, y_train)

svm2 = SVC()
svm2.fit(X_train_scaled, y_train)

print(f'SVM2 classifier accuracy: {svm2.score(X_test_scaled, y_test):.4f}')

print(f'SVM classifier accuracy: {svm.score(X_test_scaled, y_test):.4f}')

'''
svm_pipeline = make_pipeline(StandardScaler(), SVC(random_state=0))

param_grid = [
    {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10, 100]},
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto", 0.01, 0.1, 1],
    },
    {
        "svc__kernel": ["poly"],
        "svc__degree": [2, 3],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.1, 1],
    },
]

grid_svm = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring="accuracy")
grid_svm.fit(X_train, y_train)

print("Best SVM params:", grid_svm.best_params_)
print(f"Best 5-fold CV accuracy: {grid_svm.best_score_:.4f}")
print(f"Test set accuracy: {grid_svm.score(X_test, y_test):.4f}")
'''