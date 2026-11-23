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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

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

print("Best parameters logistic regression:", grid.best_params_)
print(f"Best 5-fold CV accuracy logistic regression: {grid.best_score_:.4f}")

score = grid.score(X_test, y_test)
print(f"Test set accuracy: {score:.4f}")
'''
# final logistic regression model with the highest scoring hyperparameters
clf = LogisticRegression(C=0.1, random_state=42).fit(X_train_scaled, y_train)

y_test_pred = clf.predict(X_test_scaled)
y_train_pred = clf.predict(X_train_scaled)

score_test = clf.score(X_test_scaled, y_test)
score_train = clf.score(X_train_scaled, y_train)

print("Logistic regression train score", score_train)
print("logistic regression test score", score_test)

'''
model = DecisionTreeClassifier(ccp_alpha=0.0, criterion='gini', max_depth=1, min_samples_split=5).fit(X_train, y_train)

print("Training set accuracy decision tree: {}".format(model.score(X_train, y_train)))
print("Test set accuracy decision tree: {}".format(model.score(X_test, y_test)))

param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [1, 5, 10, 20, 30],
              'ccp_alpha': [0.0, 0.005, 0.01, 0.02, 0.05],
              'max_features': [1,2,3],
              'criterion': ['gini', 'entropy']}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best DT params:", grid_search.best_params_)
print(f"Best 5-fold CV accuracy: {grid_search.best_score_:.4f}")
print(f"Test set accuracy: {grid_search.score(X_test, y_test):.4f}")
'''
# final deicision tree model with best hyperparameters

decisionTree_model = DecisionTreeClassifier(ccp_alpha=0.0, criterion='entropy', max_depth=2, max_features=2, min_samples_split=30)

decisionTree_model.fit(X_train, y_train)
print("Final decision tree train set score", decisionTree_model.score(X_train, y_train))
print("Final decision tree test set score", decisionTree_model.score(X_test, y_test))

'''
plt.figure(figsize=(10, 5))
plot_tree(
    decisionTree_model,
    feature_names=["Feature 0", "Feature 1", "Feature 2"],  
    class_names=["Class 0", "Class 1"],  
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Final Regularized Decision Tree (max_depth=2, criterion='entropy')")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300)
plt.show()
'''

svm = SVC()
svm.fit(X_train_scaled, y_train)

print(f'SVM classifier test accuracy: {svm.score(X_test_scaled, y_test):.4f}')
print(f'SVM classifier train accuracy: {svm.score(X_train_scaled, y_train):.4f}')

'''
svm_pipeline = make_pipeline(StandardScaler(), SVC(random_state=42))

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
print(f"Best 5-fold CV accuracy SVM: {grid_svm.best_score_:.4f}")
print(f"Test set accuracy SVM: {grid_svm.score(X_test, y_test):.4f}")
'''

# ensemble methods
np.random.seed(42)

X_train_ensemble, X_validation, y_train_ensemble, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

y_train_ensemble = np.array(y_train_ensemble)
y_test = np.array(y_test)
y_validation = np.array(y_validation)

'''
# adaboost ensemble
parameters_adaBoost = {
    'learning_rate': [round(x * 0.1, 1) for x in range(1, 21)],
    'n_estimators': list(range(1,51)),
}

adaBoost_model = AdaBoostClassifier(random_state=42)
adaBoost_model = GridSearchCV(adaBoost_model, parameters_adaBoost, cv=5, n_jobs=-1)

adaBoost_model.fit(X_train_ensemble, y_train_ensemble)

print("Best params:", adaBoost_model.best_params_)
print("Best cv score:", adaBoost_model.best_score_)

print("Validation score AdaBoost:", adaBoost_model.score(X_validation, y_validation))
print("Train score AdaBoost:", adaBoost_model.score(X_train_ensemble, y_train_ensemble))
print("Test score AdaBoost:", adaBoost_model.score(X_test, y_test))

# bagging ensemble

parameters_bagging = {
    'estimator':[DecisionTreeClassifier(max_depth=3), SVC(), LogisticRegression()],
    'n_estimators': [5,10,15,20,30,40,50],
    'max_features': [1, 2, 3]
}

bagging_model = BaggingClassifier(random_state=42)
bagging_model = GridSearchCV(bagging_model, parameters_bagging, cv=5, n_jobs=-1)

bagging_model.fit(X_train_ensemble, y_train_ensemble)

print("Best params:", bagging_model.best_params_)
print("Best cv score:", bagging_model.best_score_)

bagging_validation = bagging_model.score(X_validation, y_validation)
bagging_train = bagging_model.score(X_train_ensemble, y_train_ensemble)
bagging_test = bagging_model.score(X_test, y_test)

print("Validation score Bagging:", bagging_validation)
print("Train score Bagging:", bagging_train)
print("Test score Bagging:", bagging_test)

# gradient boosting ensemble

parameters_gradientBoosting = {
    'learning_rate': [round(x * 0.1, 1) for x in range(1, 11)],
    'n_estimators': [10, 25, 75, 100],
    'min_samples_split': [3,5,8,10],
    'min_samples_leaf': [3,5,8,10],
    'max_depth':[1,2,3,4,5,10],
    'max_features':[1,2,3],
    'n_iter_no_change': [10,15,30],
    'ccp_alpha':[0.0, 0.01, 0.05, 0.1, 1]
}

gradientBoosting_model = GradientBoostingClassifier(random_state=42)
gradientBoosting_model = RandomizedSearchCV(gradientBoosting_model, parameters_gradientBoosting, n_iter=100,cv=5, n_jobs=-1, random_state=42)

gradientBoosting_model.fit(X_train_ensemble, y_train_ensemble)

print("Best params:", gradientBoosting_model.best_params_)
print("Best cv score:", gradientBoosting_model.best_score_)

gradientBoosting_validation = gradientBoosting_model.score(X_validation, y_validation)
gradientBoosting_train = gradientBoosting_model.score(X_train_ensemble, y_train_ensemble)
gradientBoosting_test = gradientBoosting_model.score(X_test, y_test)

print("Validation score gradient boosting:", gradientBoosting_validation)
print("Train score gradient boosting:", gradientBoosting_train)
print("Test score gradient boosting:", gradientBoosting_test)

# random forest ensemble

parameters_randomForest = {
    'n_estimators': [10, 25, 75, 100],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth':[1,2,3,4,5,10],
    'min_samples_split': [3,5,8,10],
    'min_samples_leaf': [3,5,8,10],
    'max_features':[1,2,3],
    'ccp_alpha':[0.0, 0.01, 0.05, 0.1, 1]
}

randomForest_model = RandomForestClassifier(random_state=42)
randomForest_model = RandomizedSearchCV(randomForest_model, parameters_randomForest, n_iter=100, cv=5, n_jobs=-1, random_state=42)

randomForest_model.fit(X_train_ensemble, y_train_ensemble)

print("Best params:", randomForest_model.best_params_)
print("Best cv score:", randomForest_model.best_score_)

randomForest_validation = randomForest_model.score(X_validation, y_validation)
randomForest_train = randomForest_model.score(X_train_ensemble, y_train_ensemble)
randomForest_test = randomForest_model.score(X_test, y_test)

print("Validation score random forest:", randomForest_validation)
print("Train score random forest:", randomForest_train)
print("Test score random forest:", randomForest_test)
'''

finalEnsemble_model = RandomForestClassifier(n_estimators=100, min_samples_split=3, min_samples_leaf=8, max_features=1, max_depth=10, criterion='gini', ccp_alpha=0.1)
finalEnsemble_model.fit(X_train, y_train)

print("random forest train score", finalEnsemble_model.score(X_train, y_train))
print("random forest test score", finalEnsemble_model.score(X_test, y_test))