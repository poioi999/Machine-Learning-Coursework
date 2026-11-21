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


clf = LogisticRegression(C=0.1, penalty='l2', random_state=0).fit(X_train_scaled, y_train)

y_test_pred = clf.predict(X_test_scaled)
y_train_pred = clf.predict(X_train_scaled)

score_test = clf.score(X_test_scaled, y_test)
score_train = clf.score(X_train_scaled, y_train)


print(score_train)
print(score_test)


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

model = DecisionTreeClassifier(ccp_alpha=0.0, criterion='gini', max_depth=1, min_samples_split=5).fit(X_train, y_train)

print("Training set accuracy: {}".format(model.score(X_train, y_train)))
print("Test set accuracy: {}".format(model.score(X_test, y_test)))

print(export_text(model, feature_names=["x1", "x2", "x3"]))

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

# random forest ensemble
np.random.seed(0)

y_train = np.array(y_train)
y_test = np.array(y_test)

num_models = 20
sample_size = X_train.shape[0]
feature_sample_size = 5

all_models = []

for m in range(num_models):
    sample_idx = np.random.choice(X_train.shape[0], sample_size)
    X_train_sample, y_train_sample = X_train[sample_idx], y_train[sample_idx]

    thisModel = DecisionTreeClassifier(max_depth=4, max_features=feature_sample_size)
    thisModel.fit(X_train_sample, y_train_sample)

    all_models.append(thisModel)


def bagging_predict(test_data, all_models):
    votes = np.zeros((test_data.shape[0], len(all_models)))
    combined_predictions = np.zeros(test_data.shape[0])

    for idx, m in enumerate(all_models):
        votes[:, idx] = m.predict(test_data)

    for test_point in range(votes.shape[0]):
        combined_predictions[test_point] = np.bincount(np.int64(votes[test_point])).argmax()

    return combined_predictions

prediction_test = bagging_predict(X_test, all_models)
prediction_train = bagging_predict(X_train, all_models)

accuracy_test = np.count_nonzero(prediction_test==np.int64(y_test))/y_test.shape[0]
accuracy_train = np.count_nonzero(prediction_train==np.int64(y_train))/y_train.shape[0]

print("Train set accuracy: {}".format(accuracy_train))
print("Test set accuracy: {}".format(accuracy_test))