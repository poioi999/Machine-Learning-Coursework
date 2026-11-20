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

training_data = np.loadtxt("classification_train.txt")

X = training_data[:,:-1]
y = [int(z) for z in training_data[:,-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''

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