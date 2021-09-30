from categorical_nb import CategoricalNBMod
import numpy as np

# Data
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])


# Model
clf = CategoricalNBMod()
clf.fit(X, y)
CategoricalNBMod()

# Predicting with Manual Prior
manual_prior = [0, 0, 0.5, 0, 0, 0.5]

print(clf.predict(X, manual_prior))
print(clf.predict_proba(X, manual_prior))
