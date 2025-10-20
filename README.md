
# Decision Tree Classifier from Scratch

This project is a **Decision Tree classifier** built from scratch in Python using only **NumPy**. 🌳 It demonstrates core concepts like **Entropy** and **Information Gain** to find the best splits. The model recursively builds the tree, respecting `max_depth` and `min_samples_split`. The notebook trains and tests the classifier on the breast cancer dataset, achieving \~93% accuracy.

## Features

  * **Pure Python/NumPy:** The core classifier logic is built from the ground up without relying on high-level ML libraries.
  * **Information Gain:** Uses **entropy** to calculate **information gain** for finding the optimal split at each node.
  * **Hyperparameter Control:** Allows tuning of key parameters:
      * `max_depth`: The maximum depth of the tree.
      * `min_samples_split`: The minimum number of samples required to split an internal node.
      * `n_features`: The number of features to consider when looking for the best split.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Install dependencies:**
    The project requires `numpy` for calculations and `scikit-learn` for loading the example dataset and splitting the data.
    ```bash
    pip install numpy scikit-learn jupyterlab
    ```
3.  **Run the notebook:**
    Start Jupyter Lab and open the `.ipynb` notebook file.
    ```bash
    jupyter lab
    ```
    You can then run all the cells to see the classifier build, train, and test.

## How It Works

The implementation is broken into two main classes: `Node` and `DecisionTree`.

  * **`Node` Class**: A helper class representing a single node. A **decision node** stores the `feature` and `threshold` for the split, along with `left` and `right` children. A **leaf node** stores the final `value` (the predicted class).

  * **`DecisionTree` Class**: The main classifier.

      * `fit(X, y)`: Starts the recursive `_grow_tree` process to build the tree.
      * `_grow_tree()`: Recursively builds the tree by finding the best split, stopping when a stopping criterion (e.g., `max_depth`, `min_samples_split`, or a pure node) is met.
      * `_best_split()`: Iterates through features and their unique values to find the split that yields the highest **Information Gain**.
      * `_information_gain()`: Calculates the reduction in entropy from a split.
        $$
        $$$$IG = E\_{\\text{parent}} - \\left( \\frac{n\_{\\text{left}}}{n} E\_{\\text{left}} + \\frac{n\_{\\text{right}}}{n} E\_{\\text{right}} \\right)
        $$
        $$$$
        $$
      * `_entropy()`: Calculates the impurity of a set of labels $y$.
        $$
        $$$$E(y) = - \\sum\_{i} p\_i \\log(p\_i)
        $$
        $$$$Where $p_i$ is the probability of class $i$ in the node.
      * `predict(X)`: Traverses the tree for each sample in $X$ to return an array of class predictions.

## Example Usage

(This code is included in the notebook)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# --- Assume Node and DecisionTree classes are defined ---

# 1. Load data
data = datasets.load_breast_cancer()
X, y = data.data , data.target

# 2. Split data
X_train , X_test , y_train , y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 3. Initialize and train the classifier
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

# 4. Make predictions
y_pred = clf.predict(X_test)

# 5. Calculate accuracy
def accuracy(y_true, y_pred):
  return np.sum(y_true == y_pred) / len(y_true)

acc = accuracy(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Output: Accuracy: 0.9298
```

## License

This project is licensed under the MIT License.
