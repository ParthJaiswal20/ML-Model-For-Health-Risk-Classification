# NovaGen Labs Health Risk Classification Model

## Overview

This project develops a **Machine Learning classification model** to classify individuals based on their health-related attributes and predict their health outcome represented by the `Target` variable.

The project compares multiple supervised learning algorithms, including **Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, and a Voting Classifier ensemble**.

---

## Objective

The main objectives of this project are:

* Prepare health-related data for machine learning.
* Split the dataset into training and testing sets.
* Apply feature scaling where required.
* Train multiple classification algorithms.
* Compare their performance using **Accuracy, Recall, Precision, and F1-score**.
* Explore ensemble learning using a Voting Classifier.

---

## Dataset

The dataset used in this project is:

```text
novagen_dataset.csv
```

The dataset contains health-related information for individuals, including physiological measurements, lifestyle factors, and medical-history-related attributes.

The dataset contains **23 columns**, including the target variable.

### Features

The features used in the model include:

* Age
* BMI
* Blood_Pressure
* Cholesterol
* Glucose_Level
* Heart_Rate
* Sleep_Hours
* Exercise_Hours
* Water_Intake
* Stress_Level
* Smoking
* Alcohol
* Diet
* MentalHealth
* PhysicalActivity
* MedicalHistory
* Allergies
* Diet_Type__Vegan
* Diet_Type__Vegetarian
* Blood_Group_AB
* Blood_Group_B
* Blood_Group_O

### Target

* `Target` — target variable representing the health outcome/risk classification.

---

## Data Preprocessing

The dataset contained Boolean values for the blood-group one-hot encoded features:

```python
Blood_Group_AB
Blood_Group_B
Blood_Group_O
```

These columns were converted from Boolean values to integers:

```python
df["Blood_Group_AB"] = df["Blood_Group_AB"].astype(int)
df["Blood_Group_B"] = df["Blood_Group_B"].astype(int)
df["Blood_Group_O"] = df["Blood_Group_O"].astype(int)
```

The target variable was separated from the input features:

```python
X = df.drop("Target", axis=1)
y = df["Target"]
```

---

## Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```

### Split Configuration

* **Training data:** 70%
* **Testing data:** 30%
* `random_state = 42`
* Stratified split used to preserve the target-class distribution.

The test set contained **2,865 observations**.

---

## Feature Scaling

Feature scaling was performed using:

```python
StandardScaler()
```

The scaler was fitted on the training data and then used to transform both training and testing data:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaled data was used for:

* Logistic Regression
* KNN
* Voting Classifier

The tree-based models were trained using the original feature values.

---

# Models Used

## 1. Logistic Regression

The Logistic Regression model was configured with:

```python
LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=1000
)
```

### Results

| Metric              |      Score |
| ------------------- | ---------: |
| Accuracy            | **81.29%** |
| Recall              | **82.46%** |
| Precision (Class 1) |   **0.82** |
| F1-Score (Class 1)  |   **0.82** |

---

## 2. K-Nearest Neighbors

The KNN model was configured with:

```python
KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)
```

### Results

| Metric              |      Score |
| ------------------- | ---------: |
| Accuracy            | **88.03%** |
| Recall              | **88.22%** |
| Precision (Class 1) |   **0.89** |
| F1-Score (Class 1)  |   **0.88** |

---

## 3. Random Forest

The Random Forest model was configured with:

```python
RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    max_depth=None
)
```

### Results

| Metric              |      Score |
| ------------------- | ---------: |
| Accuracy            | **94.21%** |
| Recall              | **96.12%** |
| Precision (Class 1) |   **0.93** |
| F1-Score (Class 1)  |   **0.95** |

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.96      0.92      0.94      1371
           1       0.93      0.96      0.95      1494

    accuracy                           0.94      2865
   macro avg       0.94      0.94      0.94      2865
weighted avg       0.94      0.94      0.94      2865
```

---

## 4. Gradient Boosting

The Gradient Boosting model was configured with:

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

### Results

| Metric              |      Score |
| ------------------- | ---------: |
| Accuracy            | **92.53%** |
| Recall              | **94.78%** |
| Precision (Class 1) |   **0.91** |
| F1-Score (Class 1)  |   **0.93** |

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.94      0.90      0.92      1371
           1       0.91      0.95      0.93      1494

    accuracy                           0.93      2865
   macro avg       0.93      0.92      0.93      2865
weighted avg       0.93      0.93      0.93      2865
```

---

## 5. Voting Classifier

An ensemble model was created using:

* Logistic Regression
* KNN
* Random Forest

```python
VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, solver="liblinear")),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ],
    voting="soft"
)
```

The Voting Classifier used **soft voting**, where the models' predicted probabilities are combined to make the final prediction.

### Results

| Metric              |      Score |
| ------------------- | ---------: |
| Accuracy            | **91.76%** |
| Recall              | **93.17%** |
| Precision (Class 1) |   **0.91** |
| F1-Score (Class 1)  |   **0.92** |

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.92      0.90      0.91      1371
           1       0.91      0.93      0.92      1494

    accuracy                           0.92      2865
   macro avg       0.92      0.92      0.92      2865
weighted avg       0.92      0.92      0.92      2865
```

---

# Model Comparison

| Model               | Accuracy | Recall | F1-Score (Class 1) |
| ------------------- | -------: | -----: | -----------------: |
| Logistic Regression |   81.29% | 82.46% |               0.82 |
| KNN                 |   88.03% | 88.22% |               0.88 |
| Random Forest       |   94.21% | 96.12% |               0.95 |
| Gradient Boosting   |   92.53% | 94.78% |               0.93 |
| Voting Classifier   |   91.76% | 93.17% |               0.92 |

---

## Workflow

```text
NovaGen Health Dataset
        ↓
Load Dataset
        ↓
Convert Boolean Features to Integer
        ↓
Separate Features and Target
        ↓
Stratified Train-Test Split
        ↓
StandardScaler
        ↓
┌──────────────────────────────────────────┐
│              ML Models                   │
│                                          │
│  Logistic Regression                     │
│  KNN                                     │
│  Random Forest                           │
│  Gradient Boosting                       │
│  Voting Classifier                       │
└──────────────────────────────────────────┘
        ↓
Predictions
        ↓
Accuracy / Recall / Precision / F1-Score
        ↓
Model Comparison
```

---

## Technologies Used

* **Python**
* **Pandas**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

### Scikit-learn Algorithms & Tools

* `train_test_split`
* `StandardScaler`
* `LogisticRegression`
* `KNeighborsClassifier`
* `RandomForestClassifier`
* `GradientBoostingClassifier`
* `VotingClassifier`
* `accuracy_score`
* `recall_score`
* `classification_report`

---

## Project Structure

```text
ML-Model-For-Health-Risk-Classification/
│
├── novagen-health-risk-classification.ipynb
├── README.md
└── dataset/
    └── novagen_dataset.csv
```

---

## Key Learning Outcomes

Through this project, I learned how to:

* Prepare health-related tabular data for machine learning.
* Convert Boolean features into numerical representations.
* Perform stratified train-test splitting.
* Apply `StandardScaler` to numerical feature data.
* Train and evaluate multiple classification algorithms.
* Understand the difference between linear, distance-based, tree-based, boosting, and ensemble models.
* Evaluate classification models using accuracy, precision, recall, and F1-score.
* Implement a **Voting Classifier** using multiple machine learning models.
* Compare different supervised learning approaches on the same dataset.

---

## Author

**Parth Jaiswal** <br>
B.Tech CSE — AI/ML & Robotics
