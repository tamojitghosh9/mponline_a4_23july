# MPOnline Assignment-4 23rd July - 24th July 
# Breast Cancer Diagnosis Using K-Nearest Neighbors (KNN)

## Objective
The objective of this project is to develop a machine learning classification model to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on clinical diagnostic measurements. By deploying a K-Nearest Neighbors (KNN) classifier, the model provides a reliable medical decision-support tool designed to assist healthcare professionals in early cancer detection while minimizing critical diagnostic errors.

## Dataset Link
This project utilizes the **Breast Cancer Wisconsin Diagnostic Dataset** from the UCI Machine Learning Repository, hosted on Kaggle:
* **Kaggle Link:** [Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* **Dataset Characteristics:** 569 patient records containing 30 continuous numerical features (such as tumor radius, texture, perimeter, and smoothness) and one target classification label.


## Libraries Used
* **Pandas:** For structural data loading, dataframe manipulation, and dropping non-predictive columns.
* **NumPy:** For array operations and mathematical computation.
* **Scikit-Learn (`sklearn`):** For machine learning modeling and pipeline workflows, specifically:
  * `LabelEncoder` and `StandardScaler` for preprocessing.
  * `train_test_split` for stratified dataset partitioning.
  * `KNeighborsClassifier` for model development.
  * `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `confusion_matrix` for evaluation.
* **Matplotlib & Seaborn:** For data visualization, generating the visual confusion matrix heatmap, and plotting performance curves.


## Methodology

### 1. Data Understanding & Cleaning
* Loaded the raw dataset and inspected feature types and summary statistics.
* Identified and removed non-predictive or redundant columns: the patient tracking `id` and `Unnamed: 32` (an empty column caused by trailing commas in Kaggle CSV exports).
* Encoded the categorical target variable (`diagnosis`) into binary integers using Label Encoding: **Benign (B) → 0** and **Malignant (M) → 1**.

### 2. Data Preprocessing & Scaling
* **Stratified Splitting:** Partitioned the dataset into an **80% training set (455 patients)** and a **20% testing set (114 patients)** using stratified sampling (`stratify=y`) to maintain the exact 63:37 ratio of benign to malignant cases across both subsets.
* **Feature Standardization:** Applied `StandardScaler` to normalize all 30 continuous measurements to a mean of 0 and a variance of 1. *Note: The scaler was fitted exclusively on the training data to prevent data leakage into the test set.*

### 3. Model Development
* Developed a K-Nearest Neighbors classifier utilizing **Euclidean distance** (`metric='minkowski'`, `p=2`) to calculate spatial similarity between patient profiles.
* Established an initial baseline neighborhood size of **$K = 5$** to predict class labels for the unseen testing data.

## Results
The model was evaluated on the 114 test patients using quantitative metrics and a visual confusion matrix:

### Model Performance Metrics
* **Accuracy Score:** `0.9649` (**96.49%**) — The model correctly classified 110 out of 114 tumors.
* **Precision:** `0.9756` (**97.56%**) — When the model predicted a tumor was malignant, it was correct 97.56% of the time.
* **Recall (Sensitivity):** `0.9302` (**93.02%**) — The model successfully identified 93.02% of all actual malignant tumors in the test set.
* **F1-Score:** `0.9524` — Demonstrates a strong harmonic balance between precision and recall.

### Confusion Matrix Breakdown
| | **Predicted Benign (0)** | **Predicted Malignant (1)** |
| :--- | :---: | :---: |
| **True Benign (0)** | **70** *(True Negatives)* | **1** *(False Positive)* |
| **True Malignant (1)** | **3** *(False Negatives)* | **40** *(True Positives)* |

* **Clinical Error Analysis:** The model generated **3 False Negatives** (malignant tumors classified as benign) and **1 False Positive** (a benign tumor flagged as malignant).

### OBSERVATIONS ADDED AS MARKDOWN CELLS IN NOTEBOOK AFTER OUTPUT IMAGE, BASEDON MODEL'S PERFORMANCE 

## Conclusion
This study demonstrates the clinical viability of the K-Nearest Neighbors algorithm as an effective diagnostic decision-support tool for oncology screening. The baseline classifier ($K=5$) achieved a robust accuracy of **96.49%** and an F1-score of **0.9524**. Most importantly, the model demonstrated a high diagnostic sensitivity (recall) of **93.02%**, generating only three false negatives out of 114 test cases—a vital achievement where undetected malignancies carry severe consequences.

A foundational driver of this precision was the application of feature standardization during preprocessing. Because KNN relies entirely on geometric distance calculations, unscaled features with macroscopic magnitudes (such as tumor area, ranging from 140 to 2,500) would inherently overpower equally critical features measured on microscopic scales (such as cellular smoothness, ranging from 0.05 to 0.16). Standardizing features ensured every biological measurement contributed equitably to the majority vote.

Despite its diagnostic accuracy, a notable limitation of the KNN algorithm is its computational inefficiency during inference. As a non-parametric "lazy learner," KNN does not derive a generalized mathematical equation during training; instead, it stores the entire training dataset and must compute spatial distances against every historical patient profile whenever a new diagnosis is requested. While instantaneous on this 569-patient dataset, this memory-intensive search scales poorly, creating latency bottlenecks if deployed across massive real-time hospital registries.
