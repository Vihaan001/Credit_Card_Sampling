# Sampling Assignment - Credit Card Fraud Detection

**Assignment Number:** 1  
**Subject:** Parameter Estimation & Sampling  

## Submitted By
* **Name:** Vihaan Agarwal
* **Roll Number:** 102303658
* **Group:** 3C45
* **Batch:** L1 Batch

---

## Project Overview
This project analyzes the impact of different sampling techniques on the performance of various machine learning models using the Credit Card Fraud Detection dataset. The dataset is highly imbalanced, requiring balancing techniques (SMOTE/Random Oversampling) before generating representative samples.

### Objective
1.  **Balance** the imbalanced dataset.
2.  Create **5 different samples** using specific sampling techniques.
3.  Train **5 different Machine Learning models** on these samples.
4.  Compare the accuracy to determine the best sampling technique and model combination.

---

## Methodology

### 1. Data Preparation
* **Dataset:** Credit Card Fraud Detection Data.
* **Balancing:** Used `RandomOverSampler` to balance the minority class (Fraud) with the majority class (Legit).

### 2. Sample Size Calculation
Using **Cochran’s Formula** with a 95% confidence level and 5% margin of error, the required sample size was determined.

### 3. Sampling Techniques Used
1.  **Simple Random Sampling:** Random selection of representative samples.
2.  **Systematic Sampling:** Selection based on a fixed interval (step size).
3.  **Stratified Sampling:** Selection ensuring class proportions are maintained.
4.  **Cluster Sampling:** Grouping data into clusters and selecting random clusters.
5.  **Bootstrap Sampling:** Random sampling with replacement.

### 4. Models Evaluated
* **M1:** Logistic Regression
* **M2:** Decision Tree Classifier
* **M3:** Random Forest Classifier
* **M4:** Support Vector Machine (SVM)
* **M5:** K-Nearest Neighbors (KNN)

---

## Results & Analysis

The following table summarizes the accuracy achieved by each model across different sampling techniques:

| Model | Simple Random | Systematic | Stratified | Cluster | Bootstrap |
|-------|---------------|------------|------------|---------|-----------|
| **Logistic Regression** | 89.61% | 88.31% | 90.91% | 92.86% | 94.81% |
| **Decision Tree** | 97.40% | 97.40% | 98.70% | 100.00% | 97.40% |
| **Random Forest** | 98.70% | 100.00% | 98.70% | 100.00% | 98.70% |
| **SVM** | 70.13% | 75.32% | 75.32% | 66.07% | 80.52% |
| **KNN** | 97.40% | 93.51% | 89.61% | 94.64% | 97.40% |

### Key Observations
* **Best Performing Model:** **Random Forest** consistently achieved the highest accuracy, reaching **100%** with Systematic and Cluster sampling.
* **Best Sampling Technique:** **Bootstrap Sampling** provided the most consistent high accuracy across all models (highest average accuracy of ~93.7%).
* **Cluster Sampling** yielded excellent results for Tree-based models (Decision Tree & Random Forest) but performed poorly for SVM.

---

## Conclusion
For this specific dataset, **Random Forest** combined with **Systematic** or **Cluster Sampling** proved to be the most effective approach. However, if model stability across different algorithms is the priority, **Bootstrap Sampling** is the recommended technique.

---

## How to Run the Code
1.  Clone the repository.
2.  Ensure `Creditcard_data.csv` is in the root directory.
3.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn
    ```
4.  Run the notebook in Jupyter.
