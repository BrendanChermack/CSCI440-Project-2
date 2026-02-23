# CSCI 440 – Project 2  
Supervised Classification on German Credit Risk

## Group Members
- Brendan Chermack
- Abel Asfaw
- Yohannes Nigusse

## Objective

This project applies supervised machine learning techniques to classify applicants as good or bad credit risks using the German Credit dataset.

The goal is to build, tune, evaluate, and interpret a classification model following the end-to-end workflow from Chapters 2 and 3 of the textbook.

## Project Workflow

1. Data Preparation  
   - Loaded ARFF dataset  
   - Performed stratified train/test split  
   - Conducted exploratory data analysis  
   - Built preprocessing pipeline (scaling + one-hot encoding)

2. Model Selection  
   - Compared Random Forest and SVC using 10-fold cross-validation

3. Model Training  
   - Selected best baseline model based on CV performance

4. Model Evaluation  
   - Evaluated using accuracy, precision, recall, F1-score, ROC-AUC

5. Hyperparameter Tuning  
   - Used RandomizedSearchCV with StratifiedKFold  
   - Optimized F1-score

6. Threshold Optimization  
   - Adjusted decision threshold from 0.5 to 0.325  
   - Improved recall for bad credit cases from 0.45 to 0.80

7. Interpretation  
   - Analyzed feature importance  
   - Interpreted false-positive and false-negative tradeoffs

## Final Model Performance (Adjusted Threshold = 0.325)

- Accuracy: 0.69  
- Precision (Bad): 0.49  
- Recall (Bad): 0.80  
- F1-score: 0.61  
- ROC-AUC: 0.78  

The final model prioritizes reducing false negatives (missed risky applicants), aligning with credit-risk management goals.   

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Open Project2.ipynb

3. Run all cells in order.
