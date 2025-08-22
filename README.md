# Iris Flower Classification using Machine Learning

## Project Overview
This project focuses on building a machine learning model to classify iris flowers into three species — **Setosa, Versicolor, and Virginica** — based on **sepal** and **petal measurements**.  
It automates the classification process, reducing manual effort and improving accuracy.

---

## Problem Statement
Manually identifying flower species based on physical measurements can be time-consuming and error-prone.  
This project aims to automate the process using supervised machine learning models.

---

## Objectives
- Train ML models (Decision Tree, Logistic Regression) on the Iris dataset.  
- Evaluate performance using accuracy, confusion matrix, and classification report.  
- Visualize decision boundaries, confusion matrices, and decision tree plots.  
- Predict flower species for new input values.  

---

## Dataset
- Source: **Iris Dataset** (sklearn library)  
- Total samples: **150**  
- Features: `Sepal Length`, `Sepal Width`, `Petal Length`, `Petal Width`  
- Classes: `Setosa`, `Versicolor`, `Virginica`  

---

## Project Workflow
1. Load dataset from sklearn.  
2. Split data into training and testing sets.  
3. Train models: **Decision Tree** and **Logistic Regression**.  
4. Evaluate using accuracy & classification report.  
5. Visualize results (confusion matrix heatmaps, decision tree plot).  
6. Predict species for new input `[5.0, 3.4, 1.5, 0.2]`.  

---

## How to Run
### 1. Clone the repository:
```bash
git clone https://github.com/AnandmaniKumar/Iris-Flower-Classification.git
cd Iris_Flower_classification 

##2. install dependencies 

pip install -r requirements.txt
 ## 3.  Run the program 
python iris_classification.py
or
jupyter notebook iris_classification.ipynb

