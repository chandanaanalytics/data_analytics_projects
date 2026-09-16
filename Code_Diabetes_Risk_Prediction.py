"""#Diabetes Risk Predictions
##Project Overview
#This project analyzes the risk predictions in diabetic patients using key medical indicators such as glucose level,Blood pressure, Insulin level and BMI
#Objectives
-To clean and preprocess real-world healthcare data
-To explore relationships between patient health metrics and diabetes outcome
-To visualize key trends and correlations among features
-To build and train machine learning models to predict diabetes risk
-To evaluate model performance using healthcare-relevant metrics (recall, precision, ROC-AUC)
-To identify the most important risk factors contributing to diabetes
-To generate insights and recommendations based on the findings
# Tools Used
-Python
-Pandas
-Numpy
-matplotlib
-seaborn"""
#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#To clean and preprocess real-world healthcare data
df = pd.read_csv("Project_Diabetes_Risk_Prediction/diabetes_Risk_prediction.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.columns)
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)
print(df.duplicated().sum())
print(df.drop_duplicates())
#To explore relationships between patient health metrics and diabetes outcome
df.groupby("Outcome")[["Glucose", "BloodPressure", "Insulin", "BMI"]].mean()
sns.lineplot(x="Outcome", y="Glucose", data=df)
plt.title("Glucose Levels by Diabetes Outcome")
plt.xlabel("Diabetes Outcome")
plt.ylabel("Glucose Level")
plt.show()
# Compare average glucose levels by diabetes outcome

average_glucose = df.groupby("Outcome")["Glucose"].mean()

plt.bar(["No Diabetes", "Diabetes"], average_glucose)
plt.title("Average Glucose Level by Diabetes Outcome")
plt.xlabel("Diabetes Outcome")
plt.ylabel("Average Glucose Level")

plt.show()
#compare average BMI levels by diabetes outcome
average_BMI = df.groupby("Outcome")["BMI"].mean()
plt.bar(["NO Diabetes","Diabetes"],average_BMI)
plt.title("Average BMI by Diabetes Outcome")
plt.xlabel("Diabetes Outcome")
plt.ylabel("Average BMI Level")
plt.show()
#compare average Blood Pressure levels by diabetes outcome
average_Blood_Pressure = df.groupby("Outcome")["BloodPressure"].mean()
plt.bar(["No Diabetes","Diabetes"],average_Blood_Pressure) 
plt.xlabel("Diabetes Outcome")
plt.ylabel("Average Blood Pressure")
plt.show()
#compare average Age by diabetes outcome
average_Age = df.groupby("Outcome")["Age"].mean()
plt.bar(["No Diabetes","Diabetes"],average_Age) 
plt.xlabel("Diabetes Outcome")
plt.ylabel("Average Age")
plt.show()
#compare average SkinThickness by diabetes outcome
average_Skin_Thickness = df.groupby("Outcome")["SkinThickness"].mean()
plt.bar(["No Diabetes","Diabetes"],average_Skin_Thickness) 
plt.xlabel("Diabetes Outcome")
plt.ylabel("Average Skin Thickness")
plt.show()
#compare average Pregnancies by diabetes outcome
average_Pregnancies = df.groupby("Outcome")["Pregnancies"].mean()
plt.bar(["No Diabetes","Diabetes"],average_Pregnancies) 
plt.xlabel("Diabetes Outcome")
plt.ylabel("Average Pregnancies")
plt.show()
#For each age group, compare the percentage of women who have diabetes vs those who don't, while also considering the number of pregnancies.
bins=[20,30,40,50,60,100]
labels=["21-30", "31-40", "41-50", "51-60", "60+"] 

df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

age_diabetes = pd.crosstab(
    df["Age_Group"],
    df["Outcome"],
    normalize="index"
) * 100

print(age_diabetes)
age_diabetes.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Diabetes Percentage by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Percentage (%)")
plt.legend(["Without Diabetes", "With Diabetes"])
plt.xticks(rotation=0)
plt.show()
pregnancy_diabetes = df.groupby("Outcome")["Pregnancies"].mean()

print(pregnancy_diabetes)     
pregnancy_diabetes.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Average Number of Pregnancies by Diabetes Outcome")
plt.xlabel("Diabetes Outcome")
plt.ylabel("Average Number of Pregnancies")
plt.xticks([0, 1], ["Without Diabetes", "With Diabetes"], rotation=0)
plt.show()     
#To build and train machine learning models to predict diabetes risk.
X = df.drop(["Outcome", "Age_Group"], axis=1)
y = df["Outcome"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
#Evaluating the model
#Accuracy
y_pred=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
#Precision & Redcall
from sklearn.metrics import precision_score, recall_score
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
print("Precision",precision)
print("Recall:",recall)
#Confusion Matrix 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm=confusion_matrix (y_test,y_pred)
print("Confusion Matrix")
print(cm)
disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diaries","Diabetes"]
)
disp.plot()
plt.title("Confusiom Matrix-Logistic Regression")
plt.show()
#ROC-AUC
from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)
from sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")

# Random classifier line
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()

plt.show()
# Identify important risk factors

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

# Sort by coefficient
feature_importance = feature_importance.sort_values(
    by="Coefficient",
    ascending=False
)

print("Important Risk Factors:")
print(feature_importance)
# Visualize important risk factors

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Coefficient"]
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Risk Factor")
plt.title("Important Factors in Diabetes Prediction")

plt.show()
# Final Insights and Recommendations

print("\n===== FINAL INSIGHTS =====")

print("\n1. Diabetes Prediction:")
print("The Logistic Regression model was used to predict diabetes outcomes based on patient health indicators.")

print("\n2. Important Predictive Factors:")
print(feature_importance)

print("\n3. Model Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)

print("\n4. Recommendations:")
print("- Regular monitoring of blood glucose levels can help identify diabetes risk early.")
print("- Maintaining a healthy BMI through balanced nutrition and physical activity is important.")
print("- Individuals with multiple risk factors may benefit from regular diabetes screening.")
print("- Predictive models can support healthcare analysis but should not replace clinical diagnosis.")
# Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

# Create the model
dt_model = DecisionTreeClassifier(
    random_state=42
)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
dt_pred = dt_model.predict(X_test)

print("Decision Tree Predictions:")
print(dt_pred)
from sklearn.metrics import accuracy_score, precision_score, recall_score

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)

print("\nDecision Tree Performance:")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
# Decision Tree ROC-AUC

from sklearn.metrics import roc_auc_score

dt_prob = dt_model.predict_proba(X_test)[:, 1]

dt_roc_auc = roc_auc_score(y_test, dt_prob)

print("Decision Tree ROC-AUC:", dt_roc_auc)
# Compare Machine Learning Models

model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [accuracy, dt_accuracy],
    "Precision": [precision, dt_precision],
    "Recall": [recall, dt_recall],
    "ROC-AUC": [roc_auc, dt_roc_auc]
})

print("\n===== MODEL COMPARISON =====")
print(model_comparison)
# Visualize Model Performance

model_comparison.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Logistic Regression vs Decision Tree")
plt.xlabel("Machine Learning Model")
plt.ylabel("Performance Score")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.ylim(0, 1)

plt.show()
# Final Model Selection

if (dt_recall > recall) and (dt_roc_auc >= roc_auc):
    best_model = "Decision Tree"
    best_recall = dt_recall
    best_auc = dt_roc_auc
else:
    best_model = "Logistic Regression"
    best_recall = recall
    best_auc = roc_auc

print("\n===== FINAL MODEL SELECTION =====")
print("Best Model:", best_model)
print("Recall:", best_recall)
print("ROC-AUC:", best_auc)
# Final Project Conclusion

print("\n===== PROJECT CONCLUSION =====")

print(
    f"The project developed machine learning models to predict diabetes "
    f"risk using medical indicators such as Glucose, Blood Pressure, "
    f"Insulin, BMI, Age and Pregnancies."
)

print(
    f"Two machine learning algorithms, Logistic Regression and Decision Tree, "
    f"were trained and evaluated using Accuracy, Precision, Recall and ROC-AUC."
)

print(
    f"Based on the evaluation results, {best_model} was selected as the "
    f"better-performing model for this dataset."
)

print(
    "The analysis also helped identify important predictive factors "
    "associated with diabetes outcomes."
)

print(
    "These findings can support healthcare data analysis and early risk "
    "assessment. However, the model should be considered a supportive "
    "analytical tool and should not replace professional medical diagnosis."
)
