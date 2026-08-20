Loan Default Prediction



Machine Learning project for predicting the probability of loan default using borrower financial and credit-related information.



The project uses LightGBM for binary classification and provides an interactive web application built with Gradio and deployed on Render.



🚀 Live Demo



Live Application:

https://loan-prediction-ml-as3u.onrender.com



GitHub Repository:

https://github.com/baonguyen1210/Loan-Prediction-ML



📌 Project Overview



Loan default prediction is an important problem in credit risk management.



This project develops a Machine Learning classification model to estimate the probability that a borrower will default on a loan based on financial, credit, loan, and borrower-related characteristics.



The trained model is integrated into an interactive web application. Users can enter loan and borrower information and receive:



Probability of loan default

Probability of repayment

Risk classification

Business-oriented interpretation of the prediction



The project demonstrates an end-to-end Machine Learning workflow from data preprocessing and model training to application development and cloud deployment.



💼 Business Problem



Financial institutions need to assess borrower risk before approving loans.



Traditional credit assessment can be supported by Machine Learning models that identify patterns associated with loan default.



This project explores how borrower financial and credit information can be used to estimate default risk.



The model is designed as a decision-support demonstration, not as a replacement for professional credit assessment or lending decisions.



🎯 Project Objectives

Build a binary classification model for loan default prediction.

Estimate the probability of loan default.

Evaluate model performance using multiple classification metrics.

Convert the trained Machine Learning model into an interactive web application.

Deploy the application online.

Demonstrate an end-to-end Machine Learning project suitable for portfolio use.

📊 Dataset



The project uses the PJ2008.csv dataset.



The target variable is:



isDefault



Where:



0 = Non-default

1 = Default

Data Preprocessing



The following preprocessing steps were performed:



Load the dataset using Pandas.

Remove rows containing missing values.

Separate the target variable from predictor variables.

Identify categorical variables.

Encode the categorical feature subGrade using LabelEncoder.

Split the data into training and testing sets.

Use stratified sampling to preserve the target class distribution.

🧩 Model Features



The final model uses 13 features:



term

subGrade

annualIncome

verificationStatus

dti

ficoRangeLow

ficoRangeHigh

revolUtil

issueDate\_months

debtpca1-1

npca1-1

npca1-2

netprofit

Feature Groups



Loan Information



term

subGrade

issueDate\_months



Borrower Financial Information



annualIncome

dti

netprofit



Credit Information



ficoRangeLow

ficoRangeHigh

revolUtil



Other Financial / Verification Variables



verificationStatus

debtpca1-1

npca1-1

npca1-2

🔤 Categorical Encoding



The only categorical feature used by the final model is:



subGrade



It was encoded using LabelEncoder.



The encoding follows the following mapping:



Sub Grade	Encoded Value

A1	0

A2	1

A3	2

A4	3

A5	4

B1	5

B2	6

B3	7

B4	8

B5	9

C1	10

C2	11

C3	12

C4	13

C5	14

D1	15

D2	16

D3	17

D4	18

D5	19

E1	20

E2	21

E3	22

E4	23

E5	24

F1	25

F2	26

F3	27

F4	28

F5	29

G1	30

G2	31

G3	32

G4	33

G5	34

🤖 Machine Learning Model



The project uses:



LightGBM Classifier



LightGBM is a gradient boosting framework designed for efficient Machine Learning on structured and tabular datasets.



The model was configured for binary classification:



LGBMClassifier(

&#x20;   objective="binary",

&#x20;   metric="auc",

&#x20;   random\_state=42

)



The model produces a probability representing:



P(Default = 1)



This probability is then used by the application to generate a business-oriented risk classification.



📈 Model Evaluation



The model was evaluated on a held-out test set using:



Accuracy

Precision

Recall

ROC-AUC

Model Performance

Metric	Score

Accuracy	81.03%

Precision	57.32%

Recall	5.88%

ROC-AUC	72.17%

Metric Interpretation



Accuracy — 81.03%



Measures the overall proportion of correct predictions.



Precision — 57.32%



Among observations predicted as default, approximately 57.32% were actually default cases.



Recall — 5.88%



The model identified approximately 5.88% of the actual default cases at the classification threshold used during evaluation.



ROC-AUC — 72.17%



Measures the model's ability to distinguish between default and non-default observations across different classification thresholds.



Evaluation Note



Accuracy should not be considered sufficient on its own for a credit risk problem.



The relatively low recall indicates that the model, at its evaluated classification threshold, does not identify a large proportion of actual default cases.



This is an important limitation and an area for potential future improvement through techniques such as:



Threshold optimization

Class weighting

Resampling techniques

Hyperparameter tuning

Probability calibration

Additional feature engineering

🌐 Web Application



The trained model is integrated into an interactive web application using Gradio.



User Inputs



The application accepts the following information:



Term

Sub Grade

Annual Income

Verification Status

DTI

FICO Range Low

FICO Range High

Revolving Utilization

Issue Date (Months)

Debt PCA1-1

NPCA1-1

NPCA1-2

Net Profit



After submitting the information, the application calculates:



Repayment probability

Default probability

Risk classification

Business-oriented explanation

📊 Risk Classification



The application uses the predicted default probability to provide a simple business interpretation.



Default Probability	Risk Classification

< 30%	🟢 High Repayment Probability

30% – <50%	🟡 Medium Risk

≥ 50%	🔴 High Risk

Important Note



The 30% and 50% thresholds are business/risk interpretation thresholds defined for this portfolio project.



They are not official banking credit policy thresholds and were not automatically learned by the Machine Learning model.



🏗️ Application Architecture

Borrower / Loan Information

&#x20;           ↓

&#x20;     Gradio Interface

&#x20;           ↓

&#x20;     Input Processing

&#x20;           ↓

&#x20;     SubGrade Encoding

&#x20;           ↓

&#x20;      LightGBM Model

&#x20;           ↓

&#x20;   Default Probability

&#x20;           ↓

&#x20;    Risk Classification

&#x20;           ↓

&#x20;Business-Oriented Output

🔄 End-to-End Workflow

PJ2008.csv

&#x20;    ↓

Data Cleaning

&#x20;    ↓

Feature Selection

&#x20;    ↓

Categorical Encoding

&#x20;    ↓

Train-Test Split

&#x20;    ↓

LightGBM Classification

&#x20;    ↓

Model Evaluation

&#x20;    ↓

Model Serialization

&#x20;    ↓

Gradio Application

&#x20;    ↓

GitHub

&#x20;    ↓

Render Deployment

&#x20;    ↓

Public Web Application

🛠️ Technology Stack

Programming \& Data

Python

Pandas

NumPy

Machine Learning

Scikit-learn

LightGBM

Classification

Model Evaluation

Web Application

Gradio

Version Control \& Deployment

Git

GitHub

Render

📁 Project Structure

Loan-Prediction-ML/

│

├── app.py

├── loan\_model.pkl

├── Loan\_Prediction\_ML.ipynb

├── requirements.txt

├── README.md

├── .gitignore

└── .venv/

File Description

File	Description

app.py	Gradio web application and prediction logic

loan\_model.pkl	Trained LightGBM model

Loan\_Prediction\_ML.ipynb	Model training and evaluation notebook

requirements.txt	Python dependencies

README.md	Project documentation

.gitignore	Files excluded from Git tracking

.venv/	Local Python virtual environment



The .venv/ directory is excluded from the GitHub repository through .gitignore.



▶️ Run Locally

1\. Clone the Repository

git clone https://github.com/baonguyen1210/Loan-Prediction-ML.git

cd Loan-Prediction-ML

2\. Create a Virtual Environment

python -m venv .venv

3\. Activate the Virtual Environment



On Windows:



.venv\\Scripts\\activate

4\. Install Dependencies

pip install -r requirements.txt

5\. Run the Application

python app.py



The application will be available locally at:



http://127.0.0.1:7860

🚀 Deployment



The application is deployed using Render.



The deployment workflow is:



GitHub Repository

&#x20;       ↓

&#x20;     Render

&#x20;       ↓

&#x20;Python / Gradio App

&#x20;       ↓

&#x20;  Public URL

Live Application



https://loan-prediction-ml-as3u.onrender.com



The application is configured to use the PORT environment variable provided by Render while retaining port 7860 as the local fallback.



⚠️ Limitations



This project is developed as a Machine Learning portfolio project and has several limitations.



1\. Dataset Limitations



The model is trained using the available historical dataset. Model performance may differ when applied to other populations, institutions, economic conditions, or time periods.



2\. Low Recall



The model achieved a recall of 5.88% at the evaluated classification threshold.



This means that the current model misses a substantial proportion of actual default cases.



Therefore, the model should not be considered production-ready for real-world credit decision-making.



3\. Class Imbalance



Loan default datasets may contain an imbalance between default and non-default observations.



As a result, Accuracy can provide an incomplete view of model performance.



4\. Probability Calibration



The predicted probabilities have not been presented as formally calibrated real-world default probabilities.



Additional probability calibration analysis would be required before using them for financial decision-making.



5\. Feature Limitations



The model uses a limited set of available variables and does not represent the full underwriting process used by financial institutions.



6\. Risk Thresholds



The 30% and 50% thresholds used in the application are project-defined interpretation thresholds.



They should not be interpreted as official lending or banking policies.



🔮 Future Improvements



Potential improvements include:



Hyperparameter tuning for LightGBM

Cross-validation

Class imbalance handling

Threshold optimization

Precision-Recall curve analysis

ROC curve visualization

Feature importance analysis

SHAP-based model explainability

Probability calibration

Model monitoring

Improved UI/UX

More comprehensive credit risk features

⚖️ Disclaimer



This application is developed for educational and portfolio demonstration purposes only.



The predictions should not be used as the sole basis for actual lending decisions, credit approval, financial decisions, or rejection of loan applications.



The project demonstrates how Machine Learning can be applied to a credit risk classification problem. It should not be considered a production-ready credit scoring system.



👨‍💻 Author



Bao Nguyen



Business \& Data Analytics Enthusiast



Areas of Interest

Data Analytics

Business Intelligence

Machine Learning

Financial Analytics

Operations Analytics

Business Analytics

⭐ Project Highlights



This project demonstrates an end-to-end Machine Learning workflow:



Data preprocessing with Pandas

Categorical encoding with Scikit-learn

Binary classification with LightGBM

Model evaluation using Accuracy, Precision, Recall, and ROC-AUC

Model serialization using Joblib

Interactive Machine Learning application with Gradio

Version control with Git and GitHub

Cloud deployment with Render

Public-facing Machine Learning application

Live Demo



https://loan-prediction-ml-as3u.onrender.com



Source Code



https://github.com/baonguyen1210/Loan-Prediction-ML

