# Project Based Virtual Internship Final Task: Home Credit Indonesia Scorecard Model

<div style="display: flex; justify-content: center;">
  <img src="https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/images/logo_rakamin-f00322ea-dc6a-4540-b427-c90a1fc87691.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIATEHFT35YRRD7BK7S%2F20240306%2Fap-southeast-1%2Fs3%2Faws4_request&X-Amz-Date=20240306T174551Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0b6c98e57f6613a699eb6d0c904b5af683dca69e9e2e757fc1630cb286db67f3" width="400" height=150 />
  <img src="https://media.giphy.com/avatars/HomeCreditID/LLViA0BmMJMY.gif" width="250" /> 
</div>

## 📚 Table Of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Workflow](#project-workflow)
4. [Conclusion](#conclusion)

## 🧾 Introduction
### 🖥️ Project Overview/Problem
From the data provided by Home Credit, I identify that the problem Home Credit facing is they receive total bad debt or loss about $74.004.000.000 and because of that Home Credit want me to reduce the bad debt or loss in order to increase the net profit.
<p align="center">
  <img src="https://raw.githubusercontent.com/azaryasph/Home-Credit-Scorecard-Model/main/assets/target.jpg" alt="Target Image" width="800" height="300">
</p>

The solution that I propose is to create a scorecard model that can predict the probability of a customer to be a good or bad customer. The scorecard model will be used to determine the credit limit for each customer. The scorecard model will be developed using the data provided by Home Credit. The data provided by Home Credit is :
<p align="center">
  <img src="https://raw.githubusercontent.com/azaryasph/Home-Credit-Scorecard-Model/main/assets/data%20desc.jpg" alt="Alt text" width="750" height="520">
</p>

### 🎯 Project Goals
The goal of this project is to decrease the default rate by atleast 3% 
### 📌Project Objective
The objective of this project is to create a scorecard model that can predict the probability of a customer to be a good or bad customer. 
### 📊 Business Metrics 
The business metrics that will be used to evaluate the model are:
- **Default Rate**: The percentage of customers that are unable to pay their loan.
- **Net Profit**: The total profit made by Home Credit after deducting the bad debt or loss.

## 🔌 Installation

This project requires Python and the following Python libraries installed:

- [Pandas](http://pandas.pydata.org)
- [NumPy](http://www.numpy.org)
- [Matplotlib](http://matplotlib.org)
- [Seaborn](http://seaborn.pydata.org)
- [Toad](https://github.com/amphibian-dev/toad)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
- [SHAP](https://shap.readthedocs.io/en/latest/)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included.

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included.

### Steps:

1. Clone the repo: `git clone https://github.com/your_username/your_project.git`
2. Move into the project directory: `cd your_project`
3. Install the requirements: `pip install -r requirements.txt`

Make sure you also have the custom module `missing` available in the `modules` directory.


## 🔬 Project Workflow

### 🧹 Data Cleaning
The project started with cleaning seven different datasets. Each dataset was cleaned individually in separate notebooks to ensure data integrity and to handle specific issues in each dataset.
<!-- Detailed description of the data cleaning process, including the cleaning of individual datasets in `app_train.ipynb`, `bureau.ipynb`, `ccb.ipynb`, `ipayment.ipynb`, and `pcbalance.ipynb`. -->

### 📊 Data Aggregation
After cleaning, all the cleaned datasets were imported into the final notebook, `final_notebook.ipynb`. The bureau data was aggregated, and the behavioral data of the previous application of the customer was joined with the previous application data. This combined data was then joined with the main application train data.<br>
Here's the join and aggregation process illustration:
<p align="center">
  <img src="https://raw.githubusercontent.com/azaryasph/Home-Credit-Scorecard-Model/main/assets/data%20join.jpg" alt="Image Description" width="650" height="320">
</p>

### 🔍 Exploratory Data Analysis
On this EDA section i analyzed the data summary statistics, the target variable distribution, outlier checking, and i also checked the couple of variable distributions with the target to see how is the default rate for each category of the variable. (Correlation checking is not done because the data is too large and we are going to do feature selection based on Information value). 
Here's the table of information value brief explanation:
![Image Description](https://miro.medium.com/v2/resize:fit:1400/1*hxOouQdog6dAAlFz9rcYOQ.png)

Image Source: [Link](https://miro.medium.com/v2/resize:fit:1400/1*hxOouQdog6dAAlFz9rcYOQ.png)

### 🔄 Data Filtering & Preprocessing
Data preprocessing was done using the Toad library. This step included handling invalid values, splitting the data into training and testing sets, and data filtering. The data filtering process involved feature selection, dropping columns with more than 50% missing values (i keep the rest of missing values because toad library will bin the missing values seperately), and removing features with an information value of 0.02 (indicating that they were not good predictors for the model). Features with multicollinearity were also removed. After data filtering, outliers were handled, data binning was performed, and the features were all transformed into Weight of Evidence (WoE).

### 🤖 Machine Learning Modeling
Two machine learning models were chosen for this project: Logistic Regression and LightGBM (LGBM). Logistic Regression was chosen because the goal was to create a scorecard model. Both models were trained on the preprocessed data.

### 🧪 Model Evaluation & Hyperparameter Tuning
The models were evaluated using the same threshold. Hyperparameters were tuned to optimize the models' performance.

### 💡 Model Selection
The best model choosen with the best recall score and AUC score wich is the LightGBM model.
But for the scorecard model, we will use the tuned logistic Regression model, even though it's not perform as good as the LightGBM model, but the scorecard model is more interpretable and easier to implement.

### 🧮 Business Impact Calculation 
The final step was to calculate the business impact of the model. This involved analyzing how the model's predictions would affect business decisions and outcomes.

### 📇 Scorecard Model, Calculatiing Client's Credit score with Logistic Regression Model
The final model was used to calculate the credit score for each client. The score was calculated using the formula:
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*d4RKVz5HRoOMaGPNmpFsCA.png" alt="Image Description" width="400" height="100">
</p>

Image source: [Medium Article](https://medium.com/@yanhuiliu104/credit-scoring-scorecard-development-process-8554c3492b2b)

### Conclusion
In conclusion, this project has provided a robust and effective solution to Home Credit's problem of reducing bad debt or loss. The scorecard model developed will help Home Credit make informed decisions about credit limits for each customer, thereby reducing the default rate and increasing net profit.

### Acknowledgements
I would like to thank Home Credit for providing the data and the opportunity to work on this project. I would also like to thank Rakamin for providing the platform and resources to complete this project.
And also to thanks to the author of the medium article that i used as a reference for the scorecard model development process:<br>
- [End-to-end guide to building a credit scorecard using Machine Learning](https://towardsdatascience.com/end-to-end-guide-to-building-a-credit-scorecard-using-machine-learning-6502d8bb765a)
- [How to develop a credit risk model and scorecard](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [Credit Scoring: Scorecard Development Process](https://medium.com/@yanhuiliu104/credit-scoring-scorecard-development-process-8554c3492b2b)
<br>

<p align="center">
  <img src="https://media.giphy.com/media/l3vR4yk0X20KimqJ2/giphy.gif" alt="Thank You GIF">
</p>