# Project Based Virtual Internship Final Task: Home Credit Indonesia Scorecard Model

<p float="left">
  <img src="https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/images/logo_rakamin-f00322ea-dc6a-4540-b427-c90a1fc87691.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIATEHFT35YRRD7BK7S%2F20240306%2Fap-southeast-1%2Fs3%2Faws4_request&X-Amz-Date=20240306T174551Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0b6c98e57f6613a699eb6d0c904b5af683dca69e9e2e757fc1630cb286db67f3" width="300" />
  <div style="display: flex; justify-content: center;">
    <img src="https://media.giphy.com/avatars/HomeCreditID/LLViA0BmMJMY.gif" width="250" /> 
  </div>
</p>

## 🧾Table Of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Workflow](#project-workflow)
4. [Data Cleaning](#data-cleaning)
5. [Preliminary Analysis](#preliminary-analysis)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Data Preprocessing](#data-preprocessing)
8. [Model Development](#model-development)
9. [Evaluation](#evaluation)
10. [Hyperparameter Tuning](#hyperparameter-tuning)
11. [Feature Selection](#feature-selection)
12. [Conclusion](#conclusion)

## Introduction
### Project Overview/Problem
From the data provided by Home Credit, I identify that the problem Home Credit facing is they receive total bad debt or loss about $74.004.000.000 and because of that Home Credit want me to reduce the bad debt or loss in order to increase the net profit.
<p align="center">
  <img src="https://raw.githubusercontent.com/azaryasph/Home-Credit-Scorecard-Model/main/assets/target.jpg" alt="Target Image" width="800" height="300">
</p>

The solution that I propose is to create a scorecard model that can predict the probability of a customer to be a good or bad customer. The scorecard model will be used to determine the credit limit for each customer. The scorecard model will be developed using the data provided by Home Credit. The data provided by Home Credit is :
<p align="center">
  <img src="https://raw.githubusercontent.com/azaryasph/Home-Credit-Scorecard-Model/main/assets/data%20desc.jpg" alt="Alt text" width="750" height="520">
</p>

### Project Goals
The goal of this project is to decrease the default rate by atleast 3% 
### Project Objective
The objective of this project is to create a scorecard model that can predict the probability of a customer to be a good or bad customer. 
### Business Metrics 
The business metrics that will be used to evaluate the model are:
- **Default Rate**: The percentage of customers that are unable to pay their loan.
- **Net Profit**: The total profit made by Home Credit after deducting the bad debt or loss.

## Installation

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


## Project Workflow

### Data Cleaning
The project started with cleaning seven different datasets. Each dataset was cleaned individually in separate notebooks to ensure data integrity and to handle specific issues in each dataset.
<!-- Detailed description of the data cleaning process, including the cleaning of individual datasets in `app_train.ipynb`, `bureau.ipynb`, `ccb.ipynb`, `ipayment.ipynb`, and `pcbalance.ipynb`. -->

### Data Aggregation
After cleaning, all the cleaned datasets were imported into the final notebook, `final_notebook.ipynb`. The bureau data was aggregated, and the behavioral data of the previous application of the customer was joined with the previous application data. This combined data was then joined with the main application train data.<br>
Here's the join and aggregation process illustration:
<p align="center">
  <img src="https://raw.githubusercontent.com/azaryasph/Home-Credit-Scorecard-Model/main/assets/data%20join.jpg" alt="Image Description" width="650" height="320">
</p>

## Exploratory Data Analysis
On this EDA section i analyzed the data summary statistics, the target variable distribution, outlier checking, and i also checked the couple of variable distributions with the target to see how is the default rate for each category of the variable. (Correlation checking is not done because the data is too large and we are going to do feature selection based on Information value). 
Here's the table of information value brief explanation:
![Image Description](https://miro.medium.com/v2/resize:fit:1400/1*hxOouQdog6dAAlFz9rcYOQ.png)

Image Source: [Link](https://miro.medium.com/v2/resize:fit:1400/1*hxOouQdog6dAAlFz9rcYOQ.png)

## Data Preprocessing
Description of the data preprocessing steps taken before model development.

## Model Development
Description of the machine learning model development process.

## Evaluation
Description of the model evaluation process and metrics used.

## Hyperparameter Tuning
Description of the hyperparameter tuning process.

## Feature Selection
Description of the feature selection process.

## Conclusion
Final thoughts and conclusions from the project.