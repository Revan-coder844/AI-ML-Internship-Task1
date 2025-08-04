# AI-ML-Internship-Task1
Data cleaning and processing for the Titanic dataset.
AI & ML Internship: Task 1 - Data Cleaning & Preprocessing

# overview
This repository contains the solution for Task 1 of the AI & ML Internship, focusing on data cleaning and preprocessing for machine learning using the Titanic dataset. The objective is to prepare raw data for ML by handling missing values, encoding categorical features, scaling numerical features, and removing outliers.
# Task Description
The task involves:

Importing and exploring the dataset (checking for nulls and data types).
Handling missing values using mean/median computation.
Converting categorical features to numerical using encoding techniques.
Normalizing/standardizing numerical features.
Visualizing and removing outliers using boxplots.

# Tools Used

#### Python: Programming language for data processing.
#### Pandas: For data manipulation and analysis.
#### NumPy: For numerical computations.
#### Matplotlib/Seaborn: For data visualization (e.g., boxplots for outlier detection).

# Dataset

##### Titanic Dataset: Sourced from Kaggle Titanic Dataset.
File: titanic_dataset.csv (included in the repository).

# Steps Performed

#### Data Exploration:
Loaded the dataset using Pandas.
Checked for null values, data types, and basic statistics using .info() and .describe().


#### Handling Missing Values:
Filled missing Age values with the median.
Filled missing Embarked values with the mode.
Dropped the Cabin column due to excessive missing data.


#### Feature Scaling:
Standardized numerical features (Age, Fare) using StandardScaler from scikit-learn.


#### Outlier Detection and Removal:
Visualized outliers in Age and Fare using boxplots (Seaborn).

#### Output:
Saved the cleaned dataset as cleaned_titanic.csv.



# Repository Structure
# AI-ML-Internship-Task1/
### Table
|task1_data_cleaning.py|Python script for data cleaning and preprocessing|
|----------------------|-------------------------------------------------|
|titanic_dataset.csv|Original Titanic dataset|
|cleaned_titanic.csv|Cleaned dataset after preprocessing|
|screenshots|Boxplot for Age outliers|Boxplot for Fare outliers|
README.md|This file|

# How to Run the Code

#### Prerequisites:
Install Python 3.8+.
Install required libraries:
### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```



#### Steps:
Clone this repository:
```bash
git clone https://github.com/Revan-coder844/AI-ML-Internship-Task1.git
```

Navigate to the repository folder:
```bash
cd AI-ML-Internship-Task1
```

Run the Python script:
```bash
python titanic_preprocessing.py
```
The script will process the Titanic-Dataset.csv, generate boxplot visualizations (saved in screenshots/), and output the cleaned dataset as cleaned_titanic.csv.

# Insights Gained

Data preprocessing significantly impacts model performance by ensuring clean, consistent, and scaled data.
# Submission Details

GitHub Repository: [Link to my repository, e.g., https://github.com/revan-coder844/AI-ML-Internship-Task1]

