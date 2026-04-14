# Football Player & Match Outcome Analiysis

## Index

| Sr. No. | Contents |
|---|---|
| 1 | Abstract |
| 2 | Introduction |
| 2.1 | Overview & Importance of Data Analytics |
| 2.2 | Objective of the Project |
| 3 | Dataset Description |
| 3.1 | Dataset Name |
| 3.2 | Dataset Source and Link |
| 3.3 | Dataset Domain & Dataset Attributes Description |
| 4 | Tools and Technologies Used |
| 4.1 | Python Programming Language |
| 4.2 | Libraries Used |
| 4.3 | Development Environment |
| 5 | Data Understanding |
| 5.1 | Loading Dataset and Display Data |
| 5.2 | Dataset Shape & Data Types |
| 5.3 | Quantitative & Qualitative Data Identification |
| 6 | Exploratory Data Analysis (EDA) |
| 6.1 | Univariate Analysis |
| 6.2 | Bivariate Analysis |
| 6.3 | Correlation Matrix |
| 6.4 | Multivariate Analysis |
| 7 | Handling Missing Data and Outliers |
| 7.1 | Identifying Missing Values |
| 7.2 | Handling Missing Values (Mean/Median/Mode) |
| 7.3 | Outlier Detection & Impact of Outliers |
| 8 | Spread of Data |
| 8.1 | Statistical Analysis |
| 8.2 | Skewness & Kurtosis |
| 8.3 | Interpretation of Data Distribution |
| 9 | Automating EDA |
| 10 | Regression Analysis |
| 10.1 | Overview for Algorithm |
| 10.2 | Supervised Learning and its Techniques |
| 11 | Supervised Learning - Regression Model |
| 11.1 | Algorithm Implementation |
| 11.2 | Overfitting & Underfitting |
| 11.3 | Training vs Testing Error Comparison |
| 11.4 | Model Complexity Demonstration |
| 13 | Classification Task |
| 14 | Model Evaluation (MSE, MAE, R2) & Interpretation |
| 15 | Data Visualization |
| 16 | Results and Observations |
| 17 | Conclusion |
| 18 | Future Scope |
| 19 | References |

---

## 1. Abstract

This project is about using football data to make useful predictions. We worked on two main tasks:
1. Predicting match outcome (Win/Draw/Loss)
2. Predicting player goals

We cleaned the dataset, explored patterns using charts, trained machine learning models, and evaluated model performance. The project shows how data analytics can help in football decision-making.

Screenshot to add:
- Final model result output (both match outcome and player goal prediction)

## 2. Introduction

Football is one of the most popular sports in the world. Teams and analysts now use data to improve strategies and player performance. Instead of only depending on observation, data helps in making clear and objective decisions.

### 2.1 Overview & Importance of Data Analytics

Data analytics helps us understand past performance and predict future outcomes. In football, this is useful for:
- Team strategy planning
- Player performance analysis
- Match outcome prediction
- Goal scoring prediction

### 2.2 Objective of the Project

The objectives of this project are:
- To study and understand football player and match data
- To perform EDA for finding useful patterns
- To predict player goals using regression
- To predict match outcome using classification
- To evaluate model performance and explain the results

Screenshot to add:
- Introduction page or notebook section title screenshot

## 3. Dataset Description

### 3.1 Dataset Name

Dataset used: cleaned_football_dataset.csv

### 3.2 Dataset Source and Link

Source: Kaggle
Link: https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025

### 3.3 Dataset Domain & Dataset Attributes Description

Domain: Sports Analytics (Football)

Important columns used in this project:
- Player
- Team
- Position
- Age
- Minutes
- Matches
- Goals
- Assists
- xG
- xAG
- Shots
- Shots_on_Target

Screenshot to add:
- Dataset file preview (first 5 rows)
- Column names list

## 4. Tools and Technologies Used

### 4.1 Python Programming Language

Python was used for complete analysis and model building.

### 4.2 Libraries Used

- pandas for data handling
- numpy for numerical operations
- matplotlib and seaborn for visualization
- scikit-learn for machine learning and evaluation

### 4.3 Development Environment

- Jupyter Notebook
- VS Code
- Python virtual environment

Screenshot to add:
- Import libraries code cell

## 5. Data Understanding

### 5.1 Loading Dataset and Display Data

The dataset was loaded using pandas.read_csv(). We checked the first rows using head() to understand the data.

### 5.2 Dataset Shape & Data Types

We checked:
- Number of rows and columns using shape
- Data type of each column using info()

### 5.3 Quantitative & Qualitative Data Identification

Quantitative data examples:
- Goals, Assists, Shots, Minutes, xG, xAG

Qualitative data examples:
- Player, Team, Position

Screenshot to add:
- head() output
- shape output
- info() output

## 6. Exploratory Data Analysis (EDA)

### 6.1 Univariate Analysis

Univariate analysis means studying one variable at a time. We used histograms and boxplots for Goals, Assists, and Shots.

### 6.2 Bivariate Analysis

Bivariate analysis means studying relation between two variables. We checked:
- Shots vs Goals
- xG vs Goals
- Minutes vs Goals

### 6.3 Correlation Matrix

A correlation heatmap was used to understand which numerical features are strongly related.

### 6.4 Multivariate Analysis

We used pairplot and grouped visualizations to understand multiple variable relationships together.

Screenshot to add:
- Histogram plot
- Scatter plot
- Correlation heatmap
- Pairplot or grouped chart

## 7. Handling Missing Data and Outliers

### 7.1 Identifying Missing Values

Missing values were identified using isnull().sum().

### 7.2 Handling Missing Values (Mean / Median / Mode)

- Numerical missing values filled with mean or median
- Categorical missing values filled with mode

### 7.3 Outlier Detection (using Boxplots) & Impact of Outliers on Dataset

Outliers were detected using boxplots. Outliers can affect model accuracy, especially regression models, so this step is important.

Screenshot to add:
- Missing values before handling
- Missing values after handling
- Boxplots for outlier detection

## 8. Spread of Data

### 8.1 Statistical Analysis (Mean / Mode / Standard Deviation)

We calculated mean, mode, and standard deviation for major numerical columns.

### 8.2 Skewness & Kurtosis

- Skewness shows whether data is left or right tilted
- Kurtosis shows how heavy the tails are

### 8.3 Interpretation of Data Distribution

Based on statistics and plots, we interpreted whether data is normally distributed or skewed and how that can influence model selection.

Screenshot to add:
- describe() output
- skewness and kurtosis table/output

## 9. Automating EDA

Reusable Python functions were created for:
- describe()
- info()
- isnull()
- corr()

This reduces repeated code and makes analysis faster.

Screenshot to add:
- Function definitions
- Example function output

## 10. Regression Analysis

### 10.1 Overview for Algorithm

For player goal prediction:
- Dependent variable (target): Goals
- Independent variables: Shots, Minutes, xG, Assists, etc.
- We checked correlation and covariance before model training
- Data was split into train and test sets

### 10.2 Supervised Learning and its Techniques

Supervised learning uses labeled data.
- Regression is used for numeric prediction (player goals)
- Classification is used for category prediction (match outcome)

Screenshot to add:
- X and y selection code
- train_test_split code/output

## 11. Supervised Learning - Regression Model

### 11.1 Algorithm Implementation

Implemented models:
- Simple Linear Regression
- Multiple Linear Regression

### 11.2 Explanation of Overfitting & Underfitting

- Underfitting: model is too simple, both train and test performance are low
- Overfitting: model is too complex, train performance is high but test performance drops

### 11.3 Training vs Testing Error Comparison

Train and test errors were compared using MSE and MAE.

### 11.4 Model Complexity Demonstration

Model complexity was demonstrated with different feature combinations (or polynomial level) and their train/test performance.

Screenshot to add:
- Model training code
- Predicted vs actual output
- Train vs test error comparison chart/table

## 13. Classification Task

Match outcome prediction was treated as a classification task. The target classes were:
- Win
- Draw
- Loss

A classification algorithm (such as Logistic Regression) was trained and tested.

Screenshot to add:
- Match outcome class creation
- Classification model output
- Confusion matrix

## 14. Model Evaluation (Mean Squared Error (MSE), Mean Absolute Error (MAE), R2 Score) & Interpretation of Model Performance

For player goal prediction (regression), we used:
- MSE
- MAE
- R2 score

For match outcome prediction (classification), we used:
- Accuracy
- Precision
- Recall
- F1-score

Interpretation:
- Lower MSE and MAE means better regression predictions
- Higher R2 means better explained variance
- Higher accuracy/F1 means better classification quality

Screenshot to add:
- Evaluation metric outputs from notebook

## 15. Data Visualization (Graphs/Charts)

The following charts were used:
- Histograms
- Boxplots
- Bar charts
- Scatter plots
- Heatmap
- Confusion matrix plot

Screenshot to add:
- Best 4-6 charts used in your analysis

## 16. Results and Observations

Important findings:
- Player goal prediction improved with multiple attacking features
- Match outcome prediction showed useful class-level accuracy
- Data cleaning improved model stability
- EDA helped identify key factors affecting outcomes

Screenshot to add:
- Final result summary table

## 17. Conclusion

This project successfully analyzed football data and built models for:
- Player goal prediction
- Match outcome prediction

The project proves that data analytics can support better sports decisions with clear insights and measurable model performance.

## 18. Future Scope

- Use larger and latest season datasets
- Add team form and opponent strength features
- Try advanced models like Random Forest, XGBoost
- Build a live dashboard for prediction
- Add explainable AI methods for model interpretation

## 19. References

1. Kaggle Football Dataset
2. Scikit-learn Documentation: https://scikit-learn.org/
3. Pandas Documentation: https://pandas.pydata.org/
4. Matplotlib Documentation: https://matplotlib.org/
5. Seaborn Documentation: https://seaborn.pydata.org/

---

### Note for Submission

After adding screenshots in the marked places, export this report to PDF or Word format as required by your college/institute.
