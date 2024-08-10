# Employee Attrition Prediction Model

## Description

The Employee Attrition Prediction model evaluates the risk of employee attrition within an organization. It addresses factors such as personal events leading to resignation, career changes, and lack of growth opportunities. By analyzing various factors that influence attrition, we aim to understand how different company-specific variables affect employee turnover and identify working environments that are more likely to cause attrition. This is achieved using various machine learning algorithms with Python libraries and data visualization through Tableau.

## Tools and Technologies

- **Machine Learning Algorithms**: Logistic Regression, Random Forest Classifier, Naive Bayes, Decision Tree Classifier
- **Python Libraries**: PySpark, pandas, numpy, seaborn, matplotlib
- **Data Visualization**: Tableau

## Dataset

- **Source**: [https://data.world/aaizemberg/hr-employee-attrition]
- **Description**: The dataset includes features related to employee demographics, job roles, performance metrics, and other relevant factors.

## How It Works

1. **Data Preparation**:
   - Load and preprocess the dataset using PySpark.
   - Convert categorical variables to numeric format using `StringIndexer`.
   - Perform exploratory data analysis (EDA) and visualization using `seaborn` and `matplotlib`.

2. **Exploratory Data Analysis (EDA)**:
   - Analyze the distribution of attrition and other features.
   - Visualize distributions and relationships using histograms, bar charts, pie charts, and boxplots.

3. **Data Balancing**:
   - Balance the dataset using oversampling of the minority class and undersampling of the majority class to address class imbalance.

4. **Model Building**:
   - Train and evaluate various machine learning models including Logistic Regression, Random Forest Classifier, Naive Bayes, and Decision Tree Classifier.
   - Use `VectorAssembler` to prepare data for model training and make predictions.

5. **Model Evaluation**:
   - Evaluate model performance using metrics such as accuracy and area under the ROC curve (AUC).
   - Compare performance across different models and data balancing techniques.

## Code

### Installation

To get started, install the required Python libraries:

```bash
pip install pyspark pandas numpy seaborn matplotlib
