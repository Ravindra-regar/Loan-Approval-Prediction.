# 1. IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Import the models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 2. LOAD THE DATASET
file_path = 'loan_train.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset 'loan_train.csv' loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script.")
    exit() 
# 3. DATA CLEANING AND PREPROCESSING
# Fill missing values for categorical columns with the mode
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Fill missing values for numerical columns with the median
for col in ['LoanAmount', 'Loan_Amount_Term']:
     if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values have been handled.")

# 4. FEATURE ENGINEERING
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount_Log'] = np.log(df['LoanAmount'] + 1)
df['Total_Income_Log'] = np.log(df['Total_Income'] + 1)
df = df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income', 'Loan_ID'])
print("\nFeature Engineering complete.")

# 5. ENCODING CATEGORICAL FEATURES

categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])
print("\nCategorical variables encoded.")

# 6. SPLIT DATA INTO TRAINING AND TESTING SETS

X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nData split complete.")

# 7. TRAIN MODELS AND STORE PERFORMANCE METRICS
# Create a dictionary to hold the models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# Create an empty list to store the results for comparison
results_list = []
# Loop through the models to train, evaluate, and store results
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Get performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # We focus on the 'Approved' class (labeled '1')
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1_score = report['1']['f1-score']
    
    # Append results to the list
    results_list.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    })
    
    # Print the report for detailed view
    print(f"--- Results for {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Not Approved (0)', 'Approved (1)']))


# 8. CREATE THE COMBINED COMPARISON GRAPH
# Convert the results list to a pandas DataFrame
results_df = pd.DataFrame(results_list)

# "Melt" the DataFrame to make it suitable for seaborn's grouped bar plot
results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

print("\nGenerating comparison graph...")

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y='Score', hue='Metric', data=results_melted, palette='viridis')

# Add score labels on top of each bar
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 9),
                textcoords = 'offset points')

plt.title('Comparison of Machine Learning Models', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.1) # Set y-axis limit to be slightly above 1.0
plt.xticks(rotation=0)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

# Display the results table
print("\n--- Performance Summary Table ---")
print(results_df)
