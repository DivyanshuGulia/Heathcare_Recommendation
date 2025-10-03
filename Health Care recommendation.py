#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


blood_data = pd.read_csv('blood_count_dataset.csv')
diet_data = pd.read_csv('Personalized_Diet_Recommendations.csv')


# In[3]:


print("Blood Count Dataset Info:")
print(blood_data.info())
print("\nBlood Count Dataset Shape:", blood_data.shape)
print("\nBlood Count Dataset - First 5 rows:")
print(blood_data.head())


# In[4]:


print("Diet Recommendations Dataset Info:")
print(diet_data.info())
print("\nDiet Recommendations Dataset Shape:", diet_data.shape)
print("\nDiet Recommendations Dataset - First 5 rows:")
print(diet_data.head())


# In[5]:


print("Blood Count Dataset - Descriptive Statistics:")
print(blood_data.describe())


# In[6]:


print("Diet Recommendations Dataset - Descriptive Statistics:")
print(diet_data.describe())


# In[7]:


print("Blood Count Dataset - Missing Values:")
print(blood_data.isnull().sum())

print("\nDiet Recommendations Dataset - Missing Values:")
print(diet_data.isnull().sum())


# In[8]:


print("Blood Count Dataset - Unique values in categorical columns:")
for col in blood_data.select_dtypes(include=['object']).columns:
    print(f"{col}: {blood_data[col].unique()}")

print("\nDiet Recommendations Dataset - Unique values in categorical columns:")
for col in diet_data.select_dtypes(include=['object']).columns:
    print(f"{col}: {diet_data[col].value_counts().head()}")


# In[9]:


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

df = pd.read_csv('blood_count_dataset.csv')


# In[10]:


fig = px.scatter(df, 
                x='Age', 
                y='Hemoglobin', 
                color='Gender',
                title='Age vs Hemoglobin by Gender',
                labels={'Age': 'Age (years)', 'Hemoglobin': 'Hemoglobin (g/dL)'},
                color_discrete_map={'Male': '#1FB8CD', 'Female': '#DB4545'})


# In[11]:


fig.update_traces(
    cliponaxis=False,
    hovertemplate='<b>%{hovertext}</b><br>' +
                  'Age: %{x} years<br>' +
                  'Hemoglobin: %{y} g/dL<br>' +
                  '<extra></extra>',
    hovertext=df['Gender']
)


# In[12]:


fig.update_layout(
    xaxis_title='Age (years)',
    yaxis_title='Hemoglobin (g/dL)',
    legend=dict(
        orientation='h', 
        yanchor='bottom', 
        y=1.05, 
        xanchor='center', 
        x=0.5
    )
)


# In[13]:


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

df = pd.read_csv('Personalized_Diet_Recommendations.csv')

df_clean = df.dropna(subset=['Exercise_Frequency', 'BMI', 'Gender'])

color_map = {'Male': '#1FB8CD', 'Female': '#DB4545', 'Other': '#2E8B57'}

fig = go.Figure()


# In[14]:


for gender in df_clean['Gender'].unique():
    gender_data = df_clean[df_clean['Gender'] == gender]
    
    fig.add_trace(go.Scatter(
        x=gender_data['Exercise_Frequency'],
        y=gender_data['BMI'],
        mode='markers',
        name=gender,
        marker=dict(
            color=color_map.get(gender, '#5D878F'),
            size=6,
            opacity=0.7
        ),
        hovertemplate='Exercise: %{x}/week<br>BMI: %{y:.1f}<br>Gender: ' + gender + '<extra></extra>'
    ))


# In[15]:


fig.add_trace(go.Scatter(
    x=df_clean['Exercise_Frequency'],
    y=np.poly1d(np.polyfit(df_clean['Exercise_Frequency'], df_clean['BMI'], 1))(df_clean['Exercise_Frequency']),
    mode='lines',
    name='Trend',
    line=dict(color='#D2BA4C', width=2, dash='dash'),
    hoverinfo='skip'
))


# In[16]:


fig.update_layout(
    title='Exercise vs BMI by Gender',
    xaxis_title='Exercise/Week',
    yaxis_title='BMI',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)


# In[17]:


fig.update_traces(cliponaxis=False)


# In[18]:


print(f"Data points plotted: {len(df_clean)}")
print(f"Exercise frequency range: {df_clean['Exercise_Frequency'].min()} - {df_clean['Exercise_Frequency'].max()}")
print(f"BMI range: {df_clean['BMI'].min():.1f} - {df_clean['BMI'].max():.1f}")


# In[19]:


diet_processed = diet_data.copy()


# In[20]:


diet_processed['Chronic_Disease'].fillna('None', inplace=True)
diet_processed['Allergies'].fillna('None', inplace=True)
diet_processed['Food_Aversions'].fillna('None', inplace=True)


# In[21]:


print("Target variable distribution:")
print(diet_processed['Recommended_Meal_Plan'].value_counts())


# In[22]:


feature_cols = [
    'Age', 'Height_cm', 'Weight_kg', 'BMI',
    'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 
    'Cholesterol_Level', 'Blood_Sugar_Level',
    'Daily_Steps', 'Exercise_Frequency', 'Sleep_Hours',
    'Caloric_Intake', 'Protein_Intake', 'Carbohydrate_Intake', 'Fat_Intake',
    'Gender', 'Chronic_Disease', 'Genetic_Risk_Factor', 
    'Alcohol_Consumption', 'Smoking_Habit', 'Dietary_Habits',
    'Preferred_Cuisine', 'Allergies', 'Food_Aversions'
]


# In[23]:


X = diet_processed[feature_cols].copy()
y = diet_processed['Recommended_Meal_Plan']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

print(f"\nCategorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")


# In[24]:


# Label encoding for categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\nData preprocessing completed!")
print("Sample of processed features:")
print(X.head())


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")


# In[26]:


models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}


# In[27]:


model_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


# In[28]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')


# In[29]:


diet_data = pd.read_csv('Personalized_Diet_Recommendations.csv')


# In[30]:


print("Original Dataset Info:")
print(f"Shape: {diet_data.shape}")
print(f"Target distribution:\n{diet_data['Recommended_Meal_Plan'].value_counts()}")
print(f"Target distribution (%):\n{diet_data['Recommended_Meal_Plan'].value_counts(normalize=True) * 100}")


# In[31]:


def create_health_risk_target(data):
    """Create a more predictive target based on health risks"""
    risk_score = 0
    
    bmi_risk = pd.cut(data['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[1, 2, 3, 4])
    
    bp_risk = np.where((data['Blood_Pressure_Systolic'] >= 140) | (data['Blood_Pressure_Diastolic'] >= 90), 3,
                      np.where((data['Blood_Pressure_Systolic'] >= 130) | (data['Blood_Pressure_Diastolic'] >= 80), 2, 1))
    
    chol_risk = np.where(data['Cholesterol_Level'] >= 240, 3,
                        np.where(data['Cholesterol_Level'] >= 200, 2, 1))

    sugar_risk = np.where(data['Blood_Sugar_Level'] >= 200, 3,
                         np.where(data['Blood_Sugar_Level'] >= 140, 2, 1))

    total_risk = pd.to_numeric(bmi_risk, errors='coerce') + bp_risk + chol_risk + sugar_risk

    risk_categories = pd.cut(total_risk, bins=[3, 6, 9, 12, 16], 
                            labels=['Low_Risk', 'Medium_Risk', 'High_Risk', 'Critical_Risk'])
    
    return risk_categories

diet_data['Health_Risk'] = create_health_risk_target(diet_data)
diet_data['Health_Risk'] = diet_data['Health_Risk'].fillna('Medium_Risk') 

print(f"\nNew Health Risk Target Distribution:")
print(diet_data['Health_Risk'].value_counts())
print(f"\nNew Health Risk Target Distribution (%):")
print(diet_data['Health_Risk'].value_counts(normalize=True) * 100)


# In[32]:


def create_binary_health_target(data):
    """Create a binary target: Needs Intervention vs Healthy"""
    interventions_needed = []
    
    for idx, row in data.iterrows():
        needs_intervention = False

        if (row['BMI'] >= 30 or  # Obese
            row['Blood_Pressure_Systolic'] >= 140 or row['Blood_Pressure_Diastolic'] >= 90 or  # Hypertension
            row['Cholesterol_Level'] >= 240 or  # High cholesterol
            row['Blood_Sugar_Level'] >= 200 or  # Diabetes range
            row['Exercise_Frequency'] == 0 or  # Sedentary
            (row['Smoking_Habit'] == 'Yes' and row['Alcohol_Consumption'] == 'Yes')):  # Multiple bad habits
            needs_intervention = True
            
        interventions_needed.append('Needs_Intervention' if needs_intervention else 'Healthy')
    
    return interventions_needed


# In[33]:


diet_data['Intervention_Needed'] = create_binary_health_target(diet_data)

print("Binary Classification Target Distribution:")
print(diet_data['Intervention_Needed'].value_counts())
print(f"\nBinary Target Distribution (%):")
print(diet_data['Intervention_Needed'].value_counts(normalize=True) * 100)


# In[34]:


# Create engineered features
def engineer_features(data):
    """Create meaningful engineered features"""
    data_eng = data.copy()
    
    data_eng['Health_Score'] = (
        (data_eng['BMI'] - 25).abs() +  # Distance from ideal BMI
        (data_eng['Blood_Pressure_Systolic'] - 120).abs() +  # Distance from ideal systolic
        (data_eng['Blood_Pressure_Diastolic'] - 80).abs() +  # Distance from ideal diastolic
        (data_eng['Cholesterol_Level'] - 200).abs() +  # Distance from ideal cholesterol
        (data_eng['Blood_Sugar_Level'] - 100).abs()  # Distance from ideal blood sugar
    )
    data_eng['Lifestyle_Score'] = (
        data_eng['Exercise_Frequency'] * 2 +
        data_eng['Sleep_Hours'] +
        data_eng['Daily_Steps'] / 1000 -
        (data_eng['Smoking_Habit'] == 'Yes').astype(int) * 3 -
        (data_eng['Alcohol_Consumption'] == 'Yes').astype(int) * 2
    )
    total_calories = data_eng['Caloric_Intake']
    data_eng['Protein_Ratio'] = (data_eng['Protein_Intake'] * 4) / total_calories
    data_eng['Carb_Ratio'] = (data_eng['Carbohydrate_Intake'] * 4) / total_calories  
    data_eng['Fat_Ratio'] = (data_eng['Fat_Intake'] * 9) / total_calories

    data_eng['Age_Group'] = pd.cut(data_eng['Age'], bins=[0, 30, 45, 60, 100], 
                                  labels=['Young', 'Middle', 'Senior', 'Elderly'])

    data_eng['BMI_Category'] = pd.cut(data_eng['BMI'], 
                                     bins=[0, 18.5, 25, 30, 100],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    data_eng['Activity_Level'] = pd.cut(data_eng['Exercise_Frequency'],
                                       bins=[-1, 1, 3, 5, 7],
                                       labels=['Sedentary', 'Low', 'Moderate', 'High'])

    data_eng['Risk_Factors'] = (
        (data_eng['BMI'] >= 30).astype(int) +
        (data_eng['Blood_Pressure_Systolic'] >= 140).astype(int) +
        (data_eng['Cholesterol_Level'] >= 240).astype(int) +
        (data_eng['Blood_Sugar_Level'] >= 200).astype(int) +
        (data_eng['Exercise_Frequency'] <= 1).astype(int) +
        (data_eng['Smoking_Habit'] == 'Yes').astype(int)
    )
    
    return data_eng


# In[35]:


# Apply feature engineering
data_engineered = engineer_features(diet_data)

print(f"Original features: {diet_data.shape[1]}")
print(f"After feature engineering: {data_engineered.shape[1]}")
print("New engineered features:")
new_features = set(data_engineered.columns) - set(diet_data.columns)
for feature in new_features:
    print(f"  - {feature}")

# Select relevant features for modeling
feature_cols = [
    # Demographic
    'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI',
    
    # Health metrics
    'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 
    'Cholesterol_Level', 'Blood_Sugar_Level',
    
    # Lifestyle
    'Daily_Steps', 'Exercise_Frequency', 'Sleep_Hours',
    'Alcohol_Consumption', 'Smoking_Habit',
    
    # Nutrition
    'Caloric_Intake', 'Protein_Intake', 'Carbohydrate_Intake', 'Fat_Intake',
    'Dietary_Habits',
    
    # Engineered features
    'Health_Score', 'Lifestyle_Score', 'Protein_Ratio', 'Carb_Ratio', 'Fat_Ratio',
    'Age_Group', 'BMI_Category', 'Activity_Level', 'Risk_Factors'
]

print(f"\nSelected {len(feature_cols)} features for modeling")


# In[36]:


def create_optimized_target(data):
    """Create a well-balanced target for better prediction accuracy"""
    targets = []
    
    for idx, row in data.iterrows():
        # Focus on cardiovascular health risk - more balanced and predictable
        cv_risk = 0
        
        # Major CV risk factors with different weights
        if row['BMI'] >= 30:
            cv_risk += 3  # Obesity
        elif row['BMI'] >= 25:
            cv_risk += 2  # Overweight
            
        if row['Blood_Pressure_Systolic'] >= 140:
            cv_risk += 3  # Stage 2 Hypertension
        elif row['Blood_Pressure_Systolic'] >= 130:
            cv_risk += 2  # Stage 1 Hypertension
        elif row['Blood_Pressure_Systolic'] >= 120:
            cv_risk += 1  # Elevated
            
        if row['Cholesterol_Level'] >= 240:
            cv_risk += 3  # High
        elif row['Cholesterol_Level'] >= 200:
            cv_risk += 2  # Borderline high
            
        if row['Exercise_Frequency'] <= 1:
            cv_risk += 2  # Sedentary
        elif row['Exercise_Frequency'] <= 2:
            cv_risk += 1  # Low activity
            
        if row['Smoking_Habit'] == 'Yes':
            cv_risk += 3  # Major risk factor
            
        if row['Age'] >= 60:
            cv_risk += 2
        elif row['Age'] >= 45:
            cv_risk += 1
            
        # Create balanced categories
        if cv_risk >= 8:
            targets.append('Very_High_Risk')
        elif cv_risk >= 5:
            targets.append('High_Risk')
        elif cv_risk >= 3:
            targets.append('Moderate_Risk')
        else:
            targets.append('Low_Risk')
    
    return targets

data_engineered['CV_Risk'] = create_optimized_target(data_engineered)

print("Cardiovascular Risk Target Distribution:")
print(data_engineered['CV_Risk'].value_counts())
print(f"\nCV Risk Distribution (%):")
print(data_engineered['CV_Risk'].value_counts(normalize=True) * 100)

selected_features = [
    # Core health metrics
    'Age', 'BMI', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic',
    'Cholesterol_Level', 'Blood_Sugar_Level',
    
    # Lifestyle factors
    'Exercise_Frequency', 'Daily_Steps', 'Sleep_Hours',
    
    # Nutrition
    'Caloric_Intake', 'Protein_Intake', 'Carbohydrate_Intake', 'Fat_Intake',
    
    # Engineered features
    'Health_Score', 'Lifestyle_Score', 'Risk_Factors'
]

categorical_features = ['Gender', 'Smoking_Habit', 'Alcohol_Consumption', 'Dietary_Habits']

X = data_engineered[selected_features].copy()

for cat_col in categorical_features:
    dummies = pd.get_dummies(data_engineered[cat_col], prefix=cat_col)
    X = pd.concat([X, dummies], axis=1)

y = data_engineered['CV_Risk']

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Feature engineering and preprocessing completed!")
print(f"Total features: {X_scaled.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training target distribution:")
print(y_train.value_counts())


# In[37]:


from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV


# In[38]:


models = {
    'Optimized_Random_Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    
    'Optimized_Gradient_Boosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    ),
    
    'Optimized_Extra_Trees': ExtraTreesClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    
    'Optimized_Logistic_Regression': LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    ),
    
    'Optimized_SVM': SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    ),
    
    'AdaBoost_Classifier': AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    ),
    
    'Bagging_Classifier': BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
}

# Train and evaluate models
results = {}
best_accuracy = 0
best_model_name = ""


# In[39]:


for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name


# In[40]:


print(f"BEST MODEL: {best_model_name}")
print(f"BEST ACCURACY: {best_accuracy:.4f}")


# In[41]:


print(f"\nDETAILED RESULTS COMPARISON:")
print(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'CV-Mean':<10} {'CV-Std':<10}")

for name, result in results.items():
    print(f"{name:<25} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
          f"{result['cv_mean']:<10.4f} {result['cv_std']:<10.4f}")

# Classification report for best model
print(f"\nCLASSIFICATION REPORT FOR BEST MODEL ({best_model_name}):")
print(classification_report(y_test, results[best_model_name]['predictions']))


# In[42]:


from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

print("TRAINING OPTIMIZED MACHINE LEARNING MODELS")


# In[43]:


models = {
    'Optimized_Random_Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    
    'Optimized_Gradient_Boosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    ),
    
    'Optimized_Extra_Trees': ExtraTreesClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    
    'Optimized_Logistic_Regression': LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    ),
    
    'Optimized_SVM': SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    ),
    
    'AdaBoost_Classifier': AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    ),
    
    'Bagging_Classifier': BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ),
    
    'KNN_Classifier': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        p=2
    )
}

# Train and evaluate models
results = {}
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name


# In[44]:


print(f"BEST MODEL: {best_model_name}")
print(f"BEST ACCURACY: {best_accuracy:.4f}")

print(f"\nDETAILED RESULTS COMPARISON:")
print(f"{'Model':<30} {'Accuracy':<10} {'F1-Score':<10} {'CV-Mean':<10} {'CV-Std':<10}")

for name, result in results.items():
    print(f"{name:<30} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
          f"{result['cv_mean']:<10.4f} {result['cv_std']:<10.4f}")

print(f"\nCLASSIFICATION REPORT FOR BEST MODEL ({best_model_name}):")
print(classification_report(y_test, results[best_model_name]['predictions']))


# In[45]:


print("TRAINING OPTIMIZED MACHINE LEARNING MODELS - FAST VERSION")


# In[46]:


models = {
    'Optimized_Random_Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    ),
    
    'Optimized_Gradient_Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
    
    'Optimized_Logistic_Regression': LogisticRegression(
        C=1.0,
        penalty='l2',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    ),
    
    'Extra_Trees_Classifier': ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}


# In[47]:


results = {}
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name


# In[48]:


print(f"BEST MODEL: {best_model_name}")
print(f"BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")

print(f"\nDETAILED RESULTS COMPARISON:")
print(f"{'Model':<30} {'Accuracy':<12} {'F1-Score':<12} {'CV-Mean':<12}")
print("-" * 70)

for name, result in results.items():
    print(f"{name:<30} {result['accuracy']:<12.4f} {result['f1_score']:<12.4f} "
          f"{result['cv_mean']:<12.4f}")


# In[49]:


print(f"\nCLASSIFICATION REPORT FOR BEST MODEL ({best_model_name}):")
print(classification_report(y_test, results[best_model_name]['predictions']))


# In[50]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib


# In[51]:


diet_df = pd.read_csv('Personalized_Diet_Recommendations.csv')


# In[52]:


def create_risk_category(data):
    """
    Create risk categories (Low, Moderate, High, Very_High risk) based on health indicators.
    """
    targets = []
    for _, row in data.iterrows():
        cv_risk = 0
        # BMI contribution
        if row['BMI'] >= 30:
            cv_risk += 3  # Obese
        elif row['BMI'] >= 25:
            cv_risk += 2  # Overweight
        # Blood pressure (systolic) contribution
        if row['Blood_Pressure_Systolic'] >= 140:
            cv_risk += 3  # Stage 2 Hypertension
        elif row['Blood_Pressure_Systolic'] >= 130:
            cv_risk += 2  # Stage 1 Hypertension
        elif row['Blood_Pressure_Systolic'] >= 120:
            cv_risk += 1  # Elevated BP
        # Cholesterol level contribution
        if row['Cholesterol_Level'] >= 240:
            cv_risk += 3  # High cholesterol
        elif row['Cholesterol_Level'] >= 200:
            cv_risk += 2  # Borderline high cholesterol
        # Exercise frequency contribution
        if row['Exercise_Frequency'] <= 1:
            cv_risk += 2  # Sedentary lifestyle
        elif row['Exercise_Frequency'] <= 2:
            cv_risk += 1  # Low activity
        # Smoking habit contribution
        if str(row['Smoking_Habit']).strip().lower() == 'yes':
            cv_risk += 3  # Smoker is a major risk factor
        # Age contribution
        if row['Age'] >= 60:
            cv_risk += 2
        elif row['Age'] >= 45:
            cv_risk += 1
        # Assign risk category based on total score
        if cv_risk >= 8:
            targets.append('Very_High_Risk')
        elif cv_risk >= 5:
            targets.append('High_Risk')
        elif cv_risk >= 3:
            targets.append('Moderate_Risk')
        else:
            targets.append('Low_Risk')
    return targets


# In[53]:


diet_df['Risk_Level'] = create_risk_category(diet_df)


# In[54]:


feature_cols = [
    'Age', 'BMI', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic',
    'Cholesterol_Level', 'Blood_Sugar_Level',
    'Exercise_Frequency', 'Daily_Steps', 'Sleep_Hours',
    'Caloric_Intake', 'Protein_Intake', 'Carbohydrate_Intake', 'Fat_Intake',
    'Gender', 'Smoking_Habit', 'Alcohol_Consumption', 'Dietary_Habits'
]


# In[55]:


X = diet_df[feature_cols].copy()
y = diet_df['Risk_Level']

numeric_cols = ['Age', 'BMI', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic',
                'Cholesterol_Level', 'Blood_Sugar_Level',
                'Exercise_Frequency', 'Daily_Steps', 'Sleep_Hours',
                'Caloric_Intake', 'Protein_Intake', 'Carbohydrate_Intake', 'Fat_Intake']
cat_cols = ['Gender', 'Smoking_Habit', 'Alcohol_Consumption', 'Dietary_Habits']

X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = X[col].fillna('Unknown')
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=8,
    min_samples_split=10, min_samples_leaf=4, subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained. Test Accuracy = {accuracy:.2%}")

joblib.dump(model, 'risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')


# In[57]:


from flask import Flask, render_template_string, request
import joblib, io, base64
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

model = joblib.load('risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')


# In[58]:


form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Health Risk Prediction</title>
  <!-- Bootstrap CSS for styling -->
  <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { padding-top: 20px; }
    .result-table td, .result-table th { padding: 8px 12px; }
    .risk-high { color: #d9534f; font-weight: bold; }       /* red for high risk */
    .risk-moderate { color: #f0ad4e; font-weight: bold; }   /* orange for moderate risk */
    .risk-low { color: #5cb85c; font-weight: bold; }        /* green for low risk */
  </style>
</head>
<body>
<div class="container">
  <h1 class="mb-4">Health Risk Prediction App</h1>
  <form method="POST" action="/predict">
    <div class="row">
      <div class="col-md-6">
        <!-- Personal and Vital Stats -->
        <div class="form-group">
          <label for="age">Age</label>
          <input type="number" step="1" class="form-control" name="Age" id="age" required>
        </div>
        <div class="form-group">
          <label for="gender">Gender</label>
          <select class="form-control" name="Gender" id="gender" required>
            <option value="">--Select--</option>
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>
        </div>
        <div class="form-group">
          <label for="height">Height (cm)</label>
          <input type="number" step="0.1" class="form-control" name="Height_cm" id="height" required>
        </div>
        <div class="form-group">
          <label for="weight">Weight (kg)</label>
          <input type="number" step="0.1" class="form-control" name="Weight_kg" id="weight" required>
        </div>
        <div class="form-group">
          <label for="bp_sys">Blood Pressure Systolic (mmHg)</label>
          <input type="number" step="1" class="form-control" name="Blood_Pressure_Systolic" id="bp_sys" required>
        </div>
        <div class="form-group">
          <label for="bp_dia">Blood Pressure Diastolic (mmHg)</label>
          <input type="number" step="1" class="form-control" name="Blood_Pressure_Diastolic" id="bp_dia" required>
        </div>
        <div class="form-group">
          <label for="chol">Cholesterol Level (mg/dL)</label>
          <input type="number" step="1" class="form-control" name="Cholesterol_Level" id="chol" required>
        </div>
        <div class="form-group">
          <label for="sugar">Blood Sugar Level (mg/dL)</label>
          <input type="number" step="1" class="form-control" name="Blood_Sugar_Level" id="sugar" required>
        </div>
      </div>
      <div class="col-md-6">
        <!-- Lifestyle and Diet Inputs -->
        <div class="form-group">
          <label for="exercise">Exercise Frequency (days/week)</label>
          <input type="number" step="1" class="form-control" name="Exercise_Frequency" id="exercise" required>
        </div>
        <div class="form-group">
          <label for="steps">Daily Steps</label>
          <input type="number" step="1" class="form-control" name="Daily_Steps" id="steps" required>
        </div>
        <div class="form-group">
          <label for="sleep">Sleep Hours (per day)</label>
          <input type="number" step="0.1" class="form-control" name="Sleep_Hours" id="sleep" required>
        </div>
        <div class="form-group">
          <label for="calories">Daily Caloric Intake (kcal)</label>
          <input type="number" step="1" class="form-control" name="Caloric_Intake" id="calories" required>
        </div>
        <div class="form-group">
          <label for="protein">Daily Protein Intake (grams)</label>
          <input type="number" step="1" class="form-control" name="Protein_Intake" id="protein" required>
        </div>
        <div class="form-group">
          <label for="carbs">Daily Carbohydrate Intake (grams)</label>
          <input type="number" step="1" class="form-control" name="Carbohydrate_Intake" id="carbs" required>
        </div>
        <div class="form-group">
          <label for="fat">Daily Fat Intake (grams)</label>
          <input type="number" step="1" class="form-control" name="Fat_Intake" id="fat" required>
        </div>
        <div class="form-group">
          <label for="smoking">Smoking Habit</label>
          <select class="form-control" name="Smoking_Habit" id="smoking" required>
            <option value="">--Select--</option>
            <option>No</option>
            <option>Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label for="alcohol">Alcohol Consumption</label>
          <select class="form-control" name="Alcohol_Consumption" id="alcohol" required>
            <option value="">--Select--</option>
            <option>No</option>
            <option>Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label for="dietary">Dietary Habits</label>
          <select class="form-control" name="Dietary_Habits" id="dietary" required>
            <option value="">--Select--</option>
            <option>Vegetarian</option>
            <option>Non-Vegetarian</option>
            <option>Vegan</option>
            <option>Other</option>
          </select>
        </div>
      </div>
    </div>
    <button type="submit" class="btn btn-primary btn-lg btn-block">Predict Health Risk</button>
  </form>
</div>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Health Risk Results</title>
  <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { padding-top: 20px; }
    .risk-high { color: #d9534f; font-weight: bold; }
    .risk-moderate { color: #f0ad4e; font-weight: bold; }
    .risk-low { color: #5cb85c; font-weight: bold; }
  </style>
</head>
<body>
<div class="container">
  <h1 class="mb-4">Predicted Health Risk Outcome</h1>
  <h2 class="mb-3">Risk Level: 
    <span class="{{ risk_level_class }}">{{ risk_level_text }}</span>
  </h2>
  <div class="mb-4">
    <img src="data:image/png;base64,{{ plot_data }}" alt="Health metrics chart" class="img-fluid" />
  </div>
  <h4>Personalized Recommendations:</h4>
  <ul>
    {% for rec in recommendations %}
      <li>{{ rec }}</li>
    {% endfor %}
  </ul>
  <hr>
  <p><em>{{ recovery_note }}</em></p>
  <a href="/" class="btn btn-secondary mt-3">&#8592; Back to Input Form</a>
</div>
</body>
</html>
"""


# In[59]:


@app.route('/', methods=['GET'])
def form():
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict():
    # 2.a. Collect input values from form
    input_data = {}
    for field in ['Age','Gender','Height_cm','Weight_kg','Blood_Pressure_Systolic','Blood_Pressure_Diastolic',
                  'Cholesterol_Level','Blood_Sugar_Level','Exercise_Frequency','Daily_Steps','Sleep_Hours',
                  'Caloric_Intake','Protein_Intake','Carbohydrate_Intake','Fat_Intake',
                  'Smoking_Habit','Alcohol_Consumption','Dietary_Habits']:
        input_data[field] = request.form.get(field)
    # Calculate BMI from height and weight for consistency
    try:
        height_m = float(input_data['Height_cm']) / 100.0
        weight = float(input_data['Weight_kg'])
        bmi = weight / (height_m**2) if height_m > 0 else 0
    except:
        bmi = 0
    # Use the calculated BMI instead of any provided (if user had input BMI directly, we are computing it)
    input_data['BMI'] = bmi

    # 2.b. Prepare the input for model (apply encoding and scaling as done in training)
    user_df = pd.DataFrame([input_data])
    numeric_cols = ['Age','BMI','Blood_Pressure_Systolic','Blood_Pressure_Diastolic',
                    'Cholesterol_Level','Blood_Sugar_Level',
                    'Exercise_Frequency','Daily_Steps','Sleep_Hours',
                    'Caloric_Intake','Protein_Intake','Carbohydrate_Intake','Fat_Intake']
    for col in numeric_cols:
        user_df[col] = pd.to_numeric(user_df[col], errors='coerce').fillna(0.0)
    for col, le in label_encoders.items():
        if user_df[col].iloc[0] not in le.classes_:
            le.classes_ = np.append(le.classes_, user_df[col].iloc[0])
        user_df[col] = le.transform(user_df[col])
    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

    model_features = ['Age','BMI','Blood_Pressure_Systolic','Blood_Pressure_Diastolic',
                      'Cholesterol_Level','Blood_Sugar_Level','Exercise_Frequency','Daily_Steps','Sleep_Hours',
                      'Caloric_Intake','Protein_Intake','Carbohydrate_Intake','Fat_Intake',
                      'Gender','Smoking_Habit','Alcohol_Consumption','Dietary_Habits']
    X_user = user_df[model_features].values

    # 2.c. Get model prediction
    risk_pred = model.predict(X_user)[0]            # predicted risk category (e.g., "High_Risk")
    risk_proba = model.predict_proba(X_user)[0]     # probability distribution for each class (if needed)

    if 'Very_High' in risk_pred or 'High_Risk' == risk_pred:
        risk_class = "risk-high"
    elif 'Moderate' in risk_pred:
        risk_class = "risk-moderate"
    else:
        risk_class = "risk-low"
    display_text = risk_pred.replace("_", " ")  

    # 2.d. Generate a chart comparing user's metrics vs healthy benchmarks
    user_vals = [
        float(input_data['BMI']),
        float(input_data['Blood_Pressure_Systolic']),
        float(input_data['Cholesterol_Level']),
        float(input_data['Blood_Sugar_Level'])
    ]

    healthy_vals = [24.9, 120, 200, 100]  # BMI <25, BP <120, Cholesterol <200, Blood Sugar <100 (approx for fasting)
    metrics = ['BMI','Systolic BP','Cholesterol','Blood Sugar']

    plt.figure(figsize=(6,4))
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, healthy_vals, width, label='Healthy Max', color='#5cb85c')
    plt.bar(x + width/2, user_vals, width, label='Your Value', color='#337ab7')
    plt.xticks(x, metrics)
    plt.ylabel('Value')
    plt.title('Your Metrics vs Healthy Ranges')
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()  # close figure to free memory

    # 2.e. Prepare personalized recommendations based on the predicted risk and metrics
    recommendations = []
    if risk_pred in ['Very_High_Risk', 'High_Risk']:
        # High risk: strict diet control
        recommendations.append("Follow a strict healthy diet (e.g., a low-fat, low-sugar plan) to improve your metrics.")
        recommendations.append("Increase physical activity gradually (aim for at least 30 minutes/day).")
        recommendations.append("Regularly monitor blood pressure, blood sugar, and cholesterol levels.")
    elif risk_pred == 'Moderate_Risk':
        recommendations.append("Adopt a balanced diet with controlled portions to address moderate risk factors.")
        recommendations.append("Exercise regularly (at least 3-4 days a week) to improve fitness.")
        recommendations.append("Monitor key health metrics (weight, BP, cholesterol) to track improvements.")
    else:  # Low_Risk
        recommendations.append("Maintain a balanced, nutritious diet to keep your risk low.")
        recommendations.append("Continue regular exercise and healthy lifestyle habits.")
        recommendations.append("Schedule routine health check-ups to ensure metrics remain in normal range.")

    if float(input_data['BMI']) >= 25:
        recommendations.append("Work on weight management: consider a high-protein, lower-carb diet to reach a healthier BMI.")
    if float(input_data['Cholesterol_Level']) >= 200:
        recommendations.append("Reduce intake of saturated fats and cholesterol (focus on fruits, veggies, lean proteins).")
    if float(input_data['Blood_Pressure_Systolic']) >= 130:
        recommendations.append("Cut down on salt and processed foods to help control blood pressure.")
    if float(input_data['Blood_Sugar_Level']) >= 126:  # 126+ mg/dL could indicate diabetes risk
        recommendations.append("Limit sugary foods/beverages and refined carbs to help control blood sugar levels.")
    if str(request.form.get('Smoking_Habit')).lower() == 'yes':
        recommendations.append("Quit smoking to significantly improve cardiovascular health and reduce risk.")
    if str(request.form.get('Alcohol_Consumption')).lower() == 'yes':
        recommendations.append("Moderate your alcohol consumption in line with medical guidance to improve health.")

    if risk_pred == 'Very_High_Risk':
        recovery_note = "With strict adherence to the recommendations, you could potentially reduce your risk level in 6-12 months."
    elif risk_pred == 'High_Risk':
        recovery_note = "By following the advice, you may improve to a moderate risk level within a few months."
    elif risk_pred == 'Moderate_Risk':
        recovery_note = "By making the suggested changes, you can likely achieve a low risk status over time."
    else:
        recovery_note = "Continue your healthy lifestyle to maintain your low risk status long-term."

    # 2.g. Render the result template with dynamic values
    return render_template_string(result_html,
                                  risk_level_text=display_text,
                                  risk_level_class=risk_class,
                                  plot_data=plot_data,
                                  recommendations=recommendations,
                                  recovery_note=recovery_note)
    


# In[ ]:


if __name__ == "__main__":
    # Use port=5000 (default) and disable reloader to avoid issues in interactive env
    app.run(port=5000, debug=False, use_reloader=False)


# In[ ]:




