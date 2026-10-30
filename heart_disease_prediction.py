import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Set modern visual style
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Data Collection & Preprocessing
data_path = 'heart_disease_data.csv'
heart_data = pd.read_csv(data_path)

# 2. Splitting Features and Target
X = heart_data.drop(columns='target')
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# 3. Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

print("="*50)
print("🩺 HEART DISEASE PREDICTION SYSTEM 🩺")
print("="*50)
print("Please enter the following medical details:")

# 4. Interactive User Input Collection
try:
    age = float(input("Age (e.g., 62): "))
    sex = int(input("Sex (1 = male, 0 = female): "))
    cp = int(input("Chest Pain type [0-3]: "))
    trestbps = float(input("Resting Blood Pressure (e.g., 140): "))
    chol = float(input("Serum Cholesterol in mg/dl (e.g., 268): "))
    fbs = int(input("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false): "))
    restecg = int(input("Resting ECG results [0-2]: "))
    thalach = float(input("Maximum Heart Rate achieved (e.g., 160): "))
    exang = int(input("Exercise Induced Angina (1 = yes, 0 = no): "))
    oldpeak = float(input("ST depression induced by exercise (e.g., 3.6): "))
    slope = int(input("Slope of the peak exercise ST segment [0-2]: "))
    ca = int(input("Number of major vessels [0-3] colored by fluoroscopy: "))
    thal = int(input("Thal (1 = normal, 2 = fixed defect, 3 = reversible defect): "))

    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_array = np.asarray(input_data).reshape(1, -1)

    # 5. Prediction & Probability Calculation
    prediction = model.predict(input_array)
    probabilities = model.predict_proba(input_array)[0] # [prob_healthy, prob_disease]

    print("\n" + "="*30)
    print("📋 DIAGNOSTIC RESULTS")
    print("="*30)
    if prediction[0] == 0:
        print('Result: The person does not have Heart Disease (Healthy / Low Risk).')
    else:
        print('Result: The person has Heart Disease (High Risk).')

    print(f"Confidence (Healthy): {probabilities[0]*100:.1f}%")
    print(f"Confidence (Heart Disease): {probabilities[1]*100:.1f}%")

    # 6. Generate Clean Dashboard Visual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.2, 1]})

    # Donut Chart with External Percentage Labels
    sizes = [probabilities[0] * 100, probabilities[1] * 100]
    colors = ['#27ae60', '#e74c3c']
    labels = [f'Healthy / Low Risk\n({sizes[0]:.1f}%)', f'Heart Disease / High Risk\n({sizes[1]:.1f}%)']

    wedges, texts = ax1.pie(sizes, labels=labels, startangle=90, 
                            colors=colors, explode=(0.05, 0),
                            wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2))

    for text in texts:
        text.set_fontweight('bold')
        text.set_fontsize(10)

    ax1.set_title('Overall Patient Risk Assessment', fontsize=13, fontweight='bold', pad=10)

    # Vitals Comparison Bar Chart
    features_to_show = ['age', 'trestbps', 'chol', 'thalach']
    patient_values = [input_data[0], input_data[3], input_data[4], input_data[7]]
    feature_names = ['Age (yrs)', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate']
    dataset_means = [heart_data[col].mean() for col in features_to_show]

    y_pos = np.arange(len(feature_names))
    ax2.barh(y_pos - 0.2, patient_values, height=0.4, label='Patient', color='#34495e')
    ax2.barh(y_pos + 0.2, dataset_means, height=0.4, label='Dataset Average', color='#bdc3c7')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlabel('Value', fontsize=11, fontweight='bold')
    ax2.set_title('Patient Vitals vs. Dataset Average', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    plt.show()

except ValueError:
    print("❌ Invalid input! Please enter numbers only for the health parameters.")