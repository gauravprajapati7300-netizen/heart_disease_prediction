# Heart Disease Prediction System 🩺

An end-to-end Machine Learning Python application designed to predict the likelihood of heart disease based on patient diagnostic parameters. The system features an interactive command-line interface for manual patient data entry and generates an intuitive, clean visual dashboard for immediate risk assessment.

---

## 📌 Project Overview

Heart disease is one of the leading causes of mortality worldwide. Early and accurate detection plays a crucial role in preventative healthcare. 

This project trains a **Logistic Regression** model on clinical data to assess patient risk. Users can input their health metrics directly via the terminal and instantly receive:
1. **Diagnostic Classification**: Categorization into Low Risk (Healthy) or High Risk (Heart Disease).
2. **Confidence Scores**: Probabilistic confidence scores for the assessment.
3. **Visual Health Dashboard**: A dual-panel figure displaying an overall risk donut chart alongside a comparison of patient vitals against dataset averages.

---

## 📊 Dataset Information

The model utilizes the `heart_disease_data.csv` dataset, which contains **303 medical observations** across **13 health attributes** and 1 target label:

| Attribute | Parameter Name | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| 1 | `age` | Continuous (years) | Age of the patient |
| 2 | `sex` | `1` = Male, `0` = Female | Gender |
| 3 | `cp` | `0` - `3` | Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic) |
| 4 | `trestbps` | Continuous (mm Hg) | Resting Blood Pressure |
| 5 | `chol` | Continuous (mg/dl) | Serum Cholesterol |
| 6 | `fbs` | `1` = True, `0` = False | Fasting Blood Sugar > 120 mg/dl |
| 7 | `restecg` | `0` - `2` | Resting Electrocardiographic Results |
| 8 | `thalach` | Continuous (bpm) | Maximum Heart Rate Achieved |
| 9 | `exang` | `1` = Yes, `0` = No | Exercise Induced Angina |
| 10 | `oldpeak` | Continuous | ST Depression Induced by Exercise Relative to Rest |
| 11 | `slope` | `0` - `2` | Slope of Peak Exercise ST Segment |
| 12 | `ca` | `0` - `3` | Number of Major Vessels Colored by Fluoroscopy |
| 13 | `thal` | `1` = Normal, `2` = Fixed Defect, `3` = Reversible Defect | Thalassemia Status |
| **Target** | `target` | `0` = Healthy, `1` = Heart Disease | Outcome Label |

---

## ⚙️ Features

- **Automated Data Processing**: Cleans and splits features into stratified training (80%) and test (20%) sets.
- **Machine Learning**: Utilizes Logistic Regression optimized for convergence (`max_iter=1000`).
- **Interactive CLI Input**: Allows real-time typing of custom patient parameters directly in the terminal.
- **Dual-Panel Diagnostic Dashboard**:
  - **Overall Patient Risk Assessment**: Donut chart with external percentage labels.
  - **Patient Vitals Comparison**: Horizontal bar chart comparing patient metrics against population averages.

---

## 🛠️ Installation & Setup

### Prerequisites
Make sure you have Python installed on your computer.

### Dependencies Installation
Install all required libraries using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run

1. Place `heart_disease_prediction.py` and `heart_disease_data.csv` in the same project directory.
2. Open your terminal/PowerShell and navigate to your project folder:

```bash
cd path/to/your/project
```

3. Run the script:

```bash
python heart_disease_prediction.py
```

4. Enter the patient values as prompted in the terminal.

---

## 🧪 Sample Test Cases

### Test Case 1: Low Risk / Healthy Patient
```text
Age: 41
Sex: 0
Chest Pain Type: 1
Resting Blood Pressure: 130
Serum Cholesterol: 204
Fasting Blood Sugar: 0
Resting ECG: 0
Max Heart Rate: 172
Exercise Induced Angina: 0
ST Depression: 1.4
Slope: 2
Major Vessels: 0
Thal: 2
```
* **Expected Result**: `Healthy / Low Risk`

### Test Case 2: High Risk / Heart Disease Patient
```text
Age: 62
Sex: 1
Chest Pain Type: 0
Resting Blood Pressure: 140
Serum Cholesterol: 268
Fasting Blood Sugar: 0
Resting ECG: 0
Max Heart Rate: 160
Exercise Induced Angina: 0
ST Depression: 3.6
Slope: 0
Major Vessels: 2
Thal: 2
```
* **Expected Result**: `Heart Disease / High Risk`

---

## 📈 Model Performance

- **Training Data Accuracy**: ~85%
- **Test Data Accuracy**: ~82%
