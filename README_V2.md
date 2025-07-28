V2 is not an add-on from V1. I scrapped V1 and started from scratch because V1 had way too many bugs and the algorithm was not working at all, so I decided to start over and add features slowly so I won't screw up my whole code. This is also where I started to take this algorithm seriously.

# 🎓 Student Outcome Predictor - Algorithm V2

## Overview

A machine learning model that predicts high school student outcomes using Random Forest classification. This version focuses on core prediction functionality with basic feature engineering.

## 🚀 Features

- **3 Outcome Predictions**: High School Dropout, High School Graduate, College Bound
- **10 Core Features**: GPA, attendance, discipline, family income, parent education, extracurriculars, internet access, participation, school quality, SAT scores
- **Cross-validation**: 5-fold cross-validation for model evaluation
- **Feature Importance**: Shows which factors most influence predictions
- **SAT Score Estimation**: Automatically estimates SAT scores from GPA if not provided

## 📊 Model Details

- **Algorithm**: Random Forest Classifier (100 estimators)
- **Preprocessing**: StandardScaler for feature normalization
- **Training Data**: 1,000 synthetic student records
- **Features**: 10 core academic and demographic features

## 🛠️ Installation

### Prerequisites

```bash
pip install pandas numpy scikit-learn
```

### Quick Start

```bash
python algorithim_V2.py
```

## 📋 Input Features

| Feature          | Type    | Range     | Description                      |
| ---------------- | ------- | --------- | -------------------------------- |
| GPA              | Float   | 0.0 - 4.0 | Grade Point Average              |
| Attendance Rate  | Float   | 0.0 - 1.0 | School attendance percentage     |
| Discipline Level | Integer | 1-10      | Student discipline rating        |
| Family Income    | Float   | Any       | Annual family income             |
| Parent Education | Integer | 0-4       | Parent's highest education level |
| Extracurriculars | Integer | 0+        | Number of activities             |
| Internet Access  | Boolean | 0/1       | Has internet access              |
| Participation    | Integer | 1-10      | Class participation level        |
| School Quality   | Integer | 1-10      | School quality rating            |
| SAT Score        | Integer | 400-1600  | SAT test score                   |

## 🎯 Output Predictions

The model predicts three possible outcomes with probability scores:

1. **High School Dropout** - Student may not complete high school
2. **High School Graduate** - Student will graduate but may not attend college
3. **College Bound** - Student is likely to attend college

## 📈 Model Performance

- **Cross-validation**: 5-fold CV for robust evaluation
- **Feature Importance**: Ranked list of most influential factors
- **Probability Scores**: Confidence levels for each prediction

## 🔧 Usage Example

```python
# Run the prediction model
python algorithim_V2.py

# Example interaction:
# Enter student information:
# GPA: 3.5
# Attendance rate: 0.9
# Discipline level: 8
# Family income: 75000
# Parent education: 3 (Bachelor's degree)
# Extracurriculars: 3
# Internet access: yes
# Participation: 8
# School quality: 8
# SAT score: 1200

# Output:
# Predicted Outcome: College Bound
# Probability: 85% College Bound, 12% HS Graduate, 3% Dropout
```

## 📁 File Structure

```
college-dropout-prediction/
├── algorithim_V2.py          # Main prediction algorithm
├── requirements.txt          # Python dependencies
└── README_V2.md             # This file
```

## 🔄 Version History

### V2 Improvements over V1:

- Added SAT score handling with automatic estimation
- Improved feature engineering
- Better synthetic data generation
- Enhanced user interaction
- More detailed output formatting

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

---

**Next Version**: [V3](README_V3.md) adds location-based features and state rankings.
