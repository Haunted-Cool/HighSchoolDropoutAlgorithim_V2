import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
positive_responses = {'y', 'yes', 'ye', 'why not', 'sure', 'certainly', 'for sure', 'of course', 'obviously', 'ok', '1'}
negative_responses = {'no', 'n', 'nah', 'na', 'nope', 'not feeling it', 'obviously not', 'hell no', '2'}

class EducationPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = [
            'GPA', 'attendance_rate', 'discipline_count', 'family_income',
            'parent_edu', 'extracurriculars', 'access_to_internet',
            'participation', 'school_quality', 'sat_score'
        ]
        
    def train(self, X, y):
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Train final model
        self.model.fit(X_scaled, y)
        
        # Get feature importance
        importance = dict(zip(self.features, self.model.feature_importances_))
        print("\nFeature Importance:")
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feat}: {imp:.3f}")
    
    def predict_proba(self, student_data):
        # Convert single student data to DataFrame
        student_df = pd.DataFrame([student_data])
        
        # Standardize features
        student_scaled = self.scaler.transform(student_df)
        
        # Get probability predictions
        probas = self.model.predict_proba(student_scaled)[0]
        
        return probas

def get_prediction():
    parent_edu_levels = {
        0: "No formal education",
        1: "High school diploma or GED",
        2: "Some college or associate degree",
        3: "Bachelor's degree",
        4: "Graduate or professional degree"
    }
    
    outcomes = {
        0: "High School Dropout",
        1: "High School Graduate Only",
        2: "College Bound"
    }
    
    # Collect student data
    student_data = {}
    
    print("\nEnter student information:")
    student_data['GPA'] = float(input("What is their GPA? "))
    student_data['attendance_rate'] = float(input("What is their attendance rate (0-1)? "))
    student_data['discipline_count'] = int(input("On a scale of 1-10, how disciplined are they? (10 = very disciplined): "))
    student_data['family_income'] = float(input("What is their family income? "))
    
    print("\nParent Education Levels:")
    for key, value in parent_edu_levels.items():
        print(f"{key}: {value}")
    student_data['parent_edu'] = int(input("\nWhat's their parent's education level? (0-4): "))
    
    student_data['extracurriculars'] = int(input("How many extracurriculars do they participate in? "))
    student_data['access_to_internet'] = 1 if get_close_matches(input("Do they have internet access? ").lower(), positive_responses, n=1, cutoff=0.6) else 0
    student_data['participation'] = int(input("Participation level (1-10)? "))
    student_data['school_quality'] = int(input("School quality (1-10)? "))
    
    # Ask for SAT score
    sat_response = input("Have they taken the SAT? (yes/no) ").lower()
    if get_close_matches(sat_response.lower(), positive_responses, n=1, cutoff=0.6):
        student_data['sat_score'] = float(input("Enter their SAT score (400-1600): "))
    else:
        # Estimate SAT score from GPA if not available
        estimated_sat = student_data['GPA'] * 400
        student_data['sat_score'] = max(400, min(1600, estimated_sat))
        print(f"\nEstimated SAT score based on GPA: {student_data['sat_score']}")
    
    # Create predictor instance
    predictor = EducationPredictor()
    
    # Generate some sample historical data for demonstration
    # In real application, this would be replaced with actual historical data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic training data
    historical_data = {
        'GPA': np.random.normal(3.0, 0.5, n_samples).clip(0, 4.0),
        'attendance_rate': np.random.beta(8, 2, n_samples),
        'discipline_count': np.random.randint(1, 11, n_samples),
        'family_income': np.random.lognormal(10.5, 1, n_samples),
        'parent_edu': np.random.randint(0, 5, n_samples),
        'extracurriculars': np.random.poisson(2, n_samples),
        'access_to_internet': np.random.binomial(1, 0.9, n_samples),
        'participation': np.random.randint(1, 11, n_samples),
        'school_quality': np.random.randint(1, 11, n_samples),
        'sat_score': np.random.normal(1100, 200, n_samples).clip(400, 1600)
    }
    
    # Generate synthetic outcomes based on feature combinations
    X = pd.DataFrame(historical_data)
    y = np.where(X['GPA'] > 3.5, 2,  # College Bound
          np.where(X['GPA'] > 2.0, 1,  # HS Graduate
                  0))  # Dropout
    
    # Train the model
    print("\nTraining model with historical data...")
    predictor.train(X, y)
    
    # Make prediction for new student
    probas = predictor.predict_proba(student_data)
    
    # Display results
    print("\nPrediction Results:")
    prediction = outcomes[np.argmax(probas)]
    print(f"Predicted Outcome: {prediction}")
    
    print("\nProbability for each outcome:")
    for i, (outcome, prob) in enumerate(zip(outcomes.values(), probas)):
        print(f"{outcome}: {prob:.1%}")

if __name__ == "__main__":
    get_prediction()