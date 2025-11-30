import pandas as pd, joblib, warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

# Load your train.csv (put it in the same folder)
train = pd.read_csv("train.csv")

def enhance(df):
    df = df.copy()
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['age'] = df['age'].astype(int)
    df['gender'] = df['gender'].replace('Other', 'Female')
    df['is_senior'] = (df['age'] >= 65).astype(int)
    df['high_glucose'] = (df['avg_glucose_level'] >= 140).astype(int)
    df['very_high_glucose'] = (df['avg_glucose_level'] >= 200).astype(int)
    df['obese'] = (df['bmi'] >= 30).astype(int)
    df['extremely_obese'] = (df['bmi'] >= 40).astype(int)
    df['age_glucose_risk'] = df['age'] * df['avg_glucose_level'] / 1000
    df['hypertension_heart'] = df['hypertension'] + df['heart_disease']
    df['smoking_unknown'] = (df['smoking_status'] == 'Unknown').astype(int)
    return df

train = enhance(train)

cat = ['gender','ever_married','work_type','Residence_type','smoking_status']
num = ['age','avg_glucose_level','bmi']

prep = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat)
], remainder='passthrough')

scale = len(train[train['stroke']==0]) / len(train[train['stroke']==1])

pipe = Pipeline([
    ('prep', prep),
    ('xgb', XGBClassifier(n_estimators=3000, learning_rate=0.008, max_depth=7,
                           min_child_weight=5, subsample=0.85, colsample_bytree=0.7,
                           reg_alpha=0.3, reg_lambda=1.2, scale_pos_weight=scale,
                           eval_metric='auc', random_state=42, n_jobs=-1))
])

print("Training the TOP 1–3 model...")
pipe.fit(train.drop(['id','stroke'], axis=1), train['stroke'])

joblib.dump(pipe, 'model.pkl')
print("MODEL SAVED → model.pkl created!")
print("Now run: streamlit run app.py")