import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import joblib
import os
from pathlib import Path


def load_data(filepath='data/data.csv'):
    print("📚 Loading the addiction data...")
    df = pd.read_csv(filepath)
    print(f"✅ All set! Found {df.shape[0]} people's data with {df.shape[1]} factors")
    return df


def preprocess_data(df):
    print("\n🔧 Getting the data ready...")

    df = df.copy()

    categorical_cols = ['gender', 'stress_level', 'academic_work_impact']
    numeric_cols = [
        'age', 'daily_screen_time_hours', 'social_media_hours',
        'gaming_hours', 'work_study_hours', 'sleep_hours',
        'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time'
    ]

    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print(f"✅ All cleaned up! Working with {len(numeric_cols) + len(categorical_cols)} important factors")

    return df, label_encoders


def select_features(df):
    print("\n📊 Picking the best factors...")

    primary_features = [
        'daily_screen_time_hours',
        'sleep_hours',
        'gaming_hours',
        'notifications_per_day'
    ]

    additional_features = [
        'age', 'gender', 'social_media_hours', 'work_study_hours',
        'app_opens_per_day', 'weekend_screen_time', 'stress_level',
        'academic_work_impact'
    ]

    feature_cols = primary_features + additional_features

    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df['addicted_label']

    print(f"✅ Using {len(feature_cols)} factors to train")
    print(f"   The main 4: {', '.join(primary_features)}")
    print(f"   Plus {len(additional_features)} extra ones: {', '.join(additional_features)}")

    return X, y, feature_cols


def train_model(X, y, test_size=0.2, random_state=42):
    print("\n🤖 Teaching the computer...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    print("✅ Training done!")

    return model, scaler, X_test_scaled, y_test


def evaluate_model(model, X_test, y_test):
    print("\n📈 Checking how good it is...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 How Well It Works:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Results Breakdown:")
    print(f"   Correctly found non-addicted: {cm[0, 0]}")
    print(f"   Wrongly marked as addicted: {cm[0, 1]}")
    print(f"   Wrongly marked as not addicted: {cm[1, 0]}")
    print(f"   Correctly found addicted: {cm[1, 1]}")

    print(f"\n📄 Full Details:")
    print(classification_report(y_test, y_pred, target_names=['Not Addicted', 'Addicted']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def save_model(model, scaler, feature_cols, model_dir='models'):
    print(f"\n💾 Saving everything...")

    Path(model_dir).mkdir(exist_ok=True)

    model_path = os.path.join(model_dir, 'addiction_model.pkl')
    joblib.dump(model, model_path)
    print(f"   ✅ Brain saved: {model_path}")

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Normalizer saved: {scaler_path}")

    features_path = os.path.join(model_dir, 'features.pkl')
    joblib.dump(feature_cols, features_path)
    print(f"   ✅ Factor list saved: {features_path}")


def main():
    print("=" * 60)
    print("🎯 Train the Addiction Predictor")
    print("=" * 60)

    try:
        df = load_data('data/data.csv')
        df, label_encoders = preprocess_data(df)

        X, y, feature_cols = select_features(df)

        model, scaler, X_test, y_test = train_model(X, y)

        metrics = evaluate_model(model, X_test, y_test)

        save_model(model, scaler, feature_cols)

        print("\n" + "=" * 60)
        print("✨ All done! Ready to predict!")
        print("=" * 60)
        print("\n🚀 Next: Run 'python predict.py' to test it out")

    except FileNotFoundError:
        print("❌ Oops! Can't find the data file")
        print("   Make sure data/data.csv exists")
    except Exception as e:
        print(f"❌ Something went wrong: {str(e)}")
        raise


if __name__ == '__main__':
    main()
