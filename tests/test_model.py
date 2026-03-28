import sys
sys.path.insert(0, '../src')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def test_model_training():
    print("🧪 Testing model training...")

    X, y = make_classification(n_samples=100, n_features=12, n_informative=8, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    score = model.score(X_test_scaled, y_test)

    assert score > 0.7, f"Model accuracy too low: {score}"
    print(f"✅ Training test passed! Accuracy: {score:.2f}")


def test_model_prediction():
    print("🧪 Testing model prediction...")

    X, y = make_classification(n_samples=100, n_features=12, n_informative=8, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    assert len(predictions) == len(y_test), "Prediction length mismatch"
    assert all(p in [0, 1] for p in predictions), "Invalid predictions"
    print(f"✅ Prediction test passed! Predictions: {len(predictions)}")


if __name__ == '__main__':
    test_model_training()
    test_model_prediction()
    print("\n✨ All tests passed!")
