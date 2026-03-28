import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, Any


class AddictionPredictor:
    model: Any
    scaler: Any
    feature_cols: list

    def __init__(self, model_dir: str = 'models') -> None:
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols = []
        self.load_model()

    def load_model(self) -> None:
        print("🔄 Loading the brain...")

        model_path = os.path.join(self.model_dir, 'addiction_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        features_path = os.path.join(self.model_dir, 'features.pkl')

        if not all(Path(p).exists() for p in [model_path, scaler_path, features_path]):
            raise FileNotFoundError(
                f"Can't find the trained brain in {self.model_dir}/ folder.\n"
                "Please train it first: python train.py"
            )

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_cols = joblib.load(features_path)
            print("✅ Brain loaded!")
        except Exception as e:
            raise RuntimeError(f"Had trouble loading: {str(e)}")

    def get_user_input(self) -> Optional[Dict[str, float]]:
        print("\n" + "=" * 60)
        print("📱 Addiction Check Tool")
        print("=" * 60)
        print("\n📝 Tell me about your phone habits:")
        print("   (Type 'q' to quit anytime)\n")

        inputs = {}

        try:
            while True:
                try:
                    value = input("  📊 How many hours on phone daily? (0-24): ")
                    if value.lower() == 'q':
                        return None
                    screen_time = float(value)
                    if 0 <= screen_time <= 24:
                        inputs['daily_screen_time_hours'] = screen_time
                        break
                    else:
                        print("     ⚠️  Please enter 0 to 24")
                except ValueError:
                    print("     ⚠️  Just a number please")

            while True:
                try:
                    value = input("  💤 How many hours sleep per night? (0-24): ")
                    if value.lower() == 'q':
                        return None
                    sleep = float(value)
                    if 0 <= sleep <= 24:
                        inputs['sleep_hours'] = sleep
                        break
                    else:
                        print("     ⚠️  Please enter 0 to 24")
                except ValueError:
                    print("     ⚠️  Just a number please")

            while True:
                try:
                    value = input("  🎮 How many hours gaming daily? (0-24): ")
                    if value.lower() == 'q':
                        return None
                    gaming = float(value)
                    if 0 <= gaming <= 24:
                        inputs['gaming_hours'] = gaming
                        break
                    else:
                        print("     ⚠️  Please enter 0 to 24")
                except ValueError:
                    print("     ⚠️  Just a number please")

            while True:
                try:
                    value = input("  🔔 How many notifications daily? (0-1000): ")
                    if value.lower() == 'q':
                        return None
                    notifs = int(value)
                    if 0 <= notifs <= 1000:
                        inputs['notifications_per_day'] = notifs
                        break
                    else:
                        print("     ⚠️  Please enter 0 to 1000")
                except ValueError:
                    print("     ⚠️  Just a whole number please")

            return inputs

        except KeyboardInterrupt:
            print("\n\n⚠️  You stopped")
            return None

    def prepare_features(self, user_input: Dict[str, float]) -> pd.DataFrame:
        data = pd.DataFrame([user_input])

        defaults = {
            'age': 26.5,
            'gender': 0,
            'social_media_hours': 3.0,
            'work_study_hours': 3.5,
            'app_opens_per_day': 115,
            'weekend_screen_time': 8.0,
            'stress_level': 1,
            'academic_work_impact': 1
        }

        for feature in self.feature_cols:
            if feature not in data.columns:
                data[feature] = defaults.get(feature, 0)

        data = data[self.feature_cols]

        return data

    def predict(self, user_input: Dict[str, float]) -> Tuple[int, np.ndarray]:
        try:
            X = self.prepare_features(user_input)

            X_scaled = self.scaler.transform(X)

            prediction = self.model.predict(X_scaled)[0]
            confidence = self.model.predict_proba(X_scaled)[0]

            return prediction, confidence

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def display_result(self, prediction: int, confidence: np.ndarray, user_input: Dict[str, float]) -> None:
        print("\n" + "=" * 60)
        print("🔍 Results")
        print("=" * 60)

        print("\n📋 What You Told Me:")
        print(f"   • Phone time:     {user_input['daily_screen_time_hours']:.1f} hours/day")
        print(f"   • Sleep:          {user_input['sleep_hours']:.1f} hours/night")
        print(f"   • Gaming:         {user_input['gaming_hours']:.1f} hours/day")
        print(f"   • Notifications:  {user_input['notifications_per_day']:.0f}/day")

        if prediction == 0:
            status = "✅ HEALTHY"
            color_emoji = "🟢"
        else:
            status = "⚠️  ADDICTED"
            color_emoji = "🔴"

        healthy_conf = confidence[0] * 100
        addicted_conf = confidence[1] * 100

        print(f"\n{color_emoji} You Are: {status}")
        print(f"\n📊 Confidence:")
        print(f"   • Healthy:   {healthy_conf:.1f}%")
        print(f"   • Addicted:  {addicted_conf:.1f}%")

        print(f"\n💡 Advice:")
        if prediction == 0:
            print("   ✓ Your phone use looks balanced")
            print("   ✓ Keep up those healthy habits!")
        else:
            print("   ⚠️  Try cutting screen time")
            print("   ⚠️  Fix your sleep schedule")
            print("   ⚠️  Reduce games and alerts")
            print("   ⚠️  Take phone-free breaks")

        print("\n" + "=" * 60)

    def run_interactive(self) -> None:
        while True:
            user_input = self.get_user_input()

            if user_input is None:
                print("\n👋 Thanks for checking!")
                break

            try:
                prediction, confidence = self.predict(user_input)
                self.display_result(prediction, confidence, user_input)

                print("\n")
                again = input("Try again? (yes/no): ").lower()
                if again not in ['yes', 'y']:
                    print("\n👋 Thanks for checking!")
                    break
                print("\n")

            except Exception as e:
                print(f"\n❌ Oops: {str(e)}")
                break


def main():
    try:
        predictor = AddictionPredictor()
        predictor.run_interactive()

    except FileNotFoundError as e:
        print(f"❌ {str(e)}")
        print("\n🚀 First, train it by running:")
        print("   python train.py")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
