import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

class DatabaseService:
    def __init__(self):
        self.client: Client = None
        if SUPABASE_URL and SUPABASE_KEY:
            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            
    def is_connected(self):
        return self.client is not None

    def save_prediction(self, student_id: str, input_data: dict, prediction_data: dict, user_id: str = None):
        if not self.is_connected():
            return False
            
        record = {
            "student_id": student_id,
            "input_snapshot": input_data,
            "prediction": prediction_data["prediction"],
            "probability": prediction_data["probability"],
            "risk_level": prediction_data["risk_level"],
            "model_version": prediction_data["model_version"],
            "user_id": user_id,
            "timestamp": prediction_data["prediction_date"]
        }
        
        try:
            result = self.client.table("predictions").insert(record).execute()
            return True
        except Exception as e:
            print(f"Error saving prediction to Supabase: {e}")
            return False

db_service = DatabaseService()
