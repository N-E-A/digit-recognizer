from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import joblib
import os

# ✅ Get the absolute path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "digit_model.pkl")

# ✅ Load model once when server starts
model = joblib.load(MODEL_PATH)

def home(request):
    return render(request, "home.html")

def predict(request):
    if request.method == "POST":
        try:
            pixels = request.POST.get("pixels")
            if not pixels:
                return JsonResponse({"error": "No pixel data received"})

            # Convert and reshape
            pixels = np.array(pixels.split(","), dtype=float)
            pixels = pixels / 255.0
            pixels = pixels.reshape(1, -1)

            # Predict
            prediction = model.predict(pixels)[0]
            print("Predicted digit:", prediction)  # 👈 for debugging in terminal

            return JsonResponse({"digit": int(prediction)})
        except Exception as e:
            print("Error:", e)
            return JsonResponse({"error": str(e)})
    else:
        return JsonResponse({"error": "Invalid request method"})
