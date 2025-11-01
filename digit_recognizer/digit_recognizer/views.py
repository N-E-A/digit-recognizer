from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import joblib
from django.views.decorators.csrf import csrf_exempt

# Disable CSRF just for testing (later we can secure it)
@csrf_exempt

def predict(request):
    if request.method == "POST":
        try:
            pixels = request.POST.get("pixels")
            if not pixels:
                return JsonResponse({"error": "No data received"}, status=400)

            # Convert to numpy array
            pixels = np.array(list(map(float, pixels.split(","))))
            pixels = pixels.reshape(1, -1)

            model = joblib.load("model/digit_model.pkl")
            digit = model.predict([pixels.flatten()])[0]

            return JsonResponse({"digit": int(digit)})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)

def home(request):
    return render(request, "home.html")
