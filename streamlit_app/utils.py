import requests

API_URL = "http://127.0.0.1:8000/predict"


def predict_price(data: dict):
    try:
        response = requests.post(
            API_URL,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}") from e