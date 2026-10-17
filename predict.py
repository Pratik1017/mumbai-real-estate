import pickle
import pandas as pd

with open("model.pkl" , "rb") as f:
    model = pickle.load(f)

user_req = {
    "area": 100,
    "bedrooms": 1,
    "bathrooms": 0,
    "status": "Ready to Move",
    "neworold": "New Property",
    "furnished_status": 0,
    "type_of_building": "Flat",
    "category": "South Mumbai"
}


input_df = pd.DataFrame([user_req])

prediction = model.predict(input_df)[0]

print(f"Predicted Price: ₹{prediction:.2f} Crore")
print(f"Predicted Price: ₹{prediction * 100:.2f} Lakh")
