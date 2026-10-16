import pickle
import pandas as pd

with open("model.pkl" , "rb") as f:
    model = pickle.load(f)

user_req = {
    "area": 1200,
    "bedrooms": 2,
    "bathrooms": 2,
    "status": "Ready to Move",
    "neworold": "Resale",
    "furnished_status": 1,
    "type_of_building": "Flat",
    "category": "West Mumbai"
}


input_df = pd.DataFrame([user_req])

prediction = model.predict(input_df)[0]

print(f"Predicted Price: ₹{prediction:.2f} Crore")
print(f"Predicted Price: ₹{prediction * 100:.2f} Lakh")
