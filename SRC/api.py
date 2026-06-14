from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pickle
import numpy as np
import pandas as pd
from SRC.train_neural import AirlineNet
import os

app = FastAPI(title="Airline Passenger Satisfaction API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'Models/scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, 'Models/encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

with open(os.path.join(BASE_DIR, 'Models/feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)

with open(os.path.join(BASE_DIR, 'Models/medians.pkl'), 'rb') as f:
    medians = pickle.load(f)

input_dim = 87
model = AirlineNet(input_dim)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'Models/best_model.pt')))
model.eval()

class PassengerInput(BaseModel):
    process: str
    month: str
    connection: str
    ticket_purchased_by: str
    ticket_purchase_channel: str = "Unknown"
    transport_to_airport: str = "Unknown"
    has_disability: str = "No"
    uses_assistive_device: str = "No"
    requested_special_assistance: str = "No"
    disembarkation_method_used: str = "Unknown"
    disembarkation_method_rating: float = 3.0
    used_parking: str = "No"
    curbside_dropoff_ease: float = 3.0
    transport_options_to_airport: float = 3.0
    checkin_method: str = "Unknown"
    checkin_process: float = 3.0
    checkin_queue_wait_time: float = 3.0
    checkin_queue_organization: float = 3.0
    self_service_kiosk_quantity: float = 3.0
    checkin_counter_quantity: float = 3.0
    staff_courtesy: float = 3.0
    checkin_service_time: float = 3.0
    ticket_purchase_process: float = 3.0
    airline_service: float = 3.0
    security_screening_process: float = 3.0
    security_queue_wait_time: float = 3.0
    security_queue_organization: float = 3.0
    security_staff_service: float = 3.0
    immigration_control: float = 3.0
    immigration_queue_wait_time: float = 3.0
    immigration_queue_organization: float = 3.0
    immigration_staff_service: float = 3.0
    service_window_quantity: float = 3.0
    customs_control: float = 3.0
    customs_queue_wait_time: float = 3.0
    customs_queue_organization: float = 3.0
    customs_staff_service: float = 3.0
    food_beverage_outlets: float = 3.0
    food_beverage_outlet_quantity: float = 3.0
    food_beverage_quality_variety: float = 3.0
    food_beverage_price_quality: float = 3.0
    retail_outlets: float = 3.0
    retail_outlet_quantity: float = 3.0
    retail_quality_variety: float = 3.0
    retail_price_quality: float = 3.0
    parking: float = 3.0
    parking_facility_quality: float = 3.0
    parking_space_availability_ease: float = 3.0
    parking_terminal_access_ease: float = 3.0
    parking_value_for_money: float = 3.0
    location_and_movement: float = 3.0
    signage: float = 3.0
    flight_information_display_availability: float = 3.0
    terminal_accessibility: float = 3.0
    boarding_lounge_comfort: float = 3.0
    thermal_comfort: float = 3.0
    acoustic_comfort: float = 3.0
    seat_availability: float = 3.0
    reserved_seat_availability: float = 3.0
    power_outlet_availability: float = 3.0
    airport_internet: float = 3.0
    internet_connection_speed: float = 3.0
    network_access_ease: float = 3.0
    restrooms: float = 3.0
    restroom_quantity: float = 3.0
    restroom_cleanliness: float = 3.0
    restroom_maintenance: float = 3.0
    overall_airport_cleanliness: float = 3.0
    baggage_claim_process: float = 3.0
    baggage_carousel_identification_ease: float = 3.0
    baggage_claim_time: float = 3.0
    baggage_integrity: float = 3.0
    nationality: str = "Unknown"
    age_group: str = "26 to 35 years"
    education: str = "Unknown"
    household_income: str = "Unknown"
    traveling_alone: str = "Yes"
    number_of_companions: float = 0.0
    trip_purpose: str = "Leisure"
    trips_last_12_months: float = 1.0
    used_airport_before_last_12_months: str = "Yes"
    arrival_lead_time: float = 120.0
    connection_wait_time: float = 150.0

@app.get("/")
def root():
    return {"message": "Airline Passenger Satisfaction API is running!"}

@app.post("/predict")
def predict(passenger: PassengerInput):
    data = pd.DataFrame([passenger.dict()])

    cat_cols = [col for col in data.columns
                if isinstance(data[col][0], str)]

    for col in cat_cols:
        if col in encoders:
            val = str(data[col][0])
            if val in encoders[col].classes_:
                data[col] = encoders[col].transform([val])[0]
            else:
                data[col] = -1
        else:
            data[col] = -1
    data['comfort_score'] = data[['boarding_lounge_comfort',
                                   'thermal_comfort',
                                   'acoustic_comfort']].mean(axis=1)
    data['cleanliness_score'] = data[['overall_airport_cleanliness',
                                       'restroom_cleanliness',
                                       'restroom_maintenance']].mean(axis=1)
    data['price_quality_score'] = data[['food_beverage_price_quality',
                                         'retail_price_quality',
                                         'parking_value_for_money']].mean(axis=1)
    data['queue_score'] = data[['checkin_queue_wait_time',
                                 'security_queue_wait_time']].mean(axis=1)

    for col in feature_names:
        if col not in data.columns:
            data[col] = medians.get(col, 0)

    data = data[feature_names]

    data_scaled = scaler.transform(data)

    X = torch.FloatTensor(data_scaled)
    with torch.no_grad():
        prob = model(X).item()

    prediction = 1 if prob >= 0.5 else 0
    label = "Satisfied" if prediction == 1 else "Not Satisfied"

    return {
        "prediction": prediction,
        "label": label,
        "probability": round(prob, 4)
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)