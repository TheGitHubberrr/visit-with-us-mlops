
import streamlit as st
import pandas as pd
import joblib

# Load trained model
MODEL_PATH = "tourism_project/deployment/best_model.pkl"

model = joblib.load(MODEL_PATH)

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Wellness Tourism Package Predictor")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

st.divider()

# --------------------------------------------------
# Customer Inputs
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    persons_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

with col2:
    preferred_property_star = st.number_input(
        "Preferred Property Star",
        min_value=3,
        max_value=5,
        value=3
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Divorced", "Single", "Unmarried"]
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=30,
        value=3
    )

    passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

with col3:
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        max_value=1000000.0,
        value=25000.0,
        step=1000.0
    )

    pitch_satisfaction = st.number_input(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )

    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=10,
        value=3
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0.0,
        max_value=120.0,
        value=15.0
    )

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Package Purchase", type="primary"):

    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "🎉 Prediction: The customer is likely to purchase "
            "the Wellness Tourism Package!"
        )
    else:
        st.info(
            "Prediction: The customer is unlikely to purchase "
            "the Wellness Tourism Package."
        )

    st.subheader("Customer Input")
    st.dataframe(input_data)
