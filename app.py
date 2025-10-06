import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

st.title("Hotel Booking Cancellation Prediction")
st.markdown("This app predicts whether a hotel booking will be cancelled or not using trained Random Forest model.")

hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
lead_time = st.number_input("Lead Time (days before arrival)", min_value=0, value=30)
previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
booking_changes = st.number_input("Number of Booking Changes", min_value=0, value=0)
total_of_special_requests = st.number_input("Total Special Requests", min_value=0, value=0)
high_request = st.selectbox("High Request (1 = Yes, 0 = No)", [0, 1])
high_canceller = st.selectbox("High Canceller (1 = Yes, 0 = No)", [0, 1])
last_minute = st.selectbox("Last-Minute Booking (1 = Yes, 0 = No)", [0, 1])
is_repeated_guest = st.selectbox("Repeated Guest (1 = Yes, 0 = No)", [0, 1])
distribution_channel = st.selectbox("Distribution Channel", ["Direct", "Corporate", "TA/TO", "GDS"])
deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
cancellation_ratio = st.number_input("Cancellation Ratio", min_value=0.0, max_value=1.0, value=0.0)

input_df = pd.DataFrame({
    'hotel': [hotel],
    'lead_time': [lead_time],
    'previous_cancellations': [previous_cancellations],
    'previous_bookings_not_canceled': [previous_bookings_not_canceled],
    'booking_changes': [booking_changes],
    'high_canceller': [high_canceller],
    'high_request': [high_request],
    'total_of_special_requests': [total_of_special_requests],
    'last_minute': [last_minute],
    'is_repeated_guest': [is_repeated_guest],
    'distribution_channel': [distribution_channel],
    'deposit_type': [deposit_type],
    'customer_type': [customer_type],
    'cancellation_ratio': [cancellation_ratio]
})

input_df = pd.get_dummies(input_df, drop_first=True)

expected_cols = model.feature_names_in_
for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_cols]

if st.button("Predict Cancellation"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"Booking likely to be CANCELLED (Probability: {prob:.2f})")
        else:
            st.success(f"Booking likely to be HONOURED (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
