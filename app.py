import streamlit as st
import joblib
import pandas as pd
import calendar

# --- Konfigurasi Halaman dan Judul ---
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"  # Menggunakan layout lebar agar lebih pas untuk kolom
)

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write(
    "Use this application to analyze new bookings and "
    "identify customers who are at high risk of cancellation."
)
st.markdown("---")

# --- Fungsi untuk Memuat Model (dengan caching) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_final.joblib')
        columns = joblib.load('model_columns.joblib')
        return model, columns
    except FileNotFoundError:
        return None, None

model, model_columns = load_model()

if model is None or model_columns is None:
    st.error("❌ Model files not found. Please ensure 'model_final.joblib' and 'model_columns.joblib' are in the same folder.")
    st.stop()


# --- Area Input Pengguna di Halaman Utama ---
st.header("⚙️ Enter Booking Details")

# Membuat dua kolom untuk input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Main Booking Information")

    lead_time = st.slider(
        'Lead Time (Days)', 0, 400, 90,
        help="How many days are between the booking date and the arrival date? A longer lead time usually means a higher risk."
    )
    
    stay_length = st.slider(
        'Total Length of Stay (Nights)', 1, 30, 3,
        help="Total nights the guest will be staying. Longer stays can sometimes have different risk profiles."
    )

    month_names = list(calendar.month_name)[1:]
    arrival_month_name = st.selectbox(
        'Arrival Month', month_names, index=6,
        help="Select the guest's arrival month. This helps the model recognize seasonal patterns."
    )
    month_map = {name: i+1 for i, name in enumerate(month_names)}
    arrival_month = month_map[arrival_month_name]
    
    adr = st.number_input(
        'Average Price per Night',
        min_value=0.0, max_value=5000.0, value=100.0, step=10.0,
        help="Enter the average price per night for this booking. Price is a very important factor in predicting cancellations."
    )


with col2:
    st.subheader("Guest Profile & History")
    
    total_guests = st.number_input(
        'Total Number of Guests', min_value=1, max_value=20, value=2,
        help="Total number of adults, children, and babies for this booking."
    )

    # WIDGET BARU DITAMBAHKAN DI SINI
    required_car_parking_spaces = st.selectbox(
        'Required Parking Spaces?',
        options=[0, 1, 2],
        help="Does the guest require a car parking space? This is a strong indicator of a committed booking."
    )
    
    is_repeated_guest_str = st.radio(
        'Is Repeated Guest?', ('No', 'Yes'), horizontal=True,
        help="Has this guest stayed at the hotel before? Repeated guests are less likely to cancel."
    )
    is_repeated_guest = 1 if is_repeated_guest_str == 'Yes' else 0

    previous_cancellations = st.slider(
        'Number of Previous Cancellations', 0, 26, 0,
        help="How many times has this guest canceled a booking in the past?"
    )

    deposit_type = st.selectbox(
        'Deposit Type', ['No Deposit', 'Non Refund', 'Refundable'],
        help="'No Deposit' is the riskiest. 'Non Refund' is the safest."
    )
    
    total_of_special_requests = st.slider(
        'Number of Special Requests', 0, 5, 1,
        help="How many special requests were made (e.g., non-smoking room)? More requests usually mean a lower chance of cancellation."
    )


st.markdown("---")

# Tombol prediksi ditempatkan di tengah setelah semua input
_, col_button, _ = st.columns([2, 1, 2])
predict_button = col_button.button("Check Cancellation Risk", type="primary", use_container_width=True)

# --- Logika Prediksi dan Tampilan Hasil ---
if predict_button:
    input_data = {
        # Menambahkan fitur baru
        'adr': adr,
        'total_guests': total_guests,
        'stay_length': stay_length,

        # Fitur yang sudah ada
        'lead_time': lead_time,
        'arrival_month': arrival_month,
        'deposit_type': deposit_type,
        'total_of_special_requests': total_of_special_requests,
        'is_repeated_guest': is_repeated_guest,
        'previous_cancellations': previous_cancellations,

        # Fitur default (tidak diinputkan user)
        'booking_changes': 0,
        # MENGGUNAKAN NILAI DARI INPUT PENGGUNA
        'required_car_parking_spaces': required_car_parking_spaces, 
        'country': 'PRT',
        'market_segment': 'Online TA',
        'customer_type': 'Transient', 
        'hotel': 'Resort Hotel'
    }

    input_df = pd.DataFrame([input_data])
    input_df_processed = pd.get_dummies(input_df, drop_first=True)
    final_df = input_df_processed.reindex(columns=model_columns, fill_value=0)

    try:
        probability = model.predict_proba(final_df)[0][1]
        
        st.header("Risk Analysis Results")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric(
                label="Cancellation Probability",
                value=f"{probability:.2%}",
                delta_color="inverse"
            )

        with res_col2:
            if probability > 0.7:
                st.error("🔴 Very High Risk")
            elif probability > 0.4:
                st.warning("🟠 Medium Risk")
            else:
                st.success("🟢 Low Risk")

        st.subheader("💡 Recommended Action")
        if probability > 0.7:
            st.info(
                "**Contact Guest Immediately:** Send a personal email to reconfirm the booking. Offer a small incentive (e.g., a drink voucher) if they pay a deposit now."
            )
        elif probability > 0.4:
            st.info(
                "**Monitor Actively:** Add this booking to a watchlist. Send a standard reminder email 2 weeks before the free cancellation period ends."
            )
        else:
            st.info(
                "**No Special Action Needed:** This booking is likely secure. Focus your efforts on higher-risk customers."
            )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Please enter the booking details in the columns above and click the button to see the results.")