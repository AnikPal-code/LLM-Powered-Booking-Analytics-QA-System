import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="LLM Booking Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin: 1rem 0;
    }
    .highlight {
        color: #1E88E5;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
    }
    .result-container {
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        background-color: rgba(30, 136, 229, 0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# FastAPI Backend URL
# Change this if your API is running on a different URL
BASE_URL = "http://127.0.0.1:8000"

# Header
st.markdown('<div class="main-header">🏨LLM-Powered Booking Analytics & QA System </div>',
            unsafe_allow_html=True)
st.markdown(
    'Analyze your hotel booking data and get AI-powered insights instantly.')

# Create tabs with custom styling - removed batch upload tab
tabs = st.tabs(["💬 Ask a Question", "📈 Analytics Report"])

# ------------------ ASK A QUESTION TAB ------------------
with tabs[0]:
    st.markdown('<div class="sub-header">💬 Ask About Your Booking Data</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area("Enter your question about hotel bookings:",
                                placeholder="Example: What factors lead to booking cancellations?",
                                height=100)
        examples = st.expander("📝 Example questions")
        with examples:
            st.markdown("""
            - Which month has the highest booking rate?
            - What's the average length of stay in city hotels vs. resort hotels?
            - Is there a correlation between lead time and cancellation rate?
            """)

    with col2:
        st.markdown('<br>', unsafe_allow_html=True)
        ask_button = st.button("🔍 Get Answer", use_container_width=True)

    if ask_button and question:
        with st.spinner("Analyzing your question..."):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{BASE_URL}/ask", json={"question": question}, timeout=30)
                end_time = time.time()

                if response.status_code == 200:
                    data = response.json()

                    st.markdown('<div class="result-container">',
                                unsafe_allow_html=True)
                    st.markdown("### 🤖 Answer")
                    st.markdown(data['answer'])
                    st.markdown(
                        f"<small>⏱️ Response time: {data.get('response_time', end_time - start_time):.2f} seconds</small>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(
                        f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to the API: {str(e)}")
    elif ask_button:
        st.warning("Please enter a question first.")

# ------------------ ANALYTICS REPORT TAB ------------------
with tabs[1]:
    st.markdown('<div class="sub-header">📊 Generate Booking Analytics Report</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Hotel selection with icons
        hotel = st.selectbox(
            "🏢 Hotel Type",
            options=["City Hotel", "Resort Hotel"],
            index=0
        )

        # Arrival date selection
        arrival_date_year = st.selectbox(
            "📅 Arrival Year",
            options=["2022", "2023", "2024"],
            index=1
        )

        arrival_date_month = st.selectbox(
            "📅 Arrival Month",
            options=[
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
            index=6
        )

    with col2:
        # Date picker for reservation status date
        default_date = datetime.strptime("2023-07-15", "%Y-%m-%d")
        reservation_status_date = st.date_input(
            "📆 Reservation Status Date",
            value=default_date
        )

        # Prettier booking status selector
        is_canceled = st.radio(
            "🚫 Booking Status",
            options=["Confirmed", "Canceled"],
            index=0,
            horizontal=True
        )

        # Convert "Confirmed"/"Canceled" to 0/1 for API
        is_canceled_val = 1 if is_canceled == "Canceled" else 0

    # Generate button
    if st.button("🚀 Generate Analytics Report", use_container_width=True):
        with st.spinner("Generating comprehensive booking analysis..."):
            booking_data = (
                f"arrival_date_year,arrival_date_month,hotel,is_canceled,reservation_status_date\n"
                f"{arrival_date_year},{arrival_date_month},{hotel},{is_canceled_val},{reservation_status_date}"
            )

            try:
                response = requests.post(
                    f"{BASE_URL}/analytics", json={"data": booking_data}, timeout=45)

                if response.status_code == 200:
                    data = response.json()

                    # Show the report in a nicely formatted container
                    st.markdown('<div class="result-container">',
                                unsafe_allow_html=True)
                    st.markdown("### 📑 Booking Analytics Report")

                    # Split the report into sections for better visualization
                    report_parts = data['report'].split('\n\n')

                    for i, part in enumerate(report_parts):
                        if i == 0:  # Summary
                            st.markdown(f"**{part}**")
                        else:  # Other sections
                            st.markdown(part)
                            if i < len(report_parts) - 1:
                                st.markdown("---")

                    # Add download button for the report
                    st.download_button(
                        label="📥 Download Report",
                        data=data['report'],
                        file_name=f"booking_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(
                        f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to the API: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "© 2025 LLM Booking Analysis | Built with Streamlit and FastAPI"
    "</div>",
    unsafe_allow_html=True
)
