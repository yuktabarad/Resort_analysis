import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib
from huggingface_hub import hf_hub_download
from datetime import date, timedelta

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="HotelIQ | Hotel Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #FFFFFF;
    color: #26344A;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #F0F2F6;
    border-right: 1px solid #E0E3E9;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* Sidebar branding */

.brand {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 5px;
    background: linear-gradient(
        90deg,
        #FF4B4B,
        #F39C42,
        #159D91,
        #3977D5
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-subtitle {
    font-size: 12px;
    color: #718096;
    margin-bottom: 30px;
}

/* Main heading */

.main-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(
        90deg,
        #FF4B4B,
        #F39C42,
        #159D91,
        #3977D5,
        #6842A6
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.main-subtitle {
    color: #718096;
    font-size: 15px;
    margin-bottom: 30px;
}

/* Navigation radio buttons */

[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    font-weight: 500;
}

/* Divider */

hr {
    border-color: #DDE1E7;
}

/* Buttons */

.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #E83E3E;
    color: white;
}

/* Footer */

.footer {
    color: #9AA3B2;
    font-size: 12px;
    text-align: center;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


    # ----------------------------------
# LOAD CLEANED DATASET
# ----------------------------------

@st.cache_data
def load_data():

    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "hotel_bookings_cleaned.csv"
    )

    df = pd.read_csv(data_path)

    return df


df = load_data()

# ----------------------------------
# LOAD TRAINED ML MODEL
# ----------------------------------

@st.cache_resource
def load_model():

    local_model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "hotel_cancellation_model.pkl"
    )

    if local_model_path.exists():
        model_artifact = joblib.load(local_model_path)
    else:
        model_path = hf_hub_download(
            repo_id="yuktabarad/HotelIQ-Cancellation-Model",
            filename="hotel_cancellation_model.pkl"
        )
        model_artifact = joblib.load(model_path)

    return model_artifact




# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">🏨 HotelIQ</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'HOTEL INTELLIGENCE PLATFORM'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")

    selected_page = st.radio(
        "Go to",
        [
            "ℹ️ About HotelIQ",
            "🏠 Hotel Overview",
            "📊 Booking Insights",
            "🔍 Cancellation Risk",
            "⚠️ High-Risk Bookings"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model Status")

    st.success("Model repository configured")

    st.divider()

    st.markdown(
        '<div class="footer">HotelIQ © 2026</div>',
        unsafe_allow_html=True
    )


# ----------------------------------
# MAIN CONTENT
# ----------------------------------
if selected_page == "🏠 Hotel Overview":

    st.markdown(
        '<div class="main-title">HotelIQ Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Your hotel performance and booking intelligence at a glance.'
        '</div>',
        unsafe_allow_html=True
    )

        # ----------------------------------
    # DASHBOARD FILTERS
    # ----------------------------------

    st.subheader("Filter Dashboard")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_year = st.selectbox(
            "Select Arrival Year",
            ["All Years"] + sorted(df["arrival_date_year"].unique().tolist())
        )

    with filter_col2:
        selected_hotel = st.selectbox(
            "Select Hotel Type",
            ["All Hotels"] + sorted(df["hotel"].unique().tolist())
        )

    # Apply filters

    dashboard_df = df.copy()

    if selected_year != "All Years":
        dashboard_df = dashboard_df[
            dashboard_df["arrival_date_year"] == selected_year
        ]

    if selected_hotel != "All Hotels":
        dashboard_df = dashboard_df[
            dashboard_df["hotel"] == selected_hotel
        ]

    st.divider()

    # Calculate KPIs

    total_bookings = len(dashboard_df)

    canceled_bookings = int(dashboard_df["is_canceled"].sum())

    not_canceled = total_bookings - canceled_bookings

    cancellation_rate = (
        canceled_bookings / total_bookings
    ) * 100

    average_lead_time = dashboard_df["lead_time"].mean()

    average_adr = dashboard_df["adr"].mean()

    # Display KPI cards

    st.subheader("Hotel Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Bookings",
            value=f"{total_bookings:,}"
        )

    with col2:
        st.metric(
            label="Cancelled Bookings",
            value=f"{canceled_bookings:,}"
        )

    with col3:
        st.metric(
            label="Not Cancelled",
            value=f"{not_canceled:,}"
        )

    with col4:
        st.metric(
            label="Cancellation Rate",
            value=f"{cancellation_rate:.2f}%"
        )

    with col5:
        st.metric(
            label="Average Lead Time",
            value=f"{average_lead_time:.1f} days"
        )

    st.divider()

    st.subheader("Booking Overview")

    st.write(
        f"Average Daily Rate (ADR): **{average_adr:.2f}**"
    )

    st.divider()

    # ----------------------------------
    # BOOKING ANALYSIS
    # ----------------------------------

    st.subheader("Booking Analytics")

    col1, col2 = st.columns(2)

    # ----------------------------------
    # 1. BOOKINGS BY HOTEL TYPE
    # ----------------------------------

    hotel_bookings = (
        dashboard_df["hotel"]
        .value_counts()
        .reset_index()
    )

    hotel_bookings.columns = ["Hotel Type", "Bookings"]

    with col1:

        fig_hotel = px.bar(
            hotel_bookings,
            x="Hotel Type",
            y="Bookings",
            color="Hotel Type",
            title="Bookings by Hotel Type",
            text_auto=True,
            color_discrete_sequence=["#3977D5", "#159D91"]
        )

        fig_hotel.update_layout(
            showlegend=False,
            height=380,
            template="plotly_white"
        )

        st.plotly_chart(fig_hotel, use_container_width=True)

    # ----------------------------------
    # 2. CANCELLATION DISTRIBUTION
    # ----------------------------------

    cancellation_data = pd.DataFrame({
        "Booking Status": ["Not Cancelled", "Cancelled"],
        "Bookings": [not_canceled, canceled_bookings]
    })

    with col2:

        fig_cancellation = px.pie(
            cancellation_data,
            names="Booking Status",
            values="Bookings",
            title="Booking Cancellation Distribution",
            hole=0.55,
            color="Booking Status",
            color_discrete_map={
                "Not Cancelled": "#159D91",
                "Cancelled": "#FF4B4B"
            }
        )

        fig_cancellation.update_layout(
            height=380,
            template="plotly_white"
        )

        st.plotly_chart(
            fig_cancellation,
            use_container_width=True
        )

    st.divider()

    # ----------------------------------
    # 3. MONTHLY BOOKING TRENDS
    # ----------------------------------

    st.subheader("Monthly Booking Performance")

    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    monthly_bookings = (
        dashboard_df.groupby("arrival_date_month")
        .size()
        .reindex(month_order)
        .reset_index()
    )

    monthly_bookings.columns = ["Month", "Bookings"]

    fig_monthly = px.line(
        monthly_bookings,
        x="Month",
        y="Bookings",
        markers=True,
        title="Monthly Booking Trends",
        color_discrete_sequence=["#3977D5"]
    )

    fig_monthly.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title="Arrival Month",
        yaxis_title="Number of Bookings"
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    # ----------------------------------
    # 4. MONTHLY CANCELLATION RATE
    # ----------------------------------

    monthly_cancellation = (
        dashboard_df.groupby("arrival_date_month")
        ["is_canceled"]
        .mean()
        .mul(100)
        .reindex(month_order)
        .reset_index()
    )

    monthly_cancellation.columns = [
        "Month",
        "Cancellation Rate"
    ]

    fig_monthly_cancel = px.bar(
        monthly_cancellation,
        x="Month",
        y="Cancellation Rate",
        title="Monthly Cancellation Rate",
        text_auto=".2f",
        color="Cancellation Rate",
        color_continuous_scale=["#159D91", "#F39C42", "#FF4B4B"]
    )

    fig_monthly_cancel.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title="Arrival Month",
        yaxis_title="Cancellation Rate (%)",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig_monthly_cancel,
        use_container_width=True
    )

elif selected_page == "📊 Booking Insights":

    st.markdown(
        '<div class="main-title">Booking Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Explore booking patterns, customer behaviour and cancellation trends.'
        '</div>',
        unsafe_allow_html=True
    )

    # ----------------------------------
    # FILTERS
    # ----------------------------------

    st.subheader("Explore Booking Data")

    col1, col2 = st.columns(2)

    with col1:
        insight_year = st.selectbox(
            "Select Year",
            ["All Years"] + sorted(df["arrival_date_year"].unique().tolist()),
            key="insight_year"
        )

    with col2:
        insight_hotel = st.selectbox(
            "Select Hotel",
            ["All Hotels"] + sorted(df["hotel"].unique().tolist()),
            key="insight_hotel"
        )

    insights_df = df.copy()

    if insight_year != "All Years":
        insights_df = insights_df[
            insights_df["arrival_date_year"] == insight_year
        ]

    if insight_hotel != "All Hotels":
        insights_df = insights_df[
            insights_df["hotel"] == insight_hotel
        ]

    st.divider()

    # ----------------------------------
    # 1. CANCELLATION BY MARKET SEGMENT
    # ----------------------------------

    st.subheader("Cancellation by Market Segment")

    segment_data = (
        insights_df.groupby("market_segment")
        .agg(
            Total_Bookings=("is_canceled", "count"),
            Cancellation_Rate=("is_canceled", "mean")
        )
        .reset_index()
    )

    segment_data["Cancellation_Rate"] *= 100

    segment_data = segment_data[
        segment_data["Total_Bookings"] >= 100
    ]

    fig_segment = px.bar(
        segment_data,
        x="market_segment",
        y="Cancellation_Rate",
        color="Cancellation_Rate",
        text_auto=".2f",
        title="Cancellation Rate by Market Segment",
        color_continuous_scale=["#159D91", "#F39C42", "#FF4B4B"]
    )

    fig_segment.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Market Segment",
        yaxis_title="Cancellation Rate (%)",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_segment, use_container_width=True)

    st.caption(
        "Segments with fewer than 100 bookings are excluded from this chart."
    )

    st.divider()

    # ----------------------------------
    # 2. CANCELLATION BY CUSTOMER TYPE
    # ----------------------------------

    st.subheader("Cancellation by Customer Type")

    customer_data = (
        insights_df.groupby("customer_type")
        .agg(
            Total_Bookings=("is_canceled", "count"),
            Cancellation_Rate=("is_canceled", "mean")
        )
        .reset_index()
    )

    customer_data["Cancellation_Rate"] *= 100

    fig_customer = px.bar(
        customer_data,
        x="customer_type",
        y="Cancellation_Rate",
        color="customer_type",
        text_auto=".2f",
        title="Cancellation Rate by Customer Type",
        color_discrete_sequence=[
            "#3977D5",
            "#159D91",
            "#F39C42",
            "#6842A6"
        ]
    )

    fig_customer.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False,
        xaxis_title="Customer Type",
        yaxis_title="Cancellation Rate (%)"
    )

    st.plotly_chart(fig_customer, use_container_width=True)

    st.divider()

    # ----------------------------------
    # 3. CANCELLATION BY LEAD TIME
    # ----------------------------------

    st.subheader("Cancellation by Booking Lead Time")

    lead_time_data = insights_df.copy()

    lead_time_data["Lead Time Group"] = pd.cut(
        lead_time_data["lead_time"],
        bins=[-1, 30, 90, 180, 365, float("inf")],
        labels=[
            "0-30 Days",
            "31-90 Days",
            "91-180 Days",
            "181-365 Days",
            "Above 365 Days"
        ]
    )

    lead_data = (
        lead_time_data.groupby("Lead Time Group", observed=False)
        ["is_canceled"]
        .mean()
        .mul(100)
        .reset_index()
    )

    lead_data.columns = ["Lead Time Group", "Cancellation Rate"]

    fig_lead = px.bar(
        lead_data,
        x="Lead Time Group",
        y="Cancellation Rate",
        color="Cancellation Rate",
        text_auto=".2f",
        title="Cancellation Rate by Lead Time",
        color_continuous_scale=["#159D91", "#F39C42", "#FF4B4B"]
    )

    fig_lead.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Booking Lead Time",
        yaxis_title="Cancellation Rate (%)",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_lead, use_container_width=True)

    st.divider()

    # ----------------------------------
    # 4. ADR BY HOTEL TYPE
    # ----------------------------------

    st.subheader("Average Daily Rate by Hotel")

    adr_data = (
        insights_df.groupby("hotel")["adr"]
        .mean()
        .reset_index()
    )

    adr_data.columns = ["Hotel Type", "Average ADR"]

    fig_adr = px.bar(
        adr_data,
        x="Hotel Type",
        y="Average ADR",
        color="Hotel Type",
        text_auto=".2f",
        title="Average Daily Rate by Hotel Type",
        color_discrete_sequence=["#3977D5", "#159D91"]
    )

    fig_adr.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False,
        xaxis_title="Hotel Type",
        yaxis_title="Average Daily Rate"
    )

    st.plotly_chart(fig_adr, use_container_width=True)

    st.caption(
        "ADR represents the average daily room rate recorded in the dataset, "
        "not actual hotel revenue."
    )
    
elif selected_page == "🔍 Cancellation Risk":

    st.markdown(
        '<div class="main-title">Cancellation Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Evaluate the cancellation risk of an individual hotel booking.'
        '</div>',
        unsafe_allow_html=True
    )

    # Load trained model

    with st.spinner("Connecting to prediction model..."):
        model_artifact = load_model()

    rf_model = model_artifact["model"]
    best_threshold = model_artifact["threshold"]

    st.success("Prediction service is ready!")

    st.info(
        "This model was trained on historical hotel booking data from 2015–2017. "
        "Its performance on current or future booking patterns is not guaranteed."
    )

    st.divider()

    # ----------------------------------
    # BOOKING FORM
    # ----------------------------------

    st.subheader("Enter Booking Information")

    st.write(
        "Please provide the booking details below "
        "to assess its cancellation risk."
    )

    with st.form("booking_prediction_form"):

        # ----------------------------------
        # 1. BOOKING DETAILS
        # ----------------------------------

        st.markdown("### 🏨 Booking Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            hotel = st.selectbox(
                "Hotel Type",
                sorted(df["hotel"].unique())
            )

            booking_date = st.date_input(
                "Booking Date",
                value=date.today(),
                min_value=date.today()
            )

        with col2:

            arrival_date = st.date_input(
                "Arrival Date",
                value=date.today() + timedelta(days=30),
                min_value=booking_date
            )

            lead_time = (arrival_date - booking_date).days

            st.metric("Booking Lead Time", f"{lead_time} days")

            st.caption(
                f"Arrival: {arrival_date.strftime('%d %B %Y')} "
                f"(Week {arrival_date.isocalendar().week})"
            )

        with col3:

            stays_in_weekend_nights = st.number_input(
                "Weekend Nights",
                min_value=0,
                max_value=30,
                value=1
            )

            stays_in_week_nights = st.number_input(
                "Week Nights",
                min_value=0,
                max_value=60,
                value=2
            )

        # ----------------------------------
        # 2. GUEST DETAILS
        # ----------------------------------

        st.divider()

        st.markdown("### 👥 Guest Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            adults = st.number_input(
                "Adults",
                min_value=0,
                max_value=20,
                value=2
            )

            children = st.number_input(
                "Children",
                min_value=0,
                max_value=10,
                value=0
            )

        with col2:

            babies = st.number_input(
                "Babies",
                min_value=0,
                max_value=10,
                value=0
            )

            country = st.selectbox(
                "Guest Country",
                sorted(df["country"].unique())
            )

        with col3:

            customer_type = st.selectbox(
                "Customer Type",
                sorted(df["customer_type"].unique())
            )

            is_repeated_guest = st.selectbox(
                "Returning Guest?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        # ----------------------------------
        # 3. BOOKING HISTORY
        # ----------------------------------

        st.divider()

        st.markdown("### 📋 Booking History")

        col1, col2, col3 = st.columns(3)

        with col1:

            previous_cancellations = st.number_input(
                "Previous Cancellations",
                min_value=0,
                max_value=100,
                value=0
            )

            previous_bookings_not_canceled = st.number_input(
                "Previous Successful Bookings",
                min_value=0,
                max_value=100,
                value=0
            )

        with col2:

            deposit_type = st.selectbox(
                "Deposit Type",
                sorted(df["deposit_type"].unique())
            )

            reserved_room_type = st.selectbox(
                "Reserved Room Type",
                sorted(df["reserved_room_type"].unique())
            )

        with col3:

            meal = st.selectbox(
                "Meal Plan",
                sorted(df["meal"].unique())
            )

            agent = st.selectbox(
                "Booking Agent ID",
                sorted(df["agent"].unique()),
                format_func=lambda x: (
                    "Unknown" if x == -1 else str(int(x))
                )
            )

        # ----------------------------------
        # 4. CHANNEL AND PRICING
        # ----------------------------------

        st.divider()

        st.markdown("### 💳 Booking Channel & Preferences")

        col1, col2, col3 = st.columns(3)

        with col1:

            market_segment = st.selectbox(
                "Market Segment",
                sorted(df["market_segment"].unique())
            )

            distribution_channel = st.selectbox(
                "Distribution Channel",
                sorted(df["distribution_channel"].unique())
            )

        with col2:

            adr = st.number_input(
                "Average Daily Rate (ADR)",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                step=5.0
            )

            required_car_parking_spaces = st.number_input(
                "Required Parking Spaces",
                min_value=0,
                max_value=10,
                value=0
            )

        with col3:

            total_of_special_requests = st.number_input(
                "Special Requests",
                min_value=0,
                max_value=10,
                value=0
            )

        # ----------------------------------
        # SUBMIT BUTTON
        # ----------------------------------

        st.divider()

        submitted = st.form_submit_button(
            "Assess Cancellation Risk",
            use_container_width=True
        )

    # ----------------------------------
    # MAKE PREDICTION
    # ----------------------------------

    if submitted:

        if arrival_date < booking_date:
            st.error("Arrival date cannot be earlier than the booking date.")
            st.stop()

        lead_time = (arrival_date - booking_date).days

        booking_data = pd.DataFrame([{

            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_month": arrival_date.strftime("%B"),
            "arrival_date_week_number": arrival_date.isocalendar().week,
            "arrival_date_day_of_month": arrival_date.day,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type": reserved_room_type,
            "deposit_type": deposit_type,
            "agent": agent,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests

        }])

        # Validate guest information

        if adults + children + babies == 0:

            st.error("Please enter at least one guest.")

        else:

            cancellation_probability = rf_model.predict_proba(
                booking_data
            )[0][1]

            predicted_class = (
                cancellation_probability >= best_threshold
            )

            risk_percentage = cancellation_probability * 100

            st.divider()

            st.subheader("Cancellation Risk Assessment")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "Cancellation Probability",
                    f"{risk_percentage:.2f}%"
                )

            with result_col2:

                if predicted_class:

                    st.error("Higher Cancellation Risk")

                else:

                    st.success("Lower Cancellation Risk")

            st.progress(float(cancellation_probability))

            if predicted_class:

                st.warning(
                    "This booking has a higher predicted cancellation "
                    "risk. Consider reviewing its confirmation status."
                )

            else:

                st.success(
                    "This booking has a lower predicted cancellation "
                    "risk based on the available information."
                )

            st.caption(
                "This prediction is an estimate based on historical "
                "booking patterns. It does not guarantee the actual "
                "booking outcome."
            )

elif selected_page == "⚠️ High-Risk Bookings":

    st.markdown(
        '<div class="main-title">High-Risk Bookings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Identify bookings with a higher predicted cancellation risk.'
        '</div>',
        unsafe_allow_html=True
    )

    # Load trained model
    model_artifact = load_model()

    rf_model = model_artifact["model"]
    threshold = model_artifact["threshold"]
    feature_columns = model_artifact["feature_columns"]

    # ----------------------------------
    # PREDICT CANCELLATION RISK
    # ----------------------------------

    if "risk_scored_df" not in st.session_state:

        with st.spinner("Assessing cancellation risk for bookings..."):

            risk_df = df.copy()

            probabilities = rf_model.predict_proba(
                risk_df[feature_columns]
            )[:, 1]

            risk_df["cancellation_probability"] = probabilities

            risk_df["risk_status"] = risk_df[
                "cancellation_probability"
            ].apply(
                lambda x: "High Risk" if x >= threshold else "Low Risk"
            )

            st.session_state["risk_scored_df"] = risk_df

    risk_df = st.session_state["risk_scored_df"]

    # ----------------------------------
    # FILTERS
    # ----------------------------------

    st.subheader("Filter Bookings")

    col1, col2 = st.columns(2)

    with col1:

        selected_hotel = st.selectbox(
            "Hotel Type",
            ["All Hotels"] + sorted(df["hotel"].unique().tolist())
        )

    with col2:

        selected_risk = st.selectbox(
            "Risk Category",
            ["High Risk", "Low Risk", "All Bookings"]
        )

    filtered_df = risk_df.copy()

    if selected_hotel != "All Hotels":

        filtered_df = filtered_df[
            filtered_df["hotel"] == selected_hotel
        ]

    if selected_risk != "All Bookings":

        filtered_df = filtered_df[
            filtered_df["risk_status"] == selected_risk
        ]

    st.divider()

    # ----------------------------------
    # HIGH-RISK BOOKINGS
    # ----------------------------------

    high_risk_df = filtered_df[
        filtered_df["cancellation_probability"] >= threshold
    ]

    total_bookings = len(filtered_df)

    high_risk_count = len(high_risk_df)

    high_risk_percentage = (
        high_risk_count / total_bookings * 100
        if total_bookings > 0 else 0
    )

    average_probability = (
        filtered_df["cancellation_probability"].mean() * 100
        if total_bookings > 0 else 0
    )

    # ----------------------------------
    # KPI CARDS
    # ----------------------------------

    st.subheader("Cancellation Risk Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Bookings Assessed",
        f"{total_bookings:,}"
    )

    col2.metric(
        "High-Risk Bookings",
        f"{high_risk_count:,}"
    )

    col3.metric(
        "High-Risk Percentage",
        f"{high_risk_percentage:.2f}%"
    )

    col4.metric(
        "Average Risk Probability",
        f"{average_probability:.2f}%"
    )

    st.divider()
    st.subheader("Booking Details")

    if filtered_df.empty:
        st.info("No bookings found for the selected filters.")

    else:
        display_columns = [
            "hotel", "lead_time", "arrival_date_month",
            "adults", "children", "market_segment",
            "deposit_type", "adr",
            "cancellation_probability", "is_canceled"
        ]

        display_df = filtered_df[display_columns].copy()

        display_df["cancellation_probability"] = (
            display_df["cancellation_probability"] * 100
        ).round(2)

        display_df["is_canceled"] = display_df["is_canceled"].map({
            0: "Not Cancelled",
            1: "Cancelled"
        })

        display_df = display_df.sort_values(
            "cancellation_probability",
            ascending=False
        )

        display_df = display_df.rename(columns={
            "hotel": "Hotel Type",
            "lead_time": "Lead Time (Days)",
            "arrival_date_month": "Arrival Month",
            "adults": "Adults",
            "children": "Children",
            "market_segment": "Market Segment",
            "deposit_type": "Deposit Type",
            "adr": "ADR",
            "cancellation_probability": "Cancellation Risk (%)",
            "is_canceled": "Actual Booking Status"
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = display_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Booking Details",
            data=csv_data,
            file_name="hoteliq_booking_details.csv",
            mime="text/csv"
        )

    if high_risk_df.empty:

        st.success("No high-risk bookings found for the selected filters.")

    else:

        display_columns = [
            "hotel",
            "lead_time",
            "arrival_date_month",
            "adults",
            "children",
            "market_segment",
            "deposit_type",
            "adr",
            "cancellation_probability",
            "is_canceled"
        ]

        display_df = high_risk_df[display_columns].copy()

        display_df["cancellation_probability"] = (
            display_df["cancellation_probability"] * 100
        ).round(2)

        display_df["is_canceled"] = display_df[
            "is_canceled"
        ].map({
            0: "Not Cancelled",
            1: "Cancelled"
        })

        display_df = display_df.sort_values(
            "cancellation_probability",
            ascending=False
        )

        display_df = display_df.rename(columns={
            "hotel": "Hotel Type",
            "lead_time": "Lead Time (Days)",
            "arrival_date_month": "Arrival Month",
            "adults": "Adults",
            "children": "Children",
            "market_segment": "Market Segment",
            "deposit_type": "Deposit Type",
            "adr": "ADR",
            "cancellation_probability": "Cancellation Risk (%)",
            "is_canceled": "Actual Booking Status"
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------
        # DOWNLOAD RESULTS
        # ----------------------------------

        csv_data = display_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download High-Risk Bookings",
            data=csv_data,
            file_name="hoteliq_high_risk_bookings.csv",
            mime="text/csv"
        )

    st.divider()

    st.caption(
        f"Bookings are classified as high risk when their predicted "
        f"cancellation probability is at least {threshold * 100:.0f}%. "
        "These are historical bookings scored by the model; actual "
        "booking status is shown separately for comparison."
    )

elif selected_page == "ℹ️ About HotelIQ":

    st.markdown(
        '<div class="main-title">How HotelIQ Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Understanding the technology behind hotel booking analytics '
        'and cancellation predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Project Introduction
    # About HotelIQ
    st.subheader("🏨 About HotelIQ")

    st.markdown("""
    **HotelIQ is a Hotel Booking Analytics and Cancellation Intelligence
    Platform** developed to explore hotel booking data, understand customer
    booking behavior, and identify reservations that may have a higher
    probability of cancellation.

    The platform combines data analytics, statistical analysis, SQL, and
    machine learning to turn raw hotel booking records into meaningful
    business insights.

    Hotel cancellations can affect hotel occupancy planning, room
    availability, and business operations. HotelIQ helps explore these
    patterns and provides a machine learning-based estimate of cancellation
    risk for individual bookings.
    """)

    st.markdown("### 🎯 What can you explore with HotelIQ?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📊 Hotel Overview**

        Get an overall picture of hotel booking performance through
        important metrics such as total bookings, cancellation rates,
        hotel distribution, and booking trends.

        **🔎 Booking Insights**

        Explore booking patterns by hotel type, customer type,
        market segment, lead time, and average daily rate (ADR).
        """)

    with col2:
        st.markdown("""
        **🤖 Cancellation Risk Prediction**

        Enter booking details and use the trained machine learning
        model to estimate the probability that a booking will be cancelled.

        **⚠️ High-Risk Bookings**

        Explore historical bookings classified as high risk by the
        model and compare predicted risk with their actual booking status.
        """)

    st.info(
        "HotelIQ is an analytical and predictive platform. "
        "Its predictions are estimates based on historical booking data, "
        "not guaranteed cancellation outcomes."
    )

    st.divider()

    # Workflow
    st.subheader("⚙️ How the Platform Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. Data Preparation")
        st.write(
            """
            - Analyzed the original hotel booking dataset.
            - Handled missing values and duplicate records.
            - Cleaned and prepared 87,395 booking records.
            """
        )

        st.markdown("### 2. Exploratory Data Analysis")
        st.write(
            """
            - Studied hotel booking patterns.
            - Analyzed cancellation trends.
            - Compared hotel types and customer segments.
            """

        )

    with col2:
        st.markdown("### 3. Statistical Analysis")
        st.write(
            """
            - Applied hypothesis testing.
            - Examined relationships between booking variables.
            - Evaluated statistically significant differences.
            """
        )

        st.markdown("### 4. Machine Learning")
        st.write(
            """
            - Built a Random Forest classification model.
            - Used 24 selected input features.
            - Generated cancellation probabilities.
            - Classified bookings using a 30% threshold.
            """
        )

    st.divider()

    # Prediction Process
    st.subheader("🤖 Cancellation Prediction Process")

    st.write(
        """
        When a user enters booking information in the Cancellation Risk
        section, HotelIQ processes the information using its trained
        machine learning model.
        """
    )

    st.markdown(
        """
        **Prediction workflow:**

        1. User enters booking details.
        2. Input data is prepared for the model.
        3. The Random Forest model calculates cancellation probability.
        4. The probability is compared with the selected threshold.
        5. HotelIQ displays the predicted cancellation risk.
        """
    )

    st.info(
        "Bookings with a predicted cancellation probability of 30% or "
        "higher are classified as High Risk."
    )

    st.divider()

    # Technologies
    st.subheader("💻 Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Matplotlib",
        "Seaborn",
        "SQL Server",
        "Scikit-learn",
        "Random Forest",
        "Streamlit",
        "Plotly",
        "Hugging Face"
    ]

    cols = st.columns(4)

    for index, technology in enumerate(technologies):
        with cols[index % 4]:
            st.markdown(f"- {technology}")

    st.divider()

    # Limitations
    st.subheader("⚠️ Project Limitations")

    st.markdown(
        """
        - The dataset contains historical booking records from 2015–2017.
        - Predictions are estimates and are not guaranteed outcomes.
        - The model has not been validated on a live hotel booking system.
        - The platform does not connect to real-time hotel reservation
          systems.
        - ADR represents Average Daily Rate, not confirmed actual revenue.
        """
    )

    st.divider()

    st.markdown(
        """
        <div style="text-align:center; padding:15px;">
            <h3>HotelIQ – Hotel Intelligence Platform</h3>
            <p>Turning hotel booking data into meaningful insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )