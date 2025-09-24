import pandas as pd
import joblib
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="🏡 Pakistan House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        font-family: 'Arial', sans-serif;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: 900;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4A148C 0%, #7B1FA2 100%);
        min-width: 280px !important;
        max-width: 280px !important;
    }
    
    .sidebar-header {
        color: white;
        text-align: center;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        padding-top: 1rem;
        font-weight: bold;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-section h3 {
        color: #FFC107;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding-bottom: 0.3rem;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton>button {
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    
    .predict-btn {
        background: linear-gradient(to right, #FF5722, #E64A19);
    }
    
    /* Login/Logout buttons */
    .login-btn {
        background: linear-gradient(to right, #4CAF50, #2E7D32) !important;
    }
    
    .logout-btn {
        background: linear-gradient(to right, #F44336, #D32F2F) !important;
    }
    
    /* Plus/Minus buttons */
    .number-input-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .number-input-btn {
        background: #5E35B1;
        color: white;
        border: none;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        font-weight: bold;
        margin: 0 10px;
        cursor: pointer;
    }
    
    .number-input-value {
        font-weight: bold;
        min-width: 30px;
        text-align: center;
    }
    
    /* Flexbox for amenities */
    .amenities-flex {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: space-between;
    }
    
    .amenity-item {
        flex: 1 0 45%;
        min-width: 120px;
        margin-bottom: 10px;
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Price text */
    .price-text {
        font-size: 3rem;
        color: #1B5E20;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin: 0;
    }
    
    .pkr-text {
        font-size: 1.5rem;
        color: #388E3C;
        margin-top: 0.5rem;
        font-weight: bold;
    }
    
    /* Cards */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        height: 100%;
    }
    
    .card-header {
        color: #5E35B1;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #D1C4E9;
        padding-bottom: 0.5rem;
    }
    
    /* About section */
    .about-container {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .about-header {
        color: #F57F17;
        font-size: 1.5rem;
        font-weight: bold;
        border-bottom: 2px solid #FFD54F;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .about-content {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }
    
    .about-item {
        flex: 1 0 45%;
        min-width: 200px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        color: #0D47A1;
        text-align: center;
        margin: 2rem 0 1.5rem 0;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #757575;
        margin-top: 2rem;
        padding: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
try:
    model = joblib.load("house_price_model.pkl")
except:
    st.error("Model file 'house_price_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Feature columns expected during training
feature_cols = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning",
    "parking", "prefarea", "furnishingstatus"
]

# Encoding mappings (must match training preprocessing)
binary_map = {"yes": 1, "no": 0}
furnishing_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}

# Session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Header section - Bigger and Bolder
st.markdown('<p class="main-header">🏡 PAKISTAN HOUSING PRICE PREDICTION</p>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("<h2 class='sidebar-header'>Enter House Details</h2>", unsafe_allow_html=True)
    
    # Login/Logout buttons in sidebar
    if not st.session_state.authenticated:
        if st.button("🔐 Login", key="login_btn", use_container_width=True):
            st.session_state.authenticated = True
    else:
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            st.session_state.authenticated = False
    
    st.markdown("---")
    
    # Property specifications
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<h3>📐 PROPERTY SPECS</h3>", unsafe_allow_html=True)
    
    # Area input
    area = st.slider("**Area (sq ft)**", 500, 20000, 5000, help="Total area of the property in square feet")
    
    # Bedrooms with plus/minus buttons
    st.markdown("**Bedrooms**")
    bedrooms_col1, bedrooms_col2, bedrooms_col3 = st.columns([1, 2, 1])
    with bedrooms_col1:
        if st.button("➖", key="bed_minus"):
            if 'bedrooms' not in st.session_state or st.session_state.bedrooms > 1:
                st.session_state.bedrooms = getattr(st.session_state, 'bedrooms', 3) - 1
    with bedrooms_col2:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, key="bedrooms", label_visibility="collapsed")
    with bedrooms_col3:
        if st.button("➕", key="bed_plus"):
            if 'bedrooms' not in st.session_state or st.session_state.bedrooms < 10:
                st.session_state.bedrooms = getattr(st.session_state, 'bedrooms', 3) + 1
    
    # Bathrooms with plus/minus buttons
    st.markdown("**Bathrooms**")
    bath_col1, bath_col2, bath_col3 = st.columns([1, 2, 1])
    with bath_col1:
        if st.button("➖", key="bath_minus"):
            if 'bathrooms' not in st.session_state or st.session_state.bathrooms > 1:
                st.session_state.bathrooms = getattr(st.session_state, 'bathrooms', 2) - 1
    with bath_col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, key="bathrooms", label_visibility="collapsed")
    with bath_col3:
        if st.button("➕", key="bath_plus"):
            if 'bathrooms' not in st.session_state or st.session_state.bathrooms < 5:
                st.session_state.bathrooms = getattr(st.session_state, 'bathrooms', 2) + 1
    
    # Stories with plus/minus buttons
    st.markdown("**Stories**")
    stories_col1, stories_col2, stories_col3 = st.columns([1, 2, 1])
    with stories_col1:
        if st.button("➖", key="stories_minus"):
            if 'stories' not in st.session_state or st.session_state.stories > 1:
                st.session_state.stories = getattr(st.session_state, 'stories', 2) - 1
    with stories_col2:
        stories = st.number_input("Stories", min_value=1, max_value=5, value=2, key="stories", label_visibility="collapsed")
    with stories_col3:
        if st.button("➕", key="stories_plus"):
            if 'stories' not in st.session_state or st.session_state.stories < 5:
                st.session_state.stories = getattr(st.session_state, 'stories', 2) + 1
    
    # Parking with plus/minus buttons
    st.markdown("**Parking Spaces**")
    parking_col1, parking_col2, parking_col3 = st.columns([1, 2, 1])
    with parking_col1:
        if st.button("➖", key="parking_minus"):
            if 'parking' not in st.session_state or st.session_state.parking > 0:
                st.session_state.parking = getattr(st.session_state, 'parking', 1) - 1
    with parking_col2:
        parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1, key="parking", label_visibility="collapsed")
    with parking_col3:
        if st.button("➕", key="parking_plus"):
            if 'parking' not in st.session_state or st.session_state.parking < 5:
                st.session_state.parking = getattr(st.session_state, 'parking', 1) + 1
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Amenities - using flexbox layout
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<h3>🏠 AMENITIES</h3>", unsafe_allow_html=True)
    
    # Create flex container for amenities
    st.markdown("<div class='amenities-flex'>", unsafe_allow_html=True)
    
    # First column of amenities
    st.markdown("<div class='amenity-item'>", unsafe_allow_html=True)
    mainroad = st.selectbox("**Main Road**", ["yes", "no"], key="mainroad")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='amenity-item'>", unsafe_allow_html=True)
    guestroom = st.selectbox("**Guest Room**", ["yes", "no"], key="guestroom")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='amenity-item'>", unsafe_allow_html=True)
    basement = st.selectbox("**Basement**", ["yes", "no"], key="basement")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Second column of amenities
    st.markdown("<div class='amenity-item'>", unsafe_allow_html=True)
    hotwaterheating = st.selectbox("**Hot Water**", ["yes", "no"], key="hotwater")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='amenity-item'>", unsafe_allow_html=True)
    airconditioning = st.selectbox("**AC**", ["yes", "no"], key="ac")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='amenity-item'>", unsafe_allow_html=True)
    prefarea = st.selectbox("**Pref Area**", ["yes", "no"], key="prefarea")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close amenities-flex
    st.markdown("</div>", unsafe_allow_html=True)  # Close sidebar-section
    
    # Furnishing
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<h3>🛋️ FURNISHING</h3>", unsafe_allow_html=True)
    furnishingstatus = st.selectbox("**Furnishing Status**", ["furnished", "semi-furnished", "unfurnished"], key="furnishing")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button
    predict_button = st.button("🚀 PREDICT PRICE IN PKR", use_container_width=True, help="Click to calculate the estimated price of your property")

# Prepare input data
input_data = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": binary_map[mainroad],
    "guestroom": binary_map[guestroom],
    "basement": binary_map[basement],
    "hotwaterheating": binary_map[hotwaterheating],
    "airconditioning": binary_map[airconditioning],
    "parking": parking,
    "prefarea": binary_map[prefarea],
    "furnishingstatus": furnishing_map[furnishingstatus]
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])
input_df = input_df[feature_cols]  # Ensure column order

# Display input summary
st.markdown("---")
st.markdown("<h3 class='section-header'>📋 YOUR PROPERTY SUMMARY</h3>", unsafe_allow_html=True)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>PROPERTY DETAILS</div>", unsafe_allow_html=True)
    st.write(f"**Area:** {area} sq ft")
    st.write(f"**Bedrooms:** {bedrooms}")
    st.write(f"**Bathrooms:** {bathrooms}")
    st.write(f"**Stories:** {stories}")
    st.markdown("</div>", unsafe_allow_html=True)

with summary_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>AMENITIES</div>", unsafe_allow_html=True)
    st.write(f"**Main Road:** {mainroad.title()}")
    st.write(f"**Guest Room:** {guestroom.title()}")
    st.write(f"**Basement:** {basement.title()}")
    st.write(f"**Parking:** {parking} cars")
    st.markdown("</div>", unsafe_allow_html=True)

with summary_col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>ADDITIONAL FEATURES</div>", unsafe_allow_html=True)
    st.write(f"**Hot Water:** {hotwaterheating.title()}")
    st.write(f"**Air Conditioning:** {airconditioning.title()}")
    st.write(f"**Preferred Area:** {prefarea.title()}")
    st.write(f"**Furnishing:** {furnishingstatus.title()}")
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction
if predict_button:
    st.markdown("---")
    with st.spinner("🤖 Calculating your property value in Pakistani Rupees..."):
        try:
            prediction = model.predict(input_df)[0]
            
            # Convert to PKR (assuming the model was trained on USD, with conversion rate ~280 PKR per USD)
            conversion_rate = 280
            prediction_pkr = prediction * conversion_rate
            
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: #0D47A1; font-size: 2rem; font-weight: 800;'>PREDICTED HOUSE PRICE</h3>", unsafe_allow_html=True)
            st.markdown(f"<p class='price-text'>Rs. {prediction_pkr:,.0f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='pkr-text'>PAKISTANI RUPEES</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional info
            st.info("💡 This estimate is based on our machine learning model trained on historical housing data. Actual market price may vary based on current market conditions.")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# About section
st.markdown("---")
st.markdown("<div class='about-container'>", unsafe_allow_html=True)
st.markdown("<h3 class='about-header'>ABOUT</h3>", unsafe_allow_html=True)

about_col1, about_col2 = st.columns(2)

with about_col1:
    st.markdown("**DEVELOPER INFORMATION:**")
    st.markdown("- **Name:** Afsar Ali")
    st.markdown("- **Phone:** +92 300 1234567")
    st.markdown("- **Email:** afsar.ali@example.com")
    
with about_col2:
    st.markdown("**PROJECT DETAILS:**")
    st.markdown("- **Registration No:** FA20-BCS-001")
    st.markdown("- **Location:** Islamabad, Pakistan")
    st.markdown("- **University:** COMSATS University")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div class='footer'><p>This housing price predictor uses advanced machine learning to estimate property values in Pakistani Rupees.</p></div>", unsafe_allow_html=True)