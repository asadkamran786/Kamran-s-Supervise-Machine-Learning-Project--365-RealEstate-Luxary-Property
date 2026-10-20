# 365 Real Estate - Streamlit App for Residential Property Valuation using Linear Regression Model B (log-target) with AED Estimate, Confidence Range and Comparable Sales - Burj Khalifa Gold Luxury Edition

# Section 1: Import required libraries for ML model, data handling and Streamlit UI
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import pathlib

# Section 2: Configure Streamlit page with title, layout and luxury icon
st.set_page_config(page_title="365 Real Estate | Burj Khalifa Luxury Valuation", layout="wide", page_icon="🏙️")

# Section 3: Define luxury Gold/Black CSS with Burj Khalifa background
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;600&display=swap');
.stApp {
    background: linear-gradient(rgba(10,10,10,0.88), rgba(10,10,10,0.92)), url('https://images.unsplash.com/photo-1512453979798-5ea266f8880c?q=80&w=2090');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}
.block-container {
    background: rgba(15,15,15,0.94) !important;
    border: 1.5px solid #D4AF37 !important;
    border-radius: 28px !important;
    padding: 3rem 3.2rem !important;
    margin-top: 1.5rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 60px rgba(212,175,55,0.25), 0 25px 80px rgba(0,0,0,0.6) !important;
}
.header-container { display: flex; align-items: center; justify-content: center; gap: 22px; margin-bottom: 8px; }
.logo-box {
    width: 76px; height: 76px;
    background: linear-gradient(135deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
    border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 900; color: #0A0A0A;
    box-shadow: 0 10px 35px rgba(212,175,55,0.5);
}
.main-header {
    font-family: 'Playfair Display', serif;
    background: linear-gradient(90deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    font-size: 2.5rem; font-weight: 900; letter-spacing: 1px; margin: 0;
}
.sub-header-gold { text-align: center; font-size: 0.82rem; color: #C9B896; letter-spacing: 3px; font-weight: 600; margin-top: 4px; }
.sub-header-light { text-align: center; font-size: 1.05rem; color: #A0A0A0; margin-top: 6px; }
.badge { display: inline-block; background: #1A1A1A; border: 1px solid #D4AF37; color: #D4AF37; padding: 7px 16px; border-radius: 25px; font-size: 0.78rem; font-weight: 600; margin: 4px; }
.metric-card { background: #151515; border-left: 6px solid #D4AF37; border-top: 1px solid rgba(212,175,55,0.3); padding: 24px; border-radius: 14px; text-align: center; }
.range-card { background: #151515; border-left: 6px solid #B76E79; padding: 24px; border-radius: 14px; text-align: center; }
.price-card { background: #151515; border-left: 6px solid #E5E4E2; padding: 24px; border-radius: 14px; text-align: center; }
.stButton>button {
    background: linear-gradient(90deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
    color: #0A0A0A !important; border: none; padding: 16px 40px; border-radius: 35px;
    font-size: 1.15rem; font-weight: 800; letter-spacing: 2px; width: 100%;
    box-shadow: 0 12px 40px rgba(212,175,55,0.5);
}
</style>
""", unsafe_allow_html=True)

# Section 4: Display luxury header with logo, title and Burj Khalifa branding
st.markdown("""
<div class="header-container">
    <div class="logo-box">365</div>
    <div>
        <div class="main-header">365 Real Estate</div>
        <div class="sub-header-gold">LUXURY VALUATION • BURJ KHALIFA COLLECTION</div>
    </div>
</div>
<div class="sub-header-light">Dubai's Most Prestigious Properties • AI-Powered Elite Valuation Platform</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin: 20px 0 30px 0;">
    <span class="badge">◈ BURJ KHALIFA EDITION</span>
    <span class="badge">◈ 79,828 LUXURY SALES</span>
    <span class="badge">◈ MODEL B • LOG-TARGET</span>
    <span class="badge">◈ MAPE 16.62%</span>
</div>
""", unsafe_allow_html=True)

# Section 5: Load ML model and columns - handles missing CSV gracefully for Streamlit Cloud
@st.cache_resource
def load_assets():
    # Model files must exist in repo root
    model_path = pathlib.Path("linear_valuation_model.pkl")
    col_path = pathlib.Path("model_columns.json")
    
    if not model_path.exists():
        st.error("❌ Missing file: linear_valuation_model.pkl - Please upload it to GitHub repo root")
        st.stop()
    if not col_path.exists():
        st.error("❌ Missing file: model_columns.json - Please upload it to GitHub repo root")
        st.stop()
    
    model = joblib.load(model_path)
    with open(col_path) as f:
        model_cols = json.load(f)
    
    # Try to find CSV - if not found, create dummy comparables
    csv_candidates = ["final_cleaned_modeling.csv", "final-cleaned-modeling.csv", "cleaned_modeling.csv", "dld_transactions.csv"]
    df_ref = None
    for c in csv_candidates:
        if pathlib.Path(c).exists():
            try:
                df_ref = pd.read_csv(c)
                st.toast(f"✅ Loaded comparables from {c}")
                break
            except:
                continue
    
    if df_ref is None:
        # Create dummy comparable dataframe so app still works
        st.warning("⚠️ Comparable CSV not found in repo - Using sample data. Upload final_cleaned_modeling.csv to GitHub for real comparables.")
        df_ref = pd.DataFrame({
            'area_name_en': ['Business Bay', 'Downtown Dubai', 'Dubai Marina', 'Business Bay', 'JVC'],
            'master_project_en': ['Unknown']*5,
            'property_type_en': ['Flat', 'Flat', 'Flat', 'Flat', 'Flat'],
            'property_sub_type_en': ['1 B/R', '2 B/R', '1 B/R', '2 B/R', '2 B/R'],
            'rooms_en': ['1 B/R', '2 B/R', '1 B/R', '2 B/R', '2 B/R'],
            'area_sqft': [900, 1200, 1000, 1100, 1300],
            'year': [2024, 2024, 2023, 2024, 2024],
            'actual_worth': [1200000, 2100000, 1500000, 1800000, 1600000],
            'has_parking': [1,1,1,1,1],
            'reg_type_en': ['Existing Properties']*5
        })
    
    return model, model_cols, df_ref

model, model_cols, df_ref = load_assets()

# Section 6: Prepare dropdown options from reference data for property selection
# Use fixed lists if CSV is dummy to avoid empty dropdowns
if len(df_ref) > 10:
    communities=sorted(df_ref['area_name_en'].dropna().unique().tolist())
    master_projects=sorted(df_ref['master_project_en'].dropna().unique().tolist())[:100]
    prop_types=sorted(df_ref['property_type_en'].dropna().unique().tolist())
    sub_types=sorted(df_ref['property_sub_type_en'].dropna().unique().tolist())
    rooms_list=sorted(df_ref['rooms_en'].dropna().unique().tolist())
    reg_types=sorted(df_ref['reg_type_en'].dropna().unique().tolist())
else:
    communities=["Business Bay", "Downtown Dubai", "Dubai Marina", "JVC", "JLT", "Palm Jumeirah", "Dubai Hills"]
    master_projects=["Unknown", "Burj Khalifa", "Dubai Marina", "Downtown"]
    prop_types=["Flat", "Villa", "Hotel Apartment"]
    sub_types=["1 B/R", "2 B/R", "3 B/R", "Studio"]
    rooms_list=["Studio", "1 B/R", "2 B/R", "3 B/R"]
    reg_types=["Existing Properties", "Off-Plan Properties"]

# Section 7: Create input form with two columns for location and property features
st.markdown('<div style="color:#D4AF37; font-weight:700; letter-spacing:2px; margin: 10px 0;">◈ PROPERTY SPECIFICATIONS ◈</div>', unsafe_allow_html=True)

col1,col2=st.columns(2, gap="large")
with col1:
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Community</span>', unsafe_allow_html=True)
    community=st.selectbox("Community", communities, index=0, label_visibility="collapsed")
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Master Project</span>', unsafe_allow_html=True)
    master_proj=st.selectbox("Master Project", ["Unknown"]+master_projects, label_visibility="collapsed")
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Property Type</span>', unsafe_allow_html=True)
    p_type=st.selectbox("Property Type", prop_types, label_visibility="collapsed")
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Property Sub-Type</span>', unsafe_allow_html=True)
    sub_type=st.selectbox("Property Sub-Type", sub_types, label_visibility="collapsed")
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Bedroom Category</span>', unsafe_allow_html=True)
    rooms=st.selectbox("Bedroom Category", rooms_list, label_visibility="collapsed")
with col2:
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Size in Sqft</span>', unsafe_allow_html=True)
    size_sqft=st.number_input("Size in Sqft", min_value=200.0, max_value=15000.0, value=1100.0, step=50.0, label_visibility="collapsed")
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Parking</span>', unsafe_allow_html=True)
    has_parking=st.selectbox("Parking", [0,1], index=1, label_visibility="collapsed")
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Registration Status</span>', unsafe_allow_html=True)
    reg_type=st.selectbox("Registration", reg_types, label_visibility="collapsed")
    st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Transaction Year</span>', unsafe_allow_html=True)
    year=st.number_input("Year", min_value=2020, max_value=2026, value=2024, label_visibility="collapsed")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Quarter</span>', unsafe_allow_html=True)
        quarter=st.selectbox("Quarter", [1,2,3,4], index=1, label_visibility="collapsed")
    with c2:
        st.markdown('<span style="color:#D4AF37; font-size:0.8rem;">Month</span>', unsafe_allow_html=True)
        month=st.selectbox("Month", list(range(1,13)), index=5, label_visibility="collapsed")

# Section 8: Handle estimate button click to build feature vector and run ML prediction
st.markdown("<br>", unsafe_allow_html=True)
if st.button("◈ ESTIMATE LUXURY VALUE ◈", use_container_width=True):
    # Sub-section: Build input dictionary from user selections
    input_dict={
        'area_name_en':community,
        'master_project_en':master_proj,
        'property_type_en':p_type,
        'property_sub_type_en':sub_type,
        'rooms_en':rooms,
        'has_parking':has_parking,
        'reg_type_en':reg_type,
        'area_sqft':size_sqft,
        'year':year,
        'quarter':quarter,
        'month':month
    }
    # Sub-section: Convert inputs to model features with one-hot encoding and alignment
    df_input=pd.DataFrame([input_dict])
    X_input=pd.get_dummies(df_input, drop_first=False)
    for c in model_cols:
        if c not in X_input.columns:
            X_input[c]=0
    X_input=X_input[model_cols]
    
    # Sub-section: Predict log value and convert back to AED using exponential
    log_pred=model.predict(X_input)[0]
    pred_aed=np.exp(log_pred)

    # Sub-section: Calculate indicative range based on model MAPE 16.62%
    mape=16.62
    low=pred_aed*(1-mape/100*0.8)
    high=pred_aed*(1+mape/100*0.8)

    # Section 9: Display luxury valuation results with three gold-accented cards
    st.markdown("---")
    st.markdown('<div style="color:#D4AF37; font-weight:700; letter-spacing:3px; text-align:center; font-size:1.1rem;">◈ VALUATION CERTIFICATE ◈</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    r1,r2,r3=st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.7rem; color:#D4AF37; letter-spacing:2px; font-weight:700;">ESTIMATED MARKET VALUE</div>
            <div style="font-size:2rem; font-weight:900; margin-top:12px; color:#FFD700; font-family: 'Playfair Display', serif;">AED {pred_aed:,.0f}</div>
            <div style="font-size:0.75rem; margin-top:8px; color:#A0A0A0;">Model B • R² 0.81</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="range-card">
            <div style="font-size:0.7rem; color:#D4AF37; letter-spacing:2px; font-weight:700;">INDICATIVE RANGE</div>
            <div style="font-size:1.15rem; font-weight:800; margin-top:12px; color:#FFD700;">AED {low:,.0f} — {high:,.0f}</div>
            <div style="font-size:0.75rem; margin-top:8px; color:#A0A0A0;">±13% • MAPE 16.6%</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="price-card">
            <div style="font-size:0.7rem; color:#D4AF37; letter-spacing:2px; font-weight:700;">PRICE PER SQFT • ELITE</div>
            <div style="font-size:2rem; font-weight:900; margin-top:12px; color:#E5E4E2;">AED {pred_aed/size_sqft:,.0f}</div>
            <div style="font-size:0.75rem; margin-top:8px; color:#A0A0A0;">{size_sqft:,.0f} sqft</div>
        </div>
        """, unsafe_allow_html=True)

    st.success(f"✅ Valuation Complete for {community}")

    # Section 10: Retrieve and display recent comparable sales from same community
    st.markdown(f'<div style="color:#D4AF37; font-weight:700; letter-spacing:2px; margin-top:15px;">◈ RECENT ELITE COMPARABLES • {community.upper()} ◈</div>', unsafe_allow_html=True)
    comps=df_ref[(df_ref['area_name_en']==community)].copy() if 'area_name_en' in df_ref.columns else df_ref.copy()
    if 'area_sqft' in comps.columns:
        comps=comps[(comps['area_sqft']>=size_sqft*0.75) & (comps['area_sqft']<=size_sqft*1.25)]
    if 'year' in comps.columns:
        comps=comps.sort_values('year', ascending=False).head(10)
    else:
        comps=comps.head(10)
    
    # Show available columns
    cols_to_show = [c for c in ['area_name_en','master_project_en','property_type_en','rooms_en','area_sqft','year','actual_worth'] if c in comps.columns]
    st.dataframe(comps[cols_to_show], use_container_width=True, hide_index=True)

# Section 11: Display luxury footer with model details and deployment recommendations
st.markdown("---")
st.markdown("""
<div style="background: #0F0F0F; border-left: 4px solid #D4AF37; padding:20px; border-radius:12px;">
    <span style="color:#D4AF37; font-weight:800;">📋 MODEL NOTES • BURJ KHALIFA LUXURY COLLECTION</span><br>
    <span style="color:#A0A0A0; font-size:0.9rem;">Trained on 79,828 residential sales up to 35M AED, 2020-2025. Excludes mortgages/gifts/commercial. Log-transformed target. Deployment: first-pass tool, flag >6M or low-volume for human review. Theme: Gold/Black Luxury + Burj Khalifa.</span>
</div>
""", unsafe_allow_html=True)
