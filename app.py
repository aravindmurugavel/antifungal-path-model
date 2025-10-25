#!/usr/bin/env python3
"""
PATH — Personalized Antifungal Treatment for Humans
Unified Streamlit app that:
- Loads your real model artifacts (antifungal_model.pkl, label_encoders.pkl, feature_mappings.pkl, target_columns.pkl, feature_columns.pkl)
- Preserves your original predict_proba + estimators_ logic per target
- Adds modern PATH-styled UI (cards, header, sidebar, color badges, notes)
- (Demo mode removed)
- Outcome input removed and hardcoded to 'Live' (no UI message)
- Input fields are reorganized into Patient Profile and Clinical Context
- "Recommended Therapies" heading now appears after the Predict button is clicked
- Confidence heading text colors are explicitly set to match their pill colors.
- Main background color changed to white (#ffffff).
- Result box (prob-row) background color now changes based on confidence level.
- Includes Plotly Visualization and Treatment Summary sections.
- MODIFIED: Age input is now numerical.
- MODIFIED: Prediction results are segregated by drug class.
- MODIFIED: Antifungal drugs are sub-classified.
- MODIFIED: Removed "Without Antifungal agents" from prediction output.
- NEW: Added "Potential Combination Therapies" generation and display.
- MODIFIED: Combination therapy display now shows sum of probabilities.
- MODIFIED: Removed border from h6 (drug class) titles.
- MODIFIED: Renamed drug class titles (e.g., "Azoles" to "Azoles class of drugs").
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go # Added Plotly import

# ===========================
# Page configuration and Styling
# ===========================
st.set_page_config(
    page_title="Personalized Antifungal Treatment for Humans (PATH)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PATH styling adapted from 3.py
st.markdown(
    """
    <style>
    :root {
        --primary: #1f4e79;
        --accent: #0ea5e9;
        --bg: #ffffff; 
        --text: #0f172a;
        --muted: #475569;
        --success: #059669;
        --warn: #d97706;
        --danger: #b91c1c;
        
        /* NEW: Light backgrounds for result rows */
        --bg-success-light: #ebf8ee; 
        --bg-warn-light: #fff8e7; 
        --bg-danger-light: #fcecec;
    }
    .main > div { padding-top: 0.4rem; }
    .app-header {
        background: linear-gradient(90deg, var(--primary), #14324c);
        color: #fff; border-radius: 12px; padding: 22px 24px; margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .subtext { color: rgba(255,255,255,0.9); font-size: 0.95rem; margin-top: 4px; }
    .card {
        background: var(--card); border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 12px;
    }
    .pill {
        display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.80rem; font-weight: 600;
        color: #fff; margin-left: 8px;
    }
    .pill-high { background: var(--success); }
    .pill-mid  { background: var(--warn); }
    .pill-low  { background: var(--danger); }
    
    /* Updated .prob-row style */
    .prob-row {
        display:flex; justify-content: space-between; align-items:center;
        padding: 10px 12px; border: 1px solid #e5e7eb; border-radius: 10px; margin-bottom: 8px;
    }
    
    /* Confidence-based background/border styles */
    .prob-row-high { background: var(--bg-success-light); border-left: 5px solid var(--success); }
    .prob-row-mid { background: var(--bg-warn-light); border-left: 5px solid var(--warn); }
    .prob-row-low { background: var(--bg-danger-light); border-left: 5px solid var(--danger); }

    .prob-name { font-weight: 600; color: var(--text); }
    .prob-pct  { font-feature-settings: "tnum"; font-variant-numeric: tabular-nums; color: var(--muted); }
    
    .section-title { font-size: 1.05rem; font-weight: 700; margin: 8px 0 12px; }
    .title-high { color: var(--success); }
    .title-mid { color: var(--warn); }
    .title-low { color: var(--danger); }

    .help-note { color: var(--muted); font-size: 0.92rem; }
    .footer-note { color: var(--muted); font-size: 0.88rem; margin-top: 10px; }
    
    /* vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    EDIT THIS SECTION TO CHANGE THE SEPARATOR LINE
    vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    */
    h6 {
        font-weight: 600;
        color: var(--muted);
        margin-top: 14px;
        margin-bottom: 18px;
        padding-bottom: 5px;
        /* border-bottom: 1px solid #e5e7eb; <-- REMOVED THIS LINE */ 
        /* EXAMPLE FOR CENTERING:
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        */
    }
    /* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    END OF EDITING SECTION
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    */
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================
# Header
# ===========================
st.markdown(
    """
    <div class="app-header">
        <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
            <div style="font-size:1.35rem;font-weight:800;letter-spacing:0.2px;">
                PATH — Personalized Antifungal Treatment for Humans
            </div>
        </div>
        <div class="subtext">
            AI-driven clinical decision support to deliver patient-specific, first-line antifungal therapy recommendations in seconds.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ===========================
# Sidebar
# ===========================
with st.sidebar:
    st.markdown("### About PATH")
    st.markdown(
        "**PATH (Personalized Antifungal Treatment for Humans)** is an AI-driven, open-source **decision-support tool** designed to provide patient-specific, first-line antifungal therapy recommendations. "
        "Developed using a **Random Forest machine learning model**, PATH addresses the challenge of rapidly selecting effective antifungal treatment, aiming to optimize therapy and improve clinical outcomes in fungal infections."
    )
    
    st.markdown("### How PATH Works")
    st.markdown(
        "The model generates individualized treatment probabilities by integrating a comprehensive range of **host and pathogen factors**:"
    )
    st.markdown(
        """
        - **Patient Profile**: It considers **age, gender, and geographic location** across 58 countries.
        - **Clinical Context**: It integrates data of **662 individual patients covering 181 medical histroy** (underlying medical conditions).
        - **Fungal Pathogens**: It analyzes **41 fungal species** (primarily within the *Aspergillus*, *Candida*, and *Mucorales* genera).
        """
    )
    st.markdown(
        "By synthesizing this evidence, PATH provides **immediate, data-driven insights** to guide clinicians in selecting the most effective antifungal agents and steroids for immediate care."
    )
    
    st.markdown("---") # Optional: Add a separator line
    
    st.markdown("### How to use")
    st.markdown(
        """
        - Select patient profile fields
        - Know the Fungal Infection
        - Click **Predict Therapies**
        - Review recommendations grouped by confidence
        """
    )

# ===========================
# Load artifacts (MUST exist)
# ===========================
REQUIRED_FILES = [
    "antifungal_model.pkl",
    "label_encoders.pkl",
    "feature_mappings.pkl",
    "target_columns.pkl",
    "feature_columns.pkl",
]

# Ensure all files exist, otherwise let joblib.load fail loudly
# The Streamlit app will stop and show an error if files are missing.
if not all(os.path.exists(p) for p in REQUIRED_FILES):
    st.error(
        f"Critical Error: One or more required model artifacts were not found. "
        f"Please ensure the following files are in the same directory: {', '.join(REQUIRED_FILES)}"
    )
    st.stop() # Stop execution if files are missing.

@st.cache_data(show_spinner=False)
def load_real():
    model = joblib.load("antifungal_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    mappings = joblib.load("feature_mappings.pkl")
    target = joblib.load("target_columns.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, encoders, mappings, target, features

# Load artifacts directly
model, encoders, mappings, target, features = load_real()


# ===========================
# Small helpers
# ===========================
def age_group_hint():
    return {
        # These values must correspond to the keys in your label_encoders.pkl for Age_Group
        "infant": "0–2 years",
        "toddler": "2–4 years",
        "child": "5–12 years",
        "teen": "13–19 years",
        "adult": "20–39 years",
        "middle-aged": "40–64 years",
        "senior adult": "65+ years",
    }
    
# NEW: Helper to map numerical age to a model category
def map_age_to_group(age):
    """Maps a numerical age to its corresponding string category."""
    if age <= 2: return "infant"
    if age <= 4: return "toddler"
    if age <= 12: return "child"
    if age <= 19: return "teen"
    if age <= 39: return "adult"
    if age <= 64: return "middle-aged"
    return "senior adult" # 65+

def get_age_group_description(group):
    # Helper to get the descriptive age range for the summary
    return age_group_hint().get(group, "N/A")

def pct(p): return f"{p:.0%}"

def badge(prob):
    if prob >= 0.70:
        return f'<span class="pill pill-high">{pct(prob)}</span>'
    if prob >= 0.30:
        return f'<span class="pill pill-mid">{pct(prob)}</span>'
    return f'<span class="pill pill-low">{pct(prob)}</span>'

def format_age_group(name):
    return name.capitalize()
    
# NEW: Helper to display generated combinations
def display_combinations(combo_list):
    """
    Displays generated combination therapies in a styled row.
    Args:
        combo_list (list): List of (drug1, prob1, drug2, prob2) tuples.
    """
    if not combo_list:
        st.caption("No combinations meet the criteria.")
        return
    
    for d1, p1, d2, p2 in combo_list:
        # Determine style based on the *minimum* probability
        min_prob = min(p1, p2)
        if min_prob >= 0.70:
            style_level = 'high'
        else:
            style_level = 'mid' # All candidates are > 0.40, so 'mid' is the lowest
        
        # MODIFIED: Show individual percentages in the name
        name_html = f'<div class="prob-name" style="font-size: 0.9rem;">{d1} ({pct(p1)}) + {d2} ({pct(p2)})</div>'
        
        # NEW: Calculate sum
        total_prob = p1 + p2
        # Cap at 100%
        if total_prob > 1.0:
            total_prob = 1.0
            
        # MODIFIED: Show sum as "survival"
        pct_html = f'<div class="prob-pct" style="font-weight: 600; font-size: 0.9rem; white-space: nowrap;">= {pct(total_prob)} Survival</div>'
        
        st.markdown(
            # Added style to allow wrapping and align items center
            f'<div class="prob-row prob-row-{style_level}" style="flex-wrap: wrap; align-items: center;">{name_html}{pct_html}</div>',
            unsafe_allow_html=True
        )

# ===========================
# Input Grouping Definition
# ===========================
PATIENT_PROFILE_FEATURES = ["Gender", "Country"]
CLINICAL_CONTEXT_FEATURES = ["Species", "Medical History"]
patient_input = {}
raw_inputs = {} # Dictionary to store raw (string) inputs

# ===========================
# ===========================
# Inputs
# ===========================
left, right = st.columns([1, 1])

# A robust helper function to apply binomial nomenclature to all species, even complex strings.
def format_species_name(name):
    
    # This inner function formats a *single* species name correctly.
    def format_single_species(s):
        s = s.strip()
        s_lower = s.lower()
        
        # Terms that must be converted to "Mucor spp." (case-insensitive and accounting for misspellings)
        MUCOR_TERMS = ['mucomycosis', 'mucormycosis', 'mucorales spp.']
        
        # --- START OF CORRECTION: Handle all required mandatory renames first (case-insensitive exact match) ---
        if s_lower in MUCOR_TERMS:
            return "Mucor spp."
        # --- END OF CORRECTION ---
        
        # Handles abbreviated binomial nomenclature (e.g., C. albicans, R. microsporus)
        if '.' in s:
            parts = s.split('.', 1)
            # Genus initial (C/R) must be capitalized
            genus = parts[0].strip().capitalize()
            # Species name (albicans/microsporus) must be lowercase
            species = parts[1].strip().lower() 
            return f"{genus}. {species}"
        
        # Handles full binomial nomenclature (e.g., Candida albicans, Rhizopus microsporus)
        elif ' ' in s:
            parts = s.split(' ', 1)
            # Genus name (Candida/Rhizopus) must be capitalized
            genus = parts[0].strip().capitalize()
            # Species name (albicans/microsporus) must be lowercase
            species = parts[1].strip().lower() 
            return f"{genus} {species}"
        
        # Fallback for single-word/spp names (e.g., Aspergillus spp., Candida spp.)
        else:
            return s.capitalize()

    # Split by ' and ' to handle combined cases (e.g., Aspergillus spp. and Mucomycosis)
    if ' and ' in name.lower():
        # Split using the most common case-insensitive separator
        parts = name.lower().split(' and ')
        
        # Apply the formatting to each species name in the list
        formatted_parts = [format_single_species(part) for part in parts]
        
        # Join them back together with ' and '
        return ' and '.join(formatted_parts)
    else:
        # If there's no "and", just format the single name.
        return format_single_species(name)
        
with left:
    st.markdown("#### 🩺 Patient Profile")

    # 1. Age (numerical input) - MODIFIED
    age_numerical = st.number_input(
        "Age (in years)", 
        min_value=0, 
        max_value=120, 
        value=35, 
        step=1, 
        help="Enter the patient's age in years."
    )
    # Map numerical age to the categorical group required by the model
    age_choice = map_age_to_group(age_numerical)
    # Store the mapped group name for the summary display
    age_group_raw = age_choice
    patient_input["Age_Group"] = int(encoders["Age_Group"].transform([age_choice])[0])

    # 2. Other Patient Profile Features
    for feature in PATIENT_PROFILE_FEATURES:
        options = list(encoders[feature].classes_)

        # --- START OF FIX: Rename and deduplicate 'Country' options ---
        if feature == "Country":
            options = ['Australia' if opt == 'Austria' else opt for opt in options]
            options = sorted(list(set(options)))
        # --- END OF FIX ---

        choice = st.selectbox(feature, options)
        raw_inputs[feature] = choice # Capture raw input
        patient_input[feature] = int(encoders[feature].transform([choice])[0])

    # --- HARDCODE OUTCOME TO 'Live' silently ---
    outcome_live_encoded = int(encoders["Outcome"].transform(["Live"])[0])
    patient_input["Outcome"] = outcome_live_encoded
    outcome_raw = "Live" # Capture raw outcome
    # -------------------------------------------

with right:
    st.markdown("#### 🔬 Clinical Context")
    
    # Clinical Context Features
    for feature in CLINICAL_CONTEXT_FEATURES:
        options = sorted(list(encoders[feature].classes_)) # Sort options alphabetically

        # Check if the current feature is "Species" to apply special formatting and filtering
        if feature == "Species":
            # --- START OF MODIFICATIONS FOR SPECIES ---
            # 1. Rename 'Mucormycosis' to 'Mucor spp.' for display
            options_display = ['Mucor spp.' if opt == 'Mucormycosis' else opt for opt in options]
            
            # 2. Remove "A. baumannii" from the options list
            options_display = [opt for opt in options_display if "baumannii" not in opt.lower()]
            
            # 3. Handle mapping back to original label encoder value for model prediction
            display_to_model = {
                'Mucor spp.': 'Mucormycosis',
                **{opt: opt for opt in options_display if opt != 'Mucor spp.'}
            }
            # --- END OF MODIFICATIONS FOR SPECIES ---

            choice_display = st.selectbox(
                feature,
                options_display, # Use the modified list
                format_func=format_species_name  # Apply our robust formatting function
            )
            
            # Get the original label for the model
            choice_model = display_to_model.get(choice_display, choice_display)
            raw_inputs[feature] = choice_display # Capture raw input (display name)
            
            # Encode the original label for the model
            patient_input[feature] = int(encoders[feature].transform([choice_model])[0])

        else:
            # Create the selectbox normally for other features (like Medical History)
            choice = st.selectbox(feature, options)
            raw_inputs[feature] = choice # Capture raw input
            patient_input[feature] = int(encoders[feature].transform([choice])[0])


# Map captured raw inputs for use in the summary section
age_group = age_group_raw
gender = raw_inputs.get("Gender", "N/A")
country = raw_inputs.get("Country", "N/A")
species = raw_inputs.get("Species", "N/A")
medical_history = raw_inputs.get("Medical History", "N/A")
outcome = outcome_raw
target_drugs = target

# Ensure DataFrame follows model feature order
df = pd.DataFrame([patient_input])[features]
# ===========================
# Predict Section
# ===========================

# NEW: Drug classification for segregated results - MODIFIED
DRUG_CLASSIFICATION = {
    # Azoles
    'Voriconazole': 'Azoles', 'Posaconazole': 'Azoles', 'Isavuconazole': 'Azoles',
    'Itraconazole': 'Azoles', 'Fluconazole': 'Azoles',
    # Polyenes
    'Amphotericin B Deoxycholate': 'Polyenes', 'Liposomal Amphotericin B': 'Polyenes',
    'Amphotericin B lipid complex': 'Polyenes', 'AmBisome': 'Polyenes', 'Amphotericin B': 'Polyenes',
    # Echinocandins
    'Anidulafungin': 'Echinocandins', 'Caspofungin': 'Echinocandins', 'Micafungin': 'Echinocandins',
    # Steroids
    'Prednisolone': 'Steroids', 'Dexamethasone': 'Steroids', 'Methylprednisolone': 'Steroids',
    'Corticosteroids': 'Steroids', 'Steroid': 'Steroids',
    # Combination Therapies
    'Voriconazole + Anidulafungin': 'Combination Therapies', 'Amphotericin B + Flucytosine': 'Combination Therapies'
}
CLASS_ORDER = ['Azoles', 'Polyenes', 'Echinocandins', 'Steroids', 'Combination Therapies', 'Other']

def get_drug_class(drug_name):
    """Returns the class of a given drug, defaulting to 'Other'."""
    return DRUG_CLASSIFICATION.get(drug_name, 'Other')

def display_results_by_class(confidence_list, confidence_level):
    """
    Displays prediction results segregated by drug class.
    Args:
        confidence_list (list): List of (drug, probability) tuples.
        confidence_level (str): 'high', 'mid', or 'low' for styling.
    """
    if not confidence_list:
        st.caption("No items in this band for the current profile.")
        return

    # Group drugs by class
    grouped_by_class = {}
    for tname, p in confidence_list:
        d_class = get_drug_class(tname)
        if d_class not in grouped_by_class:
            grouped_by_class[d_class] = []
        grouped_by_class[d_class].append((tname, p))

    # NEW: Title mapping
    title_mapping = {
        "Azoles": "Azoles class of drugs",
        "Polyenes": "Polyenes class of drugs",
        "Echinocandins": "Echinocandins class of drugs",
        "Steroids": "Steroids" # Added for consistency
    }

    # Display grouped results
    for d_class in CLASS_ORDER:
        if d_class in grouped_by_class:
            # Use mapped title or default
            display_title = title_mapping.get(d_class, d_class)
            st.markdown(f"<h6>{display_title}</h6>", unsafe_allow_html=True)
            
            for tname, p in grouped_by_class[d_class]:
                st.markdown(
                    f'<div class="prob-row prob-row-{confidence_level}"><div class="prob-name">{tname}</div>'
                    f'<div class="prob-pct">{badge(p)}</div></div>',
                    unsafe_allow_html=True
                )

# The button itself is placed outside the result display logic
if st.button(" ⚕ Predict Antifungal Treatment", type="primary"):
    
    st.markdown("### Recommended Therapies for Treatment Success 💊")

    try:
        # Your original prediction logic
        probs = model.predict_proba(df)
        probability_dict = {}
        for i, est in enumerate(model.estimators_):
            classes = getattr(est, "classes_", [0, 1])
            if 1 in classes:
                pos_idx = list(classes).index(1)
                prob = float(probs[i][0][pos_idx]) 
            else:
                prob = 0.0
            probability_dict[target[i]] = prob
        
        # <<< --- START: Remove "Without Antifungal agents" per user request --- >>>
        # Check for both capitalizations as requested
        if "Without Antifungal agents" in probability_dict:
            del probability_dict["Without Antifungal agents"]
        if "Without antifungal agents" in probability_dict:
            del probability_dict["Without antifungal agents"]
        # <<< --- END: Removal --- >>>
            
    except Exception as e:
        st.error(f"Prediction Error: The model's predict_proba or estimators_ structure is incompatible. Original Error: {e}")
        st.stop()


    # Sort and group (now without the removed item)
    sorted_probs = sorted(probability_dict.items(), key=lambda x: x[1], reverse=True)
    high_conf = [(t, p) for t, p in sorted_probs if p >= 0.70]
    moderate_conf = [(t, p) for t, p in sorted_probs if 0.30 <= p < 0.70]
    low_conf = [(t, p) for t, p in sorted_probs if p < 0.30]

    # <<< --- START: Combination Therapy Generation --- >>>
    # 1. Get all candidates with prob > 40%
    candidates = [(drug, prob) for drug, prob in sorted_probs if prob > 0.40]

    # 2. Classify candidates
    antifungal_candidates = []
    steroid_candidates = []
    without_steroids_tuple = None # Will store the (drug_name, prob) tuple if found
    
    # Define antifungal classes based on existing DRUG_CLASSIFICATION
    antifungal_classes = ['Azoles', 'Polyenes', 'Echinocandins']

    for drug, prob in candidates:
        # Check for "Without Steroids" case-insensitively
        if drug.lower() == "without steroids":
            without_steroids_tuple = (drug, prob)
        else:
            d_class = get_drug_class(drug) # Use existing helper
            if d_class in antifungal_classes:
                antifungal_candidates.append((drug, prob))
            elif d_class == 'Steroids':
                steroid_candidates.append((drug, prob))

    # 3. Generate combination lists
    
    # Category 1: Antifungal + Antifungal
    combo_af_af = []
    for i in range(len(antifungal_candidates)):
        for j in range(i + 1, len(antifungal_candidates)):
            drug1, prob1 = antifungal_candidates[i]
            drug2, prob2 = antifungal_candidates[j]
            combo_af_af.append((drug1, prob1, drug2, prob2))
            
    # Category 2: Antifungal + Steroid
    combo_af_steroid = []
    for drug_af, prob_af in antifungal_candidates:
        for drug_st, prob_st in steroid_candidates:
            combo_af_steroid.append((drug_af, prob_af, drug_st, prob_st))

    # Category 3: Antifungal + "Without Steroids"
    combo_af_no_steroid = []
    if without_steroids_tuple:
        drug_ws, prob_ws = without_steroids_tuple
        for drug_af, prob_af in antifungal_candidates:
            combo_af_no_steroid.append((drug_af, prob_af, drug_ws, prob_ws))
    # <<< --- END: Combination Therapy Generation --- >>>

    colA, colB, colC = st.columns([1, 1, 1])

    # MODIFIED: Display results segregated by drug class
    with colA:
        st.markdown('<div class="section-title title-high">High Confidence (≥70%)</div>', unsafe_allow_html=True)
        display_results_by_class(high_conf, 'high')

    with colB:
        st.markdown('<div class="section-title title-mid">Moderate Confidence (30–70%)</div>', unsafe_allow_html=True)
        display_results_by_class(moderate_conf, 'mid')

    with colC:
        st.markdown('<div class="section-title title-low">Low Confidence (<30%)</div>', unsafe_allow_html=True)
        display_results_by_class(low_conf, 'low')

    st.markdown(
        '<div class="footer-note">'
        'These outputs are model predictions and should be interpreted alongside clinical judgment and local guidelines.'
        '</div>',
        unsafe_allow_html=True
    )

    # <<< --- START: Display Combination Therapies --- >>>
    st.markdown("---") # Add a separator
    st.markdown("### Potential Combination Therapies")

    colD, colE, colF = st.columns([1, 1, 1])
    
    with colD:
        st.markdown('<div class="section-title title-mid">Antifungal + Antifungal</div>', unsafe_allow_html=True)
        display_combinations(combo_af_af)

    with colE:
        st.markdown('<div class="section-title title-mid">Antifungal + Steroid</div>', unsafe_allow_html=True)
        display_combinations(combo_af_steroid)

    with colF:
        st.markdown('<div class="section-title title-mid">Antifungal + Without Steroids</div>', unsafe_allow_html=True)
        display_combinations(combo_af_no_steroid)
    # <<< --- END: Display Combination Therapies --- >>>

    # ===========================
    # Visualization
    # ===========================
    st.markdown("---")
    st.markdown("### Treatment Probability Visualization")
    
    # Create interactive bar chart
    top_10 = sorted_probs[:10]
    drugs = [item[0] for item in top_10]
    probs = [item[1] for item in top_10]
    
    colors = ['#28a745' if p > 0.7 else '#ffc107' if p > 0.3 else '#dc3545' for p in probs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=drugs,
        y=probs,
        marker_color=colors,
        text=[f'{p:.1%}' for p in probs],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Probability: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Top 10 Treatment Recommendations",
        xaxis_title="Treatment",
        yaxis_title="Survival Probability",
        showlegend=False,
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary section
    st.markdown("---")
    st.markdown("### Treatment Summary & Clinical Notes")
    
    col1_sum, col2_sum = st.columns(2)
    
    with col1_sum:
        st.markdown("#### Patient Profile Summary")
        # MODIFIED: Display numerical age and the mapped group
        st.write(f"**Age:** {age_numerical} years ({age_group.replace('_', ' ').title()})")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Country:** {country}")
        st.write(f"**Fungal Species:** {species}")
        st.write(f"**Medical History:** {medical_history}")
        st.write(f"**Expected Outcome:** {outcome}")
        
        st.markdown("#### Model Performance")
        st.success("Model Accuracy: 92%")
        st.info(f"Predictions based on {len(target_drugs)-1} treatment options") # Adjusted count
    
    with col2_sum:
        st.markdown("#### Top 3 Treatment Recommendations")
        top_3 = sorted_probs[:3] # Using sorted_probs
        for i, (drug, prob) in enumerate(top_3, 1):
            confidence_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
            confidence_color_class = "title-high" if prob > 0.7 else "title-mid" if prob > 0.3 else "title-low"
            
            st.markdown(f"**{i}. {drug}**", unsafe_allow_html=True)
            st.markdown(f'<p style="margin-left: 20px;">Confidence: <span class="{confidence_color_class}">{prob:.1%} ({confidence_level})</span></p>', unsafe_allow_html=True)

        st.markdown("#### ⚠️ Clinical Considerations")
        st.warning("""
        **Important Disclaimer:**
        - The recommendations generated by PATH (Personalized Antifungal Treatment for Humans) are intended for clinical decision support only and must not be considered a  final prescription or treatment directive.
        - **Confirmation of Antifungal Susceptibility is Mandatory:** The predictions generated by PATH model, do not account for local antimicrobial reistance patterns and are nota substitute for in-vitro Antifungal Susceptibility Test (AFST).
        - This tool is intended to assist clinicians in rapidly selecting initial antifungal therapy and does not replace expert judgment.
        """)
        
else:
    st.info("Set the patient profile and click Predict Therapies to view recommendations.")

# ===========================
# About section (always visible)
# ===========================
with st.expander("ℹ️ About this Application"):
    st.markdown("""
    ## PATH - Personalized Antifungal Treatment for Humans
    
    This application uses a **Random Forest machine learning model** trained on real-world COVID-19 associated fungal infection data.
    
    ### Model Specifications:
    - **Algorithm:** Random Forest with Multi-Output Classification
    - **Training Dataset:** 661 patient cases from 50+ countries
    - **Input Features:** Demographics, geography, pathogen, comorbidities
    - **Output:** Probability predictions for 15+ treatment options
    - **Validation Accuracy:** 92% overall performance
    
    ### Clinical Applications:
    - Treatment decision support for healthcare providers
    - Evidence-based antifungal selection
    - Geographic treatment pattern analysis
    - Outcome prediction modeling
    
    ### Data Sources:
    - Multi-center COVID-19 fungal infection studies
    - International surveillance data
    - Peer-reviewed clinical literature
    
    ### Developed by:
    - Prof. R. Jayapradha | Mr. Aravind M | Ms. P.D.L. Sahithi | Ms. Sridevi Raghunathan
    - Actinomyces Biopropecting Lab, Centre for Research in Infectious Diseases (CRID), SASTRA Deemed University 
    """)


# A test comment to force a change
