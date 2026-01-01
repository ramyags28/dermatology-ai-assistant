import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.cm as cm
from fpdf import FPDF
import urllib.parse
import os
import gdown

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Dermatology Assistant",
    page_icon="🧴",
    layout="wide"
)

# ======================================================
# SESSION STATE
# ======================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ======================================================
# CUSTOM UI CSS
# ======================================================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#e3f2fd,#ffffff); font-family:Segoe UI;}
.card {background:white;padding:20px;border-radius:18px;
box-shadow:0 8px 20px rgba(0,0,0,0.1);margin-bottom:20px;}
.mild{background:#e8f5e9;}
.moderate{background:#fffde7;}
.severe{background:#ffebee;}
.btn{display:inline-block;padding:10px 18px;border-radius:12px;
color:white;text-decoration:none;font-weight:bold;}
.btn-green{background:#2e7d32;}
.btn-blue{background:#1565c0;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# MODEL DOWNLOAD + LOAD (GOOGLE DRIVE)
# ======================================================
MODEL_PATH = "dermatology_assistant_model.keras"
GDRIVE_URL = "https://drive.google.com/uc?id=1k5QpG18JlqCetsGhqZuNdCFS_OdPDDUZ"

@st.cache_resource
def load_derm_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI model (first run only)..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_derm_model()

# ======================================================
# 23 DISEASE CLASSES
# ======================================================
CLASS_NAMES = [
    "Acne","Actinic Keratosis","Basal Cell Carcinoma","Benign Keratosis",
    "Dermatofibroma","Eczema","Melanoma","Nevus","Psoriasis","Rosacea",
    "Seborrheic Keratosis","Squamous Cell Carcinoma","Tinea Ringworm",
    "Vitiligo","Urticaria","Lichen Planus","Impetigo","Cellulitis","Warts",
    "Herpes Simplex","Chickenpox","Scabies","Contact Dermatitis"
]

# ======================================================
# MEDICINE DATABASE
# ======================================================
MEDICINE_DB = {
    "Acne":["Benzoyl Peroxide","Adapalene"],
    "Actinic Keratosis":["5-Fluorouracil","Imiquimod"],
    "Basal Cell Carcinoma":["Surgical Excision"],
    "Benign Keratosis":["No treatment needed"],
    "Dermatofibroma":["Observation"],
    "Eczema":["Hydrocortisone","Moisturizer"],
    "Melanoma":["Immediate Oncology Referral"],
    "Nevus":["Monitoring"],
    "Psoriasis":["Vitamin D Cream","Coal Tar"],
    "Rosacea":["Metronidazole","Azelaic Acid"],
    "Seborrheic Keratosis":["Cryotherapy"],
    "Squamous Cell Carcinoma":["Surgical Removal"],
    "Tinea Ringworm":["Clotrimazole","Ketoconazole"],
    "Vitiligo":["Tacrolimus"],
    "Urticaria":["Antihistamines"],
    "Lichen Planus":["Corticosteroids"],
    "Impetigo":["Mupirocin"],
    "Cellulitis":["Oral Antibiotics"],
    "Warts":["Salicylic Acid"],
    "Herpes Simplex":["Acyclovir"],
    "Chickenpox":["Calamine Lotion"],
    "Scabies":["Permethrin Cream"],
    "Contact Dermatitis":["Topical Steroids"]
}

MED_LINK = "https://www.1mg.com/search/all?name="

# ======================================================
# UTILITIES
# ======================================================
def preprocess_image(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, 0)

def severity_calc(disease, conf):
    if disease in ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]:
        return "Severe"
    elif conf >= 75:
        return "Moderate"
    else:
        return "Mild"

# ======================================================
# GRAD-CAM (NO OPENCV – CLOUD SAFE)
# ======================================================
def gradcam(img_array, model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_sum(pooled * conv_out[0], axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

def overlay(img, heatmap):
    img = np.array(img.resize((224,224)))
    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize((224,224))
    heatmap = np.array(heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3] * 255
    overlay_img = 0.6 * img + 0.4 * heatmap
    return Image.fromarray(overlay_img.astype(np.uint8))

# ======================================================
# PDF REPORT
# ======================================================
def make_pdf(name, age, gender, disease, conf, severity, meds):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"AI Dermatology Diagnosis Report",ln=True,align="C")
    pdf.ln(5)
    pdf.set_font("Arial",size=12)

    for t in [
        f"Name: {name}",
        f"Age: {age}",
        f"Gender: {gender}",
        f"Disease: {disease}",
        f"Confidence: {conf}%",
        f"Severity: {severity}"
    ]:
        pdf.cell(0,8,t,ln=True)

    pdf.ln(3)
    pdf.cell(0,8,"Medicines:",ln=True)
    for m in meds:
        pdf.cell(0,8,f"- {m}",ln=True)

    pdf.ln(3)
    pdf.set_font("Arial",size=10)
    pdf.multi_cell(0,8,"AI-based preliminary screening only. Consult a dermatologist.")
    return pdf

# ======================================================
# LOGIN PAGE
# ======================================================
def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login") and u and p:
        st.session_state.logged_in = True
        st.rerun()

# ======================================================
# MAIN APP
# ======================================================
def app():
    st.title("🧴 AI Dermatology Assistant")

    c1, c2 = st.columns(2)

    with c1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age",1,100,25)
        gender = st.selectbox("Gender",["Male","Female","Other"])
        file = st.file_uploader("Upload Skin Image",["jpg","png","jpeg"])
        predict = st.button("Predict")

        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True)

    with c2:
        if predict and file:
            arr = preprocess_image(img)
            preds = model.predict(arr)[0]
            idx = np.argmax(preds)

            disease = CLASS_NAMES[idx]
            conf = round(preds[idx]*100,2)
            sev = severity_calc(disease, conf)

            st.markdown(
                f"<div class='card'><h3>{disease}</h3><p>Confidence: {conf}%</p></div>",
                unsafe_allow_html=True
            )

            sev_class = "mild" if sev=="Mild" else "moderate" if sev=="Moderate" else "severe"
            st.markdown(
                f"<div class='card {sev_class}'><h3>Severity: {sev}</h3></div>",
                unsafe_allow_html=True
            )

            st.subheader("💊 Medicines")
            for m in MEDICINE_DB[disease]:
                st.markdown(f"- **{m}** → [Link]({MED_LINK}{urllib.parse.quote(m)})")

            st.subheader("🔥 Grad-CAM")
            st.image(overlay(img, gradcam(arr, model)))

            st.subheader("📍 Nearby Dermatologists")
            st.components.v1.iframe(
                "https://www.google.com/maps?q=dermatologist+near+me&output=embed",
                height=300
            )

            pdf = make_pdf(name, age, gender, disease, conf, sev, MEDICINE_DB[disease])
            st.download_button(
                "📄 Download PDF",
                pdf.output(dest="S").encode("latin-1"),
                "Dermatology_Report.pdf",
                "application/pdf"
            )

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ======================================================
# ROUTER
# ======================================================
if not st.session_state.logged_in:
    login()
else:
    app()
