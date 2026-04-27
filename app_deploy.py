import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from fpdf import FPDF
import tempfile
import os
import datetime
import matplotlib.pyplot as plt
import gdown

# -------------------------------
# CONFIG
# -------------------------------
MODEL_ID = "1HMLvxxL95RVFUcu7w39dyuex4ywRbrMx"
MODEL_PATH = "lung_cancer_model.keras"

# -------------------------------
# DOWNLOAD MODEL (ONCE)
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

class_names = ["Benign", "Malignant", "Normal"]

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Lung Cancer Detection", layout="wide")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🫁 Lung Cancer System")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# -------------------------------
# HOME
# -------------------------------
if page == "Home":
    st.title("🏥 Lung Cancer Detection System")
    st.markdown("""
    CNN-based system to detect lung cancer from CT scans.

    ✔ Upload CT image  
    ✔ Multi-class prediction  
    ✔ Risk level classification  
    ✔ Download PDF report  
    """)

# -------------------------------
# PREDICTION
# -------------------------------
elif page == "Prediction":

    st.title("🔍 Cancer Prediction")

    col1, col2 = st.columns(2)

    # -------------------------------
    # PATIENT DETAILS
    # -------------------------------
    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 120)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        uploaded_file = st.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])

    # -------------------------------
    # PREDICTION FUNCTION
    # -------------------------------
    def predict(image):
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        class_index = np.argmax(preds)
        confidence = preds[class_index]

        return class_index, confidence, preds

    # -------------------------------
    # RISK LEVEL
    # -------------------------------
    def get_risk(label, confidence):
        if label == "Malignant":
            if confidence > 0.7:
                return "High", "red"
            else:
                return "Medium", "orange"
        else:
            return "Low", "green"

    # -------------------------------
    # OUTPUT
    # -------------------------------
    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="CT Scan", width="stretch")

            if st.button("Predict"):

                class_index, confidence, preds = predict(image)
                label = class_names[class_index]
                risk, color = get_risk(label, confidence)

                st.markdown("### 🧾 Result Summary")

                st.markdown(f"""
                **Prediction:** :{color}[{label}]  
                **Confidence:** {confidence:.2f}  
                **Risk Level:** :{color}[{risk}]
                """)

                # -------------------------------
                # LINE GRAPH
                # -------------------------------
                fig, ax = plt.subplots()
                ax.plot(class_names, preds, marker='o')
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Confidence")
                st.pyplot(fig)

                # -------------------------------
                # SAVE IMAGE TEMP
                # -------------------------------
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name)
                    temp_image_path = tmp_file.name

                # -------------------------------
                # PDF GENERATION
                # -------------------------------
                def generate_pdf():
                    pdf = FPDF()
                    pdf.add_page()

                    # HEADER
                    pdf.set_font("Arial", "B", 18)
                    pdf.cell(200, 10, "AI Diagnostic Center", ln=True, align="C")

                    pdf.set_font("Arial", "", 12)
                    pdf.cell(200, 8, "Lung Cancer Detection Report", ln=True, align="C")

                    pdf.ln(5)
                    pdf.cell(200, 0, "", ln=True, border="T")

                    # PATIENT INFO
                    pdf.ln(8)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 10, "Patient Information", ln=True)

                    pdf.set_font("Arial", "", 12)
                    pdf.cell(100, 8, f"Name: {name}", ln=False)
                    pdf.cell(100, 8, f"Age: {age}", ln=True)
                    pdf.cell(100, 8, f"Sex: {sex}", ln=False)
                    pdf.cell(100, 8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

                    # RESULT
                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 10, "Diagnosis Summary", ln=True)

                    if risk == "High":
                        pdf.set_text_color(255, 0, 0)
                    elif risk == "Medium":
                        pdf.set_text_color(255, 140, 0)
                    else:
                        pdf.set_text_color(0, 128, 0)

                    pdf.cell(200, 10, f"Prediction: {label}", ln=True)
                    pdf.cell(200, 10, f"Risk Level: {risk}", ln=True)

                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(200, 8, f"Confidence: {confidence:.2f}", ln=True)

                    # IMAGE
                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 10, "CT Scan Image", ln=True)
                    pdf.image(temp_image_path, x=30, w=150)

                    # FOOTER
                    pdf.ln(85)
                    pdf.set_font("Arial", "I", 10)
                    pdf.set_text_color(100, 100, 100)

                    pdf.multi_cell(
                        0, 6,
                        "Disclaimer: This report is AI-generated and for educational use only. Consult a doctor."
                    )

                    pdf_path = "report.pdf"
                    pdf.output(pdf_path)
                    return pdf_path

                # -------------------------------
                # DOWNLOAD
                # -------------------------------
                pdf_path = generate_pdf()

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download Report",
                        f,
                        file_name="Lung_Cancer_Report.pdf"
                    )

                os.remove(temp_image_path)
