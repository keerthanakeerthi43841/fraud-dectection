import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import easyocr
from deepface import DeepFace
from pdf2image import convert_from_bytes
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import tempfile, os, re, datetime

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Banking Fraud Guard", page_icon="🏦", layout="wide")

# ---------------------------
# Load external CSS file
# ---------------------------
def local_css(file_name: str):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠ CSS file '{file_name}' not found. Continuing without styling.")

local_css("style.css")

# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🏦  Fraud Detector </h1>
        <div>Digital Document, KYC & Transaction Validation System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper functions
# ---------------------------
def load_image(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        pages = convert_from_bytes(uploaded_file.read())
        return pages[0]
    else:
        return Image.open(uploaded_file)

def generate_report_pdf(output_path: str, data: dict):
    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20*mm, h - 20*mm, "Banking Fraud Guard - Fraud Detection Report")
    c.setFont("Helvetica", 10)
    c.drawString(20*mm, h - 26*mm, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if data.get("customer_name"):
        c.drawString(20*mm, h - 32*mm, f"Customer: {data['customer_name']}")
    if data.get("reference"):
        c.drawString(20*mm, h - 38*mm, f"Ref: {data['reference']}")

    y = h - 48*mm
    for title, val in [
        ("Document Forgery", data.get("forgery_score")),
        ("Signature Verification", data.get("signature_score")),
        ("Aadhaar OCR", data.get("aadhaar_text")),
        ("PAN OCR", data.get("pan_text")),
        ("KYC Face Match", data.get("face_distance")),
        ("Transaction Summary", data.get("transaction_frauds_count")),
    ]:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(20*mm, y, title)
        y -= 6*mm
        c.setFont("Helvetica", 10)
        if val is None:
            c.drawString(20*mm, y, "Not analyzed")
        elif isinstance(val, str):
            c.drawString(20*mm, y, val[:200].replace("\n", " ") + "...")
        else:
            c.drawString(20*mm, y, f"{val}")
        y -= 8*mm

    remarks = data.get("remarks", "None")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y, "Remarks")
    y -= 6*mm
    c.setFont("Helvetica", 10)
    for line in (remarks[i:i+90] for i in range(0, len(remarks), 90)):
        c.drawString(20*mm, y, line)
        y -= 6*mm

    c.showPage()
    c.save()

# ---------------------------
# Initialize session
# ---------------------------
for key in [
    "forgery_score",
    "signature_score",
    "aadhaar_text",
    "pan_text",
    "face_distance",
    "face_verified",
    "transaction_frauds_count",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["📄 Doc Forgery", "✍ Signature", "🪪 Aadhaar", "🧾 PAN", "👥 KYC Match", "💸 Transactions"])

# ---- DOC FORGERY ----
with tabs[0]:
    st.header("📄 Document Forgery Detection")
    c1, c2 = st.columns(2)
    with c1:
        orig = st.file_uploader("Upload Original Document", type=["jpg","jpeg","png","pdf"], key="orig")
    with c2:
        sus = st.file_uploader("Upload Suspect Document", type=["jpg","jpeg","png","pdf"], key="sus")

    if st.button("Analyze for Forgery"):
        if not (orig and sus):
            st.error("Please upload both files.")
        else:
            pil1 = load_image(orig).convert("L").resize((600,600))
            pil2 = load_image(sus).convert("L").resize((600,600))
            a1, a2 = np.array(pil1), np.array(pil2)
            score, diff = ssim(a1, a2, full=True)
            st.session_state.forgery_score = float(score)
            diff_img = (diff * 255).astype("uint8")
            st.write(f"Similarity Score: {score:.3f}")
            if score > 0.9:
                st.success("No forgery detected.")
            elif score > 0.6:
                st.warning("Minor differences detected.")
            else:
                st.error("High chance of forgery.")
            st.image([pil1, pil2, diff_img], caption=["Original","Suspect","Diff Map"], width=250)

# ---- SIGNATURE ----
with tabs[1]:
    st.header("✍ Signature Verification")
    s1, s2 = st.columns(2)
    with s1:
        sig_orig = st.file_uploader("Original Signature", type=["jpg","jpeg","png"], key="sorig")
    with s2:
        sig_sus = st.file_uploader("Suspect Signature", type=["jpg","jpeg","png"], key="ssus")

    if st.button("Verify Signature"):
        if not (sig_orig and sig_sus):
            st.error("Upload both signatures.")
        else:
            p1 = load_image(sig_orig).convert("L").resize((300,300))
            p2 = load_image(sig_sus).convert("L").resize((300,300))
            sscore, _ = ssim(np.array(p1), np.array(p2), full=True)
            st.session_state.signature_score = float(sscore)
            st.write(f"Signature Similarity: {sscore:.3f}")
            if sscore > 0.85:
                st.success("Signatures appear genuine.")
            elif sscore > 0.6:
                st.warning("Partial match - suspicious.")
            else:
                st.error("Signatures likely forged.")
            st.image([p1, p2], caption=["Original Sig","Suspect Sig"], width=200)

# ---- AADHAAR ----
with tabs[2]:
    st.header("🪪 Aadhaar OCR & Validation")
    aad = st.file_uploader("Upload Aadhaar (front) image/pdf", type=["jpg","jpeg","png","pdf"], key="aad")
    if st.button("Extract Aadhaar"):
        if not aad:
            st.error("Upload Aadhaar file.")
        else:
            reader = easyocr.Reader(["en"])
            pil = load_image(aad).convert("RGB")
            res = reader.readtext(np.array(pil))
            text = " ".join([r[1] for r in res])
            st.session_state.aadhaar_text = text
            st.text_area("Extracted Text", text, height=220)
            found = re.findall(r"\d{4}\s*\d{4}\s*\d{4}|\d{12}", text)
            if found:
                st.success(f"Aadhaar-like number found: {found[0]}")
            else:
                st.warning("No 12-digit Aadhaar number detected.")

# ---- PAN ----
with tabs[3]:
    st.header("🧾 PAN OCR & Validation")
    pan = st.file_uploader("Upload PAN card image/pdf", type=["jpg","jpeg","png","pdf"], key="pan")
    if st.button("Extract PAN"):
        if not pan:
            st.error("Upload PAN file.")
        else:
            reader_pan = easyocr.Reader(["en"])
            pilp = load_image(pan).convert("RGB")
            res = reader_pan.readtext(np.array(pilp))
            textp = " ".join([r[1] for r in res])
            st.session_state.pan_text = textp
            st.text_area("Extracted Text", textp, height=220)
            found = re.findall(r"[A-Z]{5}[0-9]{4}[A-Z]", textp.upper())
            if found:
                st.success(f"PAN-like number found: {found[0]}")
            else:
                st.warning("PAN number not confidently detected.")

# ---- KYC ----
with tabs[4]:
    st.header("👥 KYC Face Match")
    f1, f2 = st.columns(2)
    with f1:
        id_face = st.file_uploader("ID Photo (from doc)", type=["jpg","jpeg","png"], key="idface")
    with f2:
        live_face = st.file_uploader("Live Photo", type=["jpg","jpeg","png"], key="liveface")
    if st.button("Compare Faces"):
        if not (id_face and live_face):
            st.error("Please upload both ID and live photos.")
        else:
            try:
                res = DeepFace.verify(np.array(load_image(id_face)), np.array(load_image(live_face)), enforce_detection=False)
                st.session_state.face_distance = float(res.get("distance", 0.0))
                st.session_state.face_verified = bool(res.get("verified", False))
                st.write(f"Distance: {st.session_state.face_distance:.4f}")
                if st.session_state.face_verified:
                    st.success("Face match verified.")
                else:
                    st.error("Face mismatch.")
            except Exception as e:
                st.error(f"Face verification error: {e}")

# ---- TRANSACTIONS ----
with tabs[5]:
    st.header("💸 Transaction Fraud Detection")
    txn = st.file_uploader("Upload transactions CSV (must include 'amount' column)", type=["csv"], key="txn")
    if st.button("Analyze Transactions"):
        if not txn:
            st.error("Upload CSV.")
        else:
            try:
                df = pd.read_csv(txn)
                st.write(df.head())
                if "amount" not in df.columns:
                    st.warning("CSV must have 'amount' column.")
                else:
                    threshold = df["amount"].mean() + 3 * df["amount"].std()
                    frauds = df[df["amount"] > threshold]
                    st.session_state.transaction_frauds_count = len(frauds)
                    st.metric("Potential Fraud Transactions", len(frauds))
                    if not frauds.empty:
                        st.dataframe(frauds)
            except Exception as e:
                st.error(f"CSV read error: {e}")

# ---- PDF REPORT ----
st.markdown("---")
st.header("🧾 Generate Final Fraud Detection Report (PDF)")
cust_name = st.text_input("Customer name (for report)", "")
ref = st.text_input("Account / Reference number", "")
remarks = st.text_area("Additional remarks", height=80)

if st.button("Generate PDF Report"):
    tmpd = tempfile.mkdtemp()
    out_pdf = os.path.join(tmpd, f"fraud_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    data = {
        "customer_name": cust_name,
        "reference": ref,
        "forgery_score": st.session_state.forgery_score,
        "signature_score": st.session_state.signature_score,
        "aadhaar_text": st.session_state.aadhaar_text,
        "pan_text": st.session_state.pan_text,
        "face_distance": st.session_state.face_distance,
        "face_verified": st.session_state.face_verified,
        "transaction_frauds_count": st.session_state.transaction_frauds_count,
        "remarks": remarks,
    }
    try:
        generate_report_pdf(out_pdf, data)
        with open(out_pdf, "rb") as f:
            st.download_button("📥 Download Fraud Report (PDF)", data=f, file_name=os.path.basename(out_pdf), mime="application/pdf")
    except Exception as e:
        st.error(f"PDF generation error: {e}")