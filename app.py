import streamlit as st
import pandas as pd
import hashlib
import os
import io
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import socket
from dotenv import load_dotenv  # ✅ for secure API management

# ---------------------------
# CONFIGURATION & SETUP
# ---------------------------
st.set_page_config(page_title="🐾 AI Animal Recognition", layout="wide")

# Load environment variables (if available)
load_dotenv()

# File paths
USER_DB = "user_data.csv"
LOG_FILE = "recognition_log.csv"

# ✅ Imagga API Keys (you can safely replace with your actual keys here)
# If .env file is missing, these act as direct fallbacks
IMAGGA_API_KEY = os.getenv("IMAGGA_API_KEY", "acc_cd145549a1ef021")
IMAGGA_API_SECRET = os.getenv("IMAGGA_API_SECRET", "ad678b98a21847deec859c34f0d3407a")


# ---------------------------
# UTILITIES
# ---------------------------

# Hash passwords for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Load all users
def load_users():
    if os.path.exists(USER_DB):
        return pd.read_csv(USER_DB)
    return pd.DataFrame(columns=["username", "password", "role"])

# Save a new user
def save_user(username, password, role="user"):
    df = load_users()
    if username in df["username"].values:
        return False
    new_user = pd.DataFrame([[username, hash_password(password), role]], columns=["username", "password", "role"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DB, index=False)
    return True

# Verify credentials
def verify_user(username, password):
    df = load_users()
    hashed = hash_password(password)
    user = df[(df["username"] == username) & (df["password"] == hashed)]
    if not user.empty:
        return user.iloc[0]["role"]
    return None

# Check internet connectivity
def check_internet():
    try:
        socket.create_connection(("8.8.8.8", 53))
        return True
    except OSError:
        return False

# Load model once
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# ---------------------------
# LOGIN / SIGNUP PAGE
# ---------------------------
def login_page():
    st.title("🔐 AI-Based Animal Recognition Login")

    choice = st.radio("Choose an option:", ["Login", "Sign Up"])

    if choice == "Sign Up":
        username = st.text_input("Create Username")
        password = st.text_input("Create Password", type="password")
        role = st.selectbox("Select Role", ["user", "admin"])
        if st.button("Sign Up"):
            if save_user(username, password, role):
                st.success("✅ Account created successfully! You can now log in.")
            else:
                st.error("⚠️ Username already exists.")
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            role = verify_user(username, password)
            if role:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = role
                st.success(f"🎉 Welcome back, {username}! Redirecting to your dashboard...")
                st.toast("Login successful! Please wait...", icon="✅")
                st.rerun()  # ✅ fixed function name
            else:
                st.error("❌ Invalid credentials.")

# ---------------------------
# LOGGING FUNCTION
# ---------------------------
def save_log(user, animal, confidence):
    df = pd.DataFrame({
        "User": [user],
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Animal": [animal],
        "Confidence": [confidence]
    })
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

# ---------------------------
# RECOGNITION FUNCTIONS
# ---------------------------
def preprocess_image(image):
    return image.resize((224, 224))

def recognize_with_imagga(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    response = requests.post(
        "https://api.imagga.com/v2/tags",
        auth=(IMAGGA_API_KEY, IMAGGA_API_SECRET),
        files={'image': buffered}
    )
    result = response.json()
    tags = result.get("result", {}).get("tags", [])
    return [(tag["tag"]["en"], tag["confidence"]) for tag in tags[:3]] if tags else None

def recognize_offline(image):
    img_resized = preprocess_image(image)
    x = np.expand_dims(np.array(img_resized), axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return [(label.replace("_", " ").title(), float(prob)*100) for (_, label, prob) in decoded]

# ---------------------------
# CONFIDENCE ANALYSIS
# ---------------------------
def analyze_confidence(predictions):
    avg_conf = np.mean([conf for _, conf in predictions])
    if avg_conf > 80:
        return "High Confidence ✅"
    elif avg_conf > 60:
        return "Moderate Confidence ⚠️"
    else:
        return "Low Confidence ❌"

# ---------------------------
# USER PAGE
# ---------------------------
def user_dashboard():
    st.title("🐾 AI-Based Animal Recognition System")
    st.sidebar.info(f"Welcome, {st.session_state['username']}!")

    mode = st.radio("Select Input:", ("📁 Upload Image", "📷 Capture via Webcam"))
    image = None

    if mode == "📁 Upload Image":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file:
            image = Image.open(file).convert("RGB")
    else:
        camera = st.camera_input("Capture Image")
        if camera:
            image = Image.open(camera).convert("RGB")

    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("### 🔍 Detecting...")

        try:
            if check_internet():
                results = recognize_with_imagga(preprocess_image(image))
                st.success("✅ Online Mode (Imagga API)")
            else:
                st.warning("⚠️ Offline Mode Activated")
                results = recognize_offline(image)
                st.success("✅ Offline Model (MobileNetV2)")

            if results:
                confidence_status = analyze_confidence(results)
                st.info(f"**Confidence Status:** {confidence_status}")

                for name, conf in results:
                    st.write(f"**{name}** — {conf:.2f}% confidence")

                save_log(st.session_state["username"], results[0][0], results[0][1])
            else:
                st.error("No predictions found.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.subheader("📈 Recognition History")

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        user_df = df[df["User"] == st.session_state["username"]]
        st.dataframe(user_df)

        if not user_df.empty:
            st.subheader("📊 Most Recognized Animals")
            counts = user_df["Animal"].value_counts()
            plt.bar(counts.index, counts.values)
            plt.xlabel("Animal")
            plt.ylabel("Frequency")
            st.pyplot(plt)
    else:
        st.info("No recognition logs yet.")

# ---------------------------
# ADMIN DASHBOARD
# ---------------------------
def admin_dashboard():
    st.title("🧑‍💼 Admin Dashboard")
    st.sidebar.info("You are logged in as an Admin.")

    if os.path.exists(USER_DB):
        users = pd.read_csv(USER_DB)
        st.subheader("👥 Registered Users")
        st.dataframe(users[["username", "role"]])
    else:
        st.info("No users yet.")

    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        st.subheader("📜 Recognition Logs")
        st.dataframe(logs)

        if not logs.empty:
            st.subheader("📊 Animal Recognition Frequency")
            counts = logs["Animal"].value_counts()
            plt.bar(counts.index, counts.values)
            plt.xlabel("Animal")
            plt.ylabel("Count")
            st.pyplot(plt)
    else:
        st.info("No recognition logs available.")

# ---------------------------
# LOGOUT
# ---------------------------
def logout():
    st.session_state.clear()
    st.success("You have been logged out.")
    st.experimental_rerun()

# ---------------------------
# MAIN APP LOGIC
# ---------------------------
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login_page()
else:
    role = st.session_state["role"]
    st.sidebar.button("🚪 Logout", on_click=logout)
    if role == "admin":
        admin_dashboard()
    else:
        user_dashboard()
