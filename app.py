import streamlit as st
import cv2
import torch
import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Face Attendance System", layout="centered")

DATA_DIR = "data"
EMB_FILE = os.path.join(DATA_DIR, "embeddings.pkl")
ATT_FILE = os.path.join(DATA_DIR, "attendance.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    mtcnn = MTCNN(keep_all=False)
    model = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, model

mtcnn, model = load_models()

# -------------------- UTILS --------------------
def load_embeddings():
    database = {}
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE, "rb") as f:
            while True:
                try:
                    data = pickle.load(f)
                    database.update(data)
                except EOFError:
                    break
    return database

def save_embeddings(name, embeddings):
    with open(EMB_FILE, "ab") as f:
        pickle.dump({name: embeddings}, f)

def recognize_face(embedding, database):
    min_dist = 999
    identity = "Unknown"
    for name, embs in database.items():
        for db_emb in embs:
            dist = torch.norm(embedding - torch.tensor(db_emb))
            if dist < min_dist and dist < 0.9:
                min_dist = dist
                identity = name
    return identity

def mark_attendance(name, action):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [name, action, time]

    if not os.path.exists(ATT_FILE):
        pd.DataFrame(columns=["Name", "Action", "Time"]).to_csv(ATT_FILE, index=False)

    pd.DataFrame([row], columns=["Name", "Action", "Time"]).to_csv(
        ATT_FILE, mode="a", header=False, index=False
    )

# -------------------- UI --------------------
st.title("🎯 Face Authentication Attendance System")

menu = st.sidebar.selectbox(
    "Select Action",
    ["Register Face", "Punch IN", "Punch OUT", "View Attendance"]
)

# -------------------- REGISTER --------------------
if menu == "Register Face":
    st.header("👤 Register New User")
    name = st.text_input("Enter User Name")
    samples = st.slider("Number of samples", 3, 10, 5)

    if st.button("Start Registration"):
        if not name:
            st.error("Enter user name")
        else:
            embeddings = []
            st.info("Capture face images")

            for i in range(samples):
                img = st.camera_input(f"Capture Image {i+1}")
                if img:
                    bytes_data = np.asarray(bytearray(img.read()), dtype=np.uint8)
                    frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
                    face = mtcnn(frame)
                    if face is not None:
                        emb = model(face.unsqueeze(0)).detach().numpy()
                        embeddings.append(emb)
                        st.success(f"Captured {i+1}")
                    else:
                        st.warning("Face not detected")

            if embeddings:
                save_embeddings(name, embeddings)
                st.success(f"User {name} registered successfully")

# -------------------- PUNCH IN --------------------
elif menu == "Punch IN":
    st.header("🟢 Punch IN")
    img = st.camera_input("Capture Face")

    if img:
        bytes_data = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        face = mtcnn(frame)

        if face is None:
            st.error("No face detected")
        else:
            emb = model(face.unsqueeze(0)).detach()
            database = load_embeddings()
            identity = recognize_face(emb, database)

            if identity == "Unknown":
                st.error("Face not recognized")
            else:
                mark_attendance(identity, "IN")
                st.success(f"{identity} punched IN")

# -------------------- PUNCH OUT --------------------
elif menu == "Punch OUT":
    st.header("🔴 Punch OUT")
    img = st.camera_input("Capture Face")

    if img:
        bytes_data = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        face = mtcnn(frame)

        if face is None:
            st.error("No face detected")
        else:
            emb = model(face.unsqueeze(0)).detach()
            database = load_embeddings()
            identity = recognize_face(emb, database)

            if identity == "Unknown":
                st.error("Face not recognized")
            else:
                mark_attendance(identity, "OUT")
                st.success(f"{identity} punched OUT")

# -------------------- VIEW ATTENDANCE --------------------
elif menu == "View Attendance":
    st.header("📄 Attendance Records")
    if os.path.exists(ATT_FILE):
        df = pd.read_csv(ATT_FILE)
        st.dataframe(df)
    else:
        st.info("No attendance records yet")
