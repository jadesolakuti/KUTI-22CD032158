import os
import sqlite3
import uuid
from io import BytesIO
from flask import Flask, render_template, request, redirect
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# ==============================
# Flask Config
# ==============================
app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = "uploads"
DATABASE_FILE = "database.db"
MODEL_PATH = "emotion_model.pth"   # ✅ Correct model filename

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==============================
# Model (must match training model EXACTLY)
# ==============================
class FERModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# ==============================
# Load Model
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FERModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==============================
# Emotion labels
# ==============================
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# ==============================
# DB Setup
# ==============================
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            student_id TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            emotion TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_user(student_name, student_id, filename, emotion):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (id, student_name, student_id, image_filename, emotion)
        VALUES (?, ?, ?, ?, ?)
    """, (str(uuid.uuid4()), student_name, student_id, filename, emotion))
    conn.commit()
    conn.close()

# ==============================
# Emotion Prediction
# ==============================
def classify_emotion(image_file):
    img = Image.open(BytesIO(image_file.read()))
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        return EMOTION_LABELS[predicted.item()]

# ==============================
# Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect('/')

    image_file = request.files['image']
    student_name = request.form.get("student_name")
    student_id = request.form.get("student_id")

    if image_file.filename == "":
        return redirect('/')

    predicted_emotion = classify_emotion(image_file)

    # reset file pointer and save image
    image_file.seek(0)
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)

    save_user(student_name, student_id, filename, predicted_emotion)

    return render_template('index.html', 
                           message=f"Detected Emotion: {predicted_emotion}",
                           student_name=student_name,
                           student_id=student_id)

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
