from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import urllib.request
import numpy as np
import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = Flask(__name__)

# Load your trained model
# model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("yolov8n.pt")

# ── Email Configuration (loaded from .env) ───────────────────────────
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
EMAIL_COOLDOWN = int(os.getenv("EMAIL_COOLDOWN", 60))
# ─────────────────────────────────────────────────────────────────────

# Define cameras
CAMERAS = [
    {"id": 1, "name": "Main Entrance", "url": "http://192.168.29.54:8080/shot.jpg"},
    {"id": 2, "name": "Backyard", "url": "http://192.0.0.4:8080/shot.jpg"}, 
    {"id": 3, "name": "Cam 3", "url": "http://192.0.0.4:8080/shot.jpg"}
]

# Global detection status per camera
latest_detections = {cam["id"]: False for cam in CAMERAS}

# Cooldown tracker: stores the last email timestamp per camera
last_email_sent = {cam["id"]: 0 for cam in CAMERAS}


def send_alert_email(camera_name, camera_id, detected_labels, frame):
    """Send an alert email with the detection snapshot attached (runs in a thread)."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build email
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ADMIN_EMAIL
        msg["Subject"] = f"🚨 WEAPON ALERT — {camera_name} — {timestamp}"

        # Email body
        labels_str = ", ".join(detected_labels)
        body = f"""
        <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px;">
            <div style="max-width: 600px; margin: 0 auto; background: #1e293b; border-radius: 16px; overflow: hidden; border: 1px solid rgba(239,68,68,0.3); box-shadow: 0 0 30px rgba(239,68,68,0.15);">
                
                <div style="background: linear-gradient(135deg, #dc2626, #991b1b); padding: 24px 30px; text-align: center;">
                    <h1 style="margin: 0; font-size: 22px; color: white;">⚠️ Weapon Detected</h1>
                </div>
                
                <div style="padding: 30px;">
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                        <tr>
                            <td style="padding: 10px 0; color: #94a3b8; border-bottom: 1px solid #334155;">Camera</td>
                            <td style="padding: 10px 0; color: #f8fafc; font-weight: 600; border-bottom: 1px solid #334155;">{camera_name} (ID: {camera_id})</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; color: #94a3b8; border-bottom: 1px solid #334155;">Detected</td>
                            <td style="padding: 10px 0; color: #fca5a5; font-weight: 600; border-bottom: 1px solid #334155;">{labels_str}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; color: #94a3b8;">Timestamp</td>
                            <td style="padding: 10px 0; color: #f8fafc; font-weight: 600;">{timestamp}</td>
                        </tr>
                    </table>
                    
                    <p style="color: #94a3b8; font-size: 13px; margin-bottom: 15px;">📸 Detection snapshot attached below:</p>
                    <img src="cid:detection_snapshot" style="width: 100%; border-radius: 10px; border: 2px solid #334155;" />
                    
                    <div style="margin-top: 25px; padding: 15px; background: rgba(239,68,68,0.1); border-radius: 10px; border: 1px solid rgba(239,68,68,0.2);">
                        <p style="margin: 0; font-size: 13px; color: #fca5a5;">
                            🔒 This is an automated alert from the AI Surveillance System. Please take immediate action if required.
                        </p>
                    </div>
                </div>
                
                <div style="padding: 15px 30px; background: #0f172a; text-align: center;">
                    <p style="margin: 0; font-size: 11px; color: #475569;">AI Surveillance &amp; Threat Detection System</p>
                </div>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        # Attach the detection frame as image
        _, img_buffer = cv2.imencode('.jpg', frame)
        img_data = img_buffer.tobytes()
        image_attachment = MIMEImage(img_data, name=f"detection_{camera_id}_{int(time.time())}.jpg")
        image_attachment.add_header("Content-ID", "<detection_snapshot>")
        image_attachment.add_header("Content-Disposition", "inline", filename=f"detection_{camera_id}.jpg")
        msg.attach(image_attachment)

        # Send via Gmail SMTP
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.send_message(msg)

        print(f"[ALERT] Email sent for {camera_name} at {timestamp} -- Detected: {labels_str}")

    except Exception as e:
        print(f"[ERROR] Failed to send alert email: {e}")


def generate_frames(camera_id):
    global latest_detections, last_email_sent
    
    # Find the camera info
    camera = next((cam for cam in CAMERAS if cam["id"] == camera_id), None)
    if not camera:
        return

    camera_url = camera["url"]
    camera_name = camera["name"]

    while True:
        try:
            img_resp = urllib.request.urlopen(camera_url, timeout=1)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = model(frame)

            weapon_detected = False
            detected_labels = []

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # Detection logic
                    if conf > 0.6:
                        if "Rifle" in label or "Knife" in label or "Handgun" in label or "person" in label:  # "person" added for testing
                            weapon_detected = True
                            detected_labels.append(f"{label} ({conf:.0%})")

            # Update status for this specific camera
            latest_detections[camera_id] = weapon_detected

            # ── Send alert email if weapon detected (with cooldown) ──
            if weapon_detected:
                now = time.time()
                if now - last_email_sent[camera_id] > EMAIL_COOLDOWN:
                    last_email_sent[camera_id] = now
                    annotated_frame = results[0].plot()
                    # Send email in background thread so it doesn't block the stream
                    threading.Thread(
                        target=send_alert_email,
                        args=(camera_name, camera_id, detected_labels, annotated_frame),
                        daemon=True
                    ).start()
            # ─────────────────────────────────────────────────────────

            annotated = results[0].plot()

            _, buffer = cv2.imencode('.jpg', annotated)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            continue

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

@app.route('/video/<int:camera_id>')
def video(camera_id):
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"detections": latest_detections})

if __name__ == "__main__":
    # Explicitly enable threading to handle multiple concurrent streams
    app.run(debug=True, threaded=True)