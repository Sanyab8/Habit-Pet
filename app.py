import json
import os
import random
import threading
import time
from collections import deque
from datetime import date, timedelta
from typing import Optional

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not installed. Camera features will be disabled.")

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import serial
from serial.tools import list_ports


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
DEMO_MODE = True
USE_CAMERA = True and OPENCV_AVAILABLE
PERSIST_FILE = "buddy_data.json"

# Thread management
DATA_LOCK = threading.Lock()
WORKER_LOCK = threading.Lock()
worker_thread: Optional[threading.Thread] = None
worker_stop_event = threading.Event()
worker_mode: Optional[bool] = None
camera_thread: Optional[threading.Thread] = None
camera_stop_event = threading.Event()

# State
buddy_state = {
    "streak": 0,
    "milestones": {
        "sound": False,
        "nod": False,
        "chitter": False,
    },
    "last_checkin": None,
    "sensor": 0,
    "connection": "disconnected",
    "error": None,
    "action_detected": False,
    "camera_status": "disabled",
    "last_active_date": None,
    "last_streak_date": None,
}

history = deque(maxlen=120)
sensor_history = deque(maxlen=60)


def load_persistent_data() -> None:
    """Load saved data from disk"""
    global buddy_state, history
    if not os.path.exists(PERSIST_FILE):
        return
    
    try:
        with open(PERSIST_FILE, 'r') as f:
            data = json.load(f)
            with DATA_LOCK:
                buddy_state.update({
                    "streak": data.get("streak", 0),
                    "last_active_date": data.get("last_active_date"),
                    "last_streak_date": data.get("last_streak_date"),
                    "last_checkin": data.get("last_checkin"),
                })
                update_milestones()
                if "history" in data:
                    history = deque(data["history"], maxlen=120)
        print(f"Loaded saved data: streak={buddy_state['streak']}")
    except Exception as e:
        print(f"Error loading persistent data: {e}")


def save_persistent_data() -> None:
    """Save important data to disk"""
    try:
        with DATA_LOCK:
            data = {
                "streak": buddy_state["streak"],
                "last_active_date": buddy_state["last_active_date"],
                "last_streak_date": buddy_state["last_streak_date"],
                "last_checkin": buddy_state["last_checkin"],
                "history": list(history),
            }
        with open(PERSIST_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving persistent data: {e}")


def today_key() -> str:
    """Return today's date as ISO string"""
    return date.today().isoformat()


def update_milestones() -> None:
    """Update milestone flags based on current streak"""
    buddy_state["milestones"]["sound"] = buddy_state["streak"] >= 3
    buddy_state["milestones"]["nod"] = buddy_state["streak"] >= 7
    buddy_state["milestones"]["chitter"] = buddy_state["streak"] >= 14


def evaluate_daily_streak() -> bool:
    """Evaluate and update streak based on activity. Returns True if streak was updated."""
    with DATA_LOCK:
        today = today_key()
        last_streak_date = buddy_state["last_streak_date"]
        last_active_date = buddy_state["last_active_date"]
        
        # Already evaluated today
        if last_streak_date == today:
            return False
        
        # No activity recorded yet
        if not last_active_date:
            return False
        
        # Activity is not from today
        if last_active_date != today:
            return False
        
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        
        # First day or continuing streak from yesterday
        if last_streak_date == yesterday or buddy_state["streak"] == 0:
            buddy_state["streak"] += 1
        else:
            # Missed one or more days, restart streak
            buddy_state["streak"] = 1
        
        buddy_state["last_streak_date"] = today
        buddy_state["last_checkin"] = int(time.time())
        
        # Update milestones
        update_milestones()
        
        # Save to disk
        save_persistent_data()
        
        # Add to history
        history.append({
            "streak": buddy_state["streak"],
            "timestamp": int(time.time()),
        })
        
        print(f"Streak updated: {buddy_state['streak']} days")
        return True


def record_activity(activity_date: Optional[str] = None) -> None:
    """Record that activity was detected"""
    with DATA_LOCK:
        buddy_state["last_active_date"] = activity_date or today_key()
    
    # Try to update streak
    evaluate_daily_streak()


def detect_arduino_port() -> Optional[str]:
    """Scan for an Arduino-compatible serial port."""
    ports = list_ports.comports()
    for port in ports:
        description = f"{port.description} {port.manufacturer} {port.hwid}".lower()
        if any(token in description for token in ["arduino", "ch340", "usb"]):
            return port.device
    if ports:
        return ports[0].device
    return None


def update_state_from_payload(payload: dict) -> None:
    """Update global state from Arduino payload."""
    with DATA_LOCK:
        # Update sensor reading
        buddy_state["sensor"] = int(payload.get("sensor", buddy_state["sensor"]))
        buddy_state["error"] = None
        
        # Record sensor data
        sensor_history.append({
            "timestamp": int(time.time()),
            "value": buddy_state["sensor"],
        })
        
        # If Arduino sends streak data, trust it
        if "streak" in payload:
            buddy_state["streak"] = int(payload["streak"])
            update_milestones()
        
        # If Arduino sends milestone data, use it
        if "milestones" in payload:
            buddy_state["milestones"] = payload["milestones"]
        
        # Update last checkin
        if "last_checkin" in payload:
            buddy_state["last_checkin"] = payload["last_checkin"]


def serial_reader(stop_event: threading.Event) -> None:
    """Background thread that reads JSON lines from the Arduino."""
    while not stop_event.is_set():
        port_name = detect_arduino_port()
        if not port_name:
            with DATA_LOCK:
                buddy_state["connection"] = "disconnected"
                buddy_state["error"] = "Arduino not found"
            time.sleep(2)
            continue

        try:
            with DATA_LOCK:
                buddy_state["connection"] = "reconnecting"
                buddy_state["error"] = None
            
            with serial.Serial(port_name, 9600, timeout=1) as ser:
                with DATA_LOCK:
                    buddy_state["connection"] = "connected"
                    buddy_state["error"] = None
                
                while not stop_event.is_set():
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    
                    try:
                        payload = json.loads(line)
                        update_state_from_payload(payload)
                        # Record activity when we get Arduino data
                        record_activity()
                    except json.JSONDecodeError:
                        with DATA_LOCK:
                            buddy_state["error"] = "Malformed JSON from Arduino"
                        continue
        except serial.SerialException as exc:
            with DATA_LOCK:
                buddy_state["connection"] = "disconnected"
                buddy_state["error"] = f"Serial error: {exc}"
            time.sleep(2)


def generate_fake_payload() -> dict:
    """Generate fake Arduino data for demo mode"""
    sensor_value = random.randint(200, 850)
    
    return {
        "sensor": sensor_value,
        "last_checkin": int(time.time()),
    }


def demo_loop(stop_event: threading.Event) -> None:
    """Demo mode loop - simulates Arduino"""
    while not stop_event.is_set():
        payload = generate_fake_payload()
        update_state_from_payload(payload)
        
        with DATA_LOCK:
            buddy_state["connection"] = "connected"
            buddy_state["error"] = None
            
            # Simulate camera if not using real camera
            if not USE_CAMERA:
                buddy_state["camera_status"] = "simulated"
                # Random chance of detecting action
                if random.random() < 0.15:
                    buddy_state["action_detected"] = True
                    record_activity()
                else:
                    buddy_state["action_detected"] = False
        
        time.sleep(1)


def start_background_workers() -> None:
    """Start the appropriate background worker (demo or real Arduino)"""
    global worker_thread, worker_mode
    
    with WORKER_LOCK:
        # Check if we already have the right worker running
        if worker_thread and worker_thread.is_alive() and worker_mode == DEMO_MODE:
            return
        
        # Stop existing worker
        worker_stop_event.set()
        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=1)
        
        # Start new worker
        worker_stop_event.clear()
        if DEMO_MODE:
            worker_thread = threading.Thread(
                target=demo_loop, 
                args=(worker_stop_event,), 
                daemon=True
            )
        else:
            worker_thread = threading.Thread(
                target=serial_reader, 
                args=(worker_stop_event,), 
                daemon=True
            )
        
        worker_mode = DEMO_MODE
        worker_thread.start()
        print(f"Started {'demo' if DEMO_MODE else 'Arduino'} worker")


def camera_worker(stop_event: threading.Event) -> None:
    """Background worker for camera-based motion detection"""
    if not USE_CAMERA or not OPENCV_AVAILABLE:
        with DATA_LOCK:
            buddy_state["camera_status"] = "disabled"
            buddy_state["action_detected"] = False
        return

    # Try different camera backends for Windows compatibility
    capture = None
    backends_to_try = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    for backend, name in backends_to_try:
        print(f"Trying camera with {name} backend...")
        capture = cv2.VideoCapture(0, backend)
        if capture.isOpened():
            print(f"Camera opened successfully with {name}")
            break
        capture.release()
        
        # Try secondary camera
        capture = cv2.VideoCapture(1, backend)
        if capture.isOpened():
            print(f"Secondary camera opened with {name}")
            break
        capture.release()
    
    if not capture or not capture.isOpened():
        with DATA_LOCK:
            buddy_state["camera_status"] = "disconnected"
            buddy_state["action_detected"] = False
        print("Camera could not be opened with any backend")
        return

    # Set camera properties for better performance on Windows
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with DATA_LOCK:
        buddy_state["camera_status"] = "connected"
        buddy_state["action_detected"] = False
    
    print("Camera worker started")
    background = cv2.createBackgroundSubtractorMOG2(
        history=120, 
        varThreshold=25, 
        detectShadows=False
    )

    consecutive_failures = 0
    max_failures = 10

    while not stop_event.is_set():
        success, frame = capture.read()
        if not success:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                with DATA_LOCK:
                    buddy_state["camera_status"] = "disconnected"
                    buddy_state["action_detected"] = False
                print(f"Camera failed {max_failures} times, stopping worker")
                break
            time.sleep(1)
            continue

        consecutive_failures = 0  # Reset on success
        
        try:
            mask = background.apply(frame)
            motion_ratio = cv2.countNonZero(mask) / mask.size
            action_detected = motion_ratio > 0.02
            
            with DATA_LOCK:
                buddy_state["camera_status"] = "connected"
                buddy_state["action_detected"] = action_detected
            
            if action_detected:
                record_activity()
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        time.sleep(0.2)

    capture.release()
    print("Camera worker stopped")


def start_camera_worker() -> None:
    """Start the camera worker thread"""
    global camera_thread
    
    if not USE_CAMERA:
        return
    
    if camera_thread and camera_thread.is_alive():
        return
    
    camera_stop_event.clear()
    camera_thread = threading.Thread(
        target=camera_worker, 
        args=(camera_stop_event,), 
        daemon=True
    )
    camera_thread.start()
    print("Camera thread started")


# Routes

@app.route("/")
def index():
    """Serve the main dashboard from the static folder"""
    try:
        # This tells Flask to look specifically in your 'static' folder
        return send_from_directory(app.static_folder, 'index.html')
    except Exception:
        # Fallback debug info if the file is still missing
        base_dir = os.path.dirname(os.path.abspath(__file__))
        static_path = os.path.join(base_dir, 'static')
        return f"""
        <h1>index.html Not Found in Static</h1>
        <p><strong>Checking folder:</strong> <code>{static_path}</code></p>
        <p><strong>Files found there:</strong> {os.listdir(static_path) if os.path.exists(static_path) else 'Folder missing!'}</p>
        """, 404

@app.route("/api/status")
def status():
    """Get current buddy status"""
    with DATA_LOCK:
        response = {
            "streak": buddy_state["streak"],
            "milestones": buddy_state["milestones"],
            "last_checkin": buddy_state["last_checkin"],
            "sensor": buddy_state["sensor"],
            "connection": buddy_state["connection"],
            "error": buddy_state["error"],
            "camera_status": buddy_state["camera_status"],
            "action_detected": buddy_state["action_detected"],
            "history": list(history),
            "sensor_history": list(sensor_history),
            "demo_mode": DEMO_MODE,
        }
    return jsonify(response)


@app.route("/api/checkin", methods=["POST"])
def checkin():
    """Manual check-in endpoint"""
    # Record activity for today
    record_activity()
    
    if DEMO_MODE:
        # In demo mode, just record the activity
        with DATA_LOCK:
            was_today = buddy_state["last_streak_date"] == today_key()
        
        if was_today:
            return jsonify({
                "status": "ok", 
                "message": "Check-in already logged today"
            })
        else:
            return jsonify({
                "status": "ok", 
                "message": "Daily check-in recorded!"
            })
    
    # In real mode, send command to Arduino
    port_name = detect_arduino_port()
    if not port_name:
        return jsonify({
            "status": "error", 
            "message": "Arduino not found"
        }), 404

    try:
        with serial.Serial(port_name, 9600, timeout=1) as ser:
            ser.write(b"CHECKIN\n")
        return jsonify({"status": "ok", "message": "Check-in sent to Buddy"})
    except serial.SerialException as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/history")
def status_history():
    """Get check-in history"""
    with DATA_LOCK:
        return jsonify({"history": list(history)})


@app.route("/api/demo", methods=["POST"])
def toggle_demo():
    """Toggle demo mode on/off"""
    global DEMO_MODE
    
    data = request.get_json(silent=True) or {}
    desired_state = data.get("enabled")
    
    if desired_state is None:
        return jsonify({
            "status": "error", 
            "message": "enabled flag required"
        }), 400
    
    DEMO_MODE = bool(desired_state)
    start_background_workers()
    
    return jsonify({"status": "ok", "demo_mode": DEMO_MODE})


@app.route("/api/reset", methods=["POST"])
def reset_data():
    """Reset all streak data (for testing)"""
    with DATA_LOCK:
        buddy_state["streak"] = 0
        buddy_state["last_active_date"] = None
        buddy_state["last_streak_date"] = None
        buddy_state["last_checkin"] = None
        update_milestones()
        history.clear()
        sensor_history.clear()
    
    save_persistent_data()
    
    return jsonify({"status": "ok", "message": "All data reset"})


if __name__ == "__main__":
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Load saved data
    load_persistent_data()
    
    # Start workers
    start_background_workers()
    start_camera_worker()
    
    print(f"Starting Habit Buddy server in {'DEMO' if DEMO_MODE else 'REAL'} mode")
    print(f"Camera: {'ENABLED' if USE_CAMERA else 'DISABLED'}")
    print(f"Current streak: {buddy_state['streak']} days")
    
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)