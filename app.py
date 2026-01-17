import json
import random
import threading
import time
from collections import deque
from typing import Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import serial
from serial.tools import list_ports


app = Flask(__name__)
CORS(app)

DEMO_MODE = True

DATA_LOCK = threading.Lock()
WORKER_LOCK = threading.Lock()
worker_thread: Optional[threading.Thread] = None
worker_stop_event = threading.Event()
worker_mode: Optional[bool] = None

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
}

history = deque(maxlen=120)
sensor_history = deque(maxlen=60)


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
        buddy_state["streak"] = int(payload.get("streak", buddy_state["streak"]))
        buddy_state["milestones"] = payload.get("milestones", buddy_state["milestones"])
        buddy_state["last_checkin"] = payload.get("last_checkin", buddy_state["last_checkin"])
        buddy_state["sensor"] = int(payload.get("sensor", buddy_state["sensor"]))
        buddy_state["error"] = None
        history.append({
            "streak": buddy_state["streak"],
            "timestamp": int(time.time()),
        })
        sensor_history.append({
            "timestamp": int(time.time()),
            "value": buddy_state["sensor"],
        })


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
                    except json.JSONDecodeError:
                        with DATA_LOCK:
                            buddy_state["error"] = "Malformed JSON from Arduino"
                        continue
                    update_state_from_payload(payload)
        except serial.SerialException as exc:
            with DATA_LOCK:
                buddy_state["connection"] = "disconnected"
                buddy_state["error"] = f"Serial error: {exc}"
            time.sleep(2)


def generate_fake_payload() -> dict:
    streak = buddy_state["streak"]
    if random.random() < 0.12:
        streak += 1
    sensor_value = random.randint(200, 850)
    milestones = {
        "sound": streak >= 3,
        "nod": streak >= 7,
        "chitter": streak >= 14,
    }
    return {
        "streak": streak,
        "milestones": milestones,
        "last_checkin": int(time.time()),
        "sensor": sensor_value,
    }


def demo_loop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        payload = generate_fake_payload()
        update_state_from_payload(payload)
        with DATA_LOCK:
            buddy_state["connection"] = "connected"
            buddy_state["error"] = None
        time.sleep(1)


def start_background_workers() -> None:
    global worker_thread, worker_mode
    with WORKER_LOCK:
        if worker_thread and worker_thread.is_alive() and worker_mode == DEMO_MODE:
            return
        worker_stop_event.set()
        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=1)
        worker_stop_event.clear()
        if DEMO_MODE:
            worker_thread = threading.Thread(target=demo_loop, args=(worker_stop_event,), daemon=True)
        else:
            worker_thread = threading.Thread(target=serial_reader, args=(worker_stop_event,), daemon=True)
        worker_mode = DEMO_MODE
        worker_thread.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    with DATA_LOCK:
        response = {
            "streak": buddy_state["streak"],
            "milestones": buddy_state["milestones"],
            "last_checkin": buddy_state["last_checkin"],
            "sensor": buddy_state["sensor"],
            "connection": buddy_state["connection"],
            "error": buddy_state["error"],
            "history": list(history),
            "sensor_history": list(sensor_history),
            "demo_mode": DEMO_MODE,
        }
    return jsonify(response)


@app.route("/api/checkin", methods=["POST"])
def checkin():
    if DEMO_MODE:
        payload = generate_fake_payload()
        payload["streak"] = buddy_state["streak"] + 1
        payload["last_checkin"] = int(time.time())
        update_state_from_payload(payload)
        return jsonify({"status": "ok", "message": "Demo check-in received"})

    port_name = detect_arduino_port()
    if not port_name:
        return jsonify({"status": "error", "message": "Arduino not found"}), 404

    try:
        with serial.Serial(port_name, 9600, timeout=1) as ser:
            ser.write(b"CHECKIN\n")
        return jsonify({"status": "ok", "message": "Check-in sent"})
    except serial.SerialException as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/history")
def status_history():
    with DATA_LOCK:
        return jsonify({"history": list(history)})


@app.route("/api/demo", methods=["POST"])
def toggle_demo():
    global DEMO_MODE
    data = request.get_json(silent=True) or {}
    desired_state = data.get("enabled")
    if desired_state is None:
        return jsonify({"status": "error", "message": "enabled flag required"}), 400
    DEMO_MODE = bool(desired_state)
    start_background_workers()
    return jsonify({"status": "ok", "demo_mode": DEMO_MODE})


if __name__ == "__main__":
    start_background_workers()
    app.run(host="0.0.0.0", port=5000, debug=True)
