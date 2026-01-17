# Habit Buddy Dashboard

A playful, real-time dashboard for the **Habit Buddy** Arduino desk companion. The UI updates every second, celebrates streak milestones, and includes a demo mode for development without hardware.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open: `http://localhost:5000`

## Demo mode

The backend ships with demo mode enabled for quick iteration:

```python
DEMO_MODE = True
```

You can also toggle Demo Mode from the dashboard UI.

## Example Arduino serial payload

```json
{"streak": 5, "milestones": {"sound": true, "nod": false, "chitter": false}, "last_checkin": 1234567890, "sensor": 512}
```

## Notes

- Serial baud rate: **9600**
- Auto-detection scans for **Arduino**, **CH340**, or **USB** in port descriptions.
- If the Arduino is disconnected, the UI shows a clear status with reconnection hints.
