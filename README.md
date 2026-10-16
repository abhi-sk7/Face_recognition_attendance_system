# Face Recognition Attendance System

## Setup

Use Python 3.10 or 3.11. Install dependencies and create a local config:

```powershell
python -m pip install -r requirements.txt
Copy-Item config.example.py config.py
```

Edit `config.py` with your camera source and a private admin password/secret key.
For a DroidCam virtual camera, use `CAMERA = 0` after connecting the DroidCam
desktop client to the phone. Set `CAMERA_ROTATE` to `0`, `90`, `180`, or `270`
to orient the camera image.

## Run

```powershell
python app.py
```

The Flask server, live camera feed, face recognition, and attendance engine run
in the same process. Open <http://127.0.0.1:5000>, sign in as the configured
admin, and use **Live Camera** or the enrollment controls on the admin page.
InsightFace downloads its model the first time it starts.

The local `config.py` and attendance database are intentionally git-ignored.
Keep admin credentials and employee attendance data private.
