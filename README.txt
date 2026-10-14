FACE ATTENDANCE SYSTEM
======================
1) Python 3.10 / 3.11 install karein
2) pip install -r requirements.txt
3) config.py mein CAMERA, CAMERA_ROTATE aur admin password set karein
4) Run `python app.py` -> browser: http://127.0.0.1:5000
   App, live camera feed aur attendance ek hi process mein chalenge.
5) Admin login karke "Live Camera" dekhein. Enrollment scan ke liye /admin par
   "Scan karke Save" dabayein; camera browser se share nahi hota.

Admin login: config.py ka ADMIN_USER / ADMIN_PASSWORD
Employee login: apni Employee ID + password (apni attendance dekhne ke liye)

Face attendance:
- CAMERA=0 DroidCam virtual camera use karta hai; DroidCam desktop client aur phone feed pehle connect karein.
- CAMERA_ROTATE ko 0, 90, 180, ya 270 set karke camera image seedha karein.
- LIVENESS_MODE="blink" par employee ko blink karna hoga; blink ke baad attendance mark hoti hai.
- Admin enrollment mein screen par dikhaye gaye chhe poses ko follow karein.
- Engine same camera feed ko recognition aur enrollment dono ke liye share karta hai.

Pehli baar chalane par insightface model (~100-300MB) khud download hoga (internet chahiye).
