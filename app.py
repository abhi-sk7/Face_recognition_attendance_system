import io
import threading
import time
from datetime import date
from functools import wraps

import numpy as np
from flask import (Flask, Response, request, render_template, redirect, session,
                   jsonify, send_file, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from openpyxl import Workbook

import config
from db import get_conn, init_db, get_setting, set_setting, bump_version
from engine import Engine
from report import build_report

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
init_db()

eng = Engine()
eng.start()
print("Camera aur attendance engine start ho gaya.")

pending = {}
pending_lock = threading.Lock()


def admin_required(f):
    @wraps(f)
    def w(*a, **k):
        if not session.get("admin"):
            return redirect("/login")
        return f(*a, **k)
    return w
 

def emp_required(f):
    @wraps(f)
    def w(*a, **k):
        if not session.get("emp_id"):
            return redirect("/login")
        return f(*a, **k)
    return w


@app.route("/")
def home():
    return redirect("/admin" if session.get("admin") else "/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, p = request.form["u"].strip(), request.form["p"]
        if u == config.ADMIN_USER and p == config.ADMIN_PASSWORD:
            session.clear(); session["admin"] = True
            return redirect("/admin")
        c = get_conn()
        e = c.execute("SELECT * FROM employees WHERE emp_code=?", (u,)).fetchone()
        c.close()
        if e and check_password_hash(e["password_hash"], p):
            session.clear(); session["emp_id"] = e["id"]
            return redirect("/my")
        flash("Galat ID ya password")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------------- ADMIN ----------------
@app.route("/admin")
@admin_required
def admin():
    c = get_conn()
    emps = c.execute("SELECT id, name, emp_code, created FROM employees ORDER BY name").fetchall()
    today = c.execute("""SELECT e.name, a.check_in, a.check_out FROM attendance a
                         JOIN employees e ON e.id=a.emp_id WHERE a.date=? ORDER BY a.check_in""",
                      (date.today().strftime("%Y-%m-%d"),)).fetchall()
    c.close()
    settings = {k: get_setting(k) for k in
                ["office_start", "grace_min", "half_day_after", "half_day_hours", "weekly_off"]}
    return render_template("admin.html", emps=emps, today=today, s=settings)


def filter_embeddings(embeddings):
    matrix = np.asarray(embeddings, dtype=np.float32).reshape(-1, 512)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    valid = norms[:, 0] > 0
    matrix = matrix[valid] / norms[valid]
    if not len(matrix):
        return np.empty((0, 512), dtype=np.float32)
    centroid = matrix.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return np.empty((0, 512), dtype=np.float32)
    centroid /= norm
    matrix = matrix[(matrix @ centroid) > 0.35]

    unique = []
    for embedding in matrix:
        if all(float(np.dot(embedding, kept)) < 0.97 for kept in unique):
            unique.append(embedding)
        if len(unique) == 20:
            break
    return np.asarray(unique, dtype=np.float32).reshape(-1, 512)


def duplicate_face(c, embeddings, exclude_id=None):
    rows = c.execute("SELECT id, name, embedding FROM employees").fetchall()
    for row in rows:
        if row["id"] == exclude_id:
            continue
        saved = np.frombuffer(row["embedding"], dtype=np.float32).reshape(-1, 512)
        saved_norms = np.linalg.norm(saved, axis=1, keepdims=True)
        saved = saved / np.maximum(saved_norms, 1e-12)
        similarities = embeddings @ saved.T
        scores = [
            float(np.sort(sample_sims)[::-1][:3].mean())
            for sample_sims in similarities
        ]
        if scores and max(scores) > 0.55:
            return row["name"]
    return None


@app.route("/video_feed")
@admin_required
def video_feed():
    def stream():
        while True:
            with eng.data_lock:
                jpg = eng.annot_jpg
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)

    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live")
@admin_required
def live():
    return render_template("live.html")


@app.route("/enroll_capture", methods=["POST"])
@admin_required
def enroll_capture():
    data = request.get_json(silent=True) or {}
    token = data.get("token")
    if not isinstance(token, str) or not token:
        return jsonify(ok=False, n=0, msg="Scan token invalid hai."), 400
    embedding = eng.capture_embedding()
    with pending_lock:
        samples = pending.setdefault(token, [])
        if embedding is not None:
            samples.append(embedding)
        count = len(samples)
    return jsonify(ok=embedding is not None, n=count)


@app.route("/enroll", methods=["POST"])
@admin_required
def enroll():
    data = request.get_json(silent=True) or {}
    token = data.get("token")
    if not isinstance(token, str) or not token:
        return jsonify(ok=False, msg="Scan token missing hai."), 400
    with pending_lock:
        embeddings = pending.get(token)
        if embeddings is None or len(embeddings) < 4:
            count = len(embeddings) if embeddings else 0
            return jsonify(ok=False, msg=f"Kam se kam 4 clear face samples chahiye (abhi {count}).")
        embeddings = [embedding.copy() for embedding in embeddings]

    matrix = filter_embeddings(embeddings)
    if len(matrix) < 4:
        return jsonify(ok=False, msg="Angle samples match nahi hue. Scan dobara karein.")

    employee_id = data.get("employee_id")
    name = str(data.get("name", "")).strip()
    code = str(data.get("emp_code", "")).strip()
    password = str(data.get("password", ""))
    if employee_id is None and not (name and code and password):
        return jsonify(ok=False, msg="Naam, Employee ID aur password teeno bharein.")

    conn = get_conn()
    try:
        employee = None
        if employee_id is not None:
            employee = conn.execute(
                "SELECT id, name FROM employees WHERE id=?", (employee_id,)
            ).fetchone()
            if employee is None:
                return jsonify(ok=False, msg="Employee nahi mila.")
        duplicate = duplicate_face(
            conn, matrix, exclude_id=employee["id"] if employee else None
        )
        if duplicate:
            return jsonify(ok=False, msg=f"Ye face pehle se '{duplicate}' ke naam se registered hai.")

        if employee:
            conn.execute(
                "UPDATE employees SET embedding=? WHERE id=?",
                (matrix.tobytes(), employee["id"]),
            )
            result_message = f"{employee['name']} ke face samples update ho gaye."
        else:
            conn.execute(
                "INSERT INTO employees(name,emp_code,password_hash,embedding) VALUES(?,?,?,?)",
                (name, code, generate_password_hash(password), matrix.tobytes()),
            )
            result_message = f"{name} add ho gaya."
        conn.commit()
    except Exception:
        conn.rollback()
        app.logger.exception("Employee face enrollment failed")
        return jsonify(ok=False, msg="Employee save nahi hua; ID duplicate ho sakti hai."), 400
    finally:
        conn.close()

    with pending_lock:
        pending.pop(token, None)
    bump_version()
    return jsonify(ok=True, msg=result_message)


@app.route("/delete/<int:eid>", methods=["POST"])
@admin_required
def delete(eid):
    c = get_conn()
    c.execute("DELETE FROM employees WHERE id=?", (eid,))
    c.execute("DELETE FROM attendance WHERE emp_id=?", (eid,))
    c.commit(); c.close()
    bump_version()
    return redirect("/admin")


@app.route("/settings", methods=["POST"])
@admin_required
def settings():
    for k in ["office_start", "grace_min", "half_day_after", "half_day_hours", "weekly_off"]:
        set_setting(k, request.form[k].strip())
    return redirect("/admin")


@app.route("/report")
@admin_required
def report():
    today = date.today()
    start = request.args.get("start", today.replace(day=1).strftime("%Y-%m-%d"))
    end = request.args.get("end", today.strftime("%Y-%m-%d"))
    rows, summary = build_report(start, end)
    return render_template("report.html", rows=rows, summary=summary, start=start, end=end)


@app.route("/report/download")
@admin_required
def download():
    today = date.today()
    start = request.args.get("start", today.replace(day=1).strftime("%Y-%m-%d"))
    end = request.args.get("end", today.strftime("%Y-%m-%d"))
    rows, summary = build_report(start, end)

    wb = Workbook()
    ws = wb.active; ws.title = "Attendance"
    ws.append(["Date", "Name", "Emp ID", "Check In", "Check Out", "Hours", "Status"])
    for r in rows:
        ws.append([r["date"], r["name"], r["code"], r["check_in"], r["check_out"],
                   r["hours"], r["status"]])
    ws2 = wb.create_sheet("Summary")
    ws2.append(["Name", "Emp ID", "Present", "Late", "Half Day", "Absent"])
    for s in summary:
        ws2.append([s["name"], s["code"], s["Present"], s["Late"], s["Half Day"], s["Absent"]])
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"attendance_{start}_to_{end}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------- EMPLOYEE ----------------
@app.route("/my")
@emp_required
def my():
    today = date.today()
    start = request.args.get("start", today.replace(day=1).strftime("%Y-%m-%d"))
    end = request.args.get("end", today.strftime("%Y-%m-%d"))
    rows, summary = build_report(start, end, emp_id=session["emp_id"])
    return render_template("my.html", rows=rows, summary=summary, start=start, end=end)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
