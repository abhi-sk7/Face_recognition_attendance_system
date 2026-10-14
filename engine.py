import random
import threading
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config
from db import get_conn, get_setting


def _camera_source():
    source = config.CAMERA
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def _rotate(frame):
    rotation = getattr(config, "CAMERA_ROTATE", 0)
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotation not in rotations:
        raise ValueError("CAMERA_ROTATE must be 0, 90, 180, or 270")
    code = rotations[rotation]
    return frame if code is None else cv2.rotate(frame, code)


class Reader:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="camera-reader")
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def _loop(self):
        cap = None
        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.source)
                    if not cap.isOpened():
                        print("Camera open nahi hua; 2 second mein dobara try hoga.")
                        cap.release()
                        cap = None
                        time.sleep(2)
                        continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Camera se frame nahi mila; reconnect ho raha hai.")
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue
                with self.lock:
                    self.frame = frame
            except Exception as exc:
                print(f"Camera reader error: {exc}")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(2)
        if cap is not None:
            cap.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class Engine:
    def __init__(self):
        self.reader = Reader(_camera_source())
        self.face_app = FaceAnalysis(
            name=config.MODEL_NAME,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "landmark_3d_68"],
        )
        self.face_app.prepare(ctx_id=0, det_size=config.DET_SIZE)
        self.running = False
        self.thread = None
        self.data_lock = threading.Lock()
        self.employee_lock = threading.Lock()
        self.annot_jpg = None
        self.annot_frame = None
        self.last_faces = []
        self.tracks = {}
        self.next_track_id = 1
        self.cooldowns = {}
        self.banners = {}
        self.employee_ids = []
        self.employee_names = []
        self.embeddings = np.zeros((0, 512), dtype=np.float32)
        self.embedding_owners = []
        self.db = (self.employee_ids, self.employee_names, self.embeddings, self.embedding_owners)
        self.emp_version = None
        self._load_employees()

    def start(self):
        if self.running:
            return
        self.running = True
        self.reader.start()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="attendance-engine")
        self.thread.start()

    def stop(self):
        self.running = False
        self.reader.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def _load_employees(self):
        conn = get_conn()
        try:
            rows = conn.execute("SELECT id, name, embedding FROM employees ORDER BY id").fetchall()
            version_row = conn.execute(
                "SELECT value FROM settings WHERE key='emp_version'"
            ).fetchone()
        finally:
            conn.close()

        ids, names, matrices, owners = [], [], [], []
        for row in rows:
            raw = np.frombuffer(row["embedding"], dtype=np.float32)
            if raw.size == 0 or raw.size % 512:
                print(f"Employee {row['id']} ke face embeddings invalid hain; skip kiya.")
                continue
            gallery = raw.reshape(-1, 512).copy()
            norms = np.linalg.norm(gallery, axis=1, keepdims=True)
            valid = norms[:, 0] > 0
            gallery = gallery[valid] / norms[valid]
            if not len(gallery):
                continue
            person_index = len(ids)
            ids.append(row["id"])
            names.append(row["name"])
            matrices.append(gallery)
            owners.extend([person_index] * len(gallery))

        with self.employee_lock:
            self.employee_ids = ids
            self.employee_names = names
            self.embeddings = np.concatenate(matrices, axis=0) if matrices else np.zeros(
                (0, 512), dtype=np.float32
            )
            self.embedding_owners = owners
            self.db = (self.employee_ids, self.employee_names, self.embeddings,
                       self.embedding_owners)
        self.tracks.clear()
        self.emp_version = version_row["value"] if version_row else None
        print(f"{len(rows)} employees load hue")

    @staticmethod
    def _ear(face):
        points = face.landmark_3d_68[:, :2]

        def eye(start):
            vertical = (
                np.linalg.norm(points[start + 1] - points[start + 5])
                + np.linalg.norm(points[start + 2] - points[start + 4])
            )
            horizontal = 2 * np.linalg.norm(points[start] - points[start + 3]) + 1e-6
            return vertical / horizontal

        return float((eye(36) + eye(42)) / 2)

    def _new_state(self, now):
        return {
            "stage": "blink",
            "closed": False,
            "challenge": random.choice(("left", "right")),
            "start": now,
        }

    def _liveness(self, track, face, now):
        mode = getattr(config, "LIVENESS_MODE", "blink")
        if mode in ("none", "off"):
            return True, "OK"

        state = track.setdefault("liveness", self._new_state(now))
        timeout = getattr(config, "LIVENESS_TIMEOUT", 12)
        if now - state["start"] > timeout:
            state.update(self._new_state(now))

        ear = self._ear(face)
        history = track.setdefault("ear_history", deque(maxlen=60))
        baseline = float(np.percentile(history, 80)) if len(history) >= 5 else None
        if baseline is not None and ear < baseline * getattr(config, "BLINK_RATIO", 0.75):
            state["closed"] = True
        elif state["closed"]:
            state["closed"] = False
            if mode == "full":
                state["stage"] = "turn"
                state["start"] = now
            else:
                state["stage"] = "done"
        history.append(ear)

        if state["stage"] == "turn":
            yaw = float(face.pose[1]) * getattr(config, "YAW_SIGN", 1)
            angle = getattr(config, "YAW_ANGLE", 18)
            turned = (
                state["challenge"] == "left" and yaw > angle
            ) or (
                state["challenge"] == "right" and yaw < -angle
            )
            if turned:
                state["stage"] = "done"
            else:
                return False, f"Sar {state['challenge'].upper()} ghumao"

        if state["stage"] != "done":
            return False, "Palak jhapkayein"
        return True, "OK"

    def _mark_attendance(self, employee_id, now):
        day = now.strftime("%Y-%m-%d")
        clock = now.strftime("%H:%M:%S")
        gap_seconds = getattr(config, "MIN_CHECKOUT_GAP_MIN", 60) * 60
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT id, check_in, check_out FROM attendance WHERE emp_id=? AND date=?",
                (employee_id, day),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO attendance(emp_id,date,check_in) VALUES(?,?,?)",
                    (employee_id, day, clock),
                )
                conn.commit()
                return "CHECK-IN"
            if row["check_out"]:
                return "Already marked"
            check_in = datetime.strptime(f"{day} {row['check_in']}", "%Y-%m-%d %H:%M:%S")
            if (now - check_in).total_seconds() >= gap_seconds:
                conn.execute("UPDATE attendance SET check_out=? WHERE id=?", (clock, row["id"]))
                conn.commit()
                return "CHECK-OUT"
            return "Already marked"
        finally:
            conn.close()

    def identify(self, emb):
        with self.employee_lock:
            ids, names, known, owner = self.db
            if len(names) != len(ids):
                raise RuntimeError("Employee IDs aur names ki gallery align nahi hai.")
            if not len(known):
                return None, 0.0, 0.0
            similarities = known @ emb
            owner_array = np.asarray(owner)
            scores = []
            for person_index in range(len(ids)):
                person_sims = similarities[owner_array == person_index]
                top = np.sort(person_sims)[::-1][:3]
                scores.append(float(top.mean()))
            order = np.argsort(scores)[::-1]
            best = int(order[0])
            second = scores[int(order[1])] if len(order) > 1 else -1.0
            return best, scores[best], scores[best] - second

    def _new_track(self, face, now):
        bbox = face.bbox.astype(float)
        track = {
            "id": self.next_track_id,
            "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            "width": max(1.0, bbox[2] - bbox[0]),
            "last_seen": now,
            "votes": deque(maxlen=max(1, int(getattr(config, "VOTE_FRAMES", 8)))),
            "identity": None,
            "score": 0.0,
            "margin": 0.0,
        }
        self.next_track_id += 1
        return track

    def _assign_tracks(self, faces, now):
        expiry = 1.5
        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if now - track["last_seen"] <= expiry
        }
        unused_tracks = set(self.tracks)
        assigned = []
        for face in faces:
            bbox = face.bbox.astype(float)
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            width = max(1.0, bbox[2] - bbox[0])
            candidates = []
            for track_id in unused_tracks:
                track = self.tracks[track_id]
                distance = float(np.hypot(
                    center[0] - track["center"][0],
                    center[1] - track["center"][1],
                ))
                if distance <= width * 0.6:
                    candidates.append((distance, track_id))
            if candidates:
                _, track_id = min(candidates)
                unused_tracks.remove(track_id)
                track = self.tracks[track_id]
            else:
                track = self._new_track(face, now)
                track_id = track["id"]
                self.tracks[track_id] = track
            track["center"] = center
            track["width"] = width
            track["last_seen"] = now
            assigned.append(track)
        return assigned

    def _process_frame(self, frame):
        frame = _rotate(frame)
        faces = self.face_app.get(frame)
        now = time.time()
        annotations = []
        tracks = self._assign_tracks(faces, now)

        for face, track in zip(faces, tracks):
            x1, y1, x2, y2 = face.bbox.astype(int)
            width = x2 - x1
            label, color = "Pehchan raha hai...", (0, 165, 255)
            if width < getattr(config, "MIN_FACE_PX", 70) or float(face.det_score) < 0.5:
                label = "Paas aayein"
                color = (0, 165, 255)
            else:
                yaw = float(face.pose[1]) * getattr(config, "YAW_SIGN", 1)
                if abs(yaw) <= 45:
                    person_index, score, margin = self.identify(face.normed_embedding)
                    threshold = getattr(config, "RECOG_THRESHOLD", 0.45)
                    min_margin = getattr(config, "RECOG_MARGIN", 0.08)
                    vote = person_index if score >= threshold and margin >= min_margin else None
                    track["votes"].append(vote)
                    track["score"], track["margin"] = score, margin
                    needed = max(1, int(getattr(config, "VOTE_NEEDED", 5)))
                    confirmed = next(
                        (index for index, count in Counter(
                            value for value in track["votes"] if value is not None
                        ).most_common() if count >= needed),
                        None,
                    )
                    if confirmed is not None and confirmed != track["identity"]:
                        track["identity"] = confirmed
                        track.pop("liveness", None)
                        track.pop("ear_history", None)
                        print(
                            f"Identity confirmed: {self.employee_names[confirmed]} "
                            f"score={score:.2f} margin={margin:.2f}"
                        )

                person_index = track["identity"]
                if person_index is not None:
                    employee_id = self.employee_ids[person_index]
                    name = self.employee_names[person_index]
                    if self.cooldowns.get(employee_id, 0) > now:
                        label = f"{name} - {self.banners.get(employee_id, 'Marked')}"
                        color = (0, 200, 0)
                    else:
                        passed, prompt = self._liveness(track, face, now)
                        if passed:
                            result = self._mark_attendance(employee_id, datetime.now())
                            label = f"{name} - {result}"
                            self.banners[employee_id] = result
                            self.cooldowns[employee_id] = now + getattr(
                                config, "COOLDOWN_SEC", 300
                            )
                            track.pop("liveness", None)
                            track.pop("ear_history", None)
                            print(f"[{datetime.now():%H:%M:%S}] {name}: {result}")
                            color = (0, 200, 0)
                        else:
                            label, color = f"{name}: {prompt}", (0, 165, 255)

                    if getattr(config, "SHOW_DEBUG", False):
                        label = (
                            f"{name} {track['score']:.2f} m{track['margin']:.2f} | "
                            f"{label.split(': ', 1)[-1].split(' - ', 1)[-1]}"
                        )
                elif width >= getattr(config, "MIN_FACE_PX", 70) and float(face.det_score) >= 0.5:
                    if getattr(config, "SHOW_DEBUG", False):
                        label = f"Pehchan raha hai... {track['score']:.2f} m{track['margin']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            annotations.append(face)

        encoded, jpg = cv2.imencode(".jpg", frame)
        if not encoded:
            raise RuntimeError("Annotated camera frame JPEG encode nahi hua.")
        with self.data_lock:
            self.last_faces = annotations
            self.annot_frame = frame
            self.annot_jpg = jpg.tobytes()

    def _loop(self):
        last_version_check = 0.0
        while self.running:
            try:
                now = time.monotonic()
                if now - last_version_check >= 5:
                    last_version_check = now
                    version = get_setting("emp_version")
                    if version != self.emp_version:
                        self._load_employees()

                frame = self.reader.read()
                if frame is None:
                    time.sleep(0.03)
                    continue
                self._process_frame(frame)
            except Exception as exc:
                print(f"Attendance engine error: {exc}")
                time.sleep(0.1)

    def capture_embedding(self):
        with self.data_lock:
            if len(self.last_faces) != 1:
                return None
            face = self.last_faces[0]
            bbox = face.bbox.astype(float)
            if float(face.det_score) <= 0.7 or bbox[2] - bbox[0] < 90:
                return None
            pose = np.asarray(face.pose, dtype=float)
            if abs(float(pose[1])) > 35 or abs(float(pose[0])) > 25:
                return None
            return np.asarray(face.normed_embedding, dtype=np.float32).copy()
