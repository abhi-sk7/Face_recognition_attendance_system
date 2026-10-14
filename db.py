import sqlite3, time
from config import DB_PATH

DEFAULT_SETTINGS = {
    "office_start": "09:30",     # office time
    "grace_min": "15",           # itne minute tak late nahi
    "half_day_after": "13:00",   # iske baad check-in = Half Day
    "half_day_hours": "4",       # isse kam kaam = Half Day
    "weekly_off": "6",           # 0=Mon ... 6=Sun (comma se alag: 5,6)
    "emp_version": "0",
}

def get_conn():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = get_conn()
    c.execute("""CREATE TABLE IF NOT EXISTS employees(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        emp_code TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        check_in TEXT,
        check_out TEXT,
        UNIQUE(emp_id, date))""")
    c.execute("CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)")
    for k, v in DEFAULT_SETTINGS.items():
        c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))
    c.commit(); c.close()

def get_setting(key):
    c = get_conn()
    r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    c.close()
    return r["value"] if r else DEFAULT_SETTINGS.get(key)

def set_setting(key, value):
    c = get_conn()
    c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, str(value)))
    c.commit(); c.close()

def bump_version():
    """Employee add/delete hone par recognize.py ko auto-reload ka signal"""
    set_setting("emp_version", str(time.time()))
