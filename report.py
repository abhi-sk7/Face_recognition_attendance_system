from datetime import datetime, timedelta, date
from db import get_conn, get_setting

def _t(s):
    return datetime.strptime(s, "%H:%M").time()

def build_report(start, end, emp_id=None):
    """start, end = 'YYYY-MM-DD'. Returns (rows, summary)"""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = min(datetime.strptime(end, "%Y-%m-%d").date(), date.today())

    office = _t(get_setting("office_start"))
    grace = int(get_setting("grace_min"))
    hd_after = _t(get_setting("half_day_after"))
    hd_hours = float(get_setting("half_day_hours"))
    offs = {int(x) for x in get_setting("weekly_off").split(",") if x.strip().isdigit()}
    late_limit = (datetime.combine(date.today(), office) + timedelta(minutes=grace)).time()

    c = get_conn()
    if emp_id:
        emps = c.execute("SELECT * FROM employees WHERE id=?", (emp_id,)).fetchall()
    else:
        emps = c.execute("SELECT * FROM employees ORDER BY name").fetchall()
    recs = {(r["emp_id"], r["date"]): r for r in c.execute(
        "SELECT * FROM attendance WHERE date BETWEEN ? AND ?", (start, end))}
    c.close()

    rows, summary = [], {}
    for emp in emps:
        sm = summary.setdefault(emp["id"], {"name": emp["name"], "code": emp["emp_code"],
                                            "Present": 0, "Late": 0, "Half Day": 0, "Absent": 0})
        d = s
        while d <= e:
            ds = d.strftime("%Y-%m-%d")
            r = recs.get((emp["id"], ds))
            ci = co = hours = ""
            if r:
                ci, co = r["check_in"] or "", r["check_out"] or ""
                cit = datetime.strptime(ci, "%H:%M:%S").time()
                status = "Present"
                if cit > late_limit:
                    status = "Late"
                if co:
                    h = (datetime.strptime(co, "%H:%M:%S") - datetime.strptime(ci, "%H:%M:%S")).seconds / 3600
                    hours = round(h, 2)
                    if h < hd_hours:
                        status = "Half Day"
                if cit > hd_after:
                    status = "Half Day"
                sm[status] += 1
            elif d.weekday() in offs:
                status = "Weekly Off"
            else:
                status = "Absent"
                sm["Absent"] += 1
            rows.append({"date": ds, "name": emp["name"], "code": emp["emp_code"],
                         "check_in": ci, "check_out": co, "hours": hours, "status": status})
            d += timedelta(days=1)
    rows.sort(key=lambda x: (x["date"], x["name"]), reverse=True)
    return rows, list(summary.values())
