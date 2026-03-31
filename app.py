import csv
import io
import os
import re
import sqlite3
from datetime import datetime
from functools import wraps

import cv2
import face_recognition
import joblib
import numpy as np
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.path.join(BASE_DIR, "face_encodings.pkl")
DB_FILE = os.path.join(BASE_DIR, "attendance.db")
LEGACY_ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance.csv")
LOW_ATTENDANCE_THRESHOLD = 75.0

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret-change-this")


def normalize_role(role):
    role_val = (role or "").strip().lower()
    if role_val == "teacher":
        return "faculty"
    return role_val


def slugify_username(value):
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "student"


def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def column_exists(conn, table_name, column_name):
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(col["name"] == column_name for col in columns)


def seed_default_users(conn):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    has_admin = conn.execute(
        "SELECT 1 FROM users WHERE lower(role) = 'admin' LIMIT 1"
    ).fetchone()
    if not has_admin:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin", "Admin", now),
        )

    has_faculty = conn.execute(
        "SELECT 1 FROM users WHERE lower(role) IN ('faculty', 'teacher') LIMIT 1"
    ).fetchone()
    if not has_faculty:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
            ("faculty", generate_password_hash("faculty123"), "faculty", "Faculty", now),
        )


def seed_students_from_known_faces(conn):
    if not known_names:
        return

    existing_users = {
        row["username"].lower()
        for row in conn.execute("SELECT username FROM users").fetchall()
    }
    existing_full_names = {
        (row["full_name"] or "").strip().lower()
        for row in conn.execute("SELECT full_name FROM users").fetchall()
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for full_name in sorted(set(known_names)):
        if full_name.strip().lower() in existing_full_names:
            continue

        base = slugify_username(full_name)
        username = base
        idx = 1
        while username.lower() in existing_users:
            idx += 1
            username = f"{base}{idx}"

        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                username,
                generate_password_hash("student123"),
                "student",
                full_name.strip(),
                now,
            ),
        )
        existing_users.add(username.lower())
        existing_full_names.add(full_name.strip().lower())


def init_db():
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semesters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                semester_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(semester_id, name),
                FOREIGN KEY (semester_id) REFERENCES semesters(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT DEFAULT 'camera',
                marked_by TEXT DEFAULT 'system',
                faculty_user_id INTEGER,
                semester_id INTEGER,
                subject_id INTEGER,
                FOREIGN KEY (faculty_user_id) REFERENCES users(id) ON DELETE SET NULL,
                FOREIGN KEY (semester_id) REFERENCES semesters(id) ON DELETE SET NULL,
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faculty_subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                faculty_user_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(faculty_user_id, subject_id),
                FOREIGN KEY (faculty_user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            )
            """
        )

        if not column_exists(conn, "users", "full_name"):
            conn.execute("ALTER TABLE users ADD COLUMN full_name TEXT")

        attendance_new_columns = {
            "source": "TEXT DEFAULT 'camera'",
            "marked_by": "TEXT DEFAULT 'system'",
            "faculty_user_id": "INTEGER",
            "semester_id": "INTEGER",
            "subject_id": "INTEGER",
        }
        for col_name, col_type in attendance_new_columns.items():
            if not column_exists(conn, "attendance", col_name):
                conn.execute(f"ALTER TABLE attendance ADD COLUMN {col_name} {col_type}")

        # Backfill faculty_user_id for historical rows based on marked_by username.
        conn.execute(
            """
            UPDATE attendance
            SET faculty_user_id = (
                SELECT u.id FROM users u
                WHERE lower(u.username) = lower(attendance.marked_by)
                  AND lower(u.role) IN ('faculty', 'teacher')
                LIMIT 1
            )
            WHERE faculty_user_id IS NULL AND marked_by IS NOT NULL
            """
        )

        seed_default_users(conn)
        seed_students_from_known_faces(conn)

        conn.commit()
    finally:
        conn.close()


def migrate_legacy_csv_if_needed():
    if not os.path.exists(LEGACY_ATTENDANCE_CSV):
        return

    conn = get_db_connection()
    try:
        existing = conn.execute("SELECT COUNT(*) AS c FROM attendance").fetchone()["c"]
        if existing > 0:
            return

        with open(LEGACY_ATTENDANCE_CSV, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows_to_insert = []
            for row in reader:
                name = (row.get("Name") or "").strip()
                timestamp = (row.get("Timestamp") or "").strip()
                if not name or not timestamp:
                    continue
                rows_to_insert.append((name, timestamp, "legacy-csv", "system", None, None))

        if rows_to_insert:
            conn.executemany(
                "INSERT INTO attendance (name, timestamp, source, marked_by, semester_id, subject_id) VALUES (?, ?, ?, ?, ?, ?)",
                rows_to_insert,
            )
            conn.commit()
    finally:
        conn.close()


def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        raise FileNotFoundError(
            "face_encodings.pkl not found. Run train_model.py first to generate encodings."
        )

    data = joblib.load(ENCODINGS_FILE)
    encodings = data.get("encodings", [])
    names = data.get("names", [])

    if not encodings or not names:
        raise ValueError("face_encodings.pkl exists but has no valid encoding data.")

    return encodings, names


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            if (
                request.path.startswith("/attendance")
                or request.path.startswith("/recognize")
                or request.path.startswith("/api/")
            ):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


def roles_required(*roles):
    allowed = {normalize_role(role) for role in roles}

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not session.get("user"):
                return jsonify({"error": "Unauthorized"}), 401
            if normalize_role(session.get("role")) not in allowed:
                return jsonify({"error": "Forbidden"}), 403
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


def get_faculty_context(require_selected=True):
    semester_id = session.get("active_semester_id")
    subject_id = session.get("active_subject_id")
    faculty_user_id = session.get("user_id")

    if not semester_id or not subject_id:
        if require_selected:
            return None
        return {
            "semester_id": None,
            "subject_id": None,
            "semester_name": None,
            "subject_name": None,
        }

    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT s.id AS semester_id, s.name AS semester_name,
                   sub.id AS subject_id, sub.name AS subject_name
            FROM subjects sub
            JOIN semesters s ON s.id = sub.semester_id
            JOIN faculty_subjects fs ON fs.subject_id = sub.id
            WHERE s.id = ? AND sub.id = ? AND fs.faculty_user_id = ?
            """,
            (semester_id, subject_id, faculty_user_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        session.pop("active_semester_id", None)
        session.pop("active_subject_id", None)
        return None

    return {
        "semester_id": row["semester_id"],
        "subject_id": row["subject_id"],
        "semester_name": row["semester_name"],
        "subject_name": row["subject_name"],
    }


def read_attendance_rows(
    semester_id=None,
    subject_id=None,
    student_name=None,
    faculty_user_id=None,
    limit=None,
):
    clauses = []
    params = []

    if semester_id:
        clauses.append("a.semester_id = ?")
        params.append(semester_id)
    if subject_id:
        clauses.append("a.subject_id = ?")
        params.append(subject_id)
    if student_name:
        clauses.append("lower(a.name) = lower(?)")
        params.append(student_name)
    if faculty_user_id:
        clauses.append("a.faculty_user_id = ?")
        params.append(faculty_user_id)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = "LIMIT ?" if isinstance(limit, int) and limit > 0 else ""

    query = f"""
        SELECT a.id, a.name, a.timestamp, a.source, a.marked_by,
               sem.id AS semester_id, sem.name AS semester_name,
               sub.id AS subject_id, sub.name AS subject_name
        FROM attendance a
        LEFT JOIN semesters sem ON sem.id = a.semester_id
        LEFT JOIN subjects sub ON sub.id = a.subject_id
        {where_sql}
        ORDER BY a.timestamp DESC
        {limit_sql}
    """

    if limit_sql:
        params.append(limit)

    conn = get_db_connection()
    try:
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "timestamp": row["timestamp"],
                "source": row["source"],
                "marked_by": row["marked_by"],
                "semester_id": row["semester_id"],
                "semester_name": row["semester_name"],
                "subject_id": row["subject_id"],
                "subject_name": row["subject_name"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def already_marked_today(name, semester_id, subject_id, faculty_user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM attendance
            WHERE lower(name) = lower(?)
              AND date(timestamp) = ?
              AND semester_id = ?
              AND subject_id = ?
              AND faculty_user_id = ?
            LIMIT 1
            """,
            (name, today, semester_id, subject_id, faculty_user_id),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def mark_attendance(name, semester_id, subject_id, source="camera"):
    faculty_user_id = session.get("user_id")
    if already_marked_today(name, semester_id, subject_id, faculty_user_id):
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    marked_by = session.get("user", "system")
    conn = get_db_connection()
    try:
        conn.execute(
                """
            INSERT INTO attendance (name, timestamp, source, marked_by, faculty_user_id, semester_id, subject_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                timestamp,
                source,
                marked_by,
                faculty_user_id,
                semester_id,
                subject_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return True


def recognize_from_rgb_image(
    rgb_image,
    known_encodings,
    known_names,
    semester_id,
    subject_id,
    source="camera",
    threshold=0.45,
):
    face_locations = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, face_locations)

    recognized = []
    marked_now = []

    for face_encoding in encodings:
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if face_distances.size == 0:
            continue

        best_idx = int(np.argmin(face_distances))
        best_distance = face_distances[best_idx]

        if best_distance < threshold:
            name = known_names[best_idx]
            if name not in recognized:
                recognized.append(name)
                if mark_attendance(name, semester_id, subject_id, source=source):
                    marked_now.append(name)

    return {
        "recognized": recognized,
        "marked_now": marked_now,
        "face_count": len(encodings),
    }


def build_attendance_csv_bytes(rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Name",
            "Semester",
            "Subject",
            "Timestamp",
            "Source",
            "MarkedBy",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["name"],
                row.get("semester_name") or "-",
                row.get("subject_name") or "-",
                row["timestamp"],
                row["source"],
                row["marked_by"],
            ]
        )

    bytes_buffer = io.BytesIO(output.getvalue().encode("utf-8"))
    bytes_buffer.seek(0)
    return bytes_buffer


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_semesters_with_subject_counts(faculty_user_id=None):
    conn = get_db_connection()
    try:
        if faculty_user_id:
            rows = conn.execute(
                """
                SELECT sem.id, sem.name, sem.created_at, COUNT(sub.id) AS subject_count
                FROM semesters sem
                JOIN subjects sub ON sub.semester_id = sem.id
                JOIN faculty_subjects fs ON fs.subject_id = sub.id
                WHERE fs.faculty_user_id = ?
                GROUP BY sem.id, sem.name, sem.created_at
                ORDER BY sem.id DESC
                """,
                (faculty_user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT sem.id, sem.name, sem.created_at, COUNT(sub.id) AS subject_count
                FROM semesters sem
                LEFT JOIN subjects sub ON sub.semester_id = sem.id
                GROUP BY sem.id, sem.name, sem.created_at
                ORDER BY sem.id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_subjects_for_semester(semester_id=None, faculty_user_id=None):
    conn = get_db_connection()
    try:
        if faculty_user_id and semester_id:
            rows = conn.execute(
                """
                SELECT sub.id, sub.name, sub.semester_id, sem.name AS semester_name, sub.created_at
                FROM subjects sub
                JOIN semesters sem ON sem.id = sub.semester_id
                JOIN faculty_subjects fs ON fs.subject_id = sub.id
                WHERE sub.semester_id = ? AND fs.faculty_user_id = ?
                ORDER BY sub.name
                """,
                (semester_id, faculty_user_id),
            ).fetchall()
        elif faculty_user_id:
            rows = conn.execute(
                """
                SELECT sub.id, sub.name, sub.semester_id, sem.name AS semester_name, sub.created_at
                FROM subjects sub
                JOIN semesters sem ON sem.id = sub.semester_id
                JOIN faculty_subjects fs ON fs.subject_id = sub.id
                WHERE fs.faculty_user_id = ?
                ORDER BY sem.id DESC, sub.name
                """,
                (faculty_user_id,),
            ).fetchall()
        elif semester_id:
            rows = conn.execute(
                """
                SELECT sub.id, sub.name, sub.semester_id, sem.name AS semester_name, sub.created_at
                FROM subjects sub
                JOIN semesters sem ON sem.id = sub.semester_id
                WHERE sub.semester_id = ?
                ORDER BY sub.name
                """,
                (semester_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT sub.id, sub.name, sub.semester_id, sem.name AS semester_name, sub.created_at
                FROM subjects sub
                JOIN semesters sem ON sem.id = sub.semester_id
                ORDER BY sem.id DESC, sub.name
                """
            ).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_student_summary(student_name):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT sub.id AS subject_id,
                   sub.name AS subject_name,
                   sem.name AS semester_name,
                   COUNT(DISTINCT date(a.timestamp)) AS total_classes,
                   COUNT(DISTINCT CASE WHEN lower(a.name) = lower(?) THEN date(a.timestamp) END) AS attended_classes
            FROM subjects sub
            JOIN semesters sem ON sem.id = sub.semester_id
            LEFT JOIN attendance a ON a.subject_id = sub.id
            GROUP BY sub.id, sub.name, sem.name
            ORDER BY sem.id, sub.name
            """,
            (student_name,),
        ).fetchall()
    finally:
        conn.close()

    subject_wise = []
    total_classes = 0
    total_attended = 0
    alerts = []

    for row in rows:
        total_c = int(row["total_classes"] or 0)
        att_c = int(row["attended_classes"] or 0)
        percentage = round((att_c / total_c) * 100, 2) if total_c > 0 else 0.0

        entry = {
            "subject_id": row["subject_id"],
            "subject_name": row["subject_name"],
            "semester_name": row["semester_name"],
            "total_classes": total_c,
            "attended_classes": att_c,
            "attendance_percentage": percentage,
        }
        subject_wise.append(entry)

        total_classes += total_c
        total_attended += att_c

        if total_c > 0 and percentage < LOW_ATTENDANCE_THRESHOLD:
            alerts.append(
                {
                    "semester_name": row["semester_name"],
                    "subject_name": row["subject_name"],
                    "attendance_percentage": percentage,
                    "required_threshold": LOW_ATTENDANCE_THRESHOLD,
                }
            )

    overall_percentage = round((total_attended / total_classes) * 100, 2) if total_classes > 0 else 0.0

    return {
        "student_name": student_name,
        "overall_percentage": overall_percentage,
        "subject_wise": subject_wise,
        "low_attendance_alerts": alerts,
    }


try:
    known_encodings, known_names = load_encodings()
    model_error = None
except Exception as exc:
    known_encodings, known_names = [], []
    model_error = str(exc)

init_db()
migrate_legacy_csv_if_needed()


@app.route("/")
def index():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    role = normalize_role(session.get("role"))
    if role == "admin":
        return redirect(url_for("admin_dashboard"))
    if role == "faculty":
        return redirect(url_for("faculty_dashboard"))
    session.clear()
    return redirect(url_for("login"))


@app.route("/admin/dashboard")
@login_required
@roles_required("admin")
def admin_dashboard():
    return render_template(
        "admin_dashboard.html",
        user=session.get("user"),
        role=normalize_role(session.get("role")),
    )


@app.route("/faculty/dashboard")
@login_required
@roles_required("faculty")
def faculty_dashboard():
    return render_template(
        "faculty_dashboard.html",
        user=session.get("user"),
        role=normalize_role(session.get("role")),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("user"):
            return redirect(url_for("dashboard"))
        return render_template("login.html")

    payload = request.get_json(silent=True) or request.form
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    conn = get_db_connection()
    try:
        user = conn.execute(
            """
            SELECT id, username, password_hash, role,
                   COALESCE(NULLIF(full_name, ''), username) AS full_name
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
    finally:
        conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    role = normalize_role(user["role"])
    session["user"] = user["username"]
    session["role"] = role
    session["user_id"] = user["id"]
    session["student_name"] = user["full_name"]

    if role not in {"faculty", "admin"}:
        session.clear()
        return jsonify({"error": "Only admin and faculty logins are enabled"}), 403

    if role != "faculty":
        session.pop("active_semester_id", None)
        session.pop("active_subject_id", None)

    return jsonify({"ok": True, "redirect": url_for("dashboard")})


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return jsonify({"ok": True, "redirect": url_for("login")})


@app.route("/health")
def health():
    return jsonify(
        {
            "ok": model_error is None,
            "model_loaded": model_error is None,
            "known_faces": len(known_names),
            "error": model_error,
        }
    )


@app.route("/api/semesters", methods=["GET"])
@login_required
def list_semesters():
    role = normalize_role(session.get("role"))
    faculty_user_id = session.get("user_id") if role == "faculty" else None
    return jsonify({"semesters": get_semesters_with_subject_counts(faculty_user_id)})


@app.route("/api/semesters", methods=["POST"])
@login_required
@roles_required("admin")
def create_semester():
    payload = request.get_json(silent=True) or request.form
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Semester name is required"}), 400

    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO semesters (name, created_at) VALUES (?, ?)",
            (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Semester already exists"}), 409
    finally:
        conn.close()

    return jsonify({"ok": True})


@app.route("/api/semesters/<int:semester_id>", methods=["DELETE"])
@login_required
@roles_required("admin")
def delete_semester(semester_id):
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE attendance SET semester_id = NULL, subject_id = NULL WHERE semester_id = ?",
            (semester_id,),
        )
        cur = conn.execute("DELETE FROM semesters WHERE id = ?", (semester_id,))
        conn.commit()
    finally:
        conn.close()

    if cur.rowcount == 0:
        return jsonify({"error": "Semester not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/subjects", methods=["GET"])
@login_required
def list_subjects():
    semester_id = parse_int(request.args.get("semester_id"))
    role = normalize_role(session.get("role"))
    faculty_user_id = session.get("user_id") if role == "faculty" else None
    return jsonify({"subjects": get_subjects_for_semester(semester_id, faculty_user_id)})


@app.route("/api/subjects", methods=["POST"])
@login_required
@roles_required("admin")
def create_subject():
    payload = request.get_json(silent=True) or request.form
    semester_id = parse_int(payload.get("semester_id"))
    name = (payload.get("name") or "").strip()

    if not semester_id or not name:
        return jsonify({"error": "Semester and subject name are required"}), 400

    conn = get_db_connection()
    try:
        semester = conn.execute(
            "SELECT id FROM semesters WHERE id = ?", (semester_id,)
        ).fetchone()
        if not semester:
            return jsonify({"error": "Semester not found"}), 404

        conn.execute(
            "INSERT INTO subjects (semester_id, name, created_at) VALUES (?, ?, ?)",
            (semester_id, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Subject already exists in this semester"}), 409
    finally:
        conn.close()

    return jsonify({"ok": True})


@app.route("/api/subjects/<int:subject_id>", methods=["DELETE"])
@login_required
@roles_required("admin")
def delete_subject(subject_id):
    conn = get_db_connection()
    try:
        conn.execute("UPDATE attendance SET subject_id = NULL WHERE subject_id = ?", (subject_id,))
        cur = conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
        conn.commit()
    finally:
        conn.close()

    if cur.rowcount == 0:
        return jsonify({"error": "Subject not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/faculty/context", methods=["GET"])
@login_required
@roles_required("faculty")
def get_faculty_active_context():
    context = get_faculty_context(require_selected=False)
    return jsonify({"context": context})


@app.route("/api/faculty/context", methods=["POST"])
@login_required
@roles_required("faculty")
def set_faculty_active_context():
    payload = request.get_json(silent=True) or request.form
    semester_id = parse_int(payload.get("semester_id"))
    subject_id = parse_int(payload.get("subject_id"))

    if not semester_id or not subject_id:
        return jsonify({"error": "Semester and subject are required"}), 400

    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT sem.id AS semester_id, sem.name AS semester_name,
                   sub.id AS subject_id, sub.name AS subject_name
            FROM semesters sem
            JOIN subjects sub ON sub.semester_id = sem.id
            JOIN faculty_subjects fs ON fs.subject_id = sub.id
            WHERE sem.id = ? AND sub.id = ? AND fs.faculty_user_id = ?
            """,
            (semester_id, subject_id, session.get("user_id")),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return jsonify({"error": "Invalid semester/subject combination"}), 404

    session["active_semester_id"] = semester_id
    session["active_subject_id"] = subject_id

    return jsonify(
        {
            "ok": True,
            "context": {
                "semester_id": row["semester_id"],
                "semester_name": row["semester_name"],
                "subject_id": row["subject_id"],
                "subject_name": row["subject_name"],
            },
        }
    )


@app.route("/recognize", methods=["POST"])
@login_required
@roles_required("faculty")
def recognize_image():
    if model_error:
        return jsonify({"error": model_error}), 500

    context = get_faculty_context(require_selected=True)
    if not context:
        return jsonify({"error": "Select semester and subject before marking attendance"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = request.files["image"].read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if bgr is None:
        return jsonify({"error": "Invalid image file"}), 400

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source = request.form.get("source", "image")
    result = recognize_from_rgb_image(
        rgb,
        known_encodings,
        known_names,
        semester_id=context["semester_id"],
        subject_id=context["subject_id"],
        source=source,
    )
    result["context"] = context
    return jsonify(result)


@app.route("/recognize-video", methods=["POST"])
@login_required
@roles_required("faculty")
def recognize_video():
    if model_error:
        return jsonify({"error": model_error}), 500

    context = get_faculty_context(require_selected=True)
    if not context:
        return jsonify({"error": "Select semester and subject before marking attendance"}), 400

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    uploaded = request.files["video"]
    temp_path = os.path.join(BASE_DIR, "_temp_upload_video.mp4")
    uploaded.save(temp_path)

    recognized_total = set()
    marked_total = set()
    sampled_frames = 0

    cap = cv2.VideoCapture(temp_path)
    frame_idx = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx % 15 != 0:
                continue

            sampled_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = recognize_from_rgb_image(
                rgb,
                known_encodings,
                known_names,
                semester_id=context["semester_id"],
                subject_id=context["subject_id"],
                source="video",
            )

            recognized_total.update(out["recognized"])
            marked_total.update(out["marked_now"])

            if sampled_frames >= 60:
                break
    finally:
        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(
        {
            "recognized": sorted(recognized_total),
            "marked_now": sorted(marked_total),
            "sampled_frames": sampled_frames,
            "context": context,
        }
    )


@app.route("/attendance", methods=["GET"])
@login_required
@roles_required("faculty")
def attendance_list():
    semester_id = parse_int(request.args.get("semester_id"))
    subject_id = parse_int(request.args.get("subject_id"))

    if not semester_id or not subject_id:
        context = get_faculty_context(require_selected=False)
        if context:
            semester_id = context["semester_id"]
            subject_id = context["subject_id"]

    rows = read_attendance_rows(
        semester_id=semester_id,
        subject_id=subject_id,
        faculty_user_id=session.get("user_id"),
        limit=100,
    )
    return jsonify({"records": rows})


@app.route("/attendance/summary", methods=["GET"])
@login_required
@roles_required("faculty")
def attendance_summary():
    semester_id = parse_int(request.args.get("semester_id"))
    subject_id = parse_int(request.args.get("subject_id"))

    if not semester_id or not subject_id:
        context = get_faculty_context(require_selected=False)
        if context:
            semester_id = context["semester_id"]
            subject_id = context["subject_id"]

    rows = read_attendance_rows(
        semester_id=semester_id,
        subject_id=subject_id,
        faculty_user_id=session.get("user_id"),
    )
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(1 for row in rows if row["timestamp"].startswith(today))
    unique_names = len({row["name"].lower() for row in rows})

    return jsonify(
        {
            "total_records": len(rows),
            "today_records": today_count,
            "unique_students": unique_names,
            "known_faces": len(known_names),
            "semester_id": semester_id,
            "subject_id": subject_id,
        }
    )


@app.route("/attendance/clear", methods=["POST"])
@login_required
@roles_required("admin")
def attendance_clear():
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM attendance")
        conn.commit()
    finally:
        conn.close()
    return jsonify({"ok": True})


@app.route("/attendance/clear-current", methods=["POST"])
@login_required
@roles_required("faculty")
def attendance_clear_current():
    payload = request.get_json(silent=True) or request.form
    semester_id = parse_int(payload.get("semester_id"))
    subject_id = parse_int(payload.get("subject_id"))

    if not semester_id or not subject_id:
        context = get_faculty_context(require_selected=True)
        if not context:
            return jsonify({"error": "Select semester and subject first"}), 400
        semester_id = context["semester_id"]
        subject_id = context["subject_id"]

    # Validate assignment ownership before deletion.
    conn = get_db_connection()
    try:
        valid = conn.execute(
            """
            SELECT 1
            FROM faculty_subjects
            WHERE faculty_user_id = ? AND subject_id = ?
            LIMIT 1
            """,
            (session.get("user_id"), subject_id),
        ).fetchone()
        if not valid:
            return jsonify({"error": "This subject is not assigned to your account"}), 403

        cur = conn.execute(
            """
            DELETE FROM attendance
            WHERE faculty_user_id = ?
              AND semester_id = ?
              AND subject_id = ?
            """,
            (session.get("user_id"), semester_id, subject_id),
        )
        conn.commit()
    finally:
        conn.close()

    return jsonify({"ok": True, "deleted_records": cur.rowcount})


@app.route("/attendance/download", methods=["GET"])
@login_required
@roles_required("faculty")
def attendance_download():
    semester_id = parse_int(request.args.get("semester_id"))
    subject_id = parse_int(request.args.get("subject_id"))

    if not semester_id or not subject_id:
        context = get_faculty_context(require_selected=False)
        if context:
            semester_id = context["semester_id"]
            subject_id = context["subject_id"]

    rows = read_attendance_rows(
        semester_id=semester_id,
        subject_id=subject_id,
        faculty_user_id=session.get("user_id"),
    )
    csv_data = build_attendance_csv_bytes(rows)
    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        csv_data,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/faculty-users", methods=["GET"])
@login_required
@roles_required("admin")
def list_faculty_users():
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, username, COALESCE(NULLIF(full_name, ''), username) AS full_name, created_at
            FROM users
            WHERE lower(role) IN ('faculty', 'teacher')
            ORDER BY created_at DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return jsonify({"faculty_users": [dict(row) for row in rows]})


@app.route("/api/faculty-users", methods=["POST"])
@login_required
@roles_required("admin")
def create_faculty_user():
    payload = request.get_json(silent=True) or request.form
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    full_name = (payload.get("full_name") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, full_name, created_at)
            VALUES (?, ?, 'faculty', ?, ?)
            """,
            (
                username,
                generate_password_hash(password),
                full_name or username,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 409
    finally:
        conn.close()
    return jsonify({"ok": True})


@app.route("/api/faculty-users/<int:user_id>", methods=["DELETE"])
@login_required
@roles_required("admin")
def delete_faculty_user(user_id):
    if session.get("user_id") == user_id:
        return jsonify({"error": "You cannot delete your own account"}), 400

    conn = get_db_connection()
    try:
        conn.execute("UPDATE attendance SET faculty_user_id = NULL WHERE faculty_user_id = ?", (user_id,))
        cur = conn.execute(
            "DELETE FROM users WHERE id = ? AND lower(role) IN ('faculty', 'teacher')",
            (user_id,),
        )
        conn.commit()
    finally:
        conn.close()

    if cur.rowcount == 0:
        return jsonify({"error": "Faculty user not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/faculty-assignments", methods=["GET"])
@login_required
@roles_required("admin")
def list_faculty_assignments():
    faculty_user_id = parse_int(request.args.get("faculty_user_id"))
    if not faculty_user_id:
        return jsonify({"error": "faculty_user_id is required"}), 400

    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT fs.faculty_user_id, sub.id AS subject_id, sub.name AS subject_name,
                   sem.id AS semester_id, sem.name AS semester_name
            FROM faculty_subjects fs
            JOIN subjects sub ON sub.id = fs.subject_id
            JOIN semesters sem ON sem.id = sub.semester_id
            WHERE fs.faculty_user_id = ?
            ORDER BY sem.id DESC, sub.name
            """,
            (faculty_user_id,),
        ).fetchall()
    finally:
        conn.close()
    return jsonify({"assignments": [dict(row) for row in rows]})


@app.route("/api/faculty-assignments", methods=["POST"])
@login_required
@roles_required("admin")
def create_faculty_assignment():
    payload = request.get_json(silent=True) or request.form
    faculty_user_id = parse_int(payload.get("faculty_user_id"))
    subject_id = parse_int(payload.get("subject_id"))
    if not faculty_user_id or not subject_id:
        return jsonify({"error": "faculty_user_id and subject_id are required"}), 400

    conn = get_db_connection()
    try:
        faculty = conn.execute(
            "SELECT id FROM users WHERE id = ? AND lower(role) IN ('faculty', 'teacher')",
            (faculty_user_id,),
        ).fetchone()
        if not faculty:
            return jsonify({"error": "Faculty user not found"}), 404

        subject = conn.execute("SELECT id FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subject:
            return jsonify({"error": "Subject not found"}), 404

        conn.execute(
            """
            INSERT INTO faculty_subjects (faculty_user_id, subject_id, created_at)
            VALUES (?, ?, ?)
            """,
            (faculty_user_id, subject_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Assignment already exists"}), 409
    finally:
        conn.close()
    return jsonify({"ok": True})


@app.route("/api/faculty-assignments", methods=["DELETE"])
@login_required
@roles_required("admin")
def delete_faculty_assignment():
    payload = request.get_json(silent=True) or request.form
    faculty_user_id = parse_int(payload.get("faculty_user_id"))
    subject_id = parse_int(payload.get("subject_id"))
    if not faculty_user_id or not subject_id:
        return jsonify({"error": "faculty_user_id and subject_id are required"}), 400

    conn = get_db_connection()
    try:
        cur = conn.execute(
            "DELETE FROM faculty_subjects WHERE faculty_user_id = ? AND subject_id = ?",
            (faculty_user_id, subject_id),
        )
        conn.commit()
    finally:
        conn.close()

    if cur.rowcount == 0:
        return jsonify({"error": "Assignment not found"}), 404
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True)
