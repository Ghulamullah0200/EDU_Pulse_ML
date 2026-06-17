"""
EduPulse AI – Database Service (SQLite)
Handles all persistent storage: students, predictions, audit logs.
The SQLite database file lives at  backend/edupulse.db
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "edupulse.db")

# On Vercel, filesystem is read-only except /tmp
if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/edupulse.db"


def _get_conn() -> sqlite3.Connection:
    """Return a connection with row_factory so results behave like dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")      # better concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't already exist."""
    conn = _get_conn()
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS students (
        student_id   TEXT PRIMARY KEY,
        name         TEXT NOT NULL DEFAULT '',
        age          INTEGER,
        gender       TEXT,
        department   TEXT,
        semester     INTEGER,
        study_hours_per_week       REAL DEFAULT 0,
        attendance_percentage      REAL DEFAULT 0,
        assignment_average         REAL DEFAULT 0,
        midterm_score              REAL DEFAULT 0,
        previous_gpa               REAL DEFAULT 0,
        internet_access            TEXT DEFAULT 'Yes',
        extra_academic_support     TEXT DEFAULT 'No',
        part_time_job              TEXT DEFAULT 'No',
        extracurricular_hours_per_week  REAL DEFAULT 0,
        absences                   INTEGER DEFAULT 0,
        final_score                REAL DEFAULT 0,
        at_risk                    INTEGER DEFAULT 0,
        risk_label                 TEXT DEFAULT 'Pending',
        data_origin                TEXT DEFAULT 'Enrolled',
        created_at   TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id   TEXT,
        input_snapshot TEXT,       -- JSON blob
        prediction   TEXT,
        probability  REAL,
        risk_level   TEXT,
        model_version TEXT,
        created_at   TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );

    CREATE TABLE IF NOT EXISTS audit_logs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        action       TEXT,
        details      TEXT,
        created_at   TEXT DEFAULT (datetime('now'))
    );
    """)

    conn.commit()
    conn.close()


class DatabaseService:
    """Thin wrapper around SQLite for the rest of the app."""

    def __init__(self):
        init_db()
        self._seed_from_csv()

    # ── Connection health ────────────────────────────────────────
    def is_connected(self) -> bool:
        return os.path.exists(DB_PATH)

    # ── Seed data ────────────────────────────────────────────────
    def _seed_from_csv(self):
        """On first run, import the expanded CSV into the students table."""
        conn = _get_conn()
        count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        if count > 0:
            conn.close()
            return                     # already seeded

        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "ml_pipeline", "student_performance_expanded.csv"
        )
        if not os.path.exists(csv_path):
            conn.close()
            return

        import pandas as pd, numpy as np
        df = pd.read_csv(csv_path)
        df = df.where(pd.notna(df), None)

        rows = []
        for _, r in df.iterrows():
            rows.append((
                r.get("student_id"),
                "",                             # name placeholder
                r.get("age"),
                r.get("gender"),
                r.get("department"),
                r.get("semester"),
                r.get("study_hours_per_week"),
                r.get("attendance_percentage"),
                r.get("assignment_average"),
                r.get("midterm_score"),
                r.get("previous_gpa"),
                r.get("internet_access"),
                r.get("extra_academic_support"),
                r.get("part_time_job"),
                r.get("extracurricular_hours_per_week"),
                r.get("absences"),
                r.get("final_score"),
                r.get("at_risk"),
                r.get("risk_label"),
                r.get("data_origin"),
            ))

        conn.executemany("""
            INSERT OR IGNORE INTO students
            (student_id, name, age, gender, department, semester,
             study_hours_per_week, attendance_percentage, assignment_average,
             midterm_score, previous_gpa, internet_access,
             extra_academic_support, part_time_job,
             extracurricular_hours_per_week, absences,
             final_score, at_risk, risk_label, data_origin)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
        conn.close()
        print(f"[OK] Seeded {len(rows)} student records from CSV into SQLite.")

    # ── Student CRUD ─────────────────────────────────────────────
    def list_students(self, limit: int = 200, offset: int = 0):
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM students ORDER BY student_id LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        conn.close()
        return total, [dict(r) for r in rows]

    def get_student(self, student_id: str):
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def next_student_id(self) -> str:
        conn = _get_conn()
        row = conn.execute("""
            SELECT student_id FROM students
            ORDER BY CAST(SUBSTR(student_id, 4) AS INTEGER) DESC
            LIMIT 1
        """).fetchone()
        conn.close()
        if row:
            num = int(row["student_id"][3:]) + 1
        else:
            num = 1001
        return f"STU{num}"

    def create_student(self, data: dict) -> dict:
        sid = self.next_student_id()
        conn = _get_conn()
        conn.execute("""
            INSERT INTO students
            (student_id, name, age, gender, department, semester,
             study_hours_per_week, attendance_percentage, assignment_average,
             midterm_score, previous_gpa, internet_access,
             extra_academic_support, part_time_job,
             extracurricular_hours_per_week, absences,
             final_score, at_risk, risk_label, data_origin)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,'Pending','Enrolled')
        """, (
            sid, data.get("name", ""),
            data.get("age"), data.get("gender"), data.get("department"),
            data.get("semester"), data.get("study_hours_per_week", 0),
            data.get("attendance_percentage", 0),
            data.get("assignment_average", 0), data.get("midterm_score", 0),
            data.get("previous_gpa", 0), data.get("internet_access", "Yes"),
            data.get("extra_academic_support", "No"),
            data.get("part_time_job", "No"),
            data.get("extracurricular_hours_per_week", 0),
            data.get("absences", 0),
        ))
        conn.commit()

        # Log the action
        conn.execute(
            "INSERT INTO audit_logs (action, details) VALUES (?, ?)",
            ("ENROLL_STUDENT", json.dumps({"student_id": sid, "name": data.get("name", "")}))
        )
        conn.commit()
        conn.close()

        return self.get_student(sid)

    # ── Dataset paginated read ───────────────────────────────────
    def get_dataset_page(self, skip: int = 0, limit: int = 100):
        conn = _get_conn()
        total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM students ORDER BY student_id LIMIT ? OFFSET ?",
            (limit, skip)
        ).fetchall()
        conn.close()
        return total, [dict(r) for r in rows]

    # ── Predictions ──────────────────────────────────────────────
    def save_prediction(self, student_id: str, input_data: dict,
                        prediction_data: dict):
        conn = _get_conn()
        conn.execute("""
            INSERT INTO predictions
            (student_id, input_snapshot, prediction, probability,
             risk_level, model_version)
            VALUES (?,?,?,?,?,?)
        """, (
            student_id,
            json.dumps(input_data),
            prediction_data.get("prediction"),
            prediction_data.get("probability"),
            prediction_data.get("risk_level"),
            prediction_data.get("model_version"),
        ))
        conn.commit()
        conn.close()
        return True


# Singleton
db_service = DatabaseService()
