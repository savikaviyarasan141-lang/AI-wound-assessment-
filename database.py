import json
import sqlite3
from pathlib import Path

import pandas as pd


# Store the database beside this Python file.
DB_PATH = Path(__file__).resolve().parent / "wound_history.db"


def init_db():
    """Create patient and visit tables without deleting existing records."""

    with sqlite3.connect(DB_PATH) as con:

        # Store each anonymous patient ID only once.
        con.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_code TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Store each assessment as a separate visit.
        con.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_code TEXT NOT NULL,
                visit_date TEXT NOT NULL,
                image_name TEXT,
                width_px INTEGER,
                height_px INTEGER,
                mask_fraction_pct REAL,
                mean_rgb TEXT,
                predicted_label TEXT
            )
        """)

        # Preserve existing records and register their patient IDs.
        con.execute("""
            INSERT OR IGNORE INTO patients (patient_code)
            SELECT DISTINCT TRIM(patient_code)
            FROM visits
            WHERE patient_code IS NOT NULL
              AND TRIM(patient_code) <> ''
        """)

        con.commit()


def get_patient_codes():
    """Return all registered patient IDs."""

    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("""
            SELECT patient_code
            FROM patients
            ORDER BY patient_code
        """).fetchall()

    return [row[0] for row in rows]


def save_visit(
    patient_code,
    visit_date,
    image_name,
    measurements,
    predicted_label=None
):
    """
    Register the patient if necessary and append a new visit.

    Existing visits are never overwritten.
    """

    code = str(patient_code).strip()

    if not code:
        raise ValueError("Patient ID cannot be empty.")

    with sqlite3.connect(DB_PATH) as con:

        # Register the patient without duplicating their ID.
        con.execute("""
            INSERT OR IGNORE INTO patients (patient_code)
            VALUES (?)
        """, (code,))

        # Every assessment creates a new visit record.
        con.execute("""
            INSERT INTO visits (
                patient_code,
                visit_date,
                image_name,
                width_px,
                height_px,
                mask_fraction_pct,
                mean_rgb,
                predicted_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            code,
            str(visit_date),
            image_name,
            int(measurements["width_px"]),
            int(measurements["height_px"]),
            float(measurements["mask_fraction_pct"]),
            json.dumps(measurements["mean_rgb"]),
            predicted_label
        ))

        con.commit()


def get_visits(patient_code):
    """Retrieve all visits belonging to one patient."""

    code = str(patient_code).strip()

    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("""
            SELECT
                id,
                patient_code,
                visit_date,
                image_name,
                width_px,
                height_px,
                mask_fraction_pct,
                mean_rgb,
                predicted_label
            FROM visits
            WHERE patient_code = ?
            ORDER BY visit_date, id
        """, con, params=(code,))

    return df
def get_all_visits():
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("""
            SELECT id, patient_code, visit_date, image_name,
                   width_px, height_px, mask_fraction_pct,
                   mean_rgb, predicted_label
            FROM visits
            ORDER BY patient_code, visit_date, id
        """, con)
    return df
