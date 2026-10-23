
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
DB_PATH = DATA_DIR / "careermatch_history.db"


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                resume_name TEXT,
                jd_name TEXT,
                resume_path TEXT,
                jd_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
                    ON DELETE CASCADE
            )
        """)


def create_session(resume_name="", jd_name=""):
    init_db()
    session_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    title = f"{resume_name or 'Resume'} — {jd_name or 'Job Description'}"

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO sessions (
                id, title, resume_name, jd_name,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, title, resume_name, jd_name, now, now))

    return session_id


def save_document(session_id, document_type, filename, content):
    if document_type not in {"resume", "jd"}:
        raise ValueError("document_type must be 'resume' or 'jd'")

    init_db()
    folder = SESSIONS_DIR / session_id
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{document_type}.pdf"
    path.write_bytes(content)

    name_column = "resume_name" if document_type == "resume" else "jd_name"
    path_column = "resume_path" if document_type == "resume" else "jd_path"

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            UPDATE sessions
            SET {name_column} = ?, {path_column} = ?, updated_at = ?
            WHERE id = ?
        """, (
            filename,
            str(path.relative_to(BASE_DIR)),
            datetime.now().isoformat(timespec="seconds"),
            session_id
        ))

        row = conn.execute(
            "SELECT resume_name, jd_name FROM sessions WHERE id = ?",
            (session_id,)
        ).fetchone()

        if row:
            resume_name = row[0] or "Resume"
            jd_name = row[1] or "Job Description"
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (f"{resume_name} — {jd_name}", session_id)
            )

    return path


def get_session(session_id):
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        ).fetchone()

    return dict(row) if row else None


def list_sessions():
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT s.*,
                (SELECT COUNT(*) FROM messages m
                 WHERE m.session_id = s.id) AS message_count
            FROM sessions s
            ORDER BY updated_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def add_message(session_id, question, answer, sources):
    init_db()
    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("""
            INSERT INTO messages (
                session_id, question, answer, sources_json, created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            question,
            answer,
            json.dumps(sources, ensure_ascii=False),
            now
        ))

        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )


def get_messages(session_id):
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT question, answer, sources_json, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id
        """, (session_id,)).fetchall()

    messages = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(item.pop("sources_json"))
        messages.append(item)

    return messages
