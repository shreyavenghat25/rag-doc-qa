import sqlite3
import json
from pathlib import Path
from app.config import settings


def get_db():
    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            source_type TEXT NOT NULL,
            total_chunks INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER REFERENCES documents(id),
            chunk_index INTEGER NOT NULL,
            faiss_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def insert_document(filename: str, source_type: str) -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (filename, source_type) VALUES (?, ?)",
        (filename, source_type)
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def insert_chunks(doc_id: int, chunks: list[dict]) -> list[int]:
    conn = get_db()
    cursor = conn.cursor()
    chunk_ids = []
    for chunk in chunks:
        cursor.execute(
            """INSERT INTO chunks (doc_id, chunk_index, faiss_id, text, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (
                doc_id,
                chunk["chunk_index"],
                chunk["faiss_id"],
                chunk["text"],
                json.dumps(chunk.get("metadata", {}))
            )
        )
        chunk_ids.append(cursor.lastrowid)

    # Update document chunk count
    cursor.execute(
        "UPDATE documents SET total_chunks = ? WHERE id = ?",
        (len(chunks), doc_id)
    )
    conn.commit()
    conn.close()
    return chunk_ids


def get_chunks_by_faiss_ids(faiss_ids: list[int]) -> list[dict]:
    if not faiss_ids:
        return []
    conn = get_db()
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(faiss_ids))
    cursor.execute(
        f"SELECT c.*, d.filename, d.source_type FROM chunks c "
        f"JOIN documents d ON c.doc_id = d.id "
        f"WHERE c.faiss_id IN ({placeholders})",
        faiss_ids
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        result.append(d)
    return result


def get_all_chunks() -> list[dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT c.*, d.filename FROM chunks c JOIN documents d ON c.doc_id = d.id ORDER BY c.id"
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        result.append(d)
    return result


def list_documents() -> list[dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
