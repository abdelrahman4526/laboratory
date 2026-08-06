
import logging
import os
from sqlalchemy import text
import faiss
import pickle
import hashlib
import threading
from .schemas import EntityType, VectorMetadata
from .embedding import EMBEDDING_DIM,EMBEDDING_MODEL
import numpy as np

logger = logging.getLogger(__name__)
FAISS_INDEX_PATH = "lads.faiss"
FAISS_METADATA_PATH = "labs_metadata.pkl"

_faiss_index = None
_faiss_metadata = None
_faiss_lock = threading.Lock()

def _make_faiss_id(entity_id: int, entity_type: "EntityType") -> int:
    """Derives a stable int64 id from (entity_id, entity_type) — same key always gives same id."""
    key = f"{entity_type.value}:{entity_id}".encode()
    digest = hashlib.sha1(key).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _load_faiss() -> None:
    """Loads the FAISS index + metadata from disk into memory, creating them if missing."""
    global _faiss_index, _faiss_metadata
    if _faiss_index is not None:
        return
    with _faiss_lock:
        if _faiss_index is not None:
            return
        if os.path.exists(FAISS_INDEX_PATH):
            _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            _faiss_index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBEDDING_DIM))

        if os.path.exists(FAISS_METADATA_PATH):
            with open(FAISS_METADATA_PATH, "rb") as f:
                _faiss_metadata = pickle.load(f)
        else:
            _faiss_metadata = {"key_to_faiss_id": {}, "faiss_id_to_meta": {}}


def _save_faiss() -> None:
    """Persists the current index + metadata to disk."""
    faiss.write_index(_faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump(_faiss_metadata, f)


def ensure_vector_table() -> None:
    """Ensures the FAISS index + metadata files exist """
    try:
        _load_faiss()
        if not os.path.exists(FAISS_INDEX_PATH):
            _save_faiss()
    except Exception as e:
        logger.warning("Failed to initialize FAISS index: %s", e)


def upsert_vector(metadata: "VectorMetadata", embedding: list[float]) -> None:
    """
    Inserts a new embedding or updates the existing one for (id, type).
    Metadata written is strictly {id, type, name} per spec.
    """
    _load_faiss()
    key = f"{metadata.type.value}:{metadata.id}"
    faiss_id = _make_faiss_id(metadata.id, metadata.type)
    vec = np.array(embedding, dtype="float32").reshape(1, -1)

    with _faiss_lock:
        # لو الـ key ده موجود قبل كده، شيل الـ vector القديم الأول (زي ON CONFLICT DO UPDATE)
        if key in _faiss_metadata["key_to_faiss_id"]:
            old_id = _faiss_metadata["key_to_faiss_id"][key]
            _faiss_index.remove_ids(np.array([old_id], dtype="int64"))

        _faiss_index.add_with_ids(vec, np.array([faiss_id], dtype="int64"))
        _faiss_metadata["key_to_faiss_id"][key] = faiss_id
        _faiss_metadata["faiss_id_to_meta"][faiss_id] = {
            "id": metadata.id,
            "type": metadata.type.value,
            "name": metadata.name,
        }
        _save_faiss()

    logger.info("Upserted vector for %s id=%s", metadata.type.value, metadata.id)


def delete_vector(entity_id: int, entity_type: "EntityType") -> None:
    """Removes a vector, e.g. if the Lab/Bundle is deleted."""
    _load_faiss()
    key = f"{entity_type.value}:{entity_id}"

    with _faiss_lock:
        faiss_id = _faiss_metadata["key_to_faiss_id"].pop(key, None)
        if faiss_id is not None:
            _faiss_index.remove_ids(np.array([faiss_id], dtype="int64"))
            _faiss_metadata["faiss_id_to_meta"].pop(faiss_id, None)
            _save_faiss()

def get_index_and_metadata():
    """Ensures the FAISS index + metadata are loaded, then returns them."""
    _load_faiss()
    return _faiss_index, _faiss_metadata            

    logger.info("Deleted vector for %s id=%s", entity_type.value, entity_id)