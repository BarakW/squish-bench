import json
from pathlib import Path

from pydantic import BaseModel

FileKey = tuple[str, ...]


class PersistentKVStore[V: BaseModel]:
    """Persistent KV store where the keys are tuples of strings representing paths"""

    def __init__(self, model: type[V], root_dir: Path, flush_rate: int = 1) -> None:
        """Flush rate will stage that number of changes in RAM before flushing to files"""
        self._model = model
        self._root_dir = root_dir
        root_dir.mkdir(parents=True, exist_ok=True)

        self._flush_rate = flush_rate
        self._cached_changes: dict[FileKey, V] = dict()
        self._num_changes = 0

    def __setitem__(self, key: FileKey, val: V) -> None:
        self._cached_changes[key] = val
        self._num_changes += 1
        if self._num_changes >= self._flush_rate:
            self.flush()

    def __contains__(self, key: FileKey) -> bool:
        if key in self._cached_changes:
            return True
        path = self._key_to_path(key)
        return path.exists()

    def __getitem__(self, key: FileKey) -> V | None:
        if key in self._cached_changes:
            return self._cached_changes[key]
        path = self._key_to_path(key)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            return self._model(**json.load(fh))

    def flush(self) -> None:
        if not self._cached_changes:
            return
        for key, value in self._cached_changes.items():
            path = self._key_to_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as fh:
                json.dump(value.model_dump(), fh, indent=2)
        self._cached_changes.clear()
        self._num_changes = 0

    def _key_to_path(self, key: FileKey) -> Path:
        if not key:
            raise ValueError("File keys must contain at least one path segment.")
        return self._root_dir.joinpath(*key)
