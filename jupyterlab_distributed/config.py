"""Session config for distributed kernel coordination via shared filesystem."""

import json
import os
import secrets
import socket
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _find_free_ports(count: int) -> list[int]:
    """Find N free TCP ports."""
    sockets = []
    ports = []
    for _ in range(count):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        ports.append(s.getsockname()[1])
        sockets.append(s)
    for s in sockets:
        s.close()
    return ports


@dataclass
class SessionConfig:
    kernel_id: str
    world_size: int
    gateway_port: int
    zmq_ports: dict[str, int] = field(default_factory=dict)
    auth_token: str = ""
    status: str = "waiting"
    host: str | None = None
    created_at: float = 0.0
    ttl_seconds: int = 3600
    path: Path = field(default_factory=lambda: Path(""))

    @classmethod
    def create(
        cls,
        base_dir: str,
        kernel_id: str,
        world_size: int,
        gateway_port: int,
        ttl_seconds: int = 3600,
    ) -> "SessionConfig":
        session_dir = Path(base_dir) / ".jupyter" / "distributed"
        session_dir.mkdir(parents=True, exist_ok=True)

        ports = _find_free_ports(5)
        zmq_ports = {
            "shell": ports[0],
            "iopub": ports[1],
            "stdin": ports[2],
            "control": ports[3],
            "hb": ports[4],
        }

        config = cls(
            kernel_id=kernel_id,
            world_size=world_size,
            gateway_port=gateway_port,
            zmq_ports=zmq_ports,
            auth_token=secrets.token_hex(32),
            status="waiting",
            host=None,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
            path=session_dir / f"{kernel_id}.json",
        )
        config._write()
        return config

    @classmethod
    def load(cls, path: Path | str) -> "SessionConfig":
        path = Path(path)
        data = json.loads(path.read_text())
        data["path"] = path
        return cls(**data)

    def update(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._write()

    def _write(self) -> None:
        data = asdict(self)
        data["path"] = str(data["path"])
        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self.path)
        except BaseException:
            os.unlink(tmp_path)
            raise

    @staticmethod
    def cleanup_stale(base_dir: str) -> list[Path]:
        session_dir = Path(base_dir) / ".jupyter" / "distributed"
        cleaned: list[Path] = []
        if not session_dir.exists():
            return cleaned
        now = time.time()
        for f in session_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                created = data.get("created_at", 0)
                ttl = data.get("ttl_seconds", 3600)
                if now - created > ttl:
                    f.unlink()
                    cleaned.append(f)
            except (json.JSONDecodeError, OSError):
                f.unlink()
                cleaned.append(f)
        return cleaned
