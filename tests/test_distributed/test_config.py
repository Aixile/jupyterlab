import json
import os
import tempfile
import pytest
from jupyterlab_distributed.config import SessionConfig


class TestSessionConfig:
    def test_create_writes_json(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-001",
            world_size=4,
            gateway_port=9876,
        )
        assert config.path.exists()
        data = json.loads(config.path.read_text())
        assert data["kernel_id"] == "test-kernel-001"
        assert data["world_size"] == 4
        assert data["status"] == "waiting"
        assert data["host"] is None
        assert len(data["auth_token"]) >= 32

    def test_create_sets_restrictive_permissions(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-002",
            world_size=2,
            gateway_port=9877,
        )
        stat = os.stat(config.path)
        assert oct(stat.st_mode & 0o777) == "0o600"

    def test_atomic_write_no_partial_file(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-003",
            world_size=2,
            gateway_port=9878,
        )
        # File should be valid JSON (no partial writes)
        data = json.loads(config.path.read_text())
        assert data["kernel_id"] == "test-kernel-003"

    def test_load_reads_existing(self, tmp_path):
        created = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-004",
            world_size=8,
            gateway_port=9879,
        )
        loaded = SessionConfig.load(created.path)
        assert loaded.kernel_id == "test-kernel-004"
        assert loaded.world_size == 8

    def test_update_status(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-005",
            world_size=2,
            gateway_port=9880,
        )
        config.update(status="running", host="compute-01")
        reloaded = SessionConfig.load(config.path)
        assert reloaded.status == "running"
        assert reloaded.host == "compute-01"

    def test_zmq_ports_allocated(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-006",
            world_size=2,
            gateway_port=9881,
        )
        ports = config.zmq_ports
        assert "shell" in ports
        assert "iopub" in ports
        assert "stdin" in ports
        assert "control" in ports
        assert "hb" in ports
        # All ports should be distinct
        assert len(set(ports.values())) == 5

    def test_ttl_cleanup(self, tmp_path):
        config = SessionConfig.create(
            base_dir=str(tmp_path),
            kernel_id="test-kernel-007",
            world_size=2,
            gateway_port=9882,
            ttl_seconds=0,  # Immediately stale
        )
        stale = SessionConfig.cleanup_stale(str(tmp_path))
        assert len(stale) == 1
        assert not config.path.exists()
