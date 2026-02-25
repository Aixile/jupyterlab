"""Tests for the launcher entry point (torchrun-compatible)."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from jupyterlab_distributed.launcher import detect_rank, detect_world_size, main


class TestDetectRank:
    def test_detect_rank_from_env(self):
        with patch.dict(os.environ, {"RANK": "3"}, clear=False):
            assert detect_rank() == 3

    def test_detect_rank_from_slurm(self):
        env = {k: v for k, v in os.environ.items() if k != "RANK"}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"SLURM_PROCID": "5"}):
                assert detect_rank() == 5

    def test_detect_rank_from_ompi(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "SLURM_PROCID")}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"OMPI_COMM_WORLD_RANK": "7"}):
                assert detect_rank() == 7

    def test_detect_rank_default(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK")}
        with patch.dict(os.environ, env, clear=True):
            assert detect_rank() == 0

    def test_detect_rank_priority(self):
        """RANK env var takes priority over SLURM_PROCID."""
        with patch.dict(os.environ, {"RANK": "2", "SLURM_PROCID": "5"}, clear=False):
            assert detect_rank() == 2


class TestDetectWorldSize:
    def test_detect_world_size_from_env(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "32"}, clear=False):
            assert detect_world_size() == 32

    def test_detect_world_size_from_slurm(self):
        env = {k: v for k, v in os.environ.items() if k != "WORLD_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"SLURM_NTASKS": "16"}):
                assert detect_world_size() == 16

    def test_detect_world_size_from_ompi(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("WORLD_SIZE", "SLURM_NTASKS")}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "64"}):
                assert detect_world_size() == 64

    def test_detect_world_size_default(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE")}
        with patch.dict(os.environ, env, clear=True):
            assert detect_world_size() == 1

    def test_detect_world_size_priority(self):
        """WORLD_SIZE env var takes priority over SLURM_NTASKS."""
        with patch.dict(os.environ, {"WORLD_SIZE": "8", "SLURM_NTASKS": "16"}, clear=False):
            assert detect_world_size() == 8


class TestMainArgparse:
    def test_main_requires_session_config(self):
        """main() should exit with error if --session-config is not provided."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["launcher"]):
                main()
        assert exc_info.value.code != 0

    def test_main_rank0_calls_run_rank0(self, tmp_path):
        """main() should call _run_rank0 when rank == 0."""
        # Create a minimal session config file
        config_data = {
            "kernel_id": "test-kernel",
            "world_size": 2,
            "gateway_port": 12345,
            "zmq_ports": {"shell": 1, "iopub": 2, "stdin": 3, "control": 4, "hb": 5},
            "auth_token": "abc123",
            "status": "waiting",
            "host": None,
            "created_at": 1000.0,
            "ttl_seconds": 3600,
            "path": str(tmp_path / "test.json"),
        }
        config_path = tmp_path / "test.json"
        config_path.write_text(json.dumps(config_data))

        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK")}

        with patch.dict(os.environ, env, clear=True):
            with patch("sys.argv", ["launcher", "--session-config", str(config_path)]):
                with patch("jupyterlab_distributed.launcher._run_rank0") as mock_rank0:
                    main()
                    mock_rank0.assert_called_once()
                    call_args = mock_rank0.call_args
                    assert call_args[0][0].kernel_id == "test-kernel"

    def test_main_worker_calls_run_worker(self, tmp_path):
        """main() should call _run_worker when rank != 0."""
        config_data = {
            "kernel_id": "test-kernel",
            "world_size": 4,
            "gateway_port": 12345,
            "zmq_ports": {"shell": 1, "iopub": 2, "stdin": 3, "control": 4, "hb": 5},
            "auth_token": "abc123",
            "status": "waiting",
            "host": None,
            "created_at": 1000.0,
            "ttl_seconds": 3600,
            "path": str(tmp_path / "test.json"),
        }
        config_path = tmp_path / "test.json"
        config_path.write_text(json.dumps(config_data))

        with patch("sys.argv", ["launcher", "--session-config", str(config_path), "--rank", "2"]):
            with patch("jupyterlab_distributed.launcher._run_worker") as mock_worker:
                main()
                mock_worker.assert_called_once()
                call_args = mock_worker.call_args
                assert call_args[0][0] == 2  # rank
                assert call_args[0][1].kernel_id == "test-kernel"

    def test_main_rank_override(self, tmp_path):
        """--rank flag should override environment detection."""
        config_data = {
            "kernel_id": "test-kernel",
            "world_size": 4,
            "gateway_port": 12345,
            "zmq_ports": {"shell": 1, "iopub": 2, "stdin": 3, "control": 4, "hb": 5},
            "auth_token": "abc123",
            "status": "waiting",
            "host": None,
            "created_at": 1000.0,
            "ttl_seconds": 3600,
            "path": str(tmp_path / "test.json"),
        }
        config_path = tmp_path / "test.json"
        config_path.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {"RANK": "0"}, clear=False):
            with patch("sys.argv", ["launcher", "--session-config", str(config_path), "--rank", "3"]):
                with patch("jupyterlab_distributed.launcher._run_worker") as mock_worker:
                    main()
                    # --rank 3 should override RANK=0
                    mock_worker.assert_called_once()
                    call_args = mock_worker.call_args
                    assert call_args[0][0] == 3

    def test_main_log_dir_passed_through(self, tmp_path):
        """--log-dir should be passed to _run_rank0 or _run_worker."""
        config_data = {
            "kernel_id": "test-kernel",
            "world_size": 2,
            "gateway_port": 12345,
            "zmq_ports": {"shell": 1, "iopub": 2, "stdin": 3, "control": 4, "hb": 5},
            "auth_token": "abc123",
            "status": "waiting",
            "host": None,
            "created_at": 1000.0,
            "ttl_seconds": 3600,
            "path": str(tmp_path / "test.json"),
        }
        config_path = tmp_path / "test.json"
        config_path.write_text(json.dumps(config_data))
        log_dir = str(tmp_path / "logs")

        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK")}

        with patch.dict(os.environ, env, clear=True):
            with patch("sys.argv", ["launcher", "--session-config", str(config_path), "--log-dir", log_dir]):
                with patch("jupyterlab_distributed.launcher._run_rank0") as mock_rank0:
                    main()
                    call_args = mock_rank0.call_args
                    assert call_args[0][1] == log_dir
