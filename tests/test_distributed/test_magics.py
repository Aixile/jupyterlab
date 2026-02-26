"""Tests for DistributedMagics (%distributed and %%rank magic commands)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, call

from IPython.core.interactiveshell import InteractiveShell

from jupyterlab_distributed.magics import DistributedMagics


@pytest.fixture
def shell():
    """Create and return an IPython InteractiveShell instance."""
    s = InteractiveShell.instance()
    yield s
    InteractiveShell.clear_instance()


@pytest.fixture
def magics(shell):
    """Create a DistributedMagics instance with a mocked kernel."""
    m = DistributedMagics(shell=shell)
    m.kernel = MagicMock()
    m.kernel.distributed_enabled = False
    m.kernel._gateway = MagicMock()
    m.kernel._gateway.workers = {
        0: MagicMock(hostname="host0", gpu_id=0, pid=1000, state="alive"),
        1: MagicMock(hostname="host1", gpu_id=1, pid=1001, state="alive"),
    }
    m.kernel._gateway.expected_workers = 2
    m.kernel._gateway.timeout = 30.0
    shell.register_magics(m)
    return m


class TestDistributedOn:
    def test_distributed_on(self, magics):
        """'%distributed on' sets distributed_enabled to True."""
        magics.distributed("on")
        assert magics.kernel.distributed_enabled is True

    def test_distributed_on_idempotent(self, magics):
        """Calling '%distributed on' twice keeps it True."""
        magics.distributed("on")
        magics.distributed("on")
        assert magics.kernel.distributed_enabled is True


class TestDistributedOff:
    def test_distributed_off(self, magics):
        """'%distributed off' sets distributed_enabled to False."""
        magics.kernel.distributed_enabled = True
        magics.distributed("off")
        assert magics.kernel.distributed_enabled is False

    def test_distributed_off_when_already_off(self, magics):
        """Calling '%distributed off' when already off stays False."""
        magics.distributed("off")
        assert magics.kernel.distributed_enabled is False


class TestDistributedStatus:
    def test_distributed_status_prints_worker_count(self, magics, capsys):
        """'%distributed status' prints the number of connected workers."""
        magics.distributed("status")
        out = capsys.readouterr().out
        assert "2" in out

    def test_distributed_status_shows_expected(self, magics, capsys):
        """'%distributed status' prints expected workers count."""
        magics.distributed("status")
        out = capsys.readouterr().out
        assert "expected" in out.lower() or "2" in out

    def test_distributed_status_shows_enabled_state(self, magics, capsys):
        """'%distributed status' shows whether distributed mode is enabled."""
        magics.kernel.distributed_enabled = True
        magics.distributed("status")
        out = capsys.readouterr().out
        assert "enabled" in out.lower() or "on" in out.lower()

    def test_distributed_status_shows_disabled_state(self, magics, capsys):
        """'%distributed status' shows when distributed mode is disabled."""
        magics.kernel.distributed_enabled = False
        magics.distributed("status")
        out = capsys.readouterr().out
        assert "disabled" in out.lower() or "off" in out.lower()

    def test_distributed_status_no_gateway(self, magics, capsys):
        """'%distributed status' handles missing gateway gracefully."""
        magics.kernel._gateway = None
        magics.distributed("status")
        out = capsys.readouterr().out
        assert "no gateway" in out.lower() or "not connected" in out.lower()


class TestDistributedExpect:
    def test_expect_sets_expected_workers(self, magics):
        """'%distributed expect N' sets gateway.expected_workers."""
        magics.distributed("expect 4")
        assert magics.kernel._gateway.expected_workers == 4

    def test_expect_invalid_value(self, magics, capsys):
        """'%distributed expect' with non-integer prints error."""
        magics.distributed("expect abc")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "invalid" in out.lower() or "integer" in out.lower()

    def test_expect_missing_value(self, magics, capsys):
        """'%distributed expect' without argument prints error."""
        magics.distributed("expect")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "usage" in out.lower() or "expected" in out.lower()


class TestDistributedTimeout:
    def test_timeout_sets_value(self, magics):
        """'%distributed timeout N' sets gateway.timeout."""
        magics.distributed("timeout 60")
        assert magics.kernel._gateway.timeout == 60.0

    def test_timeout_float_value(self, magics):
        """'%distributed timeout N' accepts float values."""
        magics.distributed("timeout 10.5")
        assert magics.kernel._gateway.timeout == 10.5

    def test_timeout_invalid_value(self, magics, capsys):
        """'%distributed timeout' with non-numeric value prints error."""
        magics.distributed("timeout abc")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "invalid" in out.lower() or "numeric" in out.lower()

    def test_timeout_missing_value(self, magics, capsys):
        """'%distributed timeout' without argument prints error."""
        magics.distributed("timeout")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "usage" in out.lower() or "timeout" in out.lower()


class TestDistributedFailureMode:
    def test_failure_mode_fail_fast(self, magics):
        """'%distributed failure-mode fail-fast' sets the failure mode."""
        magics.distributed("failure-mode fail-fast")
        assert magics.kernel._gateway.failure_mode == "fail-fast"

    def test_failure_mode_best_effort(self, magics):
        """'%distributed failure-mode best-effort' sets the failure mode."""
        magics.distributed("failure-mode best-effort")
        assert magics.kernel._gateway.failure_mode == "best-effort"

    def test_failure_mode_invalid(self, magics, capsys):
        """'%distributed failure-mode invalid' prints error."""
        magics.distributed("failure-mode invalid")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "invalid" in out.lower()

    def test_failure_mode_missing_value(self, magics, capsys):
        """'%distributed failure-mode' without argument prints error."""
        magics.distributed("failure-mode")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "usage" in out.lower()


class TestDistributedRestart:
    def test_restart_hard_calls_shutdown(self, magics):
        """'%distributed restart hard' calls gateway.broadcast_shutdown()."""
        magics.kernel._gateway.broadcast_shutdown = AsyncMock()
        magics.distributed("restart hard")
        magics.kernel._gateway.broadcast_shutdown.assert_called_once()

    def test_restart_without_hard(self, magics, capsys):
        """'%distributed restart' without 'hard' prints usage info."""
        magics.distributed("restart")
        out = capsys.readouterr().out
        assert "hard" in out.lower() or "usage" in out.lower()


class TestDistributedUnknownCommand:
    def test_unknown_command(self, magics, capsys):
        """Unknown subcommand prints an error or usage info."""
        magics.distributed("nonexistent")
        out = capsys.readouterr().out
        assert "unknown" in out.lower() or "usage" in out.lower() or "error" in out.lower()

    def test_empty_command(self, magics, capsys):
        """Empty argument prints usage info."""
        magics.distributed("")
        out = capsys.readouterr().out
        assert "usage" in out.lower() or "distributed" in out.lower()


class TestRank0Magic:
    def test_rank0_executes_cell(self, magics, shell):
        """'%%rank0' executes the cell locally."""
        shell.run_cell("x_rank0_test = 42")
        assert shell.user_ns["x_rank0_test"] == 42
        magics.rank0("", "y_rank0_test = x_rank0_test + 1")
        assert shell.user_ns["y_rank0_test"] == 43

    def test_rank0_does_not_broadcast(self, magics):
        """'%%rank0' does not call gateway broadcast."""
        magics.rank0("", "z = 1")
        magics.kernel._gateway.broadcast_execute.assert_not_called()


class TestRankMagic:
    def test_rank_0_executes_locally(self, magics, shell):
        """'%%rank 0' executes the cell locally."""
        magics.rank("0", "rank_local_var = 99")
        assert shell.user_ns["rank_local_var"] == 99

    def test_rank_nonzero_sends_to_gateway(self, magics):
        """'%%rank N' for N>0 sends execution to gateway for that rank."""
        magics.kernel._gateway.send_to_rank = AsyncMock()
        magics.rank("1", "remote_code = True")
        magics.kernel._gateway.send_to_rank.assert_called_once_with(
            1, "remote_code = True", cell_id=""
        )

    def test_rank_invalid_number(self, magics, capsys):
        """'%%rank abc' prints an error."""
        magics.rank("abc", "code = 1")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "invalid" in out.lower()

    def test_rank_missing_number(self, magics, capsys):
        """'%%rank' without a number prints an error."""
        magics.rank("", "code = 1")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "usage" in out.lower()

    def test_rank_not_found(self, magics, capsys):
        """'%%rank N' for unknown rank prints an error."""
        # Rank 99 is not in gateway.workers (only 0 and 1 are),
        # so synchronous validation catches it before async send.
        magics.rank("99", "code = 1")
        out = capsys.readouterr().out
        assert "not connected" in out.lower() or "error" in out.lower()


class TestNoKernel:
    def test_distributed_without_kernel(self, shell, capsys):
        """Commands print an error when kernel is not set."""
        m = DistributedMagics(shell=shell)
        # kernel is None by default
        m.distributed("on")
        out = capsys.readouterr().out
        assert "no kernel" in out.lower() or "not available" in out.lower() or "error" in out.lower()
