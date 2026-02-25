import json

from jupyterlab_distributed.handlers import read_crash_logs, read_rank_logs


class TestHandlers:
    def test_read_crash_logs_empty(self, tmp_path):
        logs = read_crash_logs(str(tmp_path))
        assert logs == []

    def test_read_crash_logs_finds_files(self, tmp_path):
        crash = {"rank": 3, "hostname": "node-2", "signal": "SIGSEGV", "traceback": "..."}
        (tmp_path / "rank-3.crash.json").write_text(json.dumps(crash))
        logs = read_crash_logs(str(tmp_path))
        assert len(logs) == 1
        assert logs[0]["rank"] == 3

    def test_read_crash_logs_nonexistent_dir(self):
        logs = read_crash_logs("/nonexistent/path")
        assert logs == []

    def test_read_crash_logs_skips_invalid_json(self, tmp_path):
        (tmp_path / "rank-0.crash.json").write_text("not json")
        (tmp_path / "rank-1.crash.json").write_text('{"rank": 1}')
        logs = read_crash_logs(str(tmp_path))
        assert len(logs) == 1
        assert logs[0]["rank"] == 1

    def test_read_rank_logs(self, tmp_path):
        (tmp_path / "rank-0.log").write_text("hello from rank 0\n")
        log = read_rank_logs(str(tmp_path), rank=0, tail=10)
        assert "hello from rank 0" in log

    def test_read_rank_logs_nonexistent(self, tmp_path):
        log = read_rank_logs(str(tmp_path), rank=99, tail=10)
        assert log == ""

    def test_read_rank_logs_tail(self, tmp_path):
        lines = "\n".join(f"line {i}" for i in range(100))
        (tmp_path / "rank-0.log").write_text(lines)
        log = read_rank_logs(str(tmp_path), rank=0, tail=5)
        assert "line 95" in log
        assert "line 99" in log
        assert "line 0" not in log
