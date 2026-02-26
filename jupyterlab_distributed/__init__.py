"""JupyterLab Distributed SPMD execution support."""

__version__ = "0.1.0"


def _ensure_kernelspec_installed() -> None:
    """Install the 'distributed-python' kernel spec if not already present."""
    import shutil
    from pathlib import Path

    try:
        from jupyter_client.kernelspec import KernelSpecManager

        ksm = KernelSpecManager()
        try:
            ksm.get_kernel_spec("distributed-python")
            return  # Already installed
        except Exception:
            pass

        # Install from the bundled kernelspec directory
        src = Path(__file__).parent / "kernelspec"
        if src.exists():
            dest = Path(ksm.user_kernel_dir) / "distributed-python"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "kernel.json", dest / "kernel.json")
    except Exception:
        pass  # Best-effort; don't crash on import


_ensure_kernelspec_installed()
