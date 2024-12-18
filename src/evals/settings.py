import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import ClassVar


@dataclass
class Settings:
    """Group settings together here."""

    app_name: ClassVar[str] = "stencila-evals"

    def get_base_dir(self) -> Path:
        """Allow for setting a custom base path."""
        base_path = os.environ.get("EVALS_BASE_PATH", "./data")
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    def get_log_dir(self) -> Path:
        logs = self.get_base_dir() / "logs"
        return logs
        # return Path(
        #     platformdirs.user_log_dir(appname=self.app_name, ensure_exists=True)
        # )

    def get_downloads_dir(self, where: str | None = None) -> Path:
        pth = self.get_base_dir() / "downloads"
        if where is not None:
            pth = pth / where
        if not pth.exists():
            pth.mkdir(parents=True)
        return pth

    # TODO: Clean up this rubbish.
    def get_frames_dir(self) -> Path:
        pth = self.get_base_dir() / "frames"
        if not pth.exists():
            pth.mkdir(parents=True)
        return pth

    def get_routing_dir(self) -> Path:
        pth = self.get_base_dir() / "routing"
        if not pth.exists():
            pth.mkdir(parents=True)
        return pth

    def get_database_path(self) -> Path:
        pth = self.get_base_dir() / "database.sqlite"
        return pth


@cache
def get_settings() -> Settings:
    settings = Settings()
    return settings
