"""
The Librarian — Cross-Platform Utilities

Centralizes all platform-specific logic so the rest of the codebase
can stay platform-agnostic. Covers:

- Install directory locations
- CLI executable naming
- GUI font selection
- PATH setup (Windows registry, macOS/Linux shell config)
"""
import os
import sys
from typing import Tuple


def get_system() -> str:
    """Return normalized platform identifier.

    Returns:
        'windows', 'darwin', or 'linux'
    """
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "darwin"
    else:
        return "linux"


def get_install_base_dir() -> str:
    """Return the platform-appropriate base install directory for The Librarian.

    - Windows: %LOCALAPPDATA%\\The Librarian
    - macOS:   ~/Library/Application Support/The Librarian
    - Linux:   ~/.local/share/The Librarian
    """
    system = get_system()
    if system == "windows":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if not local_app_data:
            local_app_data = os.path.join(os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(local_app_data, "The Librarian")
    elif system == "darwin":
        return os.path.join(
            os.path.expanduser("~"),
            "Library", "Application Support", "The Librarian"
        )
    else:
        return os.path.join(
            os.path.expanduser("~"),
            ".local", "share", "The Librarian"
        )


def get_cli_executable_name() -> str:
    """Return the CLI executable filename.

    - Windows: librarian.exe
    - macOS/Linux: librarian
    """
    return "librarian.exe" if get_system() == "windows" else "librarian"


def get_gui_font() -> Tuple[str, int]:
    """Return platform-appropriate (font_family, base_size) for tkinter GUI.

    - Windows: Segoe UI 10
    - macOS:   Helvetica Neue 13 (macOS uses larger base sizes)
    - Linux:   TkDefaultFont 10
    """
    system = get_system()
    if system == "windows":
        return ("Segoe UI", 10)
    elif system == "darwin":
        return ("Helvetica Neue", 13)
    else:
        return ("TkDefaultFont", 10)


def add_to_path(bin_dir: str) -> None:
    """Add a directory to the user's PATH using platform-specific methods.

    - Windows: Modifies HKCU\\Environment\\Path in the registry
    - macOS:   Appends export to ~/.zshrc and ~/.bash_profile
    - Linux:   Appends export to ~/.bashrc and ~/.bash_profile
    """
    system = get_system()
    if system == "windows":
        _add_to_path_windows(bin_dir)
    elif system == "darwin":
        _add_to_path_macos(bin_dir)
    else:
        _add_to_path_linux(bin_dir)


def _add_to_path_windows(bin_dir: str) -> None:
    """Windows: update user PATH via registry."""
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS
        ) as key:
            try:
                current_path, _ = winreg.QueryValueEx(key, "Path")
            except OSError:
                current_path = ""

            if bin_dir.lower() not in current_path.lower():
                new_path = current_path.rstrip(";") + ";" + bin_dir
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
    except Exception:
        pass  # Best-effort


def _add_to_path_macos(bin_dir: str) -> None:
    """macOS: update ~/.zshrc (default since Catalina) and ~/.bash_profile."""
    shell_configs = [
        os.path.expanduser("~/.zshrc"),
        os.path.expanduser("~/.bash_profile"),
    ]
    export_line = f'export PATH="{bin_dir}:$PATH"'
    marker = "# Added by The Librarian"

    for config in shell_configs:
        _append_to_shell_config(config, export_line, marker)


def _add_to_path_linux(bin_dir: str) -> None:
    """Linux: update ~/.bashrc and ~/.profile."""
    shell_configs = [
        os.path.expanduser("~/.bashrc"),
        os.path.expanduser("~/.profile"),
    ]
    export_line = f'export PATH="{bin_dir}:$PATH"'
    marker = "# Added by The Librarian"

    for config in shell_configs:
        _append_to_shell_config(config, export_line, marker)


def _append_to_shell_config(config_path: str, export_line: str, marker: str) -> None:
    """Append a PATH export to a shell config file if not already present."""
    try:
        existing = ""
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                existing = f.read()

        # Already present — nothing to do
        if export_line in existing:
            return

        # Append
        with open(config_path, "a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(f"\n{marker}\n{export_line}\n")
    except Exception:
        pass  # Best-effort — Cowork doesn't need PATH setup
