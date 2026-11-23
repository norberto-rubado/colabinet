"""Run the installed official Python CLI with account-scoped paths, preserving HOME and the SSH agent.

This file runs under the CLI's own interpreter, so it must use only the standard library and colab_cli.
Upstream 0.7.4 exposes only the session-store override; other stores are redirected before loading its launcher.
Upstream 0.8 keeps the OAuth token in a single OS-keyring entry per OS user, so a profile also redirects the
keyring reads/writes to its own token.json. Without that, every profile sees the shared CLI account and a new
sign-in overwrites it.

`cli_profile.py --shared-token get|set|delete BINARY` reads or writes the shared (non-profile) CLI keyring entry with
the CLI's own keyring code; Accounts.repair uses it when the shared account is re-authenticated. `set` reads the
token from stdin and `get` prints it to stdout, so it never appears on a command line. Exit codes: 3 = this CLI has no
keyring store (0.7.x), 4 = no token stored.
"""

import logging
import os
import runpy
import sys
import uuid
from pathlib import Path

NO_KEYRING, NO_TOKEN = 3, 4


def _import_cli(binary: str):
    # This script's directory also contains Colabinet's colab_cli.py wrapper, which must not shadow upstream.
    sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]
    # Match Python's normal launcher import path, including test/install-local launchers.
    sys.path.insert(0, str(Path(binary).resolve().parent))
    from colab_cli import auth, common
    return auth, common


def _write_private(path: Path, text: str) -> None:
    """Atomically replace `path` with a new 0600 file. A unique O_EXCL temporary never reuses a leftover file or
    follows a planted symlink (R5-05)."""
    temporary = path.with_name(f".{uuid.uuid4().hex}.token.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def shared_token(op: str, binary: str) -> int:
    try:
        auth, _ = _import_cli(binary)
    except (ImportError, AttributeError):
        sys.exit("Shared re-authentication requires the official google-colab-cli Python installation.")
    if not hasattr(auth, "_save_token_to_keyring"):
        return NO_KEYRING  # 0.7.x reads TOKEN_CONFIG_PATH, a file Accounts.repair replaces itself
    if op == "get":
        token = auth._get_token_from_keyring()
        if not token:
            return NO_TOKEN
        sys.stdout.write(token)
    elif op == "set":
        token = sys.stdin.read()
        if not token.strip():
            return 2
        auth._save_token_to_keyring(token)
    elif op == "delete":
        import keyring
        import keyring.errors
        try:
            keyring.delete_password(auth.KEYRING_SERVICE_NAME, auth.KEYRING_USERNAME)
        except keyring.errors.PasswordDeleteError:
            pass
    else:
        return 2
    return 0


def main() -> None:
    os.umask(0o077)  # also covers keyring files and directories created by --shared-token (R5-03)
    if sys.argv[1:2] == ["--shared-token"]:
        sys.exit(shared_token(*sys.argv[2:4]))
    profile, binary, *args = sys.argv[1:]
    profile = Path(profile)
    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = profile / "token.json"
    try:
        auth, common = _import_cli(binary)
        from colab_cli.history import HistoryLogger
        from colab_cli.state import SettingsStore, StateStore
        auth.TOKEN_CONFIG_PATH = str(token)
        if hasattr(auth, "_get_token_from_keyring"):  # 0.8+: one keyring entry shared by every CLI caller

            def load_token():
                try:
                    return token.read_text() or None
                except FileNotFoundError:
                    return None

            auth._get_token_from_keyring = load_token
            auth._save_token_to_keyring = lambda text: _write_private(token, text)
            # The legacy migration would import (then delete) the shared ~/.config/colab-cli/token.json.
            auth.LEGACY_TOKEN_PATH = str(profile / "legacy-token.json")
            auth._remove_legacy_token = lambda: None
        common.state._store = StateStore(str(profile / "sessions.json"))
        common.state._settings_store = SettingsStore(str(profile / "settings.json"))
        common.state._history = HistoryLogger(str(profile / "history"))
        # Upstream DEBUG logging includes authenticated URLs. Do not create its shared debug log.
        logging.disable(logging.CRITICAL)
        common.setup_logging = lambda *args, **kwargs: None
    except (ImportError, AttributeError, TypeError):
        sys.exit("Account profiles require the official google-colab-cli Python installation (0.7.4 or newer).")
    sys.argv = [binary, *args]
    runpy.run_path(binary, run_name="__main__")


if __name__ == "__main__":
    main()
