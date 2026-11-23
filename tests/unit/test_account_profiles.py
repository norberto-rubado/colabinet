"""Account profiles behind a shell wrapper and with the 0.8 keyring token store (online add-account regression)."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from colabinet.colab_cli import ColabCLI

PROFILE = Path(__file__).parents[2] / "src" / "colabinet" / "cli_profile.py"


def fake_cli(tmp_path: Path) -> tuple[Path, Path]:
    """A fake 0.8-style package: the token lives in a single shared 'keyring' file, like the OS keyring."""
    shared = tmp_path / "shared-keyring"
    package = tmp_path / "lib" / "colab_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "auth.py").write_text(f'''
from pathlib import Path
TOKEN_CONFIG_PATH = "unused"
LEGACY_TOKEN_PATH = {str(tmp_path / "legacy.json")!r}
def _get_token_from_keyring():
    p = Path({str(shared)!r})
    return p.read_text() if p.exists() else None
def _save_token_to_keyring(text):
    Path({str(shared)!r}).write_text(text)
def _remove_legacy_token(): Path(LEGACY_TOKEN_PATH).unlink(missing_ok=True)
''')
    (package / "common.py").write_text("class S:\n    _store = _settings_store = _history = None\nstate = S()\n"
                                       "def setup_logging(x): pass\n")
    (package / "state.py").write_text("class StateStore:\n    def __init__(self, p): self.p = p\n"
                                      "class SettingsStore(StateStore): pass\n")
    (package / "history.py").write_text("class HistoryLogger:\n    def __init__(self, p): self.p = p\n")
    launcher = tmp_path / "lib" / "colab"
    launcher.write_text(f'''#!{sys.executable}
import sys
from colab_cli import auth
cmd = sys.argv[-1]
if cmd == "whoami":
    print("Email: " + (auth._get_token_from_keyring() or "none"))
elif cmd == "login":
    auth._save_token_to_keyring(sys.argv[-2])
''')
    launcher.chmod(0o755)
    wrapper = tmp_path / "colab"
    wrapper.write_text(f'#!/bin/sh\n# strips PYTHONPATH\nexec env -u PYTHONPATH X=1 {launcher} "$@"\n')
    wrapper.chmod(0o755)
    return wrapper, shared


def run(cmd, *args):
    return subprocess.run([*cmd, *args], capture_output=True, text=True, check=True).stdout.strip()


def test_profiles_work_behind_a_shell_wrapper(tmp_path):
    wrapper, _ = fake_cli(tmp_path)
    cli = ColabCLI(str(wrapper), time.time, profile=tmp_path / "p")
    command = cli.command()
    assert command[0] == sys.executable and command[1] == str(PROFILE)
    assert command[3] == str(tmp_path / "lib" / "colab")


def test_wrapper_without_exec_target_is_still_rejected(tmp_path):
    wrapper = tmp_path / "colab"
    wrapper.write_text("#!/bin/sh\necho hi\n")
    with pytest.raises(OSError):
        ColabCLI(str(wrapper), time.time, profile=tmp_path / "p").command()


def test_profile_token_is_isolated_from_the_shared_keyring(tmp_path):
    wrapper, shared = fake_cli(tmp_path)
    shared.write_text("shared@example.com")
    profile = tmp_path / "p"
    cli = ColabCLI(str(wrapper), time.time, profile=profile)
    assert run(cli.command(), "whoami") == "Email: none"  # a new profile must not inherit the shared account
    run(cli.command(), "new@example.com", "login")
    assert run(cli.command(), "whoami") == "Email: new@example.com"
    assert (profile / "token.json").stat().st_mode & 0o777 == 0o600
    assert shared.read_text() == "shared@example.com"  # and must not overwrite it


def test_store_shared_token_writes_the_shared_keyring(tmp_path):
    wrapper, shared = fake_cli(tmp_path)
    token = tmp_path / "token.json"
    token.write_text("repaired@example.com")
    ColabCLI(str(wrapper), time.time).store_shared_token(token)
    assert shared.read_text() == "repaired@example.com"
