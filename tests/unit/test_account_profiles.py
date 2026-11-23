"""Account profiles behind a shell wrapper and with the 0.8 keyring token store.

Covers the online add-account fix and review round 5 (R5-01..R5-07)."""

import asyncio
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from colabinet.accounts import Accounts
from colabinet.colab_cli import ColabCLI

PROFILE = Path(__file__).parents[2] / "src" / "colabinet" / "cli_profile.py"
SHARED_EMAIL = "shared@example.com"


def fake_cli(tmp_path: Path, *, keyring: bool = True, wrapper: str | None = None) -> tuple[Path, Path]:
    """A fake official CLI. With `keyring` it behaves like 0.8: one shared token entry (a file standing in for the OS
    keyring, picked by PYTHON_KEYRING_BACKEND the way keyring picks a backend) plus the legacy plaintext migration.
    Without it, it is 0.7.4: TOKEN_CONFIG_PATH only. The token text is the account email."""
    shared = tmp_path / "keyring-plaintext"
    default = tmp_path / "keyring-default"
    legacy = tmp_path / "home-config" / "token.json"
    package = tmp_path / "lib" / "colab_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    keyring_code = f'''
KEYRING_SERVICE_NAME, KEYRING_USERNAME = "google-colab-cli", "oauth_token"
LEGACY_TOKEN_PATH = {str(legacy)!r}
def _store():
    return Path({str(shared)!r} if os.environ.get("PYTHON_KEYRING_BACKEND") == "plain" else {str(default)!r})
def _get_token_from_keyring():
    p = _store()
    return p.read_text() if p.exists() else None
def _save_token_to_keyring(text):
    _store().write_text(text)
def _remove_legacy_token():
    Path(LEGACY_TOKEN_PATH).unlink(missing_ok=True)
def token():
    value = _get_token_from_keyring()
    if not value and Path(LEGACY_TOKEN_PATH).exists():
        value = Path(LEGACY_TOKEN_PATH).read_text()
        _save_token_to_keyring(value)
        _remove_legacy_token()
    return value
def login(value):
    _save_token_to_keyring(value)
'''
    file_code = '''
def token():
    p = Path(TOKEN_CONFIG_PATH)
    return p.read_text() if p.exists() else None
def login(value):
    Path(TOKEN_CONFIG_PATH).write_text(value)
'''
    config = tmp_path / ".config" / "colab-cli" / "token.json"
    (package / "auth.py").write_text(
        f"import os\nfrom pathlib import Path\nTOKEN_CONFIG_PATH = {str(legacy if keyring else config)!r}\n"
        + (keyring_code if keyring else file_code))
    (package / "common.py").write_text("class S:\n    _store = _settings_store = _history = None\nstate = S()\n"
                                       "def setup_logging(x): pass\n")
    (package / "state.py").write_text("class StateStore:\n    def __init__(self, p): self.p = p\n"
                                      "class SettingsStore(StateStore): pass\n")
    (package / "history.py").write_text("class HistoryLogger:\n    def __init__(self, p): self.p = p\n")
    # keyring.delete_password stand-in for `--shared-token delete`
    (tmp_path / "lib" / "keyring").mkdir()
    (tmp_path / "lib" / "keyring" / "__init__.py").write_text(
        "from colab_cli import auth\ndef delete_password(s, u):\n    auth._store().unlink(missing_ok=True)\n")
    (tmp_path / "lib" / "keyring" / "errors.py").write_text("class PasswordDeleteError(Exception): pass\n")
    launcher = tmp_path / "lib" / "colab"
    launcher.write_text(f'''#!{sys.executable}
import json, os, sys
from colab_cli import auth
cmd = sys.argv[-1]
if os.environ.get("PYTHONPATH", "").startswith("/poison"):
    sys.exit(9)  # the wrapper strips this; a leaked one breaks the real CLI's imports (R5-01)
if cmd == "whoami":
    print("Email:         " + (auth.token() or "none"))
elif cmd == "login":
    auth.login(sys.argv[-2])
elif cmd == "argv":
    print(json.dumps(sys.argv[1:-1]))
''')
    launcher.chmod(0o755)
    path = tmp_path / "colab"
    path.write_text("#!/bin/sh\n" + wrapper.format(launcher=launcher) + "\n" if wrapper else
                    f'#!/bin/sh\n# strips PYTHONPATH\nexec env -u PYTHONPATH PYTHON_KEYRING_BACKEND=plain {launcher} "$@"\n')
    path.chmod(0o755)
    return path, shared


def run(cmd, *args, env=None):
    return subprocess.run([*cmd, *args], capture_output=True, text=True, check=True,
                          env={**os.environ, **(env or {})}).stdout.strip()


async def const(value):
    return value


def test_profiles_work_behind_a_shell_wrapper(tmp_path):
    wrapper, _ = fake_cli(tmp_path)
    launch = ColabCLI(str(wrapper), time.time, profile=tmp_path / "p").launcher()
    assert launch.interpreter == [sys.executable] and launch.binary == str(tmp_path / "lib" / "colab")


def test_wrapper_without_exec_target_is_still_rejected(tmp_path):
    wrapper = tmp_path / "colab"
    wrapper.write_text("#!/bin/sh\necho hi\n")
    with pytest.raises(OSError):
        ColabCLI(str(wrapper), time.time, profile=tmp_path / "p").command()


def test_r5_07_profile_token_is_isolated_from_the_shared_keyring_and_legacy_file(tmp_path):
    """R5-07 item 1: also asserts the legacy plaintext token is neither inherited nor deleted."""
    wrapper, shared = fake_cli(tmp_path)
    shared.write_text(SHARED_EMAIL)
    legacy = tmp_path / "home-config" / "token.json"
    legacy.parent.mkdir()
    legacy.write_text("legacy@example.com")
    profile = tmp_path / "p"
    cli = ColabCLI(str(wrapper), time.time, profile=profile)
    assert run(cli.command(), "whoami") == "Email:         none"
    assert legacy.read_text() == "legacy@example.com"
    run(cli.command(), "new@example.com", "login")
    assert run(cli.command(), "whoami") == "Email:         new@example.com"
    assert stat.S_IMODE((profile / "token.json").stat().st_mode) == 0o600
    assert shared.read_text() == SHARED_EMAIL and legacy.exists()


def test_r5_01_profile_commands_replay_the_wrapper_environment(tmp_path, monkeypatch):
    """R5-01: profile and shared-token commands bypass the wrapper, so they must still drop PYTHONPATH and use the
    wrapper's keyring backend even when the caller's environment has neither."""
    wrapper, shared = fake_cli(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/poison")
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "other")
    cli = ColabCLI(str(wrapper), time.time, profile=tmp_path / "p")
    assert run(cli.command(), "whoami") == "Email:         none"
    shared.write_text(SHARED_EMAIL)
    shared_cli = ColabCLI(str(wrapper), time.time)
    assert asyncio.run(shared_cli.shared_token()) == (True, SHARED_EMAIL)
    asyncio.run(shared_cli.set_shared_token("repaired@example.com"))
    assert shared.read_text() == "repaired@example.com"
    assert not (tmp_path / "keyring-default").exists()


@pytest.mark.parametrize("line, args", [
    ('exec env -u PYTHONPATH {launcher} -c /etc/colab/oauth.json "$@"', ["-c", "/etc/colab/oauth.json"]),
    ('exec {launcher} --client-oauth-config=/etc/o.json "$@"', ["--client-oauth-config=/etc/o.json"]),
    ('exec env -u PYTHONPATH \\\n  {launcher} "$@"  # trailing comment', []),
    ('if false; then\n  echo debug\nfi\nexec "{launcher}" "$@"', []),
])
def test_r5_04_wrapper_forms_and_fixed_arguments(tmp_path, line, args):
    """R5-04: fixed arguments are kept (not dropped, not mistaken for the launcher); continuations and comments work."""
    wrapper, _ = fake_cli(tmp_path, wrapper=line)
    cli = ColabCLI(str(wrapper), time.time, profile=tmp_path / "p")
    assert cli.launcher().binary == str(tmp_path / "lib" / "colab")
    assert json.loads(run(cli.command(), "argv")) == args


def test_r5_04_home_relative_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    wrapper, _ = fake_cli(tmp_path, wrapper='exec "$HOME/lib/colab" "$@"')
    assert ColabCLI(str(wrapper), time.time, profile=tmp_path / "p").launcher().binary == str(tmp_path / "lib/colab")


@pytest.mark.parametrize("body", ['exec /a/colab "$@"\nexec /b/colab "$@"', 'exec "$COLAB" "$@"', "exec /a/colab"])
def test_r5_04_ambiguous_or_dynamic_wrappers_are_rejected(tmp_path, body):
    wrapper = tmp_path / "w"
    wrapper.write_text("#!/bin/sh\n" + body + "\n")
    with pytest.raises(OSError):
        ColabCLI(str(wrapper), time.time, profile=tmp_path / "p").launcher()


def test_r5_04_polyglot_launcher(tmp_path):
    """pip/uv write `#!/bin/sh` + `'''exec' '<python>' "$0" "$@"` launchers when the interpreter path is long."""
    fake_cli(tmp_path)
    launcher = tmp_path / "lib" / "colab"
    body = launcher.read_text().split("\n", 1)[1]
    launcher.write_text(f"#!/bin/sh\n'''exec' '{sys.executable}' \"$0\" \"$@\"\n' '''\n" + body)
    cli = ColabCLI(str(launcher), time.time, profile=tmp_path / "p")
    launch = cli.launcher()
    assert launch.interpreter == [sys.executable] and launch.binary == str(launcher)
    assert run(cli.command(), "whoami") == "Email:         none"


def test_r5_05_token_write_ignores_planted_temporary_files(tmp_path):
    """R5-05: a leftover file or a symlink at a predictable temporary name must not receive the token."""
    sys.path.insert(0, str(PROFILE.parent))
    try:
        import cli_profile
    finally:
        sys.path.remove(str(PROFILE.parent))
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    target = tmp_path / "token.json"
    (tmp_path / f".{os.getpid()}.token.tmp").symlink_to(outside)
    cli_profile._write_private(target, "secret")
    assert outside.read_text() == "untouched"
    assert not target.is_symlink() and stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.read_text() == "secret"


def accounts_with_shared(tmp_path, wrapper):
    accounts = Accounts(tmp_path / "data")
    accounts.remember(SHARED_EMAIL, ColabCLI(str(wrapper), time.time))
    candidate = accounts.candidate(ColabCLI(str(wrapper), time.time))
    (candidate.profile / "token.json").write_text(SHARED_EMAIL)
    return accounts, candidate


def test_r5_03_shared_repair_writes_keyring_without_plaintext_copy(tmp_path, monkeypatch):
    """R5-03: a 0.8 shared repair swaps the keyring entry and leaves no plaintext ~/.config/colab-cli/token.json."""
    monkeypatch.setenv("HOME", str(tmp_path))
    wrapper, shared = fake_cli(tmp_path)
    shared.write_text("expired")
    accounts, candidate = accounts_with_shared(tmp_path, wrapper)
    assert asyncio.run(accounts.repair(SHARED_EMAIL, candidate, ColabCLI(str(wrapper), time.time))) == "default"
    assert shared.read_text() == SHARED_EMAIL
    assert not (tmp_path / ".config" / "colab-cli" / "token.json").exists()


def test_r5_03_failed_metadata_write_restores_the_previous_keyring_token(tmp_path, monkeypatch):
    wrapper, shared = fake_cli(tmp_path)
    shared.write_text("expired")
    accounts, candidate = accounts_with_shared(tmp_path, wrapper)

    def denied():
        raise PermissionError("disk full")

    monkeypatch.setattr(accounts, "_save", denied)
    with pytest.raises(PermissionError):
        asyncio.run(accounts.repair(SHARED_EMAIL, candidate, ColabCLI(str(wrapper), time.time)))
    assert shared.read_text() == "expired"


def test_r5_01_unverified_new_token_is_rolled_back(tmp_path):
    """R5-01 item 3: if the shared CLI does not read the new token back as the owner, restore the old one."""
    wrapper, shared = fake_cli(tmp_path)
    shared.write_text("expired")
    accounts, candidate = accounts_with_shared(tmp_path, wrapper)
    (candidate.profile / "token.json").write_text("someone-else@example.com")
    with pytest.raises(OSError):
        asyncio.run(accounts.repair(SHARED_EMAIL, candidate, ColabCLI(str(wrapper), time.time)))
    assert shared.read_text() == "expired"


def test_r5_03_restores_an_absent_keyring_entry(tmp_path, monkeypatch):
    wrapper, shared = fake_cli(tmp_path)
    accounts, candidate = accounts_with_shared(tmp_path, wrapper)
    current = ColabCLI(str(wrapper), time.time)
    monkeypatch.setattr(current, "identity", lambda: const("other@example.com"))
    with pytest.raises(OSError):
        asyncio.run(accounts.repair(SHARED_EMAIL, candidate, current))
    assert not shared.exists()


def test_r5_02_keyring_timeout_is_an_oserror_and_does_not_block_the_loop(tmp_path):
    """R5-02: a hung keyring raises OSError (rollback paths catch it) and other tasks keep running meanwhile."""
    wrapper, _ = fake_cli(tmp_path)
    auth = tmp_path / "lib" / "colab_cli" / "auth.py"
    auth.write_text(auth.read_text().replace("def _get_token_from_keyring():",
                                             "def _get_token_from_keyring():\n    import time; time.sleep(30)"))
    cli = ColabCLI(str(wrapper), time.time, timeouts={"keyring": 1.0})

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        task = asyncio.create_task(ticker())
        started = time.monotonic()
        with pytest.raises(OSError):
            await cli.shared_token()
        task.cancel()
        return ticks, time.monotonic() - started

    ticks, elapsed = asyncio.run(main())
    assert ticks >= 10 and elapsed < 5


def test_r5_07_cli_074_profiles_and_shared_repair_use_the_token_file(tmp_path, monkeypatch):
    """R5-07 item 4: 0.7.4 has no keyring helpers; profiles still isolate and the shared repair replaces the file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    wrapper, _ = fake_cli(tmp_path, keyring=False)
    assert run(ColabCLI(str(wrapper), time.time, profile=tmp_path / "p").command(), "whoami") == "Email:         none"
    shared = ColabCLI(str(wrapper), time.time)
    assert asyncio.run(shared.shared_token()) == (False, None)
    accounts, candidate = accounts_with_shared(tmp_path, wrapper)
    asyncio.run(accounts.repair(SHARED_EMAIL, candidate, shared))
    token = tmp_path / ".config" / "colab-cli" / "token.json"
    assert token.read_text() == SHARED_EMAIL
    assert stat.S_IMODE(token.parent.stat().st_mode) == 0o700


def test_r5_06_every_rendered_account_email_is_wrapped_in_email_off():
    import re
    templates = Path(__file__).parents[2] / "src" / "colabinet" / "templates"
    for path in templates.rglob("*.html"):
        for line in path.read_text().splitlines():
            for expr in re.findall(r"{{[^}]*email[^}]*}}", line):
                if "data-account-email" in line or "[0]" in expr:
                    continue  # attribute values and the avatar initial are not rewritten by Cloudflare
                assert f"<!--email_off-->{expr}" in line, (path.name, line.strip())
