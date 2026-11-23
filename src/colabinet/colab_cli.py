"""Async wrapper around the `colab` binary: the only module that runs it.

Background commands except `new` have a watchdog; on timeout the whole process group is killed. `colab new` is
spawned in its own process group and is never killed here (spec 5.2 step 1): killing it after assignment but before
the name is saved would orphan a billing VM. Interactive login inherits the terminal and has no watchdog.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .cli_parse import Listing, Usage, make_listing, parse_status, parse_usage
from .config import cli_flags

SECRET_KEY = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD)", re.I)
PROXY_TOKEN = re.compile(r"(colab-runtime-proxy-token=)[^&\s]+")

TIMEOUTS = {"sessions": 30.0, "usage": 30.0, "upload": 120.0, "exec": 90.0, "stop": 60.0, "heartbeat": 45.0,
            "keyring": 60.0}

CLI_PROFILE = str(Path(__file__).with_name("cli_profile.py"))
SHARED_NO_KEYRING, SHARED_NO_TOKEN = 3, 4  # cli_profile.py --shared-token exit codes
WRAPPER_HEAD = 8192
SHELLS = {"sh", "bash", "dash", "zsh"}
EXEC_LINE = re.compile(r"^\s*(?:'''exec'|exec)\s")
ENV_ASSIGN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=[^$`]*\Z")


@dataclass(frozen=True)
class Launcher:
    interpreter: list[str]
    binary: str
    args: tuple[str, ...] = ()  # fixed arguments the wrapper passes before the caller's
    env: tuple[str, ...] = ()  # an `env ...` prefix replaying the wrapper's environment changes


def _is_python(words: list[str]) -> bool:
    return any("python" in Path(word).name for word in words)


def _shebang(path: str) -> list[str]:
    with open(path, "rb") as file:
        header = file.readline(4096).decode().strip()
    return shlex.split(header[2:]) if header.startswith("#!") else []


def _exec_words(path: str) -> list[str]:
    """The words of the wrapper's single top-level `exec` line (continuations joined, comments dropped)."""
    with open(path, "rb") as file:
        text = file.read(WRAPPER_HEAD).decode()
    text = text.replace("\r\n", "\n").replace("\\\n", "")
    lines = [shlex.split(line, comments=True) for line in text.split("\n")[1:] if EXEC_LINE.match(line)]
    lines = [words for words in lines if words and words[0] == "exec"]
    if len(lines) != 1:  # none, or branches we cannot evaluate
        raise ValueError
    return lines[0][1:]


def _expand(word: str) -> str:
    for prefix in ("${HOME}", "$HOME", "~"):
        if word == prefix or word.startswith(prefix + "/"):
            word = str(Path.home()) + word[len(prefix):]
            break
    if "$" in word or "`" in word or not os.path.isabs(word):
        raise ValueError
    return word


def _resolve_launcher(path: str, hop: int = 0) -> Launcher:
    shebang = _shebang(path)
    if shebang and _is_python(shebang):
        return Launcher(shebang, path)
    if not shebang or Path(shebang[0]).name not in SHELLS and not (
            Path(shebang[0]).name == "env" and shebang[1:2] and shebang[1] in SHELLS):
        raise ValueError
    words = _exec_words(path)
    if not words or words[-1] != "$@":
        raise ValueError
    words, env = words[:-1], []
    if words and Path(words[0]).name == "env":
        words, env = words[1:], [shutil.which("env") or "/usr/bin/env"]
        while words and words[0] != "--" and (words[0].startswith("-") or "=" in words[0]):
            word = words.pop(0)
            if word in ("-i", "-", "--ignore-environment"):
                env.append("-i")
            elif word in ("-u", "--unset") and words:
                env += ["-u", words.pop(0)]
            elif word.startswith("-u") and len(word) > 2:
                env += ["-u", word[2:]]
            elif word.startswith("--unset="):
                env += ["-u", word[len("--unset="):]]
            elif ENV_ASSIGN.match(word):
                env.append(word)
            else:
                raise ValueError
        if words[:1] == ["--"]:
            words = words[1:]
    if not words:
        raise ValueError
    target, rest = _expand(words[0]), words[1:]
    if _is_python([target]) and rest == ["$0"]:  # pip/uv polyglot launcher: '''exec' '<python>' "$0" "$@"
        return Launcher([target], path, (), tuple(env))
    if hop or any("$" in word or "`" in word for word in rest):
        raise ValueError
    inner = _resolve_launcher(target, 1)
    merged = (*env, *inner.env[1:]) if env and inner.env else (*env, *inner.env)
    return Launcher(inner.interpreter, inner.binary, (*inner.args, *rest), tuple(merged))


@dataclass
class CmdResult:
    rc: int
    out: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out


@dataclass
class ExecResult(CmdResult):
    status: dict | None = None


def scrub(text: str) -> str:
    """Hide runtime proxy tokens that the CLI prints inside URLs (e.g. on a failed upload)."""
    return PROXY_TOKEN.sub(r"\1***", text)


def redact(args: list[str]) -> str:
    shown = []
    for a in args:
        if "=" in a and SECRET_KEY.search(a.split("=", 1)[0]):
            a = a.split("=", 1)[0] + "=***"
        shown.append(a)
    return " ".join(shown)


class ColabCLI:
    def __init__(self, binary: str, now: Callable[[], float], log: Callable[[str], None] | None = None,
                 timeouts: dict[str, float] | None = None, auth: str | None = None,
                 profile: Path | None = None, enabled: bool = True) -> None:
        self.binary = binary
        self.now = now
        self.log = log or (lambda _msg: None)
        self.timeouts = {**TIMEOUTS, **(timeouts or {})}
        self.auth_args = [f"--auth={auth}"] if auth else []
        self.auth, self.profile, self.enabled = auth, profile, enabled

    def for_profile(self, profile: Path | None, *, enabled: bool = True) -> ColabCLI:
        return ColabCLI(self.binary, self.now, self.log, self.timeouts, self.auth, profile, enabled)

    def launcher(self) -> Launcher:
        """The official Python launcher behind `--colab-bin`, with the environment a shell wrapper sets for it.

        `--colab-bin` may be a small `#!/bin/sh` wrapper ending in `exec [env -u X K=V] <launcher> [args] "$@"` (this
        host's strips PYTHONPATH and sets PYTHON_KEYRING_BACKEND). Profile commands bypass the wrapper, so they get
        its `env` operations and fixed arguments replayed (R5-01, R5-04)."""
        try:
            return _resolve_launcher(self.binary)
        except (OSError, UnicodeError, ValueError):
            raise OSError("Account profiles require the official Python colab launcher.") from None

    def command(self) -> list[str]:
        if self.profile is None:
            return [self.binary, *self.auth_args]
        launch = self.launcher()
        return [*launch.env, *launch.interpreter, CLI_PROFILE, str(self.profile), launch.binary, *launch.args,
                *self.auth_args]

    async def _shared_token_op(self, op: str, data: bytes | None = None) -> tuple[int, bytes]:
        """Read or write the shared CLI's keyring token through the official CLI's own keyring setup.

        Async with a watchdog: a slow keyring must neither freeze the event loop nor escape as a non-OSError
        (R5-02). The token travels over pipes only; it is never logged."""
        launch = self.launcher()
        proc = await asyncio.create_subprocess_exec(
            *launch.env, *launch.interpreter, CLI_PROFILE, "--shared-token", op, launch.binary,
            stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(data), self.timeouts["keyring"])
        except TimeoutError:
            self._kill(proc)
            await proc.wait()
            raise OSError("Timed out accessing the shared CLI credentials.") from None
        except asyncio.CancelledError:
            self._kill(proc)
            raise
        return proc.returncode or 0, out

    async def shared_token(self) -> tuple[bool, str | None]:
        """(keyring store in use, current token). 0.7.x CLIs keep the token in a file and report False."""
        rc, out = await self._shared_token_op("get")
        if rc == 0:
            return True, out.decode()
        if rc in (SHARED_NO_TOKEN, SHARED_NO_KEYRING):
            return rc == SHARED_NO_TOKEN, None
        raise OSError("Could not read the shared CLI credentials.")

    async def set_shared_token(self, token: str | None) -> None:
        """Store (or, for None, delete) the shared CLI's keyring token."""
        rc, _ = await (self._shared_token_op("set", token.encode()) if token is not None
                       else self._shared_token_op("delete"))
        if rc:
            raise OSError("Could not store the shared CLI credentials.")

    def login(self) -> int:
        """Let the official CLI handle consent in the user's terminal. Usage does not allocate a runtime."""
        if not self.enabled:
            return 1
        return subprocess.run([*self.command(), "usage"], check=False).returncode

    async def spawn_login(self) -> asyncio.subprocess.Process:
        """An owned, interactive usage command for the local web UI. Never log its output or input."""
        if not self.enabled:
            raise OSError("Sign in to a Google account first.")
        return await asyncio.create_subprocess_exec(
            *self.command(), "usage", stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True,
        )

    async def identity(self) -> str | None:
        """Read only the account email from the CLI's diagnostic; never expose its raw output."""
        result = await self._run(["whoami"], self.timeouts["usage"])
        match = re.search(r"^Email:\s+([^\s<>]+@[^\s<>]+)\s*$", result.out, re.M) if result.ok else None
        return match[1] if match and len(match[1]) <= 320 else None

    async def _run(self, args: list[str], timeout: float) -> CmdResult:
        if not self.enabled and args[0] != "version":
            return CmdResult(1, "Sign in to a Google account in Settings.")
        self.log(f"colab {redact(args)}")
        proc = await asyncio.create_subprocess_exec(
            *self.command(), *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            self._kill(proc)
            await proc.wait()
            self.log(f"colab {args[0]} timed out after {timeout:.0f}s")
            return CmdResult(-9, "", timed_out=True)
        except asyncio.CancelledError:  # app shutting down: don't leave a stray CLI process behind
            self._kill(proc)
            raise
        return CmdResult(proc.returncode or 0, scrub(out.decode(errors="replace")))

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass

    async def listing(self) -> Listing:
        """One pass of `sessions` + `usage` (a "trusted listing" candidate, spec 5.3)."""
        taken = self.now()
        s = await self._run(["sessions"], self.timeouts["sessions"])
        u = await self._run(["usage"], self.timeouts["usage"])
        return make_listing(0 if s.ok else 1, s.out, 0 if u.ok else 1, u.out, taken)

    async def version(self) -> str | None:
        r = await self._run(["version"], self.timeouts["usage"])
        m = re.search(r"Version:\s*(\S+)", r.out) if r.ok else None
        return m[1] if m else None

    async def usage(self) -> Usage | None:
        r = await self._run(["usage"], self.timeouts["usage"])
        return parse_usage(r.out) if r.ok else None

    async def spawn_new(self, name: str, gpu_class: str) -> asyncio.subprocess.Process:
        if not self.enabled:
            raise OSError("Sign in to a Google account first.")
        args = ["new", "-s", name, *cli_flags(gpu_class)]
        self.log(f"colab {redact(args)} (own process group, no watchdog)")
        return await asyncio.create_subprocess_exec(
            *self.command(), *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )

    async def upload(self, name: str, local: Path, remote: str) -> CmdResult:
        return await self._run(["upload", "-s", name, str(local), remote], self.timeouts["upload"])

    async def exec_file(self, name: str, path: Path, env: dict[str, str] | None = None) -> ExecResult:
        args = ["exec", "-s", name, "-f", str(path)]
        for k, v in (env or {}).items():
            args += ["--env", f"{k}={v}"]
        r = await self._run(args, self.timeouts["exec"])
        return ExecResult(r.rc, r.out, r.timed_out, parse_status(r.out))

    async def exec_code(self, name: str, code: str, timeout: float | None = None) -> CmdResult:
        """`colab exec -s NAME -f <tmp.py>` for a short snippet (e.g. the external keep-alive heartbeat).

        The code goes through a temporary file rather than stdin, so stdin stays /dev/null: an expired token makes the
        CLI wait on input(), and that must fail rather than read the snippet."""
        fd, path = tempfile.mkstemp(prefix="colabinet-exec-", suffix=".py")
        try:
            with os.fdopen(fd, "w") as file:
                file.write(code)
            return await self._run(["exec", "-s", name, "-f", path], timeout or self.timeouts["heartbeat"])
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    async def stop(self, name: str) -> CmdResult:
        return await self._run(["stop", "-s", name], self.timeouts["stop"])
