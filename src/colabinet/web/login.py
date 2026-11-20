"""Bridge Google's CLI copy/paste consent flow to the local console.

Only a validated Google URL and account summary leave this module. Codes go directly to the owned CLI's
stdin; transcripts and credential files are never logged, returned, or stored in the console database.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..cli_parse import Usage, parse_usage
from ..colab_cli import ColabCLI
from ..supervisor import Supervisor

PROMPT = "Enter the authorization code:"
MAX_OUTPUT = 64 * 1024
log = logging.getLogger("colabinet.login")


class LoginError(ValueError):
    pass


def google_url(output: str) -> str | None:
    for match in re.finditer(r"https://[^\s<>\"']+(?=\s)", output):
        url = match[0]
        try:
            parsed = urlsplit(url)
            if (parsed.scheme == "https" and parsed.hostname == "accounts.google.com"
                    and parsed.port in (None, 443) and not parsed.username and not parsed.password
                    and parsed.path in ("/o/oauth2/auth", "/o/oauth2/v2/auth")):
                return url
        except ValueError:
            pass
    return None


class GoogleLogin:
    def __init__(self, sup: Supervisor, timeout: float = 600) -> None:
        self.sup, self.mode, self.timeout = sup, sup.s.colab_auth, timeout
        self.state = "idle"
        self.flow_id = ""
        self.url: str | None = None
        self.email: str | None = None
        self.balance: float | None = None
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._changing = False

    @property
    def cli(self) -> ColabCLI:
        return self.sup.cli

    @property
    def busy(self) -> bool:
        return self._changing or (self._task is not None and not self._task.done())

    def snapshot(self) -> dict:
        accounts = self.sup.accounts.snapshot() if self.sup.accounts else []
        active = next((row for row in accounts if row["active"]), None)
        return {"state": self.state, "mode": self.mode, "busy": self.busy, "flow_id": self.flow_id,
                "url": self.url, "email": self.email, "balance": self.balance, "error": self.error,
                "accounts": accounts, "active_id": self.sup.account_id,
                "active_email": active["email"] if active else None}

    def _begin(self, state: str) -> None:
        self.state, self.flow_id = state, secrets.token_urlsafe(16)
        self.url = self.email = self.balance = self.error = None

    async def start(self, account_id: str | None = None) -> None:
        async with self._lock:
            if self.mode != "oauth2":
                raise LoginError("ADC 登录由 gcloud 管理。配置好 ADC 后，请使用“检查账号”。")
            if not self.busy:
                if self.sup.accounts is None:
                    raise LoginError("此控制台无法使用已保存的账号。")
                if account_id:
                    try:
                        self.sup.accounts.get(account_id)
                    except ValueError as exc:
                        raise LoginError(str(exc)) from None
                await self.sup.begin_account_change(account_id)
                try:
                    candidate = self.sup.accounts.candidate(self.cli)
                    candidate.command()
                except (OSError, ValueError):
                    self.sup.account_change_pending = False
                    raise LoginError("账号配置需要官方 google-colab-cli 的 Python 安装。"
                                     "请查看安装指南。") from None
                self._begin("starting")
                self._task = asyncio.create_task(self._login(candidate, account_id))

    async def check(self) -> None:
        async with self._lock:
            if not self.busy:
                if self.sup.account_id is None:
                    self.state, self.email, self.balance, self.error = "signed_out", None, None, None
                    return
                self._begin("checking")
                self._task = asyncio.create_task(self._check())

    async def switch(self, account_id: str) -> None:
        async with self._lock:
            if self.mode != "oauth2" or self.sup.accounts is None:
                raise LoginError("只有使用 OAuth2 认证时才能切换已保存的账号。")
            if self.busy:
                raise LoginError("请先完成或取消当前登录，再切换账号。")
            await self.sup.begin_account_change()
            self._changing = True
            try:
                candidate = self.sup.accounts.cli(self.cli, account_id)
                usage = await candidate.usage()
                email = await candidate.identity() if usage else None
                if not email or email.casefold() != self.sup.accounts.get(account_id)["email"].casefold():
                    raise LoginError("无法验证此账号。请重新使用 Google 登录。")
                await self.sup.activate_account(account_id, candidate, usage,
                                                lambda: self.sup.accounts.select(account_id))
                self.email, self.balance = email, usage.balance
                self.state, self.url, self.error = "connected", None, None
            except ValueError as exc:
                raise LoginError(str(exc)) from None
            except OSError:
                raise LoginError("Colab CLI 无法运行。请检查其安装和路径。") from None
            finally:
                self._changing = self.sup.account_change_pending = False

    async def logout(self, account_id: str) -> None:
        async with self._lock:
            if self.sup.accounts is None or self.mode != "oauth2":
                raise LoginError("请使用 gcloud auth application-default revoke 退出 ADC。")
            if self.busy:
                raise LoginError("请先完成或取消当前登录，再退出登录。")
            if account_id != self.sup.account_id or not account_id:
                raise LoginError("当前账号已变更。请刷新设置页后再退出登录。")
            await self.sup.begin_account_change()
            try:
                await self.sup.activate_account(None, self.cli.for_profile(None, enabled=False), None,
                                                self.sup.accounts.logout)
                self._begin("signed_out")
            finally:
                self.sup.account_change_pending = False

    async def submit(self, flow_id: str, code: str) -> None:
        async with self._lock:
            if flow_id != self.flow_id or self.state != "waiting" or self._proc is None:
                raise LoginError("此登录已不再等待授权码。"
                                 "请重新开始，或检查当前账号。")
            if not code.strip() or len(code) > 4096 or any(char in code for char in ("\n", "\r", "\0")):
                raise LoginError("请粘贴 Google 完成页上的一个授权码。")
            self.state, self.url = "verifying", None
            try:
                self._proc.stdin.write(code.strip().encode() + b"\n")
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                raise LoginError("登录进程已结束。请重新登录。") from None

    async def cancel(self, flow_id: str) -> None:
        async with self._lock:
            if flow_id != self.flow_id:
                raise LoginError("此登录已被新的登录取代。请刷新账号卡片。")
            self.state = "cancelling"
            await self.close()
            self.state, self.url, self.error = "cancelled", None, None

    async def close(self) -> None:
        """Called under cancel's lock, or by lifespan after the server has drained HTTP requests."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def _fail(self, message: str) -> None:
        self.state, self.url, self.error = "failed", None, message

    async def _verified(self, usage: Usage, cli: ColabCLI, *, added: bool = False,
                        expected_account: str | None = None) -> None:
        self.email = await cli.identity()
        if not self.email and self.mode == "oauth2":
            raise LoginError("无法验证账号邮箱。请检查网络连接后重新登录。")
        self.balance = usage.balance
        if added:
            if expected_account:
                expected = self.sup.accounts.get(expected_account)["email"]
                if expected.casefold() != self.email.casefold():
                    raise LoginError(f"请选择 {expected} 来登录此已保存的账号。")
            if expected_account and expected_account == self.sup.account_id:
                repaired = self.cli.for_profile(self.cli.profile)
                await self.sup.activate_account(expected_account, repaired, usage,
                                                lambda: self.sup.accounts.repair(self.email, cli, repaired),
                                                reconnect=True)
                await self.sup.resume_account_sessions()
            else:
                await self.sup.activate_account(None, cli, usage, lambda: self.sup.accounts.remember(self.email, cli))
        else:
            if self.sup.accounts and self.email and self.mode == "oauth2":
                current = next((row for row in self.sup.accounts.rows if row["id"] == self.sup.account_id), None)
                if current and current["email"].casefold() != self.email.casefold():
                    raise LoginError("CLI 账号已在此控制台之外被更改。请重新使用 Google 登录。")
                self.sup.accounts.remember(self.email, cli)
            self.sup.usage, self.sup.usage_at = usage, self.sup.now()
        self.state, self.url, self.error = "connected", None, None

    async def _check(self) -> None:
        try:
            async with self.sup._listing_lock:
                usage = await self.cli.usage()
                if usage is None:
                    self._fail("无法验证账号访问权限。请使用 Google 登录，或检查网络连接。")
                else:
                    await self._verified(usage, self.cli)
            await self.sup.resume_account_sessions()
        except LoginError as exc:
            self._fail(str(exc))
        except OSError:
            self._fail("Colab CLI 无法运行。请检查其安装和路径。")

    async def _login(self, cli: ColabCLI, expected_account: str | None = None) -> None:
        proc = None
        error = None
        try:
            async with asyncio.timeout(self.timeout):
                self._proc = proc = await cli.spawn_login()
                output = bytearray()
                while chunk := await proc.stdout.read(1024):
                    output.extend(chunk)
                    if len(output) > MAX_OUTPUT:
                        raise LoginError("CLI 返回了意外的输出。请检查其版本后重试。")
                    transcript = output.decode(errors="replace")
                    if self.state == "starting" and PROMPT in transcript:
                        self.url = google_url(transcript)
                        if not self.url:
                            raise LoginError("CLI 没有提供受支持的 Google 登录链接。"
                                             "请检查其版本。")
                        parsed = urlsplit(self.url)
                        query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "prompt"]
                        query.append(("prompt", "select_account consent"))
                        self.url = urlunsplit(parsed._replace(query=urlencode(query)))
                        self.state = "waiting"
                result = await proc.wait()
                usage = parse_usage(output.decode(errors="replace")) if result == 0 else None
                output.clear()
                if usage is None:
                    raise LoginError("无法完成 Google 登录。授权码可能无效或已过期；"
                                     "请重新开始并检查网络连接。")
                self.state, self.url = "verifying", None
                await self._verified(usage, cli, added=True, expected_account=expected_account)
        except TimeoutError:
            error = "等待 Google 超时，此登录已过期。请重新登录。"
        except ValueError as exc:
            error = str(exc)
        except OSError:
            error = "Colab CLI 无法运行。请检查其安装和路径。"
        finally:
            if proc is not None:
                if proc.stdin:
                    proc.stdin.close()
                if proc.returncode is None:
                    cli._kill(proc)
                await proc.wait()
            self._proc = None
            try:
                self.sup.accounts.discard(cli)
            except OSError:
                log.warning("Could not remove an unused sign-in profile. Check data-directory permissions.")
            finally:
                self.sup.account_change_pending = False
        if error:
            self._fail(error)
