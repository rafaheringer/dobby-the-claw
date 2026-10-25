"""Who may reach the web UI.

Two kinds of request arrive on the local port:
- direct ones from this PC (Host is localhost, no proxy headers): trusted;
- ones forwarded by the Cloudflare tunnel: they must carry a valid Cloudflare Access JWT for
  this application, otherwise they are refused. If Access is not configured, remote access is
  off (fails closed) rather than open.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from aiohttp import web

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
_PROXY_HEADERS = ("Cf-Ray", "Cf-Connecting-Ip", "Cf-Access-Jwt-Assertion", "X-Forwarded-For", "X-Forwarded-Host", "Forwarded")
_JWT_HEADER = "Cf-Access-Jwt-Assertion"
_JWT_COOKIE = "CF_Authorization"
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@dataclass(frozen=True)
class AccessPolicy:
    team_domain: str  # e.g. "myteam.cloudflareaccess.com"
    audience: str  # the Access application's AUD tag
    allowed_emails: tuple[str, ...] = ()
    frame_ancestors: tuple[str, ...] = ()  # sites allowed to embed the page (e.g. Home Assistant)

    @property
    def configured(self) -> bool:
        return bool(self.team_domain and self.audience)


class AccessVerifier:
    def __init__(self, policy: AccessPolicy) -> None:
        self._policy = policy
        self._jwks = None
        if policy.configured:
            import jwt  # PyJWT

            team = policy.team_domain.removeprefix("https://").strip("/")
            self._issuer = f"https://{team}"
            self._jwks = jwt.PyJWKClient(f"{self._issuer}/cdn-cgi/access/certs", cache_keys=True)

    def verify(self, token: str) -> Optional[str]:
        """Return the caller's email when the token is valid for this application, else None."""
        if self._jwks is None or not token:
            return None
        import jwt

        try:
            key = self._jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=self._policy.audience, issuer=self._issuer)
        except Exception as exc:
            logger.warning("Web UI: rejected Access token (%s)", type(exc).__name__)
            return None
        email = str(claims.get("email", "")).lower()
        if self._policy.allowed_emails and email not in self._policy.allowed_emails:
            logger.warning("Web UI: %s is not in WEB_UI_ALLOWED_EMAILS", email or "unknown identity")
            return None
        return email or "access"


def _host_only(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):  # [::1]:18900
        return host.split("]")[0] + "]"
    return host.split(":")[0]


def is_direct_local(request: web.Request) -> bool:
    return _host_only(request.host) in _LOCAL_HOSTS and not any(h in request.headers for h in _PROXY_HEADERS)


def same_origin(request: web.Request) -> bool:
    """Browsers send Origin on WebSocket upgrades and cross-site writes; it must be this very site."""
    origin = request.headers.get("Origin")
    if origin is None:
        return request.method not in _UNSAFE_METHODS
    return urlsplit(origin).netloc.lower() == request.host.lower()


def make_guard(verifier: AccessVerifier):
    ancestors = " ".join(("'self'",) + verifier._policy.frame_ancestors)

    @web.middleware
    async def guard(request: web.Request, handler):
        if not same_origin(request):
            raise web.HTTPForbidden(text="origem não permitida")
        if is_direct_local(request):
            request["identity"] = "local"
        else:
            if not verifier._policy.configured:
                raise web.HTTPServiceUnavailable(text="acesso remoto desligado: Cloudflare Access não configurado")
            token = request.headers.get(_JWT_HEADER) or request.cookies.get(_JWT_COOKIE, "")
            identity = await asyncio.get_running_loop().run_in_executor(None, verifier.verify, token)
            if identity is None:
                raise web.HTTPUnauthorized(text="não autenticado")
            request["identity"] = identity
        response = await handler(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            # wasm-unsafe-eval + blob workers: the Draco decoder of the 3D robot model (no JS eval).
            "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; "
            "img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            f"connect-src 'self' ws: wss:; frame-ancestors {ancestors}",
        )
        return response

    return guard
