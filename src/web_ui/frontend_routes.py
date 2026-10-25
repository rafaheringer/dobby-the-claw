"""Serve the built frontend (src/web_ui/frontend/dist) as a single-page app.

Real files are served from dist/; any other path is a client-side route and gets index.html, so
reload and deep links work. Hashed files under /assets/ never change and are cached for good;
everything else is revalidated so a new build is picked up at once.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
_IMMUTABLE = "public, max-age=31536000, immutable"
_NO_CACHE = "no-cache"
_TYPES = {".webmanifest": "application/manifest+json", ".wasm": "application/wasm", ".glb": "model/gltf-binary"}
_MISSING = """<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Dobby</title>
<body style="font-family:system-ui;background:#0f111b;color:#eef0ff;padding:2rem">
<h1>Painel web não compilado</h1>
<p>Rode na raiz do projeto:</p>
<pre>npm --prefix src/web_ui/frontend ci
npm --prefix src/web_ui/frontend run build</pre>
<p>e recarregue a página.</p></body></html>"""


def warn_if_missing(dist: Path) -> None:
    if not (dist / "index.html").is_file():
        logger.warning("Web UI frontend is not built (%s missing): run `npm --prefix src/web_ui/frontend ci && npm --prefix src/web_ui/frontend run build`", dist)


def _file(dist: Path, tail: str) -> Path | None:
    """The file under dist/ that tail names, or None (missing, a folder, or outside dist/)."""
    root = dist.resolve()
    try:
        path = (root / tail).resolve()
    except (OSError, ValueError):
        return None
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path


def _serve(path: Path, cache: str) -> web.FileResponse:
    headers = {"Cache-Control": cache}
    content_type = _TYPES.get(path.suffix)
    if content_type:
        headers["Content-Type"] = content_type
    return web.FileResponse(path, headers=headers)


def add_frontend_routes(app: web.Application, dist: Path) -> None:
    """Register the catch-all GET route; call it after every API route."""

    async def frontend(request: web.Request) -> web.StreamResponse:
        index = dist / "index.html"
        if not index.is_file():
            return web.Response(status=503, text=_MISSING, content_type="text/html", headers={"Cache-Control": _NO_CACHE})
        tail = request.match_info["tail"]
        path = _file(dist, tail)
        if tail.startswith("assets/"):
            if path is None:
                raise web.HTTPNotFound()
            return _serve(path, _IMMUTABLE)
        return _serve(path or index, _NO_CACHE)

    app.router.add_get("/{tail:.*}", frontend)
