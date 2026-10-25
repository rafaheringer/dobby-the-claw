"""The built SPA is served with a history fallback, safe paths and the right cache headers."""

import asyncio
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web_ui.access import AccessPolicy, AccessVerifier, make_guard
from web_ui.frontend_routes import add_frontend_routes

INDEX = "<html>app</html>"


def _dist(root: Path) -> Path:
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX, encoding="utf-8")
    (dist / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "sw.js").write_text("// sw", encoding="utf-8")
    (dist / "manifest.webmanifest").write_text("{}", encoding="utf-8")
    return dist


def _get_all(app: web.Application, *paths: str) -> list[tuple[int, dict, str]]:
    async def run():
        results = []
        async with TestClient(TestServer(app)) as client:
            for path in paths:
                response = await client.get(path)
                results.append((response.status, dict(response.headers), await response.text()))
        return results

    return asyncio.run(run())


def _get(app: web.Application, path: str) -> tuple[int, dict, str]:
    return _get_all(app, path)[0]


def _app(dist: Path) -> web.Application:
    app = web.Application()
    add_frontend_routes(app, dist)
    return app


def test_spa_route_returns_index(tmp_path):
    paths = ("/logs?q=x", "/sistema/nome", "/twin", "/")
    for path, (status, headers, body) in zip(paths, _get_all(_app(_dist(tmp_path)), *paths)):
        assert status == 200, path
        assert body == INDEX
        assert headers["Cache-Control"] == "no-cache"


def test_hashed_asset_is_immutable(tmp_path):
    status, headers, body = _get(_app(_dist(tmp_path)), "/assets/app-abc123.js")
    assert status == 200
    assert body == "console.log(1)"
    assert headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_root_files_no_cache(tmp_path):
    (status, headers, _), (_, manifest_headers, _) = _get_all(_app(_dist(tmp_path)), "/sw.js", "/manifest.webmanifest")
    assert status == 200
    assert headers["Cache-Control"] == "no-cache"
    headers = manifest_headers
    assert headers["Content-Type"].startswith("application/manifest+json")


def test_missing_asset_is_404(tmp_path):
    status, _, _ = _get(_app(_dist(tmp_path)), "/assets/nope.js")
    assert status == 404


def test_traversal_is_refused(tmp_path):
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    paths = ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/..%2f..%2fsecret.txt", "/..%5csecret.txt")
    for path, (status, _, body) in zip(paths, _get_all(_app(_dist(tmp_path)), *paths)):
        assert "TOP SECRET" not in body, path
        assert status in (200, 404), path


def test_missing_dist_explains_build(tmp_path):
    status, _, body = _get(_app(tmp_path / "missing"), "/")
    assert status == 503
    assert "npm --prefix src/web_ui/frontend ci" in body
    assert "run build" in body


def test_csp_allows_wasm_and_blob_workers(tmp_path):
    app = web.Application(middlewares=[make_guard(AccessVerifier(AccessPolicy(team_domain="", audience="")))])
    add_frontend_routes(app, _dist(tmp_path))
    _, headers, _ = _get(app, "/")
    csp = headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self' 'wasm-unsafe-eval'" in csp
    assert "worker-src 'self' blob:" in csp
