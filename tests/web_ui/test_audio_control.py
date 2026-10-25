"""Volume and sound direction through the daemon's REST API, against a stand-in daemon."""

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestServer

from web_ui.audio_control import DaemonAudio


def _daemon(state):
    async def current(request):
        kind = "mic" if "microphone" in request.path else "speaker"
        return web.json_response(
            {"volume": state[kind], "device": f"{kind} dev", "platform": "Windows"}
        )

    async def set_volume(request):
        kind = "mic" if "microphone" in request.path else "speaker"
        body = await request.json()
        if state.get("fail"):
            return web.json_response({"detail": "nope"}, status=500)
        state[kind] = body["volume"]
        return web.json_response(
            {"volume": body["volume"], "device": f"{kind} dev", "platform": "Windows"}
        )

    async def doa(request):
        return web.json_response({"angle": 1.5708, "speech_detected": True})

    app = web.Application()
    app.router.add_get("/api/volume/current", current)
    app.router.add_get("/api/volume/microphone/current", current)
    app.router.add_post("/api/volume/set", set_volume)
    app.router.add_post("/api/volume/microphone/set", set_volume)
    app.router.add_get("/api/state/doa", doa)
    return app


def _run(state, scenario):
    async def main():
        server = TestServer(_daemon(state))
        await server.start_server()
        audio = DaemonAudio(str(server.make_url("")))
        try:
            return await scenario(audio)
        finally:
            await server.close()

    return asyncio.run(main())


def test_reads_and_sets_both_volumes():
    state = {"speaker": 40, "mic": 70}

    async def scenario(audio):
        volumes = await audio.refresh()
        assert volumes["speaker"] == {"volume": 40, "device": "speaker dev", "platform": "Windows"}
        assert volumes["mic"]["volume"] == 70
        assert await audio.set_volume("mic", 150) is None  # clamped to 100
        return audio.volumes

    assert _run(state, scenario)["mic"]["volume"] == 100 and state["mic"] == 100


def test_a_refused_set_reports_and_rereads():
    state = {"speaker": 40, "mic": 70, "fail": True}

    async def scenario(audio):
        assert await audio.set_volume("speaker", 10) == "não foi possível mudar o volume"
        assert await audio.set_volume("tv", 10) == "dispositivo inválido"
        return audio.volumes

    assert _run(state, scenario)["speaker"]["volume"] == 40


def test_direction_of_arrival():
    async def scenario(audio):
        return await audio.doa()

    assert _run({"speaker": 0, "mic": 0}, scenario) == {"angle": 1.571, "speech": True}


def test_daemon_down_gives_nothing():
    async def main():
        audio = DaemonAudio("http://127.0.0.1:9")
        return await audio.refresh(), await audio.doa()

    volumes, doa = asyncio.run(main())
    assert volumes == {"speaker": None, "mic": None} and doa is None
