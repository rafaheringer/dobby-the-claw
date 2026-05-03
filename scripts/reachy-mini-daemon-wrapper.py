#!/usr/bin/env python3
"""Reachy Mini daemon wrapper for Raspberry Pi camera detection.

This wrapper patches the current SDK camera discovery path before starting the
daemon. On some Raspberry Pi Ubuntu setups, the raw Gst.DeviceMonitor call can
return an empty device list unless the GLib main loop is allowed to process V4L2
events briefly, and the native V4L2 provider exposes `device.path` rather than
the `api.v4l2.path` property expected later by the SDK.
"""

from __future__ import annotations

import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import GLib, Gst

import reachy_mini.media.device_detection as device_detection


Gst.init(None)


def _patched_get_video_device():
    monitor = Gst.DeviceMonitor()
    monitor.add_filter("Video/Source")
    monitor.start()
    try:
        loop = GLib.MainLoop()
        GLib.timeout_add(2000, loop.quit)
        loop.run()
        raw_devices = monitor.get_devices()
    finally:
        monitor.stop()

    infos = []
    for index, device in enumerate(raw_devices):
        info = device_detection.gst_device_to_device_info(device, index)
        if "device.path" in info.properties and "api.v4l2.path" not in info.properties:
            info.properties["api.v4l2.path"] = info.properties["device.path"]
        infos.append(info)

    return device_detection.find_video_device(infos)


device_detection.get_video_device = _patched_get_video_device


from reachy_mini.daemon.app.main import main


if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".py")
    main()