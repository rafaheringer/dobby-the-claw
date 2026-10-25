# Web UI (PWA)

A page for the computer or the phone: live camera with detections, conversation, filterable logs,
service status, controls and settings (a test of the wake word: it shows what Dobby heard). Installable on the phone's home screen.

## How it is wired

```
phone ──https──> Cloudflare (Access login) ──tunnel──> cloudflared on the Pi
                                                          │ http://localhost:18900
                                         reverse SSH (-R 18900) ◄── PC (bridge, web UI on 127.0.0.1:18900)
```

- The web server runs **inside the bridge on the PC** (it needs the camera, the logs and the orchestrator).
  It listens on `127.0.0.1:18900` only.
- `scripts/run-windows.ps1` already opens the SSH tunnel to the Pi with `-R 18900:127.0.0.1:18900`, so the Pi's
  `localhost:18900` reaches the PC.
- The Pi's `cloudflared` publishes `dobby.example.com` → `http://localhost:18900`
  (same tunnel as `ha.example.com`).
- **Cloudflare Access** is the login. The bridge also checks the Access JWT itself, and refuses every non-local
  request when Access is not configured.

Locally (on the PC) just open <http://localhost:18900>: no login, direct requests from this machine are trusted.

## One-time setup

1. **DNS + tunnel (Pi).** In `/etc/cloudflared/config.yml` add, above the catch-all `http_status:404` rule:

   ```yaml
   - hostname: dobby.example.com
     service: http://localhost:18900
   ```

   Then `cloudflared tunnel route dns <tunnel-id> dobby.example.com`
   and `sudo systemctl restart cloudflared`.

2. **Access application (Cloudflare dashboard → Zero Trust → Access → Applications).** Add a *Self-hosted*
   application for `dobby.example.com`, with a policy that **allows only your e-mail** (one-time PIN or
   Google). From the application's overview copy the **Application Audience (AUD) Tag**; the team domain is
   `<team>.cloudflareaccess.com` (Zero Trust → Settings → Custom pages shows it).

3. **`.env` on the PC:**

   ```
   WEB_UI_ACCESS_TEAM_DOMAIN=<team>.cloudflareaccess.com
   WEB_UI_ACCESS_AUDIENCE=<AUD tag>
   WEB_UI_ALLOWED_EMAILS=you@example.com   # optional extra allow-list
   ```

4. Restart the bridge **and the SSH tunnel** (close the old `ssh` process so the script reopens it with `-R 18900`).

5. On the phone open `https://dobby.example.com`, log in, and use *Add to Home Screen*.

## Optional: panel inside Home Assistant

HA can show the page in its sidebar (`panel_iframe` in `configuration.yaml`, URL
`https://dobby.example.com`). It is only a shortcut: the login that protects Dobby is Cloudflare Access,
not the HA session. Set `WEB_UI_FRAME_ANCESTORS=https://ha.example.com` so the browser lets HA embed it.

## Security notes

- Never expose port 18900 any other way (no `0.0.0.0`, no port forwarding on the router).
- Texts that start with `/` are rejected from the network (they are orchestrator controls such as `/quit`).
- Credentials are masked in the logs sent to the browser, but logs still contain conversation text.
- *Câmera ligada* in the Visão page is a kill switch for the video stream.
- Voice from the phone is not implemented yet; the microphone in use is the robot's.
