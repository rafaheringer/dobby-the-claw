#!/usr/bin/env python3
"""Print the plain-text body of the newest Gmail message matching a query (Dobby account)."""
import base64, glob, html, json, os, re, sys, urllib.parse, urllib.request
# OAuth credentials saved by the workspace-mcp server (single Google account on the Pi).
tok = json.load(open(glob.glob(os.path.expanduser("~/.google_workspace_mcp/credentials/*.json"))[0]))
data = urllib.parse.urlencode({"client_id": tok["client_id"], "client_secret": tok["client_secret"],
    "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"}).encode()
at = json.load(urllib.request.urlopen(tok.get("token_uri", "https://oauth2.googleapis.com/token"), data))["access_token"]
def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + at})))
base = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
ms = get(f"{base}?maxResults=1&q={urllib.parse.quote(sys.argv[1])}").get("messages", [])
if not ms:
    sys.exit("no message")
def parts(p):
    yield p
    for c in p.get("parts", []):
        yield from parts(c)
mid = ms[0]["id"]
msg = get(f"{base}/{mid}?format=full")
texts = [base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8", "replace")
         for p in parts(msg["payload"]) if p.get("body", {}).get("data") and p["mimeType"].startswith("text/")]
t = html.unescape(re.sub(r"<[^>]+>", " ", " ".join(texts)))
print(re.sub(r"\s+", " ", t)[:3000])
