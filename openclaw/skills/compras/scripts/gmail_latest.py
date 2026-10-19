#!/usr/bin/env python3
"""Print recent Gmail messages (sender, subject, snippet) for the Dobby account."""
import glob, json, os, sys, urllib.parse, urllib.request
# OAuth credentials saved by the workspace-mcp server (single Google account on the Pi).
tok = json.load(open(glob.glob(os.path.expanduser("~/.google_workspace_mcp/credentials/*.json"))[0]))
data = urllib.parse.urlencode({"client_id": tok["client_id"], "client_secret": tok["client_secret"],
    "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"}).encode()
at = json.load(urllib.request.urlopen(tok.get("token_uri", "https://oauth2.googleapis.com/token"), data))["access_token"]
def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + at})))
q = urllib.parse.quote(sys.argv[1] if len(sys.argv) > 1 else "newer_than:1h")
base = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
for m in get(f"{base}?maxResults=5&q={q}").get("messages", []):
    msg = get(f"{base}/{m['id']}?format=metadata&metadataHeaders=From&metadataHeaders=Subject")
    h = {x["name"]: x["value"] for x in msg["payload"]["headers"]}
    print(h.get("From"), "|", h.get("Subject"), "|", msg["snippet"][:200])
