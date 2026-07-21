"""Small quota-aware client for the free API-Football account."""
import json
import os
import urllib.parse
import urllib.request


BASE = "https://v3.football.api-sports.io"


class Client:
    def __init__(self, key=None, limit=90, opener=urllib.request.urlopen):
        self.key = key or os.environ.get("API_FOOTBALL_KEY")
        if not self.key:
            raise RuntimeError("API_FOOTBALL_KEY is not set")
        self.limit = limit
        self.used = 0
        self.opener = opener

    def get(self, path, **params):
        if self.used >= self.limit:
            raise RuntimeError(f"API-Football run budget exhausted ({self.limit})")
        query = urllib.parse.urlencode({k: v for k, v in params.items()
                                       if v is not None})
        url = f"{BASE}/{path.lstrip('/')}" + (f"?{query}" if query else "")
        request = urllib.request.Request(
            url, headers={"x-apisports-key": self.key,
                          "User-Agent": "henrys-match-engine/1.0"})
        with self.opener(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.used += 1
        errors = payload.get("errors")
        if errors:
            raise RuntimeError(f"API-Football error: {errors}")
        return payload.get("response") or []
