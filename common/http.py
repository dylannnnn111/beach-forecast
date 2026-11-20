"""One place for web requests, so every job survives a slow or dropped connection.

The 18 Nov 2026 daily run failed on a single TLS handshake timeout with no retry.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


def get_json(url: str, user_agent: str, tries: int = 4, timeout: float = 45.0, pause: float = 20.0, sleep=time.sleep, opener=urllib.request.urlopen):
    """GET a URL and parse JSON. Retries timeouts, dropped connections, 429 and 5xx; other HTTP errors fail at once."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with opener(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = e
        if attempt < tries - 1:
            wait = pause * (attempt + 1)
            print(f"  request failed ({last}), retrying in {wait:.0f}s", file=sys.stderr)
            sleep(wait)
    raise last
