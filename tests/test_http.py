import io
import json
import unittest
import urllib.error

from common.http import get_json


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def opener(script):
    calls = []

    def run(req, timeout):
        calls.append(req.full_url)
        step = script[len(calls) - 1]
        if isinstance(step, Exception):
            raise step
        return Resp(json.dumps(step).encode())

    return run, calls


class HttpTest(unittest.TestCase):
    def test_retries_a_timeout_then_succeeds(self):
        o, calls = opener([urllib.error.URLError("handshake timed out"), TimeoutError(), {"ok": 1}])
        waits = []
        self.assertEqual(get_json("https://x.test", "t", sleep=waits.append, opener=o), {"ok": 1})
        self.assertEqual((len(calls), waits), (3, [20.0, 40.0]))

    def test_gives_up_after_the_last_try(self):
        o, calls = opener([urllib.error.URLError("down")] * 4)
        with self.assertRaises(urllib.error.URLError):
            get_json("https://x.test", "t", sleep=lambda s: None, opener=o)
        self.assertEqual(len(calls), 4)

    def test_a_404_fails_at_once_but_a_503_is_retried(self):
        o, calls = opener([urllib.error.HTTPError("u", 404, "nf", {}, None)])
        with self.assertRaises(urllib.error.HTTPError):
            get_json("https://x.test", "t", sleep=lambda s: None, opener=o)
        self.assertEqual(len(calls), 1)
        o, calls = opener([urllib.error.HTTPError("u", 503, "busy", {}, None), {"ok": 2}])
        self.assertEqual(get_json("https://x.test", "t", sleep=lambda s: None, opener=o), {"ok": 2})


if __name__ == "__main__":
    unittest.main()
