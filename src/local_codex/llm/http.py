from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HTTPClientError(RuntimeError):
    pass


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 600,
) -> dict:
    merged_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        merged_headers.update(headers)

    data = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data, headers=merged_headers, method="POST")

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HTTPClientError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise HTTPClientError(f"Could not connect to {url}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPClientError(f"Non-JSON response from {url}: {raw[:400]}") from exc
