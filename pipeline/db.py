"""Thin PostgREST client for Supabase (service key). No ORM, no magic."""
from __future__ import annotations

import json
import time
from typing import Any, Iterable

import requests


def vec_to_pg(vector: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in vector) + "]"


def pg_to_vec(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(x) for x in json.loads(value)]


class Db:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.base = base_url.rstrip("/") + "/rest/v1"
        self.timeout = timeout
        self.session = requests.Session()
        headers = {"apikey": api_key, "Content-Type": "application/json"}
        # new-style sb_secret_* keys authenticate via apikey alone; JWTs need Bearer too
        if not api_key.startswith("sb_"):
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)

    # -- low level -----------------------------------------------------------
    def _request(self, method: str, path: str, *, params=None, json_body=None, headers=None, retries=3):
        url = f"{self.base}/{path}"
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.session.request(
                    method, url, params=params, json=json_body, headers=headers, timeout=self.timeout
                )
                if resp.status_code in (429, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
                if resp.status_code == 204 or not resp.content:
                    return None
                return resp.json()
            except requests.RequestException as exc:  # network hiccup
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"{method} {path} failed after {retries} attempts: {last_exc}")

    # -- helpers -------------------------------------------------------------
    def select(self, table: str, **params: Any) -> list[dict]:
        return self._request("GET", table, params=params) or []

    def select_in(self, table: str, column: str, values: list[str], select: str = "*", chunk: int = 100) -> list[dict]:
        rows: list[dict] = []
        for i in range(0, len(values), chunk):
            batch = values[i : i + chunk]
            quoted = ",".join('"' + v.replace('"', '\\"') + '"' for v in batch)
            rows.extend(self.select(table, select=select, **{column: f"in.({quoted})"}))
        return rows

    def insert(self, table: str, rows: list[dict] | dict, *, returning: bool = True,
               on_conflict: str | None = None, ignore_duplicates: bool = False) -> list[dict]:
        prefer = ["return=representation" if returning else "return=minimal"]
        params: dict[str, Any] = {}
        if on_conflict:
            params["on_conflict"] = on_conflict
            prefer.append("resolution=ignore-duplicates" if ignore_duplicates else "resolution=merge-duplicates")
        return self._request("POST", table, params=params, json_body=rows, headers={"Prefer": ",".join(prefer)}) or []

    def update(self, table: str, filters: dict[str, str], payload: dict, *, returning: bool = False) -> list[dict]:
        prefer = "return=representation" if returning else "return=minimal"
        return self._request("PATCH", table, params=filters, json_body=payload, headers={"Prefer": prefer}) or []

    def rpc(self, name: str, payload: dict) -> Any:
        return self._request("POST", f"rpc/{name}", json_body=payload)

    def count(self, table: str, **filters: str) -> int:
        url = f"{self.base}/{table}"
        resp = self.session.get(url, params={"select": "id", **filters}, headers={"Prefer": "count=exact", "Range": "0-0"}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"count {table} -> {resp.status_code}: {resp.text[:300]}")
        content_range = resp.headers.get("Content-Range", "*/0")
        return int(content_range.split("/")[-1])
