"""Read how each reel performed through Instagram's official insights API."""
from __future__ import annotations

import requests

from .config import Config

REEL_METRICS = ["reach", "views", "likes", "comments", "shares", "saved", "total_interactions",
                "ig_reels_avg_watch_time", "ig_reels_video_view_total_time"]


class InsightsError(RuntimeError):
    pass


def _get(cfg: Config, path: str, params: dict | None = None) -> dict:
    r = requests.get(f"https://{cfg.graph_host}/{cfg.graph_version}/{path}", params=params or {}, timeout=40,
                     headers={"Authorization": f"Bearer {cfg.ig_token}"})
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code >= 400 or "error" in data:
        e = data.get("error", {})
        raise InsightsError(f"HTTP {r.status_code} code={e.get('code')} {e.get('message', '')[:200]}")
    return data


def media_metrics(cfg: Config, media_id: str) -> dict:
    """All metrics in one call; if Instagram rejects one, ask for them one by one and keep what works."""
    def values(d):
        return {m["name"]: (m.get("values") or [{}])[0].get("value", m.get("total_value", {}).get("value"))
                for m in d.get("data", [])}
    try:
        return values(_get(cfg, f"{media_id}/insights", {"metric": ",".join(REEL_METRICS)}))
    except InsightsError:
        out = {}
        for m in REEL_METRICS:
            try:
                out.update(values(_get(cfg, f"{media_id}/insights", {"metric": m})))
            except InsightsError:
                continue
        return out


def collect(cfg: Config, limit: int = 40) -> dict:
    """{'media': {id: {timestamp, permalink, type, likes, comments, metrics{...}}}, 'account': {...}}"""
    listing = _get(cfg, f"{cfg.ig_user_id}/media", {"fields": "id,timestamp,permalink,media_product_type,like_count,comments_count",
                                                    "limit": limit})
    media = {}
    for m in listing.get("data", []):
        met = media_metrics(cfg, m["id"]) if m.get("media_product_type") in ("REELS", "FEED") else {}
        media[m["id"]] = {"timestamp": m.get("timestamp"), "permalink": m.get("permalink"), "type": m.get("media_product_type"),
                          "like_count": m.get("like_count"), "comments_count": m.get("comments_count"), "metrics": met}
    account = {}
    try:
        acc = _get(cfg, cfg.ig_user_id, {"fields": "followers_count,media_count"})
        account = {"followers": acc.get("followers_count"), "media_count": acc.get("media_count")}
    except InsightsError:
        pass
    return {"media": media, "account": account}


def comments_for(cfg: Config, media_ids: list[str], per: int = 15) -> list[dict]:
    out = []
    for mid in media_ids:
        try:
            for c in _get(cfg, f"{mid}/comments", {"fields": "id,text,username,timestamp", "limit": per}).get("data", []):
                out.append({"media_id": mid, "id": c["id"], "text": c.get("text", ""), "username": c.get("username", "")})
        except InsightsError:
            continue
    return out
