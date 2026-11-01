"""Host the finished video on a throw-away `media` branch, then publish it as a
Reel through Instagram's official Content Publishing API.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import tempfile
import time

import requests

from .config import Config


def push_media(cfg: Config, video_path: str, cover_path: str | None = None, branch: str = "media") -> str:
    """Force-push a single-commit branch holding only today's video; returns its public URL."""
    name = dt.datetime.utcnow().strftime("reel-%Y%m%d-%H%M%S.mp4")
    tmp = tempfile.mkdtemp(prefix="media-")
    try:
        os.makedirs(os.path.join(tmp, "reels"))
        shutil.copy(video_path, os.path.join(tmp, "reels", name))
        if cover_path:
            shutil.copy(cover_path, os.path.join(tmp, "reels", name.replace(".mp4", ".jpg")))
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")

        def git(*args, check=True):
            return subprocess.run(["git", "-C", tmp, *args], check=check, env=env,
                                  capture_output=True, text=True)
        git("init", "-q")
        git("config", "user.name", "reel-bot")
        git("config", "user.email", "reel-bot@users.noreply.github.com")
        git("checkout", "-q", "-b", branch)
        git("add", "-A")
        git("commit", "-q", "-m", "media: latest reel")
        remote = f"https://x-access-token:{cfg.gh_token}@github.com/{cfg.repo}.git"
        r = subprocess.run(["git", "-C", tmp, "push", "-q", "--force", remote, f"{branch}:{branch}"],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("could not push media branch: " + r.stderr.replace(cfg.gh_token, "***")[:300])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    base = f"https://raw.githubusercontent.com/{cfg.repo}/{branch}/reels/"
    urls = {"video": base + name, "cover": (base + name.replace(".mp4", ".jpg")) if cover_path else None}
    for key, url in urls.items():
        if not url:
            continue
        for _ in range(12):
            h = requests.head(url, timeout=20, allow_redirects=True)
            if h.status_code == 200:
                print(f"{key} url ok, content-type:", h.headers.get("content-type"))
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"uploaded {key} is not publicly reachable (is the repository public?)")
    return urls


def _graph(cfg: Config, path: str) -> str:
    return f"https://{cfg.graph_host}/{cfg.graph_version}/{path}"


def _check(r: requests.Response, what: str) -> dict:
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code >= 400 or "error" in data:
        err = data.get("error", {})
        raise RuntimeError(f"{what} failed: HTTP {r.status_code} code={err.get('code')} {err.get('message', '')[:200]}")
    return data


def publish_reel(cfg: Config, video_url: str, caption: str, cover_url: str | None = None, timeout_s: int = 600) -> dict:
    auth = {"Authorization": f"Bearer {cfg.ig_token}"}
    r = requests.post(_graph(cfg, f"{cfg.ig_user_id}/media"), headers=auth, timeout=60,
                      data={"media_type": "REELS", "video_url": video_url, "caption": caption,
                            "share_to_feed": "true", **({"cover_url": cover_url} if cover_url else {})})
    container = _check(r, "create container")["id"]
    deadline = time.time() + timeout_s
    while True:
        s = _check(requests.get(_graph(cfg, container), headers=auth, timeout=30,
                                params={"fields": "status_code,status"}), "poll container")
        code = s.get("status_code")
        print("container status:", code)
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram rejected the video: {s.get('status')}")
        if time.time() > deadline:
            raise RuntimeError("timed out waiting for Instagram to process the video")
        time.sleep(10)
    r = requests.post(_graph(cfg, f"{cfg.ig_user_id}/media_publish"), headers=auth, timeout=60,
                      data={"creation_id": container})
    media_id = _check(r, "publish")["id"]
    info = {}
    try:
        info = requests.get(_graph(cfg, media_id), headers=auth, timeout=30,
                            params={"fields": "permalink"}).json()
    except requests.RequestException:
        pass
    return {"id": media_id, "permalink": info.get("permalink")}


def refresh_token(token: str) -> str:
    r = requests.get("https://graph.instagram.com/refresh_access_token", timeout=30,
                     params={"grant_type": "ig_refresh_token", "access_token": token})
    data = _check(r, "token refresh")
    return data["access_token"]
