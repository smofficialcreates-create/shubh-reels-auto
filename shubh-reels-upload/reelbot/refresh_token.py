"""Weekly: renew the 60-day Instagram token and store the new one as a repo secret.

Needs a fine-grained personal access token (secret GH_ADMIN_TOKEN) that can write
this repository's Actions secrets. GITHUB_TOKEN itself cannot write secrets.
"""
import os
import subprocess
import sys

from .config import Config
from .publish import refresh_token


def main() -> int:
    cfg = Config()
    admin = os.environ.get("GH_ADMIN_TOKEN", "").strip()
    if not cfg.ig_token:
        print("IG_ACCESS_TOKEN is not set; nothing to refresh")
        return 0
    if "facebook" in cfg.graph_host:
        print("Unified-login system-user tokens never expire; nothing to refresh")
        return 0
    new = refresh_token(cfg.ig_token)
    if not admin:
        print("::error::Token refreshed but GH_ADMIN_TOKEN is missing, so it cannot be saved. "
              "Add that secret or the token will expire in 60 days.")
        return 1
    env = dict(os.environ, GH_TOKEN=admin)
    r = subprocess.run(["gh", "secret", "set", "IG_ACCESS_TOKEN", "--repo", cfg.repo],
                       input=new, text=True, env=env, capture_output=True)
    if r.returncode != 0:
        print("::error::could not save the refreshed token:", r.stderr.replace(admin, "***")[:200])
        return 1
    print("token refreshed and saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
