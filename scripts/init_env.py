"""Create development secrets once without printing them."""

from pathlib import Path
import secrets

path = Path(".env")
if path.exists():
    raise SystemExit(".env already exists; existing settings preserved.")
path.write_text(
    "\n".join(
        f"{key}={secrets.token_urlsafe(32)}"
        for key in ("MLOPS_ADMIN_TOKEN", "POSTGRES_PASSWORD", "GRAFANA_ADMIN_PASSWORD")
    )
    + "\n",
    encoding="utf-8",
)
print("Development secrets saved to .env (ignored by Git).")
