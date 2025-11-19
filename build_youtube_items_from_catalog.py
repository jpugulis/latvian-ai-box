import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "yt_catalog.json"

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

authors = data.get("authors", {})

print("YOUTUBE_ITEMS = {")
idx = 1

for author, works in authors.items():
    for work_title, parts in works.items():
        if not parts:
            continue

        # find the first part by position (0,1,2,...)
        first = min(parts, key=lambda p: p.get("position", 0))
        video_id = first["videoId"]

        # what she will say (approx)
        spoken_name = f"{author} {work_title} pirmā daļa".lower()

        # internal ID – only you see this
        internal_id = f"yt_{idx}"

        url = f"https://www.youtube.com/watch?v={video_id}"

        print(f'    "{spoken_name}": ("{internal_id}", "{url}"),')
        idx += 1

print("}")