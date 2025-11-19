import os
import json
import sys
from typing import Dict, Any, List

import requests

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY nav iestatīts vides mainīgajos.")
    sys.exit(1)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


def yt_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params)
    params["key"] = API_KEY
    resp = requests.get(f"{YOUTUBE_API_BASE}/{endpoint}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def find_channel_id_by_name(name: str) -> str:
    """Find channelId by searching by channel name."""
    data = yt_get(
        "search",
        {
            "part": "snippet",
            "q": name,
            "type": "channel",
            "maxResults": 1,
        },
    )
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"Neizdevās atrast kanālu pēc nosaukuma: {name}")
    return items[0]["snippet"]["channelId"]


def get_uploads_playlist_id(channel_id: str) -> str:
    data = yt_get(
        "channels",
        {
            "part": "contentDetails",
            "id": channel_id,
        },
    )
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"Neizdevās atrast channels ierakstu kanālam: {channel_id}")
    return (
        items[0]
        ["contentDetails"]["relatedPlaylists"]["uploads"]
    )


def list_playlists_for_channel(channel_id: str) -> List[Dict[str, Any]]:
    """Get all playlists owned by the channel (these often correspond to books)."""
    results = []
    page_token = None
    while True:
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": 50,
        }
        if page_token:
            params["pageToken"] = page_token
        data = yt_get("playlists", params)
        results.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return results


def list_videos_in_playlist(playlist_id: str) -> List[Dict[str, Any]]:
    """Return ordered list of videos (title, videoId, position) in the playlist."""
    results = []
    page_token = None
    while True:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,
        }
        if page_token:
            params["pageToken"] = page_token
        data = yt_get("playlistItems", params)
        for item in data.get("items", []):
            snip = item["snippet"]
            vid = {
                "title": snip["title"],
                "videoId": snip["resourceId"]["videoId"],
                "position": snip.get("position", 0),
            }
            results.append(vid)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    # sort by position just in case
    results.sort(key=lambda v: v["position"])
    return results


def parse_author_and_story_from_playlist_title(title: str):
    """
    Heuristics for titles like:
      'Aleksandrs Dimā — Grāfs Monte-Kristo. Lasa Aivars Bogdanovičs'
      'Rūdolfs Blaumanis - Raudupiete (Lasa ...)'
      'Jānis Poruks. PĒRĻU ZVEJNIEKS'
    Returns (author, story) or (None, None) if fails.
    """
    # Normalize dash variants
    t = title.replace(" — ", " - ").strip()
    # Split off 'Lasa ...' and other trailing stuff
    parts = t.split("Lasa", 1)
    main = parts[0].strip()
    # Try author - story
    if " - " in main:
        a, s = main.split(" - ", 1)
        return a.strip(), s.strip().strip(".")
    # Try author. story
    if ". " in main:
        a, s = main.split(". ", 1)
        return a.strip(), s.strip().strip(".")
    return None, None


def build_catalog_for_channel(channel_name: str) -> Dict[str, Any]:
    """
    Build a nested catalog:
        {
          "authors": {
            "Aleksandrs Dimā": {
              "Grāfs Monte-Kristo": [
                  {"part_title": "...nodaļa 1", "videoId": "..."},
                  ...
              ]
            },
            ...
          }
        }
    """
    channel_id = find_channel_id_by_name(channel_name)
    print("Atrasts kanāla ID:", channel_id)

    playlists = list_playlists_for_channel(channel_id)
    print(f"Atrastās atskaņošanas sarakstu skaits: {len(playlists)}")

    catalog: Dict[str, Dict[str, list]] = {}

    for pl in playlists:
        pl_title = pl["snippet"]["title"]
        playlist_id = pl["id"]
        author, story = parse_author_and_story_from_playlist_title(pl_title)
        if not author or not story:
            # Skip playlists that are not audiobooks
            print("   [SKIP] Nevarēju saprast autoru/stāstu no:", pl_title)
            continue

        print(f"-> Apstrādāju: {author} — {story}")
        videos = list_videos_in_playlist(playlist_id)

        author_entry = catalog.setdefault(author, {})
        story_entry = author_entry.setdefault(story, [])

        for v in videos:
            story_entry.append(
                {
                    "part_title": v["title"],
                    "videoId": v["videoId"],
                    "position": v["position"],
                }
            )

    return {"authors": catalog}


def main():
    channel_name = "Latvijas Neredzīgo bibliotēka"
    catalog = build_catalog_for_channel(channel_name)
    out_path = os.path.join(os.path.dirname(__file__), "yt_catalog.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print("Saglabāts katalogs:", out_path)


if __name__ == "__main__":
    main()