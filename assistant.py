import os
import sys
import tempfile
import subprocess
import glob
import webbrowser
import requests
import threading
import time
import json
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI


import sys
if sys.platform.startswith("win"):
    import keyboard


# ===== CONFIG =====
SAMPLE_RATE = 16_000
MAX_RECORD_SECONDS = 30  # hard safety cap
TRANSCRIBE_MODEL = "gpt-4o-transcribe"   # or "whisper-1"
CHAT_MODEL = "gpt-4.1"                   # main brain
TTS_MODEL = "gpt-4o-mini-tts"            # newer text-to-speech model
TTS_VOICE = "onyx"                       # voice (multilingual, more male)
SCHEDULED_BOOK_ID = os.getenv("SCHEDULED_BOOK_ID")  # book ID used for scheduled chapter reads
DEFAULT_READING_TIMES = ["23:10"]        # default nightly reading times (HH:MM, 24h)
DEFAULT_FILES_PER_SESSION = 1            # how many mp3 files to play per scheduled run
SCHEDULED_READING_TIMES = DEFAULT_READING_TIMES.copy()
SCHEDULED_FILES_PER_SESSION = DEFAULT_FILES_PER_SESSION

# Available TTS voices user can choose from (spoken label -> OpenAI voice name)
AVAILABLE_VOICES = {
    "vīrieša balss": "onyx",
    "sieviešu balss": "opal",
    "neitrāla balss": "coral",
}

# mutable current voice used by tts_latvian_to_file
CURRENT_VOICE = AVAILABLE_VOICES["vīrieša balss"]

# Root folder for local audiobooks (subfolders with .mp3 files)
AUDIOBOOKS_ROOT = os.path.join(os.path.dirname(__file__), "audiobooks")

# Human title (what the user says) -> folder ID (subdirectory name under AUDIOBOOKS_ROOT)
# Add more entries here when you download new audiobooks.
AUDIOBOOKS = {
    # "zvejnieka dēls": "zvejnieka_dels",
    # "nāves ēnā": "naves_ena",
}

# YouTube "audiobook" items: spoken title -> (ID, URL)
# Šeit ir sākuma saraksts – tikai pirmās daļas, lai izvēle būtu pārskatāma.
YOUTUBE_ITEMS = {
    "aleksandrs dimā grāfs monte-kristo pirmā daļa": ("yt_1", "https://www.youtube.com/watch?v=PvCmfek20BI"),
    "andrievs niedra līduma dūmos pirmā daļa": ("yt_2", "https://www.youtube.com/watch?v=opy9L2kmdIE"),
    "augusts deglavs zeltenīte pirmā daļa": ("yt_3", "https://www.youtube.com/watch?v=1yv-G8llzoQ"),
    "jānis poruks pērļu zvejnieks pirmā daļa": ("yt_4", "https://www.youtube.com/watch?v=vBjfb-HUZwg"),
    "jēkabs janševskis varkaļu pagasta skolotājs pirmā daļa": ("yt_5", "https://www.youtube.com/watch?v=kS1ie2jzRA0"),
    "jēkabs janševskis līgava pirmā daļa": ("yt_6", "https://www.youtube.com/watch?v=__1luC8XYSQ"),
    "jēkabs janševskis bandavā pirmā daļa": ("yt_7", "https://www.youtube.com/watch?v=Ksj2KHcxBvQ"),
    "jēkabs janševskis mežvidus ļaudis pirmā daļa": ("yt_8", "https://www.youtube.com/watch?v=VPtxXi7yhts"),
    "pāvils rozītis ceplis pirmā daļa": ("yt_9", "https://www.youtube.com/watch?v=7XZmnhhV7dA"),
    "kārlis skalbe pasakas pirmā daļa": ("yt_10", "https://www.youtube.com/watch?v=t30quv7oZQs"),
    "r blaumaņa stāsti un noveles pirmā daļa": ("yt_11", "https://www.youtube.com/watch?v=cd6SPdnCJQ0"),
}

# Default place for weather queries – tuned for her home region
DEFAULT_WEATHER_PLACE = "Vecpiebalga, Latvia"
RIGA_WEATHER_PLACE = "Rīga, Latvia"

# Local MP3 library root (Windows desktop path by default; override with LOCAL_BOOKS_ROOT env)
LOCAL_BOOKS_ROOT = os.getenv("LOCAL_BOOKS_ROOT") or r"C:\\Users\\user\\Desktop\\Grāmatas"
# Where we persist playback progress (JSON with last played index per book_id)
LOCAL_BOOK_STATE_PATH = os.path.join(os.path.dirname(__file__), "local_book_state.json")
# Populated at startup by scanning LOCAL_BOOKS_ROOT
LOCAL_BOOKS: dict[str, dict] = {}
# Conversation log file
CONVERSATION_LOG = os.path.join(os.path.dirname(__file__), "logs", "conversation.log")
# Where we persist the selected nightly book and schedule
SCHEDULED_BOOK_CONFIG = os.path.join(os.path.dirname(__file__), "scheduled_book.json")
# Manual extra book paths (if present)
EXTRA_BOOKS = {
    "dzoja_adamsone_dzimusi_brivibai": {
        "display": "Džoja Ādamsone - Dzimusi Brīvībai",
        "path": os.path.join(LOCAL_BOOKS_ROOT, "Džoja Ādamsone - Dzimusi Brīvībai"),
    },
}
# Shared runtime flags/locks
recording_flag = threading.Event()
playback_lock = threading.Lock()
scheduler_stop_event = threading.Event()
scheduler_thread = None
exit_event = threading.Event()
# ==================


def _wait_for_enter() -> None:
    """Block until Enter/Return is pressed."""
    if sys.platform.startswith("win"):
        import keyboard
        keyboard.wait("enter")
        return
    try:
        import termios, tty  # type: ignore
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\n", "\r"):
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        input()


def _add_temp_enter_hotkey(callback):
    """Register a temporary Enter hotkey (Windows only); returns a remover."""
    if not sys.platform.startswith("win"):
        return lambda: None
    import keyboard
    hotkey_id = keyboard.add_hotkey("enter", callback)
    return lambda: keyboard.remove_hotkey(hotkey_id)


def _slugify_id(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9āčēģīķļņōŗšūž\s-]", "", text)
    text = text.replace(" ", "_").replace("-", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "book"


def refresh_local_books():
    """Scan LOCAL_BOOKS_ROOT for mp3 folders and populate LOCAL_BOOKS."""
    LOCAL_BOOKS.clear()
    root = LOCAL_BOOKS_ROOT
    if not root or not os.path.isdir(root):
        print(f"[*] LOCAL_BOOKS_ROOT nav atrasts: {root}")
        return

    for author in sorted(os.listdir(root)):
        author_path = os.path.join(root, author)
        if not os.path.isdir(author_path):
            continue

        subfolders = [f for f in sorted(os.listdir(author_path)) if os.path.isdir(os.path.join(author_path, f))]
        if subfolders:
            for book in subfolders:
                book_path = os.path.join(author_path, book)
                files = glob.glob(os.path.join(book_path, "*.mp3"))
                if not files:
                    continue
                display = f"{author} - {book}"
                book_id = _slugify_id(display)
                LOCAL_BOOKS[book_id] = {"display": display, "path": book_path}
        else:
            files = glob.glob(os.path.join(author_path, "*.mp3"))
            if files:
                display = author
                book_id = _slugify_id(display)
                LOCAL_BOOKS[book_id] = {"display": display, "path": author_path}

    # Extra manual entries (e.g., fragments in a dedicated folder)
    for book_id, info in EXTRA_BOOKS.items():
        path = info["path"]
        if os.path.isdir(path) or os.path.isfile(path):
            files = glob.glob(os.path.join(path, "*.mp3")) if os.path.isdir(path) else [path]
            if files:
                LOCAL_BOOKS[book_id] = {"display": info["display"], "path": path}


def list_local_books_for_prompt() -> str:
    """Return human-friendly list of local book folders for the system prompt."""
    if not LOCAL_BOOKS:
        return "(Pašlaik neatradu mp3 mapes vietā 'Grāmatas'.)"
    lines = []
    for book_id, info in LOCAL_BOOKS.items():
        lines.append(f"- {info['display']} (ID: {book_id})")
    return "\n".join(lines)


def _load_local_progress() -> dict:
    try:
        with open(LOCAL_BOOK_STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
            # migrate old int-only format -> dict
            for k, v in list(raw.items()):
                if isinstance(v, int):
                    raw[k] = {"file": v, "ms": 0}
            return raw
    except Exception:
        return {}


def _get_book_progress(book_id: str) -> dict:
    state = _load_local_progress()
    val = state.get(book_id, {"file": 0, "ms": 0})
    if isinstance(val, int):
        val = {"file": val, "ms": 0}
    return val


def _save_local_progress(state: dict):
    try:
        with open(LOCAL_BOOK_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[*] Neizdevās saglabāt progresu: {e}")


def _parse_time_string(value: str) -> Optional[str]:
    match = re.match(r"^(\d{1,2}):(\d{2})$", value.strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if 0 <= hour < 24 and 0 <= minute < 60:
        return f"{hour:02d}:{minute:02d}"
    return None


def _normalize_reading_times(raw) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    times: list[str] = []
    if isinstance(raw, list):
        for entry in raw:
            parsed = _parse_time_string(str(entry))
            if parsed and parsed not in times:
                times.append(parsed)
    return times


def _load_scheduled_config() -> dict:
    """Return persisted scheduler settings with defaults."""
    cfg = {
        "book_id": SCHEDULED_BOOK_ID,
        "reading_times": DEFAULT_READING_TIMES.copy(),
        "files_per_session": DEFAULT_FILES_PER_SESSION,
    }
    try:
        with open(SCHEDULED_BOOK_CONFIG, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                if data.get("book_id") is not None:
                    cfg["book_id"] = data.get("book_id")
                times = _normalize_reading_times(data.get("reading_times") or data.get("reading_time"))
                if times:
                    cfg["reading_times"] = times
                fps = data.get("files_per_session")
                if isinstance(fps, int) and fps >= 1:
                    cfg["files_per_session"] = fps
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[*] Neizdevās nolasīt scheduled_book.json: {e}")
    return cfg


def _persist_scheduled_config():
    try:
        payload = {
            "book_id": SCHEDULED_BOOK_ID,
            "reading_times": SCHEDULED_READING_TIMES or DEFAULT_READING_TIMES,
            "files_per_session": max(1, SCHEDULED_FILES_PER_SESSION),
        }
        with open(SCHEDULED_BOOK_CONFIG, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[*] Neizdevās saglabāt scheduled_book.json: {e}")


def _apply_scheduled_config(cfg: dict):
    """Load scheduler settings into globals without persisting."""
    global SCHEDULED_BOOK_ID, SCHEDULED_READING_TIMES, SCHEDULED_FILES_PER_SESSION
    SCHEDULED_BOOK_ID = cfg.get("book_id")
    times = _normalize_reading_times(cfg.get("reading_times"))
    SCHEDULED_READING_TIMES = times if times else DEFAULT_READING_TIMES.copy()
    fps = cfg.get("files_per_session", DEFAULT_FILES_PER_SESSION)
    if not isinstance(fps, int) or fps < 1:
        fps = DEFAULT_FILES_PER_SESSION
    SCHEDULED_FILES_PER_SESSION = fps


def set_scheduled_book_id(book_id: Optional[str]):
    """Update the global scheduled book ID and persist it for future runs."""
    global SCHEDULED_BOOK_ID
    SCHEDULED_BOOK_ID = book_id
    _persist_scheduled_config()


def set_scheduled_reading_times(times):
    """Update scheduled reading times (list of HH:MM strings)."""
    global SCHEDULED_READING_TIMES
    normalized = _normalize_reading_times(times)
    if not normalized:
        normalized = DEFAULT_READING_TIMES.copy()
    SCHEDULED_READING_TIMES = normalized
    _persist_scheduled_config()


def set_scheduled_files_per_session(count: int):
    """Update how many files are played per scheduled run."""
    global SCHEDULED_FILES_PER_SESSION
    try:
        num = int(count)
    except Exception:
        num = DEFAULT_FILES_PER_SESSION
    if num < 1:
        num = DEFAULT_FILES_PER_SESSION
    SCHEDULED_FILES_PER_SESSION = num
    _persist_scheduled_config()


def _get_effective_reading_times() -> list[str]:
    """Return current reading times with defaults applied."""
    times = _normalize_reading_times(SCHEDULED_READING_TIMES)
    return times if times else DEFAULT_READING_TIMES.copy()


def _reading_time_phrase() -> str:
    """Short human phrase describing the active schedule."""
    times = _get_effective_reading_times()
    if not times:
        return "norādītais lasījuma laiks"
    if len(times) == 1:
        return f"laiks {times[0]}"
    return f"viens no iestatītajiem laikiem ({', '.join(times)})"


def describe_scheduled_book() -> str:
    """Return human-readable status for the scheduled book."""
    refresh_local_books()
    if not SCHEDULED_BOOK_ID:
        return "Vakara lasījumam nav izvēlēta grāmata."

    item = LOCAL_BOOKS.get(SCHEDULED_BOOK_ID)
    if not item:
        return f"Vakara lasījumam saglabātais ID '{SCHEDULED_BOOK_ID}' vairs nav atrodams."

    files = sorted(glob.glob(os.path.join(item["path"], "*.mp3")), key=_natural_key)
    progress = _get_book_progress(SCHEDULED_BOOK_ID)
    idx = max(0, min(progress.get("file", 0), max(len(files) - 1, 0)))
    next_file = os.path.basename(files[idx]) if files else "nav mp3 failu"
    times_text = ", ".join(_get_effective_reading_times())

    return (
        f"Vakara lasījumam izvēlēts: {item['display']} (ID: {SCHEDULED_BOOK_ID}). "
        f"Nākamais fails: {next_file}. "
        f"Lasījuma laiki: {times_text or '23:10'}. "
        f"Faili vienā reizē: {SCHEDULED_FILES_PER_SESSION}."
    )


def _natural_key(path: str):
    """Sort helper: keeps 01, 02, 10 order."""
    name = os.path.basename(path)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", name)]


def play_local_book(book_id: str, resume: bool = False):
    """Sequentially play mp3 files in the chosen folder; remember progress between runs."""
    item = LOCAL_BOOKS.get(book_id)
    if not item:
        print(f"[*] Nav atrasts lokāls ieraksts ar ID: {book_id}")
        return

    files = sorted(glob.glob(os.path.join(item["path"], "*.mp3")), key=_natural_key)
    if not files:
        print(f"[*] Mapē nav mp3 failu: {item['path']}")
        return

    state = _load_local_progress()
    book_state = _get_book_progress(book_id)

    start_index = book_state.get("file", 0) if resume else 0
    start_index = max(0, min(start_index, len(files) - 1))
    start_ms = book_state.get("ms", 0) if resume else 0

    print(f"-> Atskaņoju lokālo grāmatu: {item['display']}")
    if resume and start_index:
        print(f"   Turpinu no faila #{start_index + 1}.")
        if start_ms:
            print(f"   Turpinu no aptuveni {start_ms/1000:.1f} sekundes.")
    print("   (Ctrl+C, lai apturētu.)")

    try:
        with playback_lock:
            for idx in range(start_index, len(files)):
                fpath = files[idx]
                print(f"   Atskaņoju: {os.path.basename(fpath)}")
                offset = start_ms if idx == start_index else 0
                stopped, pos_ms = play_mp3_file(fpath, start_ms=offset)
                if stopped:
                    state[book_id] = {"file": idx, "ms": pos_ms}
                    _save_local_progress(state)
                    print("   Atskaņošana apturēta (Enter).")
                    return
                state[book_id] = {"file": idx + 1, "ms": 0}  # nākamais fails
                _save_local_progress(state)
    except KeyboardInterrupt:
        print("\n[*] Atskaņošana apturēta.")
        _save_local_progress(state)
    else:
        print("-> Grāmata pabeigta.")
        state[book_id] = {"file": len(files) - 1, "ms": 0}
        _save_local_progress(state)


# Ielādējam kataloga informāciju startā
refresh_local_books()
_apply_scheduled_config(_load_scheduled_config())

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file on this machine.")

client = OpenAI(api_key=api_key)


def record_audio_to_wav(path: str):
    """
    Press Enter to start recording, speak, press Enter again to stop.
    Audio is saved as mono 16kHz WAV.
    """
    print("\nNospied Enter, tad runā. Kad pabeidz, nospied Enter vēlreiz.")
    # Logitech presenter button mappings only on Windows
    if sys.platform.startswith("win"):
        import keyboard
        keyboard.add_hotkey('pagedown', lambda: keyboard.press_and_release('enter'))
        keyboard.add_hotkey('pageup', lambda: keyboard.press_and_release('enter'))
        keyboard.add_hotkey('.', lambda: keyboard.press_and_release('enter'))
        keyboard.add_hotkey('space', lambda: keyboard.press_and_release('enter'))

    def _wait_enter_or_space():
        if sys.platform.startswith("win"):
            keyboard.wait('enter')
            return
        try:
            import termios, tty  # type: ignore
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ("\n", "\r", " "):
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            input()

    _wait_enter_or_space()
    print("Ieraksts sākts – runā mikrofona virzienā... (Enter vai Space = stop)")

    frames = []
    recording_flag.set()

    def callback(indata, frames_count, time_info, status):
        if status:
            print(f"[sounddevice status] {status}", file=sys.stderr)
        frames.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            # This keyboard.wait() blocks while callback keeps filling frames
            _wait_enter_or_space()
            print("Ieraksts apturēts, apstrādāju...")
    finally:
        recording_flag.clear()

    if not frames:
        print("Netika ierakstīts neviens kadrs.")
        return False

    audio = np.concatenate(frames, axis=0)
    sf.write(path, audio, SAMPLE_RATE)
    return True


def transcribe_latvian(path: str) -> str:
    """
    Send WAV file to OpenAI speech-to-text and return Latvian text.
    """
    print("-> Sūtu audio uz OpenAI transkripcijai (ASR)...")
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=f,
            response_format="text",  # get plain text back  [oai_citation:5‡OpenAI Platform](https://platform.openai.com/docs/guides/speech-to-text?utm_source=chatgpt.com)
        )
    text = result.strip()
    print(f"[Transkripcija LV]: {text}")
    return text


def list_youtube_items_for_prompt() -> str:
    """Return a bullet list of YouTube items for inclusion in the system prompt.

    Mēģinām sadalīt nosaukumu autorā un darbā, lai modelis varētu meklēt
    gan pēc autora, gan pēc darba nosaukuma.
    """
    if not YOUTUBE_ITEMS:
        return "(Pašlaik nav definētu YouTube ierakstu.)"

    lines = []
    for title, (item_id, url) in YOUTUBE_ITEMS.items():  # noqa: F841 - url unused in prompt
        t = title.lower().strip()
        # noņemam "pirmā daļa" asti, ja tāda ir
        if t.endswith("pirmā daļa"):
            t = t[: -len("pirmā daļa")].strip()
        words = t.split()
        if len(words) >= 3:
            # pieņemam, ka autors ir pirmie divi vārdi, pārējais – darbs
            author = " ".join(words[:2])
            work = " ".join(words[2:])
        else:
            author = t
            work = ""
        if work:
            lines.append(f"- Autors: {author}; Darbs: {work}; ID: {item_id}")
        else:
            lines.append(f"- Autors/darbs: {author}; ID: {item_id}")
    return "\n".join(lines)


def get_weather_text(place: str = DEFAULT_WEATHER_PLACE) -> str:
    """Fetch a short weather summary.

    - For Rīga we use LVĢMC data (punkt: P269) from videscentrs.lvgmc.lv.
    - Otherwise we fall back to wttr.in one-line summary.
    """
    print("-> Vaicos par laikapstākļiem internetā...")

    def _wind_dir(deg: float) -> str:
        dirs = ["ziemeļiem", "ziemeļaustrumiem", "austrumiem", "dienvidaustrumiem", "dienvidiem", "dienvidrietumiem", "rietumiem", "ziemeļrietumiem"]
        idx = int((deg + 22.5) // 45) % 8
        return dirs[idx]

    if place.lower().startswith("rīga"):
        try:
            url = "https://videscentrs.lvgmc.lv/data/weather_forecast_for_location_hourly"
            resp = requests.get(url, params={"punkts": "P269"}, timeout=6)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                raise ValueError("empty forecast")

            now = time.strftime("%Y%m%d%H%M", time.localtime())
            # pick the first forecast at or after current time
            best = min(data, key=lambda e: (e.get("laiks") < now, abs(int(e.get("laiks", "0")) - int(now))))

            temp = float(best.get("temperatura", 0))
            wind = float(best.get("veja_atrums", 0))
            wind_dir = float(best.get("veja_virziens", 0))
            precip = best.get("nokrisni_1h") or best.get("nokrisni_12h") or "0"
            pressure = best.get("spiediens")
            time_str = best.get("laiks", "")[-4:]
            hhmm = f"{time_str[:2]}:{time_str[2:]}" if len(time_str) == 4 else "drīzumā"

            parts = [
                f"Laikapstākļi Rīgā ({hhmm} prognoze):",
                f"temperatūra {temp:+.0f}°C",
                f"vējš {wind:.1f} m/s no { _wind_dir(wind_dir) }",
            ]
            try:
                precip_val = float(precip)
                parts.append(f"nokrišņi {precip_val:.1f} mm")
            except Exception:
                pass
            if pressure:
                try:
                    parts.append(f"spiediens {float(pressure):.0f} hPa")
                except Exception:
                    pass
            return ", ".join(parts) + "."
        except Exception as e:
            print(f"[*] LVĢMC laikapstākļu kļūda: {e}")

    try:
        location_string = place.replace(" ", "+")
        url = f"https://wttr.in/{location_string}"
        resp = requests.get(url, params={"format": "3", "lang": "lv"}, timeout=5)
        if resp.status_code != 200:
            return "Neizdevās iegūt laikapstākļu informāciju."
        line = resp.text.strip()
        return f"Laikapstākļi vietā {place}: {line}"
    except Exception:
        return "Neizdevās pieslēgties laikapstākļu servisam."


def get_text_forecast_latvia() -> str:
    """Fetch 'Teksta prognoze' from LVĢMC."""
    try:
        url = "https://videscentrs.lvgmc.lv/data/sinopt_prognozes"
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return "Teksta prognoze pašlaik nav pieejama."
        entry = data[0]  # API returns latest first
        text = entry.get("teksti", {}).get("teksts") or entry.get("teksts")
        date = entry.get("datums")
        from_time = entry.get("laiks_no")
        to_time = entry.get("laiks_lidz")
        if text:
            prefix = "Teksta prognoze"
            if date:
                prefix += f" ({date}"
                if from_time and to_time:
                    prefix += f" {from_time}-{to_time}"
                prefix += ")"
            return f"{prefix}: {text}"
        return "Teksta prognoze nav atrasta."
    except Exception as e:
        print(f"[*] Teksta prognozes kļūda: {e}")
        return "Neizdevās iegūt teksta prognozi."


def get_time_text() -> str:
    """Return a short Latvian string with the current local time."""
    now = time.localtime()
    return time.strftime("Šobrīd ir %H:%M.", now)


def chat_latvian_elderly(user_text: str) -> str:
    """Send Latvian text to GPT and get a Latvian reply,
    tuned for an elderly, blind user.
    """
    refresh_local_books()  # aktualizējam lokālo sarakstu katram piegājienam
    print("-> Sūtu tekstu GPT modelim...")
    system_prompt = (
        "Tu esi balss asistents, kas runā latviešu valodā ar 88 gadus vecu, "
        "neredzīgu kundzi. Viņa nesen zaudēja redzi un bija dzīvojusi vairāk "
        "nekā 30 gadus Druviņās, Kaives pagastā, Cēsu novadā, lielāko daļu "
        "laika viena pati savā mājā, pēdējos gados kopā ar savu mazo suni Elsu. "
        "Esi ļoti pacietīgs, empātisks un mierīgs. Runā lēni, ar īsiem un "
        "vienkāršiem teikumiem. Nelieto sarežģītus svešvārdus. Nepiedāvā garus "
        "sarakstus. Ja viņa ko nesaprot, piedāvā paskaidrot to vēlreiz citādi.\n\n"
        "Tev ir pieejamas šādas audiogrāmatas (ja kāda ir definēta):\n"
        + "\n".join(f"- {name} (ID: {bid})" for name, bid in AUDIOBOOKS.items())
        + "\n\n"
        "Tev ir pieejami arī šādi YouTube ieraksti (atvērsi tos pārlūkā):\n"
        + list_youtube_items_for_prompt()
        + "\n\n"
        "Tev ir pieejamas arī lokālās mp3 grāmatas mapē 'Grāmatas' uz Windows darbvirsmas:\n"
        + list_local_books_for_prompt()
        + "\n\n"
        "Ja lietotāja lūdz pastāstīt, KO tu vari atskaņot no YouTube, "
        "atbildi parastā tekstā, īsi uzskaitot autorus un darbus, "
        "piemēram: 'No Jāņa Poruka varu atskaņot Pērļu zvejnieks; no Jēkaba "
        "Janševska – Līgava, Bandavā, Mežvidus ļaudis' utt.\n"
        "Ja lietotāja pēc tam skaidri izvēlas konkrētu autoru un darbu, "
        "piemēram: 'Es gribu klausīties Pērļu zvejnieku no Jāņa Poruka', "
        "TAD NEATBILDI parastā tekstā. Atbildi tikai ar komandu:\n"
        "  CMD:PLAY_YT:<ID>\n"
        "kur <ID> ir attiecīgā YouTube ieraksta ID. Tu drīksti izvēlēties "
        "atbilstošu ID arī tad, ja lietotāja frāze ir nedaudz citādāka, bet "
        "skaidri attiecas uz kādu no sarakstā esošajiem darbiem.\n"
        "Ja no konkrēta autora tev ir pieejams TIKAI viens darbs un lietotāja "
        "pasaka tikai autoru (piemēram: 'Gribu klausīties Rūdolfu Blaumani'), "
        "uzskati, ka tas ir skaidrs lūgums atskaņot šo vienīgo darbu un uzreiz "
        "atbildi ar atbilstošu CMD:PLAY_YT:<ID> komandu, bez papildu jautājumiem.\n"
        "Ja tu pats tikko esi piedāvājis konkrētu autoru un darbu (piemēram: "
        "'Varu atskaņot Blaumaņa \"Stāsti un noveles\"'), un nākamā lietotājas "
        "atbilde ir apstiprinoša ('jā', 'jā, vēlos', 'labi, sāc' u.tml.), tad "
        "TĀ IR SKAIDRA IZVĒLE. Šādā gadījumā atbildi tikai ar CMD:PLAY_YT:<ID> "
        "komandu šim darbam, nevis parastu tekstu.\n"
        "Ja lietotāja vēlas apturēt YouTube atskaņošanu, atbildi tikai ar:\n"
        "  CMD:STOP_YT\n"
        "Ja lietotāja vēlas klausīties lokālu mp3 grāmatu no mapes 'Grāmatas', "
        "uzreiz atbildi ar komandu:\n"
        "  CMD:PLAY_LOCAL:<ID>\n"
        "kur <ID> atbilst pieejamajam sarakstam augstāk. Ja lietotāja saka "
        "turpināt iepriekšējo grāmatu, atbildi ar:\n"
        "  CMD:RESUME_LOCAL:<ID>\n"
        "kur <ID> ir tas pats lokālais ID, kuru viņa klausījās.\n"
        "Ja lietotāja vēlme nav klausīties audiogrāmatu vai YouTube, atbildi "
        "parastā veidā sirsnīgā, mierīgā tonī.\n"
        "Tev ir pieejamas šādas balsis: 'vīrieša balss', 'sieviešu balss', "
        "'neitrāla balss'. Ja lietotāja skaidri lūdz nomainīt balsi uz kādu "
        "no šīm, TAD NEATBILDI parastā tekstā, bet atbildi tikai ar komandu:\n"
        "  CMD:SET_VOICE:<balses_nosaukums>\n"
        "piemēram: CMD:SET_VOICE:vīrieša balss.\n"
        "Tu vari arī atbildēt ar pašreizējo laiku (pulkstenis) un laikapstākļiem "
        "Rīgā, ja lietotāja to vaicā.\n"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    reply = resp.choices[0].message.content.strip()
    print(f"[GPT atbilde LV]: {reply}")
    return reply


def tts_latvian_to_file(text: str, out_path: str):
    """Use OpenAI TTS model to synthesize Latvian speech.

    Tuned to be lēnāka un dabiskāka balss.
    """
    print("-> Ģenerēju balsi (TTS)...")

    instructions = (
        "Runā skaidrā, dabiskā latviešu valodā, lēnā tempā, "
        "ar maigu vīrieša balsi, kā mierīgs radio diktors, "
        "bez pārliekas robotiskas ritma vienmērības."
    )

    # Use streaming TTS so we can write straight to the file
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=CURRENT_VOICE,
        input=text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(out_path)

    print(f"[TTS audio saglabāts]: {out_path}")


def set_tts_voice_by_label(label: str) -> str:
    """Set CURRENT_VOICE from a spoken Latvian label, return a short LV status message."""
    global CURRENT_VOICE
    normalized = label.strip().lower()
    # try exact match first
    for human_name, voice_name in AVAILABLE_VOICES.items():
        if human_name.lower() == normalized:
            CURRENT_VOICE = voice_name
            return f"Balsi nomainīju uz: {human_name}."
    # fuzzy contains match as a fallback
    for human_name, voice_name in AVAILABLE_VOICES.items():
        if normalized in human_name.lower():
            CURRENT_VOICE = voice_name
            return f"Balsi nomainīju uz: {human_name}."
    return "Neizdevās saprast, uz kādu balsi nomainīt. Tev ir pieejamas: vīrieša balss, sieviešu balss un neitrāla balss."


def play_audiobook(book_id: str):
    """Very simple audiobook player.

    - Looks for audiobooks/<book_id>/*.mp3
    - Plays files in sorted order using macOS 'afplay'.
    - Blocks until finished; Ctrl+C to stop playback.
    """
    folder = os.path.join(AUDIOBOOKS_ROOT, book_id)
    if not os.path.isdir(folder):
        print(f"[*] Audiogrāmatas mape nav atrasta: {folder}")
        return

    files = sorted(glob.glob(os.path.join(folder, "*.mp3")))
    if not files:
        print(f"[*] Mapē {folder} nav atrastu .mp3 failu.")
        return

    print(f"-> Sāku atskaņot audiogrāmatu: {book_id}")
    print("   (Ctrl+C, lai apturētu atskaņošanu un atgrieztos pie asistenta.)")

    try:
        with playback_lock:
            for fpath in files:
                print(f"   Atskaņoju: {os.path.basename(fpath)}")
                subprocess.run(["afplay", fpath])
    except KeyboardInterrupt:
        print("\n[*] Audiogrāmata apturēta ar Ctrl+C.")


def get_youtube_url_by_id(item_id: str) -> str | None:
    """Find the YouTube URL corresponding to a configured item_id."""
    for title, (yt_id, url) in YOUTUBE_ITEMS.items():
        if yt_id == item_id:
            return url
    return None


def play_youtube_item(item_id: str):
    """Open the YouTube item using Chrome with forced autoplay on Windows."""
    url = get_youtube_url_by_id(item_id)
    if not url:
        print(f"[*] Neatradu YouTube ierakstu ar ID: {item_id}")
        return

    # Add autoplay=1 parameter
    if "?" in url:
        url = url + "&autoplay=1"
    else:
        url = url + "?autoplay=1"

    print(f"-> Atveru YouTube ar autoplay: {url}")

    # ----- WINDOWS SPECIAL HANDLING -----
    if sys.platform.startswith("win"):
        chrome_paths = [
            r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
            r"%ProgramFiles%\\Google\\Chrome\\Application\\chrome.exe",
            r"%ProgramFiles(x86)%\\Google\\Chrome\\Application\\chrome.exe",
        ]

        chrome_path = None
        for p in chrome_paths:
            expanded = os.path.expandvars(p)
            if os.path.isfile(expanded):
                chrome_path = expanded
                break

        if chrome_path:
            try:
                subprocess.Popen([
                    chrome_path,
                    "--autoplay-policy=no-user-gesture-required",
                    "--new-window",
                    "--start-maximized",
                    url,
                ])
                print("-> Chrome palaists ar autoplay atļauju.")
            except Exception as e:
                print(f"[*] Chrome startēšana neizdevās: {e}")
        else:
            print("[*] Chrome nav atrasts — izmantoju webbrowser fallback.")
            webbrowser.open(url, new=1)

        # Return focus to PowerShell
        try:
            subprocess.run([
                "powershell",
                "-command",
                "(New-Object -ComObject WScript.Shell).AppActivate((Get-Process -Id $PID).MainWindowTitle)"
            ], check=False)
        except:
            pass

        return

    # ----- MACOS / LINUX fallback -----
    try:
        webbrowser.open(url, new=1)
    except:
        print("[*] Neizdevās atvērt pārlūku.")


def stop_youtube_playback():
    """Try to stop YouTube playback by closing browser windows.

    macOS: izmanto AppleScript, lai aizvērtu Google Chrome logus.
    Windows: mēģina aizvērt Chrome un Edge procesus ar taskkill.
    Šī ir radikāla pieeja, paredzēta speciālai ierīcei, kur pārlūks
    netiek izmantots citām lietām vienlaikus.
    """
    try:
        if sys.platform == "darwin":
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "Google Chrome" to close windows',
                ],
                check=False,
            )
            print("[*] Mēģinu aizvērt Google Chrome logus (macOS).")
        elif sys.platform.startswith("win"):
            # Mēģinām aizvērt Chrome un Edge, ja tie atskaņo YouTube
            subprocess.run(["taskkill", "/IM", "chrome.exe", "/F"], check=False)
            subprocess.run(["taskkill", "/IM", "msedge.exe", "/F"], check=False)
            print("[*] Mēģinu aizvērt Chrome/Edge logus (Windows).")
        else:
            print("[*] Pārlūka aizvēršana šajā platformā nav implementēta.")
    except Exception as e:
        print(f"[*] Neizdevās aizvērt pārlūku: {e}")


def play_audio_file(path: str):
    """
    Play an audio file (mp3/wav) through default speakers.
    """
    print("-> Atskaņoju atbildi...")
    stop_flag = False

    def on_enter():
        nonlocal stop_flag
        stop_flag = True
        try:
            sd.stop()
        except Exception:
            pass

    remover = _add_temp_enter_hotkey(on_enter)
    try:
        with playback_lock:
            data, samplerate = sf.read(path, dtype="float32")
            sd.play(data, samplerate)
            sd.wait()
    finally:
        try:
            remover()
        except Exception:
            pass
    return stop_flag


def _play_mp3_windows(path: str, start_ms: int = 0):
    """Play mp3 on Windows using winmm (no external deps). Returns (stopped, last_pos_ms)."""
    import ctypes

    alias = f"mp3_{int(time.time() * 1000)}"
    mci = ctypes.windll.winmm.mciSendStringW

    def mci_cmd(cmd: str) -> str:
        buf = ctypes.create_unicode_buffer(255)
        mci(cmd, buf, 254, None)
        return buf.value

    stopped = False
    mci_cmd(f'open "{path}" type mpegvideo alias {alias}')
    if start_ms > 0:
        mci_cmd(f"seek {alias} to {int(start_ms)}")

    def on_enter():
        nonlocal stopped
        stopped = True
        mci_cmd(f"stop {alias}")

    remover = _add_temp_enter_hotkey(on_enter)
    try:
        mci_cmd(f"play {alias}")
        while True:
            mode = mci_cmd(f"status {alias} mode")
            if mode != "playing":
                break
            if stopped:
                break
            time.sleep(0.1)
        pos = mci_cmd(f"status {alias} position")
    finally:
        try:
            remover()
        except Exception:
            pass
        mci_cmd(f"close {alias}")

    try:
        pos_ms = int(pos)
    except Exception:
        pos_ms = 0
    return stopped, pos_ms


def play_mp3_file(path: str, start_ms: int = 0):
    """Cross-platform mp3 playback without playsound dependency. Returns (stopped, last_pos_ms)."""
    if sys.platform.startswith("win"):
        return _play_mp3_windows(path, start_ms=start_ms)
    elif sys.platform == "darwin":
        proc = subprocess.Popen(["afplay", path])
        proc.wait()
        return False, 0
    else:
        # Linux fallback: try ffplay if present
        try:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", path], check=False)
            return False, 0
        except FileNotFoundError:
            print("[*] mp3 atskaņošana nav atbalstīta šajā platformā (nav ffplay).")
            return False, 0


# ===== SCHEDULED UI HELPERS =====
def _prompt_book_selection(max_items: int = 100) -> Optional[str]:
    """Interactive prompt to choose a book ID from LOCAL_BOOKS."""
    refresh_local_books()
    if not LOCAL_BOOKS:
        print("Nav atrastas lokālās grāmatas mapē 'Grāmatas'.")
        return None

    ordered = sorted(LOCAL_BOOKS.items(), key=lambda pair: pair[1]["display"].lower())
    ordered = ordered[:max_items]
    print("\nPieejamās grāmatas (1–100):")
    for idx, (book_id, info) in enumerate(ordered, start=1):
        marker = "  <-- vakara grāmata" if book_id == SCHEDULED_BOOK_ID else ""
        print(f"{idx:3d}. {info['display']} (ID: {book_id}){marker}")

    choice = input("Ievadi grāmatas numuru (Enter, lai atceltu): ").strip()
    if not choice:
        return None
    try:
        num = int(choice)
    except ValueError:
        print("Lūdzu ievadi skaitli.")
        return None
    if not (1 <= num <= len(ordered)):
        print("Numurs ārpus saraksta.")
        return None
    return ordered[num - 1][0]


def _prompt_file_selection(book_id: str, max_items: int = 100):
    """Let user pick the next file/chapter to use for scheduled reading."""
    item = LOCAL_BOOKS.get(book_id)
    if not item:
        print("Neizdevās atrast izvēlēto grāmatu.")
        return

    files = sorted(glob.glob(os.path.join(item["path"], "*.mp3")), key=_natural_key)
    if not files:
        print("Mapē nav mp3 failu.")
        return

    max_show = min(len(files), max_items)
    progress = _get_book_progress(book_id)
    current_idx = max(0, min(progress.get("file", 0), len(files) - 1))

    print(f"\nFaili grāmatai {item['display']}:")
    for idx, fpath in enumerate(files[:max_show], start=1):
        marker = "  <-- nākamais" if idx - 1 == current_idx else ""
        print(f"{idx:3d}. {os.path.basename(fpath)}{marker}")
    if len(files) > max_show:
        print(f"... (parādīti pirmie {max_show} faili no {len(files)})")

    prompt = input(f"Ievadi faila numuru (Enter, lai atstātu #{current_idx + 1}): ").strip()
    if not prompt:
        return
    try:
        choice = int(prompt)
    except ValueError:
        print("Lūdzu ievadi skaitli.")
        return
    if not (1 <= choice <= max_show):
        print("Numurs ārpus parādītā saraksta.")
        return

    target_idx = choice - 1
    state = _load_local_progress()
    state[book_id] = {"file": target_idx, "ms": 0}
    _save_local_progress(state)
    print(f"-> Vakara lasījums sāksies no faila #{choice}: {os.path.basename(files[target_idx])}")


def _prompt_reading_times():
    """Let user set one or multiple scheduled times (HH:MM)."""
    current = ", ".join(SCHEDULED_READING_TIMES or DEFAULT_READING_TIMES)
    prompt = input(f"Ievadi lasījuma laikus HH:MM (komatiem atdalīti) [esošie: {current}]: ").strip()
    if not prompt:
        return

    parts = [p.strip() for p in re.split(r"[;,]", prompt) if p.strip()]
    parsed = []
    for part in parts:
        normalized = _parse_time_string(part)
        if not normalized:
            print(f"Neizdevās saprast laiku: {part}")
            continue
        if normalized not in parsed:
            parsed.append(normalized)

    if not parsed:
        print("Neizdevās iestatīt nevienu laiku (formāts: HH:MM, piem., 23:10).")
        return

    set_scheduled_reading_times(parsed)
    print(f"-> Lasījuma laiki iestatīti: {', '.join(parsed)}")


def _prompt_files_per_session():
    """Let user set how many files get played in each scheduled run."""
    prompt = input(f"Cik failus atskaņot vienā lasījumā? [esošais: {SCHEDULED_FILES_PER_SESSION}]: ").strip()
    if not prompt:
        return
    try:
        count = int(prompt)
    except ValueError:
        print("Lūdzu ievadi skaitli.")
        return
    if count < 1:
        print("Skaitlim jābūt vismaz 1.")
        return
    set_scheduled_files_per_session(count)
    print(f"-> Katrā lasījumā atskaņošu {count} failu(s).")


def run_scheduler_ui():
    """Simple console UI to view and change the nightly audiobook + file."""
    print("\n=== Vakara lasījuma iestatījumi ===")
    while True:
        print(describe_scheduled_book())
        print("Komandas: [L] izvēlēties grāmatu  [F] izvēlēties failu  [R] iestatīt laikus  [C] failu skaits  [H] pārbaudīt laika paziņojumu  [W] pārbaudīt laikapstākļus Rīgā  [T] teksta prognoze  [S] sākt vakara lasījumu tagad  [Q] turpināt")
        choice = input(">> ").strip().lower()
        if choice in {"q", "quit", ""}:
            break
        if choice in {"l", "book", "g"}:
            book_id = _prompt_book_selection()
            if book_id:
                set_scheduled_book_id(book_id)
                refresh_local_books()
                book_name = LOCAL_BOOKS.get(book_id, {}).get("display", book_id)
                print(f"-> Vakara grāmata iestatīta uz: {book_name}")
                _prompt_file_selection(book_id)
            continue
        if choice in {"f", "file"}:
            if not SCHEDULED_BOOK_ID:
                print("Vispirms izvēlies vakara grāmatu.")
                continue
            _prompt_file_selection(SCHEDULED_BOOK_ID)
            continue
        if choice in {"r", "times", "laiks"}:
            _prompt_reading_times()
            continue
        if choice in {"c", "count", "skaits"}:
            _prompt_files_per_session()
            continue
        if choice in {"h", "hour", "time"}:
            if not _speak_scheduled_message(get_time_text(), "sched_time_manual"):
                print("Paziņojums izlaists — sistēma pašlaik atskaņo/ieraksta.")
            continue
        if choice in {"w", "weather"}:
            if not _speak_scheduled_message(get_weather_text(RIGA_WEATHER_PLACE), "sched_weather_manual"):
                print("Paziņojums izlaists — sistēma pašlaik atskaņo/ieraksta.")
            continue
        if choice in {"t", "text"}:
            if not _speak_scheduled_message(get_text_forecast_latvia(), "sched_text_forecast_manual"):
                print("Paziņojums izlaists — sistēma pašlaik atskaņo/ieraksta.")
            continue
        if choice in {"s", "start", "play"}:
            if not _play_scheduled_chapter():
                print("Neizdevās sākt lasījumu (iespējams, atskaņošana vai ieraksts jau notiek).")
            continue
        print("Neatpazīta izvēle.")


def log_conversation(user_text: str, reply_text: str, note: Optional[str] = None):
    """Append interaction to a log file for later review."""
    try:
        os.makedirs(os.path.dirname(CONVERSATION_LOG), exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts}\n")
            f.write(f"USER: {user_text}\n")
            if note:
                f.write(f"ASSISTANT: {reply_text} [{note}]\n")
            else:
                f.write(f"ASSISTANT: {reply_text}\n")
            f.write("---\n")
    except Exception as e:
        print(f"[*] Neizdevās ierakstīt žurnālā: {e}")


# ===== SCHEDULED ANNOUNCEMENTS =====
def _speak_scheduled_message(text: str, note: str) -> bool:
    """TTS helper for scheduler that skips if recording/playback is busy."""
    if recording_flag.is_set() or playback_lock.locked():
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "scheduled.mp3")
        tts_latvian_to_file(text, out_path)
        play_audio_file(out_path)

    log_conversation("<<scheduler>>", text, note=note)
    return True


def _play_scheduled_chapter() -> bool:
    """
    Play the next chapter(s) from SCHEDULED_BOOK_ID at the configured times.
    Returns True if we handled the trigger (even with an error message), False if deferred.
    """
    if recording_flag.is_set() or playback_lock.locked():
        return False

    if not SCHEDULED_BOOK_ID:
        _speak_scheduled_message(
            f"Ir {_reading_time_phrase()}, bet vakara lasījumam nav izvēlēta grāmata.",
            "sched_evening_missing_book",
        )
        return True

    refresh_local_books()
    item = LOCAL_BOOKS.get(SCHEDULED_BOOK_ID)
    if not item:
        _speak_scheduled_message("Vakara lasījumam norādītā grāmata nav atrodama. Lūdzu pārbaudi konfigurāciju.", "sched_evening_missing_book")
        return True

    files = sorted(glob.glob(os.path.join(item["path"], "*.mp3")), key=_natural_key)
    if not files:
        _speak_scheduled_message(f"Mapē {item['display']} nav atrasti mp3 faili.", "sched_evening_no_files")
        return True

    progress = _get_book_progress(SCHEDULED_BOOK_ID)
    idx = progress.get("file", 0)
    if idx >= len(files):
        _speak_scheduled_message(f"Grāmata {item['display']} jau ir pabeigta.", "sched_evening_complete")
        return True
    idx = max(0, idx)
    files_to_play = min(max(1, SCHEDULED_FILES_PER_SESSION), len(files) - idx)
    chapter_paths = files[idx : idx + files_to_play]

    def _play():
        with playback_lock:
            current_idx = idx
            stopped = False
            pos_ms = 0
            for chapter_path in chapter_paths:
                stopped, pos_ms = play_mp3_file(chapter_path)
                if stopped:
                    break
                current_idx += 1
        state = _load_local_progress()
        if stopped:
            state[SCHEDULED_BOOK_ID] = {"file": current_idx, "ms": pos_ms}
        else:
            state[SCHEDULED_BOOK_ID] = {"file": min(current_idx, len(files)), "ms": 0}
        _save_local_progress(state)

    threading.Thread(target=_play, daemon=True).start()
    played_count = len(chapter_paths)
    if played_count == 1:
        start_msg = f"Tagad lasu vakara nodaļu no {item['display']}."
    else:
        start_msg = f"Tagad lasu {played_count} vakara nodaļas no {item['display']}."
    _speak_scheduled_message(start_msg, "sched_evening_start")
    return True


def _scheduler_loop():
    last_hourly = None  # (yearday, hour)
    last_weather_day = None
    last_evening_by_time = {}
    last_text_morning_day = None
    last_text_evening_day = None

    while not scheduler_stop_event.is_set():
        now = time.localtime()

        # Hourly time announcement between 08:00 and 00:00
        if ((8 <= now.tm_hour <= 23) or now.tm_hour == 0) and now.tm_min == 0:
            key = (now.tm_yday, now.tm_hour)
            if key != last_hourly:
                if _speak_scheduled_message(get_time_text(), "sched_time"):
                    last_hourly = key

        # Morning weather around 09:00
        if now.tm_hour == 9 and now.tm_min == 0 and now.tm_yday != last_weather_day:
            if _speak_scheduled_message(get_weather_text(RIGA_WEATHER_PLACE), "sched_weather"):
                last_weather_day = now.tm_yday

        # Morning text forecast after the 09:00 announcement
        if now.tm_hour == 9 and now.tm_min == 1 and now.tm_yday != last_text_morning_day:
            if _speak_scheduled_message(get_text_forecast_latvia(), "sched_text_forecast_morning"):
                last_text_morning_day = now.tm_yday

        # Evening text forecast after the 16:00 announcement
        if now.tm_hour == 16 and now.tm_min == 1 and now.tm_yday != last_text_evening_day:
            if _speak_scheduled_message(get_text_forecast_latvia(), "sched_text_forecast_evening"):
                last_text_evening_day = now.tm_yday

        # Scheduled chapters (tolerate first 5 minutes of each configured time)
        for time_str in _get_effective_reading_times():
            parsed = _parse_time_string(time_str)
            if not parsed:
                continue
            hour, minute = map(int, parsed.split(":"))
            if now.tm_hour == hour and 0 <= now.tm_min - minute < 5:
                last_run = last_evening_by_time.get(parsed)
                if last_run != now.tm_yday:
                    if _play_scheduled_chapter():
                        last_evening_by_time[parsed] = now.tm_yday

        scheduler_stop_event.wait(20)


def start_scheduler_thread():
    global scheduler_thread
    if scheduler_thread and scheduler_thread.is_alive():
        return
    scheduler_stop_event.clear()
    scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    scheduler_thread.start()


def stop_scheduler_thread():
    scheduler_stop_event.set()


# ===== THINKING SOUND HELPERS =====
THINKING_SOUND_FREQ = 1200
THINKING_SOUND_DURATION = 0.03
THINKING_SOUND_INTERVAL = 0.15
thinking_stop_event = threading.Event()
thinking_thread = None

def _thinking_sound_loop():
    tone_sr = SAMPLE_RATE
    while not thinking_stop_event.is_set():
        # Generate a very short noise burst with fast decay, like a soft tick
        t = np.linspace(0, THINKING_SOUND_DURATION, int(tone_sr * THINKING_SOUND_DURATION), False)
        noise = np.random.uniform(-1.0, 1.0, size=t.shape)
        envelope = np.exp(-t * 60.0)  # very fast decay for clicky feel
        tone = (0.12 * noise * envelope).astype(np.float32)

        sd.play(tone, tone_sr)
        sd.wait()

        slept = 0.0
        chunk = 0.02
        while slept < THINKING_SOUND_INTERVAL:
            if thinking_stop_event.is_set():
                break
            time.sleep(min(chunk, THINKING_SOUND_INTERVAL - slept))
            slept += chunk

def start_thinking_sound():
    global thinking_thread
    if thinking_thread is not None and thinking_thread.is_alive():
        return
    thinking_stop_event.clear()
    thinking_thread = threading.Thread(target=_thinking_sound_loop, daemon=True)
    thinking_thread.start()

def stop_thinking_sound():
    thinking_stop_event.set()


def maybe_show_scheduler_menu() -> bool:
    """
    Offer a simple menu to manage the nightly audiobook before starting voice mode.
    Returns False if the program should exit after the menu.
    """
    if "--schedule-ui-only" in sys.argv:
        run_scheduler_ui()
        return False

    if "--schedule-ui" in sys.argv or "--ui" in sys.argv:
        run_scheduler_ui()
        return True

    if sys.stdin.isatty():
        print(describe_scheduled_book())
        ans = input("Enter - sākt balsi; ieraksti 'menu', lai pārvaldītu vakara lasījumu: ").strip().lower()
        if ans in {"menu", "m", "1"}:
            run_scheduler_ui()
    return True


def main_loop():
    print("=== Latviešu balss asistents (prototips) ===")
    print("Ctrl+C, lai izietu.\n")

    esc_remover = None
    if sys.platform.startswith("win"):
        import keyboard
        esc_remover = keyboard.add_hotkey("esc", lambda: exit_event.set())

    while True:
        if exit_event.is_set():
            print("Saņemta ESC komanda – iziešu.")
            break
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "input.wav")
            out_path = os.path.join(tmpdir, "reply.mp3")

            ok = record_audio_to_wav(wav_path)
            if not ok:
                continue

            try:
                text = transcribe_latvian(wav_path)
                if not text:
                    print("Neko neizdevās atpazīt. Pamēģini vēlreiz.")
                    continue

                if text.lower().strip() in {"beigt", "iziet", "stop", "exit", "quit"}:
                    print("Saņemts komandas vārds – iziešu no programmas.")
                    break

                lower_text = text.lower()
                if any(k in lower_text for k in ["pulksten", "cikos", "cik ir laiks", "laiks ir cik"]):
                    time_reply = get_time_text()
                    tts_latvian_to_file(time_reply, out_path)
                    play_audio_file(out_path)
                    log_conversation(text, time_reply, note="time")
                    continue

                if "laikapstāk" in lower_text or "kāds laiks" in lower_text or "kads laiks" in lower_text:
                    place = RIGA_WEATHER_PLACE if "rīg" in lower_text or "riga" in lower_text else DEFAULT_WEATHER_PLACE
                    weather_reply = get_weather_text(place)
                    tts_latvian_to_file(weather_reply, out_path)
                    play_audio_file(out_path)
                    log_conversation(text, weather_reply, note="weather")
                    continue

                start_thinking_sound()
                try:
                    reply_text = chat_latvian_elderly(text)
                finally:
                    stop_thinking_sound()

                # Ja modelis atgriež komandu atskaņot audiogrāmatu,
                # izsaucam lokālo atskaņotāju un nelasām TTS atbildi.
                if reply_text.startswith("CMD:PLAY_BOOK:"):
                    book_id = reply_text.split("CMD:PLAY_BOOK:", 1)[1].strip()
                    print(f"-> Saņemta komanda atskaņot audiogrāmatu: {book_id}")
                    log_conversation(text, reply_text, note="play_book")
                    play_audiobook(book_id)
                    # pēc audiogrāmatas beigām turpinām ciklu
                    continue

                if reply_text.startswith("CMD:PLAY_YT:"):
                    yt_id = reply_text.split("CMD:PLAY_YT:", 1)[1].strip()
                    print(f"-> Saņemta komanda atvērt YouTube ierakstu: {yt_id}")
                    log_conversation(text, reply_text, note="play_yt")
                    play_youtube_item(yt_id)
                    print("   (Enter, lai aizvērtu YouTube un turpinātu sarunu.)")
                    _wait_for_enter()
                    stop_youtube_playback()
                    continue

                if reply_text.startswith("CMD:PLAY_LOCAL:"):
                    local_id = reply_text.split("CMD:PLAY_LOCAL:", 1)[1].strip()
                    print(f"-> Saņemta komanda atskaņot lokālu grāmatu: {local_id}")
                    log_conversation(text, reply_text, note="play_local")
                    progress = _get_book_progress(local_id)
                    should_resume = progress.get("file", 0) or progress.get("ms", 0)
                    play_local_book(local_id, resume=bool(should_resume))
                    continue

                if reply_text.startswith("CMD:RESUME_LOCAL:"):
                    local_id = reply_text.split("CMD:RESUME_LOCAL:", 1)[1].strip()
                    print(f"-> Saņemta komanda turpināt lokālu grāmatu: {local_id}")
                    log_conversation(text, reply_text, note="resume_local")
                    play_local_book(local_id, resume=True)
                    continue

                if reply_text.strip() == "CMD:STOP_YT":
                    print("-> Saņemta komanda apstādināt YouTube atskaņošanu.")
                    log_conversation(text, reply_text, note="stop_yt")
                    stop_youtube_playback()
                    continue

                if reply_text.startswith("CMD:SET_VOICE:"):
                    voice_label = reply_text.split("CMD:SET_VOICE:", 1)[1].strip()
                    status_msg = set_tts_voice_by_label(voice_label)
                    log_conversation(text, reply_text, note="set_voice")
                    tts_latvian_to_file(status_msg, out_path)
                    play_audio_file(out_path)
                    continue

                tts_latvian_to_file(reply_text, out_path)
                play_audio_file(out_path)
                log_conversation(text, reply_text)

            except Exception as e:
                print(f"!!! Kļūda darbībā ar OpenAI API: {e}", file=sys.stderr)
                print("Pamēģini vēlreiz vai pārbaudi savu interneta savienojumu / API key.")
    if esc_remover:
        try:
            import keyboard
            keyboard.remove_hotkey(esc_remover)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        if not maybe_show_scheduler_menu():
            sys.exit(0)
        start_scheduler_thread()
        main_loop()
    except KeyboardInterrupt:
        print("\nIziešana. Uz redzēšanos!")
    finally:
        stop_scheduler_thread()
