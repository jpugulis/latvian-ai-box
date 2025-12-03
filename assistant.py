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


def _save_local_progress(state: dict):
    try:
        with open(LOCAL_BOOK_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[*] Neizdevās saglabāt progresu: {e}")


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
    book_state = state.get(book_id, {"file": 0, "ms": 0})
    if isinstance(book_state, int):
        book_state = {"file": book_state, "ms": 0}

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

    def callback(indata, frames_count, time_info, status):
        if status:
            print(f"[sounddevice status] {status}", file=sys.stderr)
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        # This keyboard.wait() blocks while callback keeps filling frames
        _wait_enter_or_space()
        print("Ieraksts apturēts, apstrādāju...")

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
    """Fetch a very short weather summary for the given place using wttr.in.

    This runs on *your* Mac (not inside OpenAI), so it can reach the public
    internet while the assistant script is running.
    """
    print("-> Vaicos par laikapstākļiem internetā...")
    try:
        location_string = place.replace(" ", "+")
        url = f"https://wttr.in/{location_string}"
        resp = requests.get(url, params={"format": "3", "lang": "lv"}, timeout=5)
        if resp.status_code != 200:
            return "Neizdevās iegūt laikapstākļu informāciju."
        line = resp.text.strip()
        # wttr.in already returns a very short one-line summary
        return f"Laikapstākļi vietā {place}: {line}"
    except Exception:
        return "Neizdevās pieslēgties laikapstākļu servisam."


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


def main_loop():
    print("=== Latviešu balss asistents (prototips) ===")
    print("Ctrl+C, lai izietu.\n")

    while True:
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
                    play_local_book(local_id, resume=False)
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


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nIziešana. Uz redzēšanos!")
