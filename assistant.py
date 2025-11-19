import os
import sys
import tempfile
import subprocess
import glob
import webbrowser
import requests

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

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
# ==================

client = OpenAI()


def record_audio_to_wav(path: str):
    """
    Press Enter to start recording, speak, press Enter again to stop.
    Audio is saved as mono 16kHz WAV.
    """
    print("\nNospied Enter, tad runā. Kad pabeidz, nospied Enter vēlreiz.")
    input(">>> Nospied Enter, lai sāktu ierakstu...")
    print("Ieraksts sākts – runā mikrofona virzienā... (Enter = stop)")

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
        # This input() blocks while callback keeps filling frames
        input()
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


def chat_latvian_elderly(user_text: str) -> str:
    """Send Latvian text to GPT and get a Latvian reply,
    tuned for an elderly, blind user.
    """
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
        "Ja lietotāja vēlme nav klausīties audiogrāmatu vai YouTube, atbildi "
        "parastā veidā sirsnīgā, mierīgā tonī.\n"
        "Tev ir pieejamas šādas balsis: 'vīrieša balss', 'sieviešu balss', "
        "'neitrāla balss'. Ja lietotāja skaidri lūdz nomainīt balsi uz kādu "
        "no šīm, TAD NEATBILDI parastā tekstā, bet atbildi tikai ar komandu:\n"
        "  CMD:SET_VOICE:<balses_nosaukums>\n"
        "piemēram: CMD:SET_VOICE:vīrieša balss.\n"
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
    """Open a configured YouTube item in Google Chrome in the background.

    Izmanto macOS komandu 'open -g -a "Google Chrome" <url>', lai Chrome
    atvērtos fonā un Terminālis paliktu aktīvs.
    """
    url = get_youtube_url_by_id(item_id)
    if not url:
        print(f"[*] Neatradu YouTube ierakstu ar ID: {item_id}")
        return
    print(f"-> Atveru YouTube Chrome pārlūkā (fonā): {url}")
    try:
        subprocess.run(["open", "-g", "-a", "Google Chrome", url], check=False)
    except Exception as e:
        print(f"[*] Neizdevās atvērt Google Chrome: {e}")


def stop_youtube_playback():
    """Try to stop YouTube playback by closing Google Chrome windows on macOS.

    Šī ir vienkārša pieeja, kas paredzēta speciālai ierīces lietošanai – tiek
    aizvērtI visi Chrome logi, tādēļ Chrome nedrīkst būt nepieciešams citām
    lietām vienlaikus.
    """
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "Google Chrome" to close windows',
            ],
            check=False,
        )
        print("[*] Mēģinu aizvērt Google Chrome logus.")
    except Exception as e:
        print(f"[*] Neizdevās aizvērt Google Chrome: {e}")


def play_audio_file(path: str):
    """
    Play an audio file (mp3/wav) through default speakers.
    """
    print("-> Atskaņoju atbildi...")
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


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
                if "laikapstāk" in lower_text or "kāds laiks" in lower_text or "kads laiks" in lower_text:
                    weather_reply = get_weather_text(DEFAULT_WEATHER_PLACE)
                    tts_latvian_to_file(weather_reply, out_path)
                    play_audio_file(out_path)
                    continue

                reply_text = chat_latvian_elderly(text)

                # Ja modelis atgriež komandu atskaņot audiogrāmatu,
                # izsaucam lokālo atskaņotāju un nelasām TTS atbildi.
                if reply_text.startswith("CMD:PLAY_BOOK:"):
                    book_id = reply_text.split("CMD:PLAY_BOOK:", 1)[1].strip()
                    print(f"-> Saņemta komanda atskaņot audiogrāmatu: {book_id}")
                    play_audiobook(book_id)
                    # pēc audiogrāmatas beigām turpinām ciklu
                    continue

                if reply_text.startswith("CMD:PLAY_YT:"):
                    yt_id = reply_text.split("CMD:PLAY_YT:", 1)[1].strip()
                    print(f"-> Saņemta komanda atvērt YouTube ierakstu: {yt_id}")
                    play_youtube_item(yt_id)
                    continue

                if reply_text.strip() == "CMD:STOP_YT":
                    print("-> Saņemta komanda apstādināt YouTube atskaņošanu.")
                    stop_youtube_playback()
                    continue

                if reply_text.startswith("CMD:SET_VOICE:"):
                    voice_label = reply_text.split("CMD:SET_VOICE:", 1)[1].strip()
                    status_msg = set_tts_voice_by_label(voice_label)
                    tts_latvian_to_file(status_msg, out_path)
                    play_audio_file(out_path)
                    continue

                tts_latvian_to_file(reply_text, out_path)
                play_audio_file(out_path)

            except Exception as e:
                print(f"!!! Kļūda darbībā ar OpenAI API: {e}", file=sys.stderr)
                print("Pamēģini vēlreiz vai pārbaudi savu interneta savienojumu / API key.")


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nIziešana. Uz redzēšanos!")