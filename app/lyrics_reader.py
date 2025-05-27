import os
import shutil
import tempfile
import zipfile

from flask import current_app

from app.model import Track


def load_lyrics_from_folder(root_folder: str) -> list[Track]:
    raw_lyrics_list: list[Track] = []
    for label in os.listdir(root_folder):
        label_dir = os.path.join(root_folder, label)
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.endswith(".txt"):
                artist, title = filename[:-4].split(" - ")
                filepath = os.path.join(label_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    lyrics = f.read()
                raw_lyrics_list.append(Track(artist, title, lyrics))
    return raw_lyrics_list


def read_lyrics() -> list[Track]:
    path = current_app.config["LYRICS_FOLDER_STRUCTURE_PATH"]
    if isinstance(path, str) and path.lower().endswith(".zip"):
        tmpdir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
            folder_name = os.path.basename(path).replace(".zip", "")
            tmpdir = os.path.join(tmpdir, folder_name)
            raw = load_lyrics_from_folder(tmpdir)
        finally:
            shutil.rmtree(tmpdir)
    else:
        raw = load_lyrics_from_folder(path)
    return raw


def lyrics_by_artist(artist: str) -> Track:
    raw = read_lyrics()
    return next(filter(lambda rl: rl.artist == artist, raw))
