"""Swiss geographic ground truth: cantons, postal codes, cities, languages.

The PLZ ↔ city ↔ canton table comes from Geonames CH (CC-BY-4.0). The file is
loaded once on first use from ``data/raw/CH.txt`` (relative to repo root) and
cached. Each row gives us a real Swiss postal code attached to a real city in
a real canton.

Canton → primary language is fixed (well-known mapping), used to route street
naming into the correct linguistic register.
"""
from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

# Repo-rooted path to the Geonames CH dump (downloaded in Phase 1a).
_GEONAMES_PATH = Path(__file__).resolve().parents[5] / "data" / "raw" / "CH.txt"

CantonCode = str  # 2-letter ISO 3166-2:CH (e.g. "ZH", "GE", "TI")
Lang = Literal["de", "fr", "it", "rm"]

# Canton → primary written-language register for street/address naming. A few
# cantons are bilingual; we pick the dominant one for synthetic generation.
# (RM-primary cantons don't exist; Romansh appears in GR alongside DE/IT.)
CANTON_LANG: dict[CantonCode, Lang] = {
    "AG": "de", "AI": "de", "AR": "de", "BE": "de", "BL": "de", "BS": "de",
    "FR": "fr",  # bilingual but FR-majority outside the city
    "GE": "fr", "GL": "de", "GR": "de",  # GR is DE/IT/RM mixed; default DE
    "JU": "fr", "LU": "de", "NE": "fr", "NW": "de", "OW": "de",
    "SG": "de", "SH": "de", "SO": "de", "SZ": "de",
    "TG": "de", "TI": "it", "UR": "de", "VD": "fr",
    "VS": "fr",  # bilingual; default FR (lower Valais)
    "ZG": "de", "ZH": "de",
}


@dataclass(frozen=True, slots=True)
class Place:
    plz: str
    city: str
    canton: CantonCode
    canton_full: str  # e.g. "Kanton Aargau"
    district: str
    lang: Lang


@functools.lru_cache(maxsize=1)
def _load() -> list[Place]:
    if not _GEONAMES_PATH.exists():
        raise FileNotFoundError(
            f"Geonames CH dump not found at {_GEONAMES_PATH}. "
            "Run: curl -sLo data/raw/CH.zip https://download.geonames.org/export/zip/CH.zip "
            "&& cd data/raw && unzip -o CH.zip"
        )
    out: list[Place] = []
    with _GEONAMES_PATH.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7 or parts[0] != "CH":
                continue
            plz = parts[1]
            city = parts[2]
            canton_full = parts[3]
            canton = parts[4]
            district = parts[5] if len(parts) > 5 else ""
            lang = CANTON_LANG.get(canton)
            if lang is None:
                continue
            out.append(Place(plz=plz, city=city, canton=canton,
                             canton_full=canton_full, district=district, lang=lang))
    if not out:
        raise RuntimeError(f"loaded 0 places from {_GEONAMES_PATH}")
    return out


def all_places() -> list[Place]:
    return _load()


def sample_place(
    lang: Annotated[Lang | None, "Linguistic register; if None, weighted by canton."] = None,
) -> Place:
    places = _load()
    if lang is None:
        return random.choice(places)
    pool = [p for p in places if p.lang == lang]
    if not pool:
        # Fall back: GR has IT/RM speakers though we tagged it as DE.
        if lang == "rm":
            pool = [p for p in places if p.canton == "GR"]
        elif lang == "it":
            pool = [p for p in places if p.canton in ("TI", "GR")]
        if not pool:
            raise ValueError(f"no places for lang={lang!r}")
    return random.choice(pool)
