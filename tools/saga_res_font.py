#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
saga_res_text.py

Outil CLI pour extraire / réinjecter des ressources *texte* (tables de chaînes)
depuis des fichiers SAGA .RES / .RSC (ScummVM SAGA engine).

Cible principale : IHNM / SCREAM.RES (I Have No Mouth and I Must Scream),
mais le format "RSC table at end" est le même.

Principe :
- Le fichier contient N ressources concaténées.
- À la fin, on trouve un marqueur: <uint32 table_offset><uint32 entry_count>
- À table_offset, on trouve entry_count entrées <uint32 offset><uint32 size>
- Certaines ressources sont des "String Tables" (Object Name List, verb list, etc.):
  elles commencent par une table d'offsets uint16 LE.

Ce script :
- export : scanne les ressources, détecte les String Tables de façon *structurelle*,
  et exporte en JSON.
- import : prend ce JSON (éventuellement modifié), reconstruit les ressources
  concernées, réécrit un nouveau .RES avec une table repointée recalculée.

Usage :
  python saga_res_text.py list   SCREAM.RES
  python saga_res_text.py export SCREAM.RES -o scream_text.json
  python saga_res_text.py import SCREAM.RES scream_text.json -o SCREAM.mod.RES

Astuce :
- Si tu veux cibler uniquement certaines ressources : --ids 21,40,44
- Encodage : par défaut cp437 (classique DOS). Change via --encoding cp1252 si besoin.

Limitations assumées :
- Ce script ne traite *que* les ressources au format "table de chaînes" (offset table).
- Il n'essaie pas d'extraire le texte depuis les scripts / dialogues encodés autrement.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# -------------------------
# Utilitaires de base
# -------------------------

class FormatError(Exception):
    pass


def u16le(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]


def u32le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def is_printable_for_humans(s: str) -> float:
    """
    Ratio de caractères imprimables (au sens str.isprintable()).
    On compte aussi les espaces / \t / \n.
    """
    if not s:
        return 1.0
    ok = 0
    for ch in s:
        if ch.isprintable() or ch in "\t\r\n":
            ok += 1
    return ok / len(s)


# -------------------------
# Lecture / écriture RSC/RES
# -------------------------

@dataclass(frozen=True)
class ResEntry:
    rid: int
    offset: int
    size: int


def read_rsc_entries(buf: bytes) -> Tuple[int, List[ResEntry]]:
    """
    Retourne: (table_offset, entries).
    """
    if len(buf) < 8:
        raise FormatError("Fichier trop petit pour contenir le marqueur RSC/RES.")
    table_offset = u32le(buf, len(buf) - 8)
    entry_count = u32le(buf, len(buf) - 4)

    table_size = entry_count * 8
    expected_len = table_offset + table_size + 8
    if expected_len != len(buf):
        # Beaucoup de fichiers respectent exactement ça; si non, on reste permissif.
        # Mais on garde une vérification de bornes.
        if table_offset + table_size + 8 > len(buf):
            raise FormatError(
                f"Table hors limites: table_offset={table_offset}, entry_count={entry_count}, "
                f"len={len(buf)}"
            )

    entries: List[ResEntry] = []
    for i in range(entry_count):
        off = u32le(buf, table_offset + i * 8)
        size = u32le(buf, table_offset + i * 8 + 4)
        if off + size > table_offset:
            # Ressource qui empiète sur la table : invalide.
            raise FormatError(
                f"Entrée {i} invalide: off={off} size={size} dépasse table_offset={table_offset}"
            )
        entries.append(ResEntry(rid=i, offset=off, size=size))
    return table_offset, entries


def write_rsc_file(resources: List[bytes]) -> bytes:
    """
    Construit un fichier RES/RSC à partir d'une liste de ressources (bytes), en
    reconstruisant la table et le marqueur.
    """
    out = bytearray()
    offsets_sizes: List[Tuple[int, int]] = []
    cur = 0
    for blob in resources:
        offsets_sizes.append((cur, len(blob)))
        out.extend(blob)
        cur += len(blob)

    table_offset = cur
    # table d'entrées
    for off, size in offsets_sizes:
        out.extend(struct.pack("<II", off, size))
    # marqueur final
    out.extend(struct.pack("<II", table_offset, len(resources)))
    return bytes(out)



# -------------------------
# Bitmap Font (SAGA / IHNM) parsing / building
# -------------------------
#
# Dans IHNM (SCREAM.RES), les polices sont des ressources "bitmap 1-bit"
# structurées ainsi (header fixe 1286 octets) :
#   INT16 c_height
#   INT16 c_width      (max width; peu utilisé)
#   INT16 row_length   (largeur de la bande en octets, donc largeur_px = row_length * 8)
#   INT16 index[256]   (x en octets depuis le début de la ligne)
#   BYTE  width[256]   (largeur du glyph en pixels)
#   BYTE  flag[256]    (marge/flag; souvent 0/1)
#   BYTE  tracking[256] (espacement; souvent 0..)
#   BYTE  bitmap[row_length * c_height]  (1-bit, MSB=left)
#
# Le header minimal est référencé côté ScummVM comme "FONT_DESCSIZE = 1286".
FONT_DESCSIZE = 1286
FONT_CHARCOUNT = 256


@dataclass
class FontResource:
    rid: int
    file_offset: int
    file_size: int

    c_height: int
    c_width: int
    row_length: int

    index_u16: List[int]      # 256
    width_u8: List[int]       # 256
    flag_u8: List[int]        # 256
    tracking_u8: List[int]    # 256

    bitmap: bytes             # row_length * c_height

    def atlas_width_px(self) -> int:
        return self.row_length * 8


def parse_font_resource(rid: int, blob: bytes, file_offset: int) -> Optional[FontResource]:
    """Retourne un FontResource si la ressource correspond exactement au format attendu."""
    if len(blob) < FONT_DESCSIZE:
        return None
    c_height, c_width, row_length = struct.unpack_from("<HHH", blob, 0)
    if c_height == 0 or row_length == 0:
        return None
    expected = FONT_DESCSIZE + row_length * c_height
    if expected != len(blob):
        return None

    idx_off = 6
    index_u16 = list(struct.unpack_from("<" + "H" * FONT_CHARCOUNT, blob, idx_off))
    w_off = idx_off + 2 * FONT_CHARCOUNT
    width_u8 = list(blob[w_off:w_off + FONT_CHARCOUNT])
    f_off = w_off + FONT_CHARCOUNT
    flag_u8 = list(blob[f_off:f_off + FONT_CHARCOUNT])
    t_off = f_off + FONT_CHARCOUNT
    tracking_u8 = list(blob[t_off:t_off + FONT_CHARCOUNT])

    bm_off = FONT_DESCSIZE
    bitmap = blob[bm_off:bm_off + row_length * c_height]

    return FontResource(
        rid=rid,
        file_offset=file_offset,
        file_size=len(blob),
        c_height=c_height,
        c_width=c_width,
        row_length=row_length,
        index_u16=index_u16,
        width_u8=width_u8,
        flag_u8=flag_u8,
        tracking_u8=tracking_u8,
        bitmap=bitmap,
    )


def font_bitmap_to_image(fr: FontResource):
    """Convertit le bitmap 1-bit en image PIL (mode 'L')."""
    try:
        from PIL import Image
    except Exception as e:
        raise FormatError("Pillow (PIL) est requis pour export/import des fonts.") from e

    w_px = fr.atlas_width_px()
    h_px = fr.c_height
    img = Image.new("L", (w_px, h_px), 0)

    # Convention: MSB (bit 7) = pixel le plus à gauche dans l'octet
    pixels = img.load()
    row_len = fr.row_length
    bm = fr.bitmap
    for y in range(h_px):
        row = bm[y * row_len:(y + 1) * row_len]
        x = 0
        for b in row:
            for bit in range(7, -1, -1):
                pixels[x, y] = 255 if ((b >> bit) & 1) else 0
                x += 1
    return img


def image_to_font_bitmap(img, row_length: int, height: int) -> bytes:
    """Convertit une image PIL en bitmap 1-bit (bytes/ligne = row_length)."""
    try:
        from PIL import Image
    except Exception as e:
        raise FormatError("Pillow (PIL) est requis pour export/import des fonts.") from e

    img = img.convert("L")
    w_px, h_px = img.size
    if h_px != height:
        raise FormatError(f"Hauteur image ({h_px}) != c_height attendu ({height}).")
    if w_px != row_length * 8:
        raise FormatError(f"Largeur image ({w_px}) != row_length*8 attendu ({row_length*8}).")

    pix = img.load()
    out = bytearray(row_length * height)
    for y in range(height):
        for bx in range(row_length):
            val = 0
            base_x = bx * 8
            for bit in range(8):
                if pix[base_x + bit, y] > 0:
                    val |= (1 << (7 - bit))
            out[y * row_length + bx] = val
    return bytes(out)


def build_font_resource_bytes(fr_dict: Dict[str, Any], img_path: Path) -> bytes:
    """Reconstruit bytes d'une ressource Font depuis JSON + image."""
    c_height = int(fr_dict["c_height"])
    row_length = int(fr_dict["row_length"])
    c_width = int(fr_dict.get("c_width", 0))

    index_u16 = fr_dict["index_u16"]
    width_u8 = fr_dict["width_u8"]
    flag_u8 = fr_dict["flag_u8"]
    tracking_u8 = fr_dict["tracking_u8"]

    if not (isinstance(index_u16, list) and len(index_u16) == FONT_CHARCOUNT):
        raise FormatError("index_u16 doit être une liste de 256 entiers.")
    if not (isinstance(width_u8, list) and len(width_u8) == FONT_CHARCOUNT):
        raise FormatError("width_u8 doit être une liste de 256 entiers.")
    if not (isinstance(flag_u8, list) and len(flag_u8) == FONT_CHARCOUNT):
        raise FormatError("flag_u8 doit être une liste de 256 entiers.")
    if not (isinstance(tracking_u8, list) and len(tracking_u8) == FONT_CHARCOUNT):
        raise FormatError("tracking_u8 doit être une liste de 256 entiers.")

    try:
        from PIL import Image
    except Exception as e:
        raise FormatError("Pillow (PIL) est requis pour export/import des fonts.") from e

    img = Image.open(img_path)
    bm = image_to_font_bitmap(img, row_length=row_length, height=c_height)

    if c_width <= 0:
        c_width = max(int(x) for x in width_u8) if width_u8 else 0

    header = bytearray()
    header.extend(struct.pack("<HHH", c_height, c_width, row_length))
    header.extend(struct.pack("<" + "H" * FONT_CHARCOUNT, *[int(x) & 0xFFFF for x in index_u16]))
    header.extend(bytes(int(x) & 0xFF for x in width_u8))
    header.extend(bytes(int(x) & 0xFF for x in flag_u8))
    header.extend(bytes(int(x) & 0xFF for x in tracking_u8))
    if len(header) != FONT_DESCSIZE:
        raise FormatError(f"Header font invalide: {len(header)} octets (attendu {FONT_DESCSIZE}).")

    return bytes(header) + bm


def find_font_resources(buf: bytes, entries: List[ResEntry]) -> List[FontResource]:
    out: List[FontResource] = []
    for e in entries:
        blob = buf[e.offset:e.offset + e.size]
        fr = parse_font_resource(e.rid, blob, e.offset)
        if fr is not None:
            out.append(fr)
    return out


# -------------------------
# String Table (Object Name List) parsing / building
# -------------------------

@dataclass
class StringItem:
    string_id: int
    offset: int
    text: str
    orig_b64: str


@dataclass
class StringTableResource:
    rid: int
    file_offset: int
    file_size: int

    encoding: str
    table_len: int
    table_entries_total: int
    trailing_table_u16: List[int]  # valeurs brutes uint16 après la dernière chaîne (incluant le sentinel)
    strings: List[StringItem]

    avg_printable: float
    total_chars: int


def parse_string_table(
    rid: int,
    blob: bytes,
    file_offset: int,
    encoding: str,
    strict: bool = True
) -> Optional[StringTableResource]:
    """
    Détection/parse structurel d'une String Table, proche de la logique ScummVM SAGA :
    - table_len = uint16le au début
    - table_entries_total = table_len/2 (table d'offsets en uint16)
    - on lit ces offsets, on applique la correction de wrap 16-bit (offset += 65536 si baisse)
    - on s'arrête sur l'offset == len(blob) (sentinel "fin de ressource")
    """
    if len(blob) < 4:
        return None

    table_len = u16le(blob, 0)
    if table_len < 2 or table_len > len(blob) or (table_len % 2) != 0:
        return None

    table_entries_total = table_len // 2
    if table_entries_total < 1 or table_entries_total * 2 > len(blob):
        return None

    raw_u16 = [u16le(blob, i * 2) for i in range(table_entries_total)]

    offsets: List[int] = []
    prev = 0
    break_idx: Optional[int] = None
    for i, raw in enumerate(raw_u16):
        off = raw
        # Correction wrap 16-bit : si l'offset "retourne en arrière", on ajoute 65536.
        # (On le fait en boucle au cas où on franchit plusieurs fois la barrière.)
        while off < prev:
            off += 65536
        prev = off

        if off == len(blob):
            break_idx = i
            break
        if off > len(blob):
            return None
        if strict and (off % 2) != 0:
            return None
        offsets.append(off)

    if not offsets:
        return None

    # Dans ce format, le premier offset est en général égal à table_len (début des strings).
    if strict and offsets[0] != table_len:
        return None

    # monotonicité
    if strict:
        for a, b in zip(offsets, offsets[1:]):
            if b < a:
                return None

    strings: List[StringItem] = []
    total_chars = 0
    printable_acc = 0.0

    for si, off in enumerate(offsets):
        end = offsets[si + 1] if si + 1 < len(offsets) else len(blob)
        chunk = blob[off:end]
        if b"\x00" in chunk:
            chunk = chunk.split(b"\x00", 1)[0]

        # Décodage "texte" (pour affichage / édition).
        # Note: on garde aussi la version binaire d'origine (base64) pour pouvoir reconstruire
        # même si un texte n'est pas représentable dans l'encodage choisi.
        try:
            text = chunk.decode(encoding)
        except UnicodeDecodeError:
            text = chunk.decode(encoding, errors="replace")

        total_chars += len(text)
        printable_acc += is_printable_for_humans(text)

        strings.append(
            StringItem(
                string_id=si,
                offset=off,
                text=text,
                orig_b64=b64e(chunk),
            )
        )

    avg_printable = printable_acc / max(1, len(strings))

    trailing: List[int] = []
    if break_idx is not None:
        trailing = raw_u16[break_idx:]  # inclut le sentinel (=len(res)) et éventuellement d'autres valeurs

    return StringTableResource(
        rid=rid,
        file_offset=file_offset,
        file_size=len(blob),
        encoding=encoding,
        table_len=table_len,
        table_entries_total=table_entries_total,
        trailing_table_u16=trailing,
        strings=strings,
        avg_printable=avg_printable,
        total_chars=total_chars,
    )


def build_string_table_bytes(
    strings: List[bytes],
    table_entries_total: int,
    trailing_table_u16: List[int],
    pad_byte: int = 0x00
) -> bytes:
    """
    Reconstruit une ressource StringTable à partir d'une liste de chaînes (bytes sans NUL).
    - Conserve table_entries_total (donc la taille de la table en entrée).
    - Conserve trailing_table_u16 (même longueur), mais met à jour le 1er élément
      pour pointer vers la fin de ressource (sentinel).
    - Assure que chaque chaîne commence à un offset pair (padding si nécessaire).
    """
    n = len(strings)
    if table_entries_total < n:
        raise FormatError(
            f"table_entries_total ({table_entries_total}) < nombre de chaînes ({n})."
        )

    table_len = table_entries_total * 2

    # Calcul offsets (32-bit internes), alignés sur 2.
    offsets: List[int] = []
    cur = table_len
    for s in strings:
        if cur % 2 == 1:
            cur += 1
        offsets.append(cur)
        cur += len(s) + 1  # NUL

    # Construction (table puis data)
    out = bytearray(b"\x00" * table_len)

    for off, s in zip(offsets, strings):
        if len(out) < off:
            out.extend(bytes([pad_byte]) * (off - len(out)))
        out.extend(s)
        out.append(0)

    end_offset = len(out)

    # Prépare la table complète de uint16
    # - Les n premiers = offsets des strings
    # - Le reste = trailing_table_u16 (si fourni), avec sentinel mis à jour
    table_vals: List[int] = []
    table_vals.extend([o % 65536 for o in offsets])

    remaining = table_entries_total - n
    if remaining > 0:
        # trailing_table_u16 doit avoir au moins remaining entrées; sinon on complète.
        tail = list(trailing_table_u16) if trailing_table_u16 else []
        if len(tail) < remaining:
            # on complète avec des zéros
            tail.extend([0] * (remaining - len(tail)))

        # met à jour le "sentinel fin de ressource" si on en a un
        tail[0] = end_offset % 65536

        table_vals.extend([v & 0xFFFF for v in tail[:remaining]])

    if len(table_vals) != table_entries_total:
        raise AssertionError("Incohérence table_vals/table_entries_total")

    # Écrit la table uint16
    for i, v in enumerate(table_vals):
        struct.pack_into("<H", out, i * 2, v)

    return bytes(out)


# -------------------------
# CLI
# -------------------------

def parse_ids_csv(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a, 0)
            b_i = int(b, 0)
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            out.extend(list(range(a_i, b_i + 1)))
        else:
            out.append(int(part, 0))
    # unique, conservant ordre
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def cmd_list(args: argparse.Namespace) -> int:
    buf = Path(args.res_file).read_bytes()
    table_offset, entries = read_rsc_entries(buf)

    wanted: Optional[set] = None
    if args.ids:
        wanted = set(parse_ids_csv(args.ids))

    hits: List[StringTableResource] = []
    for e in entries:
        if wanted is not None and e.rid not in wanted:
            continue
        blob = buf[e.offset : e.offset + e.size]
        st = parse_string_table(e.rid, blob, e.offset, args.encoding, strict=args.strict)
        if st is None:
            continue
        hits.append(st)

    # tri "les plus intéressantes" d'abord
    hits.sort(key=lambda r: (-(r.avg_printable), -r.total_chars, -len(r.strings)))

    print(f"Fichier: {args.res_file}")
    print(f"Ressources totales: {len(entries)} | table_offset={table_offset}")
    print(f"String Tables détectées: {len(hits)} (encoding={args.encoding})")
    print()
    print(" rid |   size | strings | printable | total_chars | aperçu")
    print("-----+--------+---------+-----------+------------+---------------------------")

    for st in hits[: args.limit]:
        preview = ""
        for it in st.strings:
            if it.text:
                preview = it.text[:30].replace("\n", "\\n")
                break
        print(
            f"{st.rid:4d} | {st.file_size:6d} | {len(st.strings):7d} | "
            f"{st.avg_printable:9.2f} | {st.total_chars:10d} | {preview}"
        )

    if len(hits) > args.limit:
        print(f"\n… {len(hits) - args.limit} autres (augmente --limit).")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    buf = Path(args.res_file).read_bytes()
    table_offset, entries = read_rsc_entries(buf)

    wanted: Optional[set] = None
    if args.ids:
        wanted = set(parse_ids_csv(args.ids))

    resources_json: List[Dict[str, Any]] = []
    for e in entries:
        if wanted is not None and e.rid not in wanted:
            continue
        blob = buf[e.offset : e.offset + e.size]
        st = parse_string_table(e.rid, blob, e.offset, args.encoding, strict=args.strict)
        if st is None:
            continue

        if args.filter_readable:
            if st.avg_printable < args.min_printable or st.total_chars < args.min_chars:
                continue

        resources_json.append(
            {
                "rid": st.rid,
                "file_offset": st.file_offset,
                "file_size": st.file_size,
                "encoding": st.encoding,
                "table_len": st.table_len,
                "table_entries_total": st.table_entries_total,
                "trailing_table_u16": st.trailing_table_u16,
                "metrics": {
                    "avg_printable": st.avg_printable,
                    "total_chars": st.total_chars,
                    "string_count": len(st.strings),
                },
                "strings": [
                    {
                        "string_id": it.string_id,
                        "offset": it.offset,
                        "text": it.text,
                        "orig_b64": it.orig_b64,
                    }
                    for it in st.strings
                ],
            }
        )

    out_obj = {
        "tool": "saga_res_text.py",
        "source_res": os.path.basename(args.res_file),
        "encoding": args.encoding,
        "note": "Édite le champ strings[].text, puis réinjecte avec la commande import.",
        "resources": resources_json,
    }

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(out_obj, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )

    print(f"OK: export -> {out_path} ({len(resources_json)} ressources)")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    base_buf = Path(args.res_file).read_bytes()
    _, entries = read_rsc_entries(base_buf)

    obj = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    resources = obj.get("resources", [])
    if not isinstance(resources, list):
        raise FormatError("JSON invalide: resources doit être une liste.")

    # map rid -> json resource
    by_rid: Dict[int, Dict[str, Any]] = {}
    for r in resources:
        rid = int(r["rid"])
        by_rid[rid] = r

    # Extraire blobs originaux une fois
    original_blobs: List[bytes] = []
    for e in entries:
        original_blobs.append(base_buf[e.offset : e.offset + e.size])

    # Construire blobs modifiés
    new_blobs: List[bytes] = list(original_blobs)

    for rid, jr in by_rid.items():
        if rid < 0 or rid >= len(entries):
            raise FormatError(f"RID {rid} hors limites (0..{len(entries)-1}).")

        enc = jr.get("encoding") or obj.get("encoding") or "cp437"
        table_entries_total = int(jr["table_entries_total"])
        trailing = jr.get("trailing_table_u16") or []

        jstrings = jr.get("strings", [])
        if not isinstance(jstrings, list):
            raise FormatError(f"JSON invalide pour rid={rid}: strings doit être une liste.")

        # Enforce count by default (sécurité)
        if not args.allow_count_change:
            # On compare au JSON exporté (pas au fichier), car c'est le référentiel de l'utilisateur.
            expected = int(jr.get("metrics", {}).get("string_count", len(jstrings)))
            if len(jstrings) != expected:
                raise FormatError(
                    f"rid={rid}: nombre de strings dans JSON ({len(jstrings)}) != attendu ({expected}). "
                    f"Utilise --allow-count-change si tu sais ce que tu fais."
                )        # Reconstruction bytes string par string
        rebuilt_strings: List[bytes] = []
        for it in jstrings:
            text = it.get("text", "")
            orig_b64 = it.get("orig_b64", "")

            # Si demandé, et si le texte "n'a pas changé", on réutilise exactement les bytes originaux
            # (ça évite de perdre des octets non représentables dans l'encodage, ou de changer
            # des caractères de contrôle).
            if args.keep_original_when_unchanged and isinstance(text, str) and orig_b64:
                orig_raw = b64d(orig_b64)
                try:
                    orig_text = orig_raw.decode(enc, errors="replace")
                except LookupError as e:
                    raise FormatError(f"Encodage inconnu: {enc}") from e
                if orig_text == text:
                    rebuilt_strings.append(orig_raw)
                    continue

            if text is None and orig_b64:
                raw = b64d(orig_b64)
            else:
                if not isinstance(text, str):
                    raise FormatError(f"rid={rid}: strings[].text doit être une string (ou null).")
                try:
                    raw = text.encode(enc, errors=args.encode_errors)
                except LookupError as e:
                    raise FormatError(f"Encodage inconnu: {enc}") from e
            rebuilt_strings.append(raw)

        new_blob = build_string_table_bytes(
            rebuilt_strings,
            table_entries_total=table_entries_total,
            trailing_table_u16=trailing,
            pad_byte=args.pad_byte,
        )
        new_blobs[rid] = new_blob
    out_bytes = write_rsc_file(new_blobs)
    Path(args.output).write_bytes(out_bytes)
    print(f"OK: import -> {args.output} (ressources modifiées: {len(by_rid)})")
    return 0


# -------------------------
# Commandes Fonts
# -------------------------

def cmd_font_list(args: argparse.Namespace) -> int:
    buf = Path(args.res_file).read_bytes()
    _, entries = read_rsc_entries(buf)
    fonts = find_font_resources(buf, entries)

    if args.ids:
        wanted = set(parse_ids_csv(args.ids))
        fonts = [f for f in fonts if f.rid in wanted]

    if not fonts:
        print("Aucune ressource font détectée.")
        return 0

    for fr in fonts:
        print(
            f"RID {fr.rid:4d}  off=0x{fr.file_offset:08X}  size={fr.file_size:6d}  "
            f"h={fr.c_height:2d}  row_len={fr.row_length:3d}  width_px={fr.atlas_width_px():4d}"
        )
    return 0


def cmd_font_export(args: argparse.Namespace) -> int:
    buf = Path(args.res_file).read_bytes()
    _, entries = read_rsc_entries(buf)
    fonts = find_font_resources(buf, entries)

    wanted: Optional[set] = None
    if args.ids:
        wanted = set(parse_ids_csv(args.ids))

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fonts_json: List[Dict[str, Any]] = []
    for fr in fonts:
        if wanted is not None and fr.rid not in wanted:
            continue

        img = font_bitmap_to_image(fr)
        img_name = f"font_rid{fr.rid:04d}.png"
        img_path = out_dir / img_name
        img.save(img_path)

        fonts_json.append(
            {
                "rid": fr.rid,
                "file_offset": fr.file_offset,
                "file_size": fr.file_size,
                "c_height": fr.c_height,
                "c_width": fr.c_width,
                "row_length": fr.row_length,
                "atlas_width_px": fr.atlas_width_px(),
                "image_file": img_name,
                "index_u16": fr.index_u16,
                "width_u8": fr.width_u8,
                "flag_u8": fr.flag_u8,
                "tracking_u8": fr.tracking_u8,
            }
        )

    out_obj = {
        "tool": "saga_res_tool.py",
        "source_res": os.path.basename(args.res_file),
        "note": "Modifie fonts[].image_file (PNG) et/ou les tableaux index/width/flag/tracking, puis réinjecte avec font-import.",
        "fonts": fonts_json,
    }

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(out_obj, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    print(f"OK: font-export -> {out_path} (fonts: {len(fonts_json)}) ; images -> {out_dir}")
    return 0


def cmd_font_import(args: argparse.Namespace) -> int:
    base_buf = Path(args.res_file).read_bytes()
    table_offset, entries = read_rsc_entries(base_buf)

    obj = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    fonts = obj.get("fonts", [])
    if not isinstance(fonts, list):
        raise FormatError("JSON invalide: fonts doit être une liste.")

    by_rid: Dict[int, Dict[str, Any]] = {}
    for fr in fonts:
        rid = int(fr["rid"])
        by_rid[rid] = fr

    # Extraire blobs originaux une fois
    original_blobs: List[bytes] = []
    for e in entries:
        original_blobs.append(base_buf[e.offset:e.offset + e.size])

    new_blobs: List[bytes] = list(original_blobs)

    indir = Path(args.indir) if args.indir else Path(args.json_file).parent

    wanted: Optional[set] = None
    if args.ids:
        wanted = set(parse_ids_csv(args.ids))

    changed = 0
    for rid, fr_dict in by_rid.items():
        if wanted is not None and rid not in wanted:
            continue
        if rid < 0 or rid >= len(entries):
            raise FormatError(f"RID {rid} hors limites (0..{len(entries)-1}).")

        img_file = fr_dict.get("image_file")
        if not img_file:
            raise FormatError(f"rid={rid}: image_file manquant.")
        img_path = (indir / img_file).resolve()
        if not img_path.exists():
            raise FormatError(f"rid={rid}: image introuvable: {img_path}")

        new_blob = build_font_resource_bytes(fr_dict, img_path)
        new_blobs[rid] = new_blob
        changed += 1

    out_bytes = write_rsc_file(new_blobs)
    Path(args.output).write_bytes(out_bytes)
    print(f"OK: font-import -> {args.output} (fonts modifiées: {changed})")
    return 0



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="saga_res_tool.py",
        description="Extraction/réinjection de String Tables depuis des fichiers SAGA .RES/.RSC (ex: SCREAM.RES).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Lister les String Tables détectées.")
    p_list.add_argument("res_file", help="Chemin vers .RES/.RSC")
    p_list.add_argument("--encoding", default="cp437", help="Encodage pour décoder les textes (défaut: cp437)")
    p_list.add_argument("--strict", action="store_true", help="Validation structurelle stricte")
    p_list.add_argument("--ids", help="Limiter aux IDs (ex: 21,40,44 ou 10-30)")
    p_list.add_argument("--limit", type=int, default=50, help="Nb de lignes à afficher")
    p_list.set_defaults(func=cmd_list)

    p_exp = sub.add_parser("export", help="Exporter en JSON.")
    p_exp.add_argument("res_file", help="Chemin vers .RES/.RSC")
    p_exp.add_argument("-o", "--output", required=True, help="Fichier JSON de sortie")
    p_exp.add_argument("--encoding", default="cp437", help="Encodage pour décoder les textes (défaut: cp437)")
    p_exp.add_argument("--strict", action="store_true", help="Validation structurelle stricte")
    p_exp.add_argument("--ids", help="Limiter aux IDs (ex: 21,40,44 ou 10-30)")
    p_exp.add_argument("--pretty", action="store_true", help="JSON indenté lisible")
    p_exp.add_argument("--filter-readable", action="store_true",
                       help="Ne garder que les ressources 'lisibles' (selon métriques simples)")
    p_exp.add_argument("--min-printable", type=float, default=0.85, help="Seuil lisibilité (0..1)")
    p_exp.add_argument("--min-chars", type=int, default=20, help="Seuil total de caractères")
    p_exp.set_defaults(func=cmd_export)

    p_imp = sub.add_parser("import", help="Réinjecter depuis un JSON et recalculer la table/pointeurs.")
    p_imp.add_argument("res_file", help="Fichier .RES/.RSC d'origine")
    p_imp.add_argument("json_file", help="JSON exporté (et éventuellement modifié)")
    p_imp.add_argument("-o", "--output", required=True, help="Fichier .RES/.RSC de sortie")
    p_imp.add_argument("--allow-count-change", action="store_true",
                       help="Autoriser ajout/suppression de strings (dangereux selon le jeu)")
    p_imp.add_argument("--encode-errors", default="strict",
                       choices=["strict", "replace", "ignore"],
                       help="Politique d'erreur d'encodage lors de l'injection (défaut: strict)")
    p_imp.add_argument("--pad-byte", type=lambda x: int(x, 0), default=0x00,
                       help="Octet de padding pour l'alignement (défaut: 0x00)")
    p_imp.add_argument("--keep-original-when-unchanged", action="store_true",
                       help="(option future) Tenter de réutiliser les bytes originaux si texte inchangé")
    p_imp.set_defaults(func=cmd_import)


    # Fonts (bitmap 1-bit) - IHNM / SAGA
    p_flist = sub.add_parser("font-list", help="Lister les ressources 'font' détectées.")
    p_flist.add_argument("res_file", help="Chemin vers .RES/.RSC")
    p_flist.add_argument("--ids", help="Limiter aux IDs (ex: 2-8 ou 3,7)")
    p_flist.set_defaults(func=cmd_font_list)

    p_fexp = sub.add_parser("font-export", help="Exporter les fonts (JSON + PNG).")
    p_fexp.add_argument("res_file", help="Chemin vers .RES/.RSC")
    p_fexp.add_argument("-o", "--output", required=True, help="JSON de sortie (métadonnées font)")
    p_fexp.add_argument("--outdir", required=True, help="Dossier où écrire les PNG")
    p_fexp.add_argument("--ids", help="Limiter aux IDs (ex: 2-8 ou 3,7)")
    p_fexp.add_argument("--pretty", action="store_true", help="JSON indenté lisible")
    p_fexp.set_defaults(func=cmd_font_export)

    p_fimp = sub.add_parser("font-import", help="Réinjecter des fonts depuis JSON + PNG.")
    p_fimp.add_argument("res_file", help="Fichier .RES/.RSC d'origine")
    p_fimp.add_argument("json_file", help="JSON exporté (et éventuellement modifié)")
    p_fimp.add_argument("-o", "--output", required=True, help="Fichier .RES/.RSC de sortie")
    p_fimp.add_argument("--indir", help="Dossier contenant les PNG (défaut: dossier du JSON)")
    p_fimp.add_argument("--ids", help="Limiter aux IDs (ex: 2-8 ou 3,7)")
    p_fimp.set_defaults(func=cmd_font_import)

    return p


def main(argv: List[str]) -> int:
    try:
        p = build_argparser()
        args = p.parse_args(argv)
        return int(args.func(args))
    except FormatError as e:
        print(f"[ERREUR FORMAT] {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"[ERREUR FICHIER] {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"[ERREUR JSON] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
