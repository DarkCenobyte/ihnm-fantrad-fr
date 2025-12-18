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


def write_rsc_file_preserve_gaps(
    base_buf: bytes,
    table_offset: int,
    entries: List[ResEntry],
    new_blobs: List[bytes],
) -> bytes:
    """\
    Reconstruit un fichier RES/RSC en **préservant** autant que possible
    la mise en page d'origine.

    Contrairement à write_rsc_file() (qui "repack" tout à la suite), ici on:
    - conserve l'ordre physique des ressources (tri par offset)
    - conserve les "gaps" (octets entre deux ressources) tels quels
    - conserve le gap de fin entre la dernière ressource et la table

    Résultat :
    - un export->import sans modification doit produire un fichier **identique**.
    - si certaines ressources changent de taille, les offsets suivants bougent,
      mais les gaps (leurs octets) restent inchangés.
    """

    if len(entries) != len(new_blobs):
        raise FormatError("new_blobs doit avoir la même longueur que entries")

    entries_by_off = sorted(entries, key=lambda e: e.offset)
    if not entries_by_off:
        raise FormatError("Aucune ressource trouvée.")

    # Vérifie non-chevauchement et calcule les gaps originaux.
    leading_gap = base_buf[0 : entries_by_off[0].offset]

    gaps_after: Dict[int, bytes] = {}
    for i, e in enumerate(entries_by_off):
        end = e.offset + e.size
        if end > table_offset:
            raise FormatError(
                f"Ressource rid={e.rid} empiète sur la table (end={end} > table_offset={table_offset})."
            )
        if i + 1 < len(entries_by_off):
            nxt = entries_by_off[i + 1]
            if end > nxt.offset:
                raise FormatError(
                    f"Ressources qui se chevauchent: rid={e.rid} end={end} > next.offset={nxt.offset} (rid={nxt.rid})."
                )
            gaps_after[e.rid] = base_buf[end : nxt.offset]
        else:
            # dernier: gap jusqu'à la table
            gaps_after[e.rid] = base_buf[end : table_offset]

    out = bytearray()
    out.extend(leading_gap)

    new_offsets_sizes: List[Tuple[int, int]] = [(0, 0)] * len(entries)

    for e in entries_by_off:
        new_off = len(out)
        blob = new_blobs[e.rid]
        new_offsets_sizes[e.rid] = (new_off, len(blob))
        out.extend(blob)
        out.extend(gaps_after.get(e.rid, b""))

    new_table_offset = len(out)
    # table d'entrées (ordre par RID)
    for rid in range(len(entries)):
        off, size = new_offsets_sizes[rid]
        out.extend(struct.pack("<II", off, size))
    # marqueur final
    out.extend(struct.pack("<II", new_table_offset, len(entries)))
    return bytes(out)


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


@dataclass
class _StringTableStruct:
    table_len: int
    table_entries_total: int
    raw_u16: List[int]
    offsets: List[int]  # offsets corrigés (uint16 + wrap)
    break_idx: Optional[int]  # index dans raw_u16 du sentinel (= len(blob))


def _read_string_table_struct(blob: bytes, strict: bool = True) -> Optional[_StringTableStruct]:
    """Parse structurel de la table d'offsets (sans décoder les strings)."""
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
        # Correction wrap 16-bit (IHNM): tant que l'offset "revient en arrière".
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
    if strict and offsets[0] != table_len:
        return None
    if strict:
        for a, b in zip(offsets, offsets[1:]):
            if b < a:
                return None

    return _StringTableStruct(
        table_len=table_len,
        table_entries_total=table_entries_total,
        raw_u16=raw_u16,
        offsets=offsets,
        break_idx=break_idx,
    )


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
    st_struct = _read_string_table_struct(blob, strict=strict)
    if st_struct is None:
        return None

    table_len = st_struct.table_len
    table_entries_total = st_struct.table_entries_total
    raw_u16 = st_struct.raw_u16
    offsets = st_struct.offsets
    break_idx = st_struct.break_idx

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
    table_offset, entries = read_rsc_entries(base_buf)

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
    original_blobs: List[bytes] = [base_buf[e.offset : e.offset + e.size] for e in entries]
    new_blobs: List[bytes] = list(original_blobs)

    # Appliquer les modifications ressource par ressource, en mode "lossless" :
    # - si aucune string ne change => on garde le blob original (padding inclus)
    # - si les nouveaux textes rentrent dans les segments existants => patch in-place, taille inchangée
    # - sinon => rebuild complet de la string table (avec recalcul des offsets)
    for rid, jr in by_rid.items():
        if rid < 0 or rid >= len(entries):
            raise FormatError(f"RID {rid} hors limites (0..{len(entries)-1}).")

        enc = jr.get("encoding") or obj.get("encoding") or "cp437"
        trailing = jr.get("trailing_table_u16") or []
        table_entries_total = int(jr.get("table_entries_total", 0))

        jstrings = jr.get("strings", [])
        if not isinstance(jstrings, list):
            raise FormatError(f"JSON invalide pour rid={rid}: strings doit être une liste.")

        # Trie par string_id pour être robuste
        try:
            jstrings_sorted = sorted(jstrings, key=lambda x: int(x.get("string_id", 0)))
        except Exception:
            jstrings_sorted = jstrings

        orig_blob = original_blobs[rid]
        st_struct = _read_string_table_struct(orig_blob, strict=False)
        if st_struct is None:
            raise FormatError(f"rid={rid}: la ressource n'est pas une string table valide.")

        orig_offsets = st_struct.offsets
        orig_count = len(orig_offsets)

        if not args.allow_count_change and len(jstrings_sorted) != orig_count:
            raise FormatError(
                f"rid={rid}: nombre de strings dans JSON ({len(jstrings_sorted)}) != dans le fichier ({orig_count}). "
                f"Utilise --allow-count-change si tu sais ce que tu fais."
            )

        # Encoder tous les textes demandés
        encoded_new: List[bytes] = []
        for it in jstrings_sorted:
            text = it.get("text", "")
            if text is None:
                # null => garder l'original (binaire)
                ob = it.get("orig_b64")
                if not ob:
                    raise FormatError(f"rid={rid}: text=null mais orig_b64 manquant.")
                encoded_new.append(b64d(ob))
                continue
            if not isinstance(text, str):
                raise FormatError(f"rid={rid}: strings[].text doit être une string (ou null).")
            try:
                encoded_new.append(text.encode(enc, errors=args.encode_errors))
            except LookupError as e:
                raise FormatError(f"Encodage inconnu: {enc}") from e

        # Détecte si rien ne change (comparaison sur le texte décodé comme à l'export)
        unchanged = True
        for i in range(min(orig_count, len(encoded_new))):
            start = orig_offsets[i]
            end = orig_offsets[i + 1] if i + 1 < orig_count else len(orig_blob)
            seg = orig_blob[start:end]
            seg_text_bytes = seg.split(b"\x00", 1)[0]
            try:
                seg_text = seg_text_bytes.decode(enc, errors="replace")
            except LookupError as e:
                raise FormatError(f"Encodage inconnu: {enc}") from e
            # On compare au JSON (donc ré-encode/décode roundtrip) :
            # si l'utilisateur n'a rien touché, ça doit matcher.
            try:
                json_text = encoded_new[i].decode(enc, errors="replace")
            except Exception:
                json_text = ""
            if seg_text != json_text:
                unchanged = False
                break

        if unchanged and len(encoded_new) == orig_count:
            new_blobs[rid] = orig_blob
            continue

        # Tentative patch in-place (ne change pas la taille du blob)
        need_rebuild = False
        patched = bytearray(orig_blob)
        # Si on change le compte => rebuild obligatoire (car table / offsets changent)
        if len(encoded_new) != orig_count:
            need_rebuild = True
        else:
            for i in range(orig_count):
                start = orig_offsets[i]
                end = orig_offsets[i + 1] if i + 1 < orig_count else len(orig_blob)
                capacity = end - start
                if capacity <= 0:
                    need_rebuild = True
                    break
                nb = encoded_new[i]
                if len(nb) + 1 > capacity:
                    need_rebuild = True
                    break
                # Écrit nouveau texte + NUL; le reste du segment reste inchangé
                patched[start : start + len(nb)] = nb
                patched[start + len(nb)] = 0

        if not need_rebuild:
            new_blobs[rid] = bytes(patched)
            continue

        # Rebuild complet de la ressource string table
        # -> on conserve la taille de table (table_entries_total) si dispo
        #    sinon on réutilise la taille du fichier.
        if table_entries_total <= 0:
            table_entries_total = st_struct.table_entries_total

        if table_entries_total < len(encoded_new):
            raise FormatError(
                f"rid={rid}: table_entries_total ({table_entries_total}) < nb de strings ({len(encoded_new)})."
            )

        new_blob = build_string_table_bytes(
            encoded_new,
            table_entries_total=table_entries_total,
            trailing_table_u16=trailing,
            pad_byte=args.pad_byte,
        )
        new_blobs[rid] = new_blob

    out_bytes = write_rsc_file_preserve_gaps(base_buf, table_offset, entries, new_blobs)
    Path(args.output).write_bytes(out_bytes)
    print(f"OK: import -> {args.output} (ressources modifiées: {len(by_rid)})")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="saga_res_text.py",
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
    p_imp.set_defaults(func=cmd_import)

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
