import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

# constants
ENCODING = "cp1252"
FOOTER_ENTRY_COUNT = 34
## ASCII pattern preceding pointers tables
PATTERN_PRECEDING_SCENES = [
    "Cybspace1Scene".encode("ascii"),
    "Cybspace2Scene".encode("ascii"),
    "Cybspace3Scene".encode("ascii"),
    "Cybspace4Scene".encode("ascii"),
    "DarkMonitor5".encode("ascii"),
    "Cybspace6Scene".encode("ascii"),
    "DecisionScene".encode("ascii"),
    "SpaceScene".encode("ascii")
]

# init
root = tk.Tk()
root.title("IHNMAIMS - Translator")
root.geometry("800x600")

notebook = ttk.Notebook(root)
notebook.pack(side="top", fill="x")

tabs = [
    {'id': 1, 'name': "Chapter 1"},
    {'id': 2, 'name': "Chapter 2"},
    {'id': 3, 'name': "Chapter 3"},
    {'id': 4, 'name': "Chapter 4"},
    {'id': 5, 'name': "Chapter 5"},
    {'id': 6, 'name': "Chapter 6"},
    {'id': 7, 'name': "Chapter 7"},
    {'id': 8, 'name': "Chapter 8"}
]

container = tk.Frame(root)
container.pack(fill="both", expand=True)
canvas = tk.Canvas(container)
scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)
res_path = None

def configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
def _on_canvas_configure(event):
    canvas.itemconfig(window_id, width=event.width)

scrollable_frame.bind("<Configure>", configure)
window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

canvas.bind("<Configure>", _on_canvas_configure)

def _on_mousewheel(event):
    """
    Support mouse scrolling:
    - event.delta Windows/macOS
    - event.num (4/5) Linux
    """
    if event.delta != 0: # Windows / macOS
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    else: # Linux : Button-4 / Button-5
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel) # Windows / macOS
canvas.bind_all("<Button-4>", _on_mousewheel) # Linux
canvas.bind_all("<Button-5>", _on_mousewheel) # Linux

entries = []
fields = []
frames = []
scene_layout = []
footer = None

def load_footer(data: bytes):
    global footer

    footer_size = FOOTER_ENTRY_COUNT * 4
    footer_start = len(data) - footer_size
    chunk = data[footer_start:footer_start + footer_size]

    raw_entries = []
    offsets = []

    # Decode the 34 dwords as-is: they are already absolute offsets/sizes
    for i in range(FOOTER_ENTRY_COUNT):
        start = i * 4
        val = int.from_bytes(chunk[start:start + 4], byteorder="little")
        raw_entries.append(val)
        offsets.append(val)  # NO shift

    # 1) Find original table_start for each scene from patterns
    scene_table_starts = []  # one per scene, in the same order as PATTERN_PRECEDING_SCENES
    for scene_id, pattern in enumerate(PATTERN_PRECEDING_SCENES):
        pattern_pos = data.find(pattern)
        if pattern_pos == -1:
            raise ValueError(f"Can't find pattern for scene {scene_id+1}: {pattern!r}")
        table_start = pattern_pos + len(pattern) + 1
        scene_table_starts.append(table_start)

    # 2) Recompute original chunk_end for each scene by scanning pointers + texts
    pointer_tables_pos = find_pointer_tables_pos(data)
    scene_chunk_ends = []

    for scene_id, pos in enumerate(pointer_tables_pos):
        table_start = pos['start']
        table_end = pos['end']  # original first text address

        pointer_entries = load_pointer_table(table_start, table_end, data)

        last_text_end = table_end  # fallback
        for entry in pointer_entries:
            base = entry['base']
            ptr = entry['pointer']
            text_start = base + ptr
            text_end = data.find(b'\x00', text_start)
            if text_end == -1:
                raise ValueError(
                    f"Could not find string terminator for scene {scene_id+1} at {hex(text_start)}"
                )
            if text_end + 1 > last_text_end:
                last_text_end = text_end + 1  # include terminator

        scene_chunk_ends.append(last_text_end)

    # 3) Pour chaque scène, trouver dans le footer :
    #    - start_idx : entrée == table_start
    #    - size_idx  : entrée == (chunk_end - table_start)
    #    - end_idx   : entrée == chunk_end
    scene_start_idx = {}
    scene_size_idx = {}
    scene_end_idx = {}

    for scene_id, table_start in enumerate(scene_table_starts):
        chunk_end = scene_chunk_ends[scene_id]
        size = chunk_end - table_start

        # start index
        try:
            start_idx = offsets.index(table_start)
            scene_start_idx[scene_id] = start_idx
        except ValueError:
            pass

        # size index
        try:
            size_idx = offsets.index(size)
            scene_size_idx[scene_id] = size_idx
        except ValueError:
            pass

        # end index
        try:
            end_idx = offsets.index(chunk_end)
            scene_end_idx[scene_id] = end_idx
        except ValueError:
            pass

    footer = {
        "start": footer_start,
        "raw": raw_entries,
        "offsets": offsets,
        "scene_table_starts": scene_table_starts,
        "scene_start_idx": scene_start_idx,
        "scene_size_idx": scene_size_idx,
        "scene_end_idx": scene_end_idx,
    }


def rebuild_footer() -> bytes:
    global footer, scene_layout

    offsets = footer["offsets"][:]  # copy current footer values

    scene_start_idx = footer["scene_start_idx"]
    scene_size_idx = footer["scene_size_idx"]
    scene_end_idx = footer["scene_end_idx"]

    for info in scene_layout:
        scene_id = info["scene_id"]
        new_table_start = info["table_start"]
        new_end_offset = info["end_offset"]
        new_size = new_end_offset - new_table_start

        start_idx = scene_start_idx.get(scene_id)
        size_idx = scene_size_idx.get(scene_id)
        end_idx = scene_end_idx.get(scene_id)

        # Update table start for this scene
        if start_idx is not None:
            offsets[start_idx] = new_table_start

        # Update end offset for this scene
        if end_idx is not None:
            offsets[end_idx] = new_end_offset

        # Keep size = end - start consistent
        if size_idx is not None:
            offsets[size_idx] = new_size

    # Re-encode footer: each entry is a plain 32-bit little-endian value
    new_footer = b"".join(off.to_bytes(4, byteorder="little") for off in offsets)
    return new_footer

def create_tabs():
    for tab in tabs:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=tab['name'])
        frames.append(frame)
    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

def on_tab_changed(event):
    index = notebook.index(notebook.select())
    tab = tabs[index]
    render_tab(tab['id'])

def render_tab(tab_id):
    for widget in scrollable_frame.winfo_children():
        widget.destroy()
    entries.clear()

    for field in fields:
        if field['chapter'] == tab_id:
            frame_line = tk.Frame(scrollable_frame)
            frame_line.pack(padx=10, pady=5, fill="x")
            label = tk.Label(frame_line, text=field['idx'], width=15, anchor="w")
            label.pack(side="left")
            # Use a StringVar so we can trace changes
            var = tk.StringVar(value=field['text'])
            # Keep a reference on the field, so trace can update it
            field['var'] = var  # keep var alive and tied to this field

            def on_var_change(*args, f=field, v=var):
                # Update field text whenever the entry content changes
                f['text'] = v.get()

            # Trace content changes in the StringVar
            var.trace_add("write", on_var_change)
            entry = tk.Entry(frame_line, textvariable=var)
            entry.pack(side="left", fill="x", expand=True)
            entries.append(entry)

def get_pointer(value: bytes):
    return int.from_bytes(value, byteorder="little")

def get_text_pos(table_address: int, pointer_address: int):
    return table_address + pointer_address

def find_pointer_tables_pos(data: bytes):
    pos = []
    i = 1
    for pattern in PATTERN_PRECEDING_SCENES:
        pattern_pos = data.find(pattern)
        if pattern_pos == -1:
            print(f"Scene {i} ({pattern!r}) is missing!!! Abort.")
            return []

        # juste après "Scene\0"
        start_pos = pattern_pos + len(pattern) + 1

        first_ptr = get_pointer(data[start_pos:start_pos + 2])
        end_pos = get_text_pos(start_pos, first_ptr)

        pos.append({'start': start_pos, 'end': end_pos})
        print(f"Scene {i} pointer table found! ({hex(start_pos)}-{hex(end_pos)})")
        i += 1
    return pos

def find_strings(base: int, pointer: int, data: bytes):
    start = base + pointer
    end = data.find(b'\x00', start)
    print(hex(start))
    print(hex(end))
    print(data[start:end])
    return data[start:end].decode(ENCODING)

def load_pointer_table(table_start: int, table_end: int, data: bytes):
    """
    Read the whole 16-bit pointer table [table_start..table_end) and resolve
    pointers across multiple 64K banks.

    Bank change rule:
      - whenever a pointer value decreases (wrap), we consider it moved to next bank.
      - base for bank N is: table_start + N*0x10000
    """
    pointers = []
    offset = table_start
    prev = None
    bank = 0

    while offset + 2 <= table_end:
        value = get_pointer(data[offset:offset + 2])

        # Detect wrap -> next bank
        if prev is not None and value < prev:
            bank += 1

        base = table_start + bank * 0x10000
        pointers.append({'base': base, 'pointer': value})

        prev = value
        offset += 2

    return pointers

def on_open():
    global fields, res_path

    res_path = filedialog.askopenfilename(
        title="Open SCRIPTS.RES",
        filetypes=(
            ("Script RES", "*.RES"),
            ("All files", "*.*"),
        )
    )

    if not res_path:
        return # cancel

    try:
        data = Path(res_path).read_bytes()
        load_footer(data)
        pointer_tables_pos = find_pointer_tables_pos(data)
        idx = 0
        chapter = 1
        for pos in pointer_tables_pos:
            # pos['start'] is the actual pointer table start for this scene
            table_start = pos['start']
            local_idx = 0

            pointer_entries = load_pointer_table(table_start, pos['end'], data)
            for entry in pointer_entries:
                base = entry['base']      # base used to resolve this pointer (e.g. table_start, table_start+0x10000, ...)
                pointer = entry['pointer']

                text = find_strings(base, pointer, data)

                # Bank index: how many 0x10000 segments above the table_start is this base?
                bank = (base - table_start) // 0x10000

                fields.append({
                    'chapter': chapter,
                    'idx': idx,                 # global index (for display)
                    'local_idx': local_idx,     # pointer index inside this chapter
                    'text': text,
                    'orig_table_start': table_start,
                    'orig_pointer': pointer,
                    'bank': bank
                })
                idx += 1
                local_idx += 1
            chapter += 1

        if not fields:
            messagebox.showwarning("Empty file", "Invalid file")
            return

        create_tabs()
        render_tab(1)
        open_button.config(state=tk.DISABLED)
        export_button.config(state=tk.NORMAL)
        csv_export_button.config(state=tk.NORMAL)
        csv_import_button.config(state=tk.NORMAL)
    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Error", f"Can't read:\n{e}")

def on_export():
    import shutil
    from pathlib import Path

    global scene_layout

    if not res_path:
        return

    # Backup original file
    shutil.copy2(res_path, res_path + ".bak")

    original_data = Path(res_path).read_bytes()
    data = original_data

    # New buffer we'll build from scratch
    buf = bytearray()
    scene_layout = []

    # Where the original footer starts
    footer_start = footer["start"]

    # Recompute original pointer table bounds for each scene
    pointer_tables_pos = find_pointer_tables_pos(data)
    cursor_src = 0  # current position in original_data
    for scene_id, pos in enumerate(pointer_tables_pos):
        chapter_id = scene_id + 1

        table_start_orig = pos['start']
        table_end_orig = pos['end']  # original first text address for this scene

        # 1) Copy everything from last cursor to the start of this pointer table
        buf.extend(data[cursor_src:table_start_orig])

        # 2) Compute original chunk end by scanning all pointers of this scene
        pointer_entries = load_pointer_table(table_start_orig, table_end_orig, data)

        last_text_end = table_end_orig  # fallback
        for entry in pointer_entries:
            base = entry['base']
            ptr = entry['pointer']
            text_start = base + ptr
            text_end = data.find(b'\x00', text_start)
            if text_end == -1:
                raise ValueError(
                    f"Could not find string terminator for scene {scene_id+1} at {hex(text_start)}"
                )
            if text_end + 1 > last_text_end:
                last_text_end = text_end + 1  # include the 0x00

        chunk_end_orig = last_text_end

        # 3) Build new block [pointer table + texts] for this scene
        chapter_fields = [f for f in fields if f['chapter'] == chapter_id]

        if not chapter_fields:
            # No fields for this scene, copy original block as-is
            new_table_start = len(buf)
            buf.extend(data[table_start_orig:chunk_end_orig])
            end_offset_new = len(buf)
        else:
            # Sort by local_idx to preserve original pointer order
            chapter_fields_sorted = sorted(chapter_fields, key=lambda f: f['local_idx'])

            pointer_count = len(chapter_fields_sorted)
            pointer_table_size = pointer_count * 2  # 2 bytes per pointer

            # New table start in the rebuilt buffer
            new_table_start = len(buf)
            pointer_table_start = new_table_start

            # Reserve space for the pointer table (we'll fill it later)
            buf.extend(b"\x00" * pointer_table_size)

            # Cursor where we start writing texts
            cursor = pointer_table_start + pointer_table_size

            # For each field (pointer) in order, write its text using a MIN bank (original one),
            # and auto-promote to the next bank whenever the 16-bit pointer would overflow.
            for field in chapter_fields_sorted:
                bank_min = int(field.get("bank", 0) or 0)
                text_bytes = field["text"].encode(ENCODING) + b"\x00"

                # Safety: a single string must fit inside one 64K bank (optional but strongly recommended)
                if len(text_bytes) > 0x10000:
                    raise ValueError(
                        f"Scene {chapter_id}: field #{field['idx']} is too long ({len(text_bytes)} bytes) "
                        "to fit inside a single 64K bank."
                    )

                bank = bank_min
                while True:
                    bank_base = pointer_table_start + bank * 0x10000

                    # Ensure the write cursor is inside (or at start of) that bank
                    if cursor < bank_base:
                        buf.extend(b"\x00" * (bank_base - cursor))
                        cursor = bank_base

                    pointer16 = cursor - bank_base  # offset inside current bank (0..0xFFFF)

                    # If pointer doesn't fit OR the string would cross bank boundary -> promote bank
                    if pointer16 <= 0xFFFF and (pointer16 + len(text_bytes)) <= 0x10000:
                        break

                    bank += 1

                # (Optional debug) keep the effective bank used for this export
                field["_export_bank"] = bank

                # Write text bytes at current cursor
                if cursor == len(buf):
                    buf.extend(text_bytes)
                else:
                    if cursor > len(buf):
                        buf.extend(b"\x00" * (cursor - len(buf)))
                    buf[cursor:cursor + len(text_bytes)] = text_bytes

                # Write pointer16 into the reserved pointer table slot
                ptr_index = field["local_idx"]
                ptr_loc = pointer_table_start + ptr_index * 2
                buf[ptr_loc:ptr_loc + 2] = pointer16.to_bytes(2, "little")

                cursor += len(text_bytes)

            end_offset_new = len(buf)

        # 4) Register new layout for this scene (for footer rebuild)
        scene_layout.append({
            "scene_id": scene_id,
            "table_start": new_table_start,
            "end_offset": end_offset_new,
        })

        # 5) Move source cursor to the end of this scene's old block
        cursor_src = chunk_end_orig

    # ------------------------------------------------------------------
    # 2) Copy everything after the last scene up to the original footer
    # ------------------------------------------------------------------
    buf.extend(data[cursor_src:footer_start])

    # ------------------------------------------------------------------
    # 3) Rebuild and append new footer
    # ------------------------------------------------------------------
    new_footer = rebuild_footer()
    buf.extend(new_footer)

    # ------------------------------------------------------------------
    # 4) Save the new file
    # ------------------------------------------------------------------
    export_path = filedialog.asksaveasfilename(
        title="Export SCRIPTS.RES",
        initialfile=Path(res_path).name,
        defaultextension=".RES",
        filetypes=(
            ("Script RES", "*.RES"),
            ("All files", "*.*"),
        ),
    )

    if not export_path:
        return

    Path(export_path).write_bytes(buf)
    messagebox.showinfo("Export", "File exported successfully.")


def _current_tab_id():
    """Return the currently selected chapter id (1..8)."""
    try:
        index = notebook.index(notebook.select())
        return tabs[index]['id']
    except Exception:
        return 1

def csv_export():
    """Export all fields['text'] to a CSV file, 1 line per field, no header."""
    if not fields:
        messagebox.showwarning("CSV export", "Open a SCRIPTS.RES first.")
        return

    export_path = filedialog.asksaveasfilename(
        title="Export CSV",
        defaultextension=".csv",
        filetypes=(
            ("CSV", "*.csv"),
            ("All files", "*.*"),
        ),
    )
    if not export_path:
        return

    # Write one field per row, in the current 'fields' order (global order).
    with open(export_path, "w", encoding="utf-8", newline="\n") as f:
        for field in fields:
            f.write((field.get("text", "") or "").replace("\r\n", "\n").replace("\r", "\n"))
            f.write("\n")

    messagebox.showinfo("CSV export", "CSV exported successfully.")

def csv_import():
    """Import all fields['text'] from a CSV file (1 line -> 1 field), then refresh current tab."""
    if not fields:
        messagebox.showwarning("CSV import", "Open a SCRIPTS.RES first.")
        return

    import_path = filedialog.askopenfilename(
        title="Import CSV",
        filetypes=(
            ("CSV", "*.csv"),
            ("All files", "*.*"),
        ),
    )
    if not import_path:
        return

    # Try UTF-8 first (common for translators), then cp1252 fallback (Windows/legacy).
    values = None
    for enc in ("utf-8-sig", "utf-8", ENCODING):
        try:
            with open(import_path, "r", encoding=enc, newline="") as f:
                values = f.read().splitlines()
            break
        except UnicodeDecodeError:
            continue

    if values is None:
        messagebox.showerror("CSV import", "Unable to read the CSV with UTF-8 or cp1252.")
        return

    if len(values) != len(fields):
        messagebox.showerror(
            "CSV import",
            f"Line count mismatch: CSV has {len(values)} lines, but the project has {len(fields)} fields.\n"
            "Import aborted."
        )
        return

    # Apply imported texts in order, regardless of active tab.
    for i, v in enumerate(values):
        fields[i]["text"] = v
        # If this field currently has a StringVar (rendered in UI at least once), keep it in sync.
        if isinstance(fields[i].get("var"), tk.StringVar):
            fields[i]["var"].set(v)

    # Refresh the currently visible tab to reflect imported changes.
    render_tab(_current_tab_id())

    messagebox.showinfo("CSV import", "CSV imported successfully.")


# Buttons frame
buttons_frame = tk.Frame(root)
buttons_frame.pack(side="bottom", pady=10, fill="x")

open_button = tk.Button(buttons_frame, text="Open", command=on_open)
open_button.pack(side="left", padx=10)

csv_import_button = tk.Button(buttons_frame, text="CSV Import", command=csv_import, state=tk.DISABLED)
csv_import_button.pack(side="left", padx=10)

csv_export_button = tk.Button(buttons_frame, text="CSV Export", command=csv_export, state=tk.DISABLED)
csv_export_button.pack(side="left", padx=10)

export_button = tk.Button(buttons_frame, text="Export", command=on_export, state=tk.DISABLED)
export_button.pack(side="right", padx=10)

root.mainloop()
