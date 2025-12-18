Usages:
# import extracted/edited font:
python saga_res_font.py font-import -o SCREAM_11a.RES --indir SCREAM_OUT_FONT_FR SCREAM.RES SCREAM_OUT_FONT_FR.json

# import extracted/edited text:
python saga_res_text.py import -o SCREAM_11b.RES SCREAM_11a.RES SCREAM_OUT.json

# Scripts Translation:
scripts_translation.old.py => Was used for v1.0
scripts_translation.dedup.py => Used for v1.1 (fix the text glitch in the last chapter of the game... but kind of unstable if unlucky...)

# Call one of those scripts will make a GUI appear, you can then load SCRIPTS.RES files, edit them on the GUI (not recommended...) or export them in a CSV file (recommended...) then import the CSV file into the GUI before performing an export into a new RES file.

# /!\ WARNING /!\
# The last chapter of the game has more text than others... And this can cause glitches... There is 2 implementations of the game with 2 different way to handle this...
# ScummVM perform a loop during buffer overflow and this any pointer looking backward is a buffer overflow... This lead to missing texts and let the game lost in SCRIPTS.RES ...
# Dosbox/native S.EXE/S.prg meet also a weird behavior that CAN trigger a glitch, however the glitch seems fixable if you don't push the file too far... # However the binary allow you to deduplicate some texts and make pointers look backward!

# Solution: use script_translator.old.py for ScummVM version of a patch, and script_translator.dedup.py if needed for others if needed.
# You can share the CSV export/CSV import feature between both scripts !
