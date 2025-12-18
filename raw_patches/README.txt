Si vous avez un soucis avec l'installeur ou que vous voulez simplement utiliser directement les patchs:
Ce sont des patch au format zstd, vous aurez besoin de zstd pour les appliquer, à l'aide de commandes:

zstd -d --patch-from=SCREAM.RES SCREAM.patch.zst -o SCREAM_FR.RES

Si vous voulez patcher une version pour y jouer avec ScummVM:
zstd -d --patch-from=SCRIPTS.RES SCRIPTS_SCUMMVM.patch.zst -o SCRIPTS_FR.RES
Sinon:
zstd -d --patch-from=SCRIPTS.RES SCRIPTS.patch.zst -o SCRIPTS_FR.RES

Si vous disposez d'un fichier S.prg ou S.EXE à patcher:
zstd -d --patch-from=S.prg Sprg.patch.zst -o S_FR.prg # mettre .EXE au lieu de .prg selon votre cas, c'est le même fichier derrière le nom

Puis de remplacer les fichiers SCREAM.RES et SCRIPTS.RES (et éventuellement votre S.prg / S.EXE) par les nouveaux fichiers *_FR.* en renommant/déplaçant/écrasant (gardez de préférence une copie de l'original quelque part)
