Si vous avez un soucis avec l'installeur ou que vous voulez simplement appliquer les patchs vous même,
choisissez le format que vous préférez entre ZSTD ou XDELTA:

- zstd:
Vous aurez besoin du binaire zstd pour les appliquer, à l'aide de commandes:

zstd -d --patch-from=SCREAM.RES SCREAM.patch.zst -o SCREAM_FR.RES

Si vous voulez patcher une version pour y jouer avec ScummVM:
zstd -d --patch-from=SCRIPTS.RES SCRIPTS_SCUMMVM.patch.zst -o SCRIPTS_FR.RES
Sinon:
zstd -d --patch-from=SCRIPTS.RES SCRIPTS.patch.zst -o SCRIPTS_FR.RES

Si vous disposez d'un fichier S.prg ou S.EXE à patcher:
zstd -d --patch-from=S.prg Sprg.patch.zst -o S_FR.prg # mettre .EXE au lieu de .prg selon votre cas, c'est le même fichier derrière le nom

Puis de remplacer les fichiers SCREAM.RES et SCRIPTS.RES (et éventuellement votre S.prg / S.EXE) par les nouveaux fichiers *_FR.* en renommant/déplaçant/écrasant (gardez de préférence une copie de l'original quelque part)

- xdelta:
Vous pouvez utiliser n'importe quel patcheur compatible xdelta pour appliquer les patchs (des versions "online" et des versions avec GUI locale existes)

Note:
Le fichier SCREAM.RES est le même dans la plupart des éditions anglaises et devrait fonctionner.
Le fichier SCRIPTS.RES est patché de façon spécifique selon si vous voulez jouer via ScummVM ou via DOSBOX (ou via le menu de la "Definitive Edition" par Nightdive).
Dans le cas où votre version dispose d'un fichier S.EXE ou S.prg et que vous n'utilisez pas ScummVM, vous pouvez également lui appliquer son patch.
