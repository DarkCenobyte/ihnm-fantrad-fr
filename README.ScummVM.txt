ScummVM nécessite qu'une variante d'un jeu soit connue de son code afin d'accepter de la lancer.

Si ce patch est suffisamment stable, je soumettrais à validation de ScummVM les informations sur la variante afin qu'il soit supporté directement, en attendant (si ce n'est toujours pas fait), vous pouvez quand même lancer le jeu en version ScummVM à condition de télécharger le code source de ScummVM, puis de rajouter vous-même une entrée dans ce fichier:
https://github.com/scummvm/scummvm/blob/master/engines/saga/detection_tables.h#L1436

Sur le modèle de la section concernant le patch allemand, mais en remplaçant les tailles (L. 1446 et 1447) à côté des hashes, puis en remplaçant le réglage de langue (L. 1452) par Common::FR_FRA .

