# DeepFace-docker

## Lancer
    docker compose run --rm face

(la 1re fois : build de l'image + téléchargement des modèles dans ./weights, ça prend quelques minutes)

## Mode verify (2 photos -> pourcentage)
Vider `data/`, puis y mettre exactement 2 images :

    data/
      a.jpg
      b.jpg

## Mode find (photo témoin + dossier)
    data/
      temoin.jpg
      base/
        photo1.jpg
        photo2.png
        sous-dossier/...

Seules les photos dont le visage correspond (sous le seuil du modèle) sont listées.
DeepFace crée un fichier cache `representations_*.pkl` dans data/base/ ;
supprime-le si tu veux forcer la réindexation.

## Changer de modèle / détecteur
    MODEL=ArcFace DETECTOR=retinaface docker compose run --rm face

Modèles : VGG-Face (défaut), Facenet, Facenet512, ArcFace...
Détecteurs : opencv (défaut), retinaface, mtcnn, ssd...

## Note sur le pourcentage
C'est un score dérivé de la distance (distance = seuil -> 50 %), pas une probabilité.
Le verdict officiel est basé sur le seuil du modèle.
