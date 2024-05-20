# Utilisez une image de base appropriée
FROM tensorflow/tensorflow:2.3.0-custom-op-gpu-ubuntu16

# Copiez les fichiers du projet dans le conteneur
COPY . /app

# Déplacez-vous dans le répertoire du projet
WORKDIR /app/custom-ops

# Exécutez les commandes nécessaires
RUN ./configure.sh
RUN bazel build build_pip_pkg
RUN bazel-bin/build_pip_pkg artifacts

# Spécifiez le répertoire de sortie pour les fichiers générés
WORKDIR /app/custom-ops/bazel-bin

# Exposez le répertoire de sortie pour la récupération du .whl généré
VOLUME /app/custom-ops/bazel-bin

# Définissez la commande par défaut pour exécuter dans le conteneur
CMD ["echo", "Les fichiers générés se trouvent dans /app/custom-ops/bazel-bin"]