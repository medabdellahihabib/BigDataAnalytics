# Base image Spark
FROM apache/spark:3.5.1

# Exposer les ports pour Spark UI et master/worker
EXPOSE 4040
EXPOSE 8080
EXPOSE 7077

# Installer numpy
USER root
RUN pip install --no-cache-dir numpy

# Revenir à l'utilisateur spark (optionnel, sécurité)
USER spark

# Lancer bash par défaut
CMD ["/bin/bash"]
