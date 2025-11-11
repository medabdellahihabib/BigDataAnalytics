# ✅ Base image Spark
FROM apache/spark:3.5.1

# ✅ Exposer les ports importants
EXPOSE 4040   # Spark UI
EXPOSE 8080   # History / Web UI
EXPOSE 7077   # Spark master

# ✅ Installer Python deps + numpy
USER root
RUN apt-get update -y && apt-get install -y python3 python3-pip && \
    pip install --no-cache-dir numpy pandas matplotlib

# ✅ Créer répertoires pour les scripts et données
RUN mkdir -p /opt/spark/work-dir /opt/spark/data /opt/spark/models

# ✅ Copier ton dataset Titanic & scripts Spark
COPY titanic.csv /opt/spark/data/
COPY titanic.scala /opt/spark/work-dir/

# ✅ Remettre l’utilisateur spark
USER spark

# ✅ Commande par défaut → ouvre bash
CMD ["/bin/bash"]
