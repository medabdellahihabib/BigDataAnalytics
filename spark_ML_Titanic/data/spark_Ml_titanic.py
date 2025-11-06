from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("TitanicAnalysis").getOrCreate()
df = spark.read.csv("/opt/spark/data/titanic.csv", header=True, inferSchema=True)



print("Nombre de lignes :", df.count())
print("Nombre de colonnes :", len(df.columns))


df.printSchema()

df.groupBy("sex").count().show()
df.groupBy("pclass", "survived").count().show()



from pyspark.sql.functions import col, count, when

df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()



df_selected = df.drop("survived", "pclass", "sex", "age", "fare", "embarked")
df_selected.show(5)


from pyspark.sql.functions import when, col, mean

# 1️⃣ Sélectionner uniquement les colonnes importantes
df_selected = df.select("survived", "pclass", "sex", "age", "fare", "embarked", "class")

# 2️⃣ Remplir les valeurs manquantes dans 'age' par la moyenne
age_mean = df_selected.select(mean("age")).collect()[0][0]
df_filled = df_selected.na.fill({"age": age_mean})

# 3️⃣ Encoder 'sex' en entier
df_filled = df_filled.withColumn(
    "sex_int",
    when(col("sex") == "male", 0)
    .when(col("sex") == "female", 1)
    .otherwise(3)
)

# 4️⃣ Renommer 'class' pour éviter le mot réservé
df_filled = df_filled.withColumnRenamed("class", "class_str")

# 5️⃣ Encoder 'class' en entier
df_filled = df_filled.withColumn(
    "class_int",
    when(col("class_str") == "First", 1)
    .when(col("class_str") == "Second", 2)
    .when(col("class_str") == "Third", 3)
    .otherwise(4)
)

# 6️⃣ Vérifier
df_filled.select("class_str", "class_int", "sex", "sex_int", "age").show(10)



from pyspark.sql.functions import when, col

df_filled = df_filled.withColumn(
    "embarked_int",
    when(col("embarked") == "S", 1)
    .when(col("embarked") == "C", 2)
    .when(col("embarked") == "Q", 3)
    .otherwise(4)  # toutes les autres valeurs
)


df_final = df_filled.drop("pclass", "sex", "embarked", "class_str")
df_final.show(5)


train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.feature import VectorAssembler
feature_cols = ["age", "fare", "sex_int", "class_int", "embarked_int"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_data_assembled = assembler.transform(train_data)
test_data_assembled = assembler.transform(test_data)   












from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
lr = LogisticRegression(labelCol="survived", featuresCol="features")
lr_model = lr.fit(train_data_assembled)
lr_predictions = lr_model.transform(test_data_assembled)
lr_predictions.select("survived", "probability", "prediction").show(10)


evaluator = BinaryClassificationEvaluator(labelCol="survived")
lr_auc = evaluator.evaluate(lr_predictions)
print(f"AUC de la régression logistique: {lr_auc:.4f}")





from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(labelCol="survived", featuresCol="features", modelType="multinomial")
nb_model = nb.fit(train_data_assembled)
nb_predictions = nb_model.transform(test_data_assembled)
nb_predictions.select("survived", "probability", "prediction").show(10)
nb_auc = evaluator.evaluate(nb_predictions)
print(f"AUC de Naive Bayes: {nb_auc:.4f}")


lr_model_path = "/opt/spark/work-dir/logistic_regression_model"
nb_model_path = "/opt/spark/work-dir/naive_bayes_model"

lr_model.write().overwrite().save(lr_model_path)
nb_model.write().overwrite().save(nb_model_path)


from pyspark.ml.classification import LogisticRegressionModel, NaiveBayesModel
lr_model_path = "/opt/spark/work-dir/logistic_regression_model"
nb_model_path = "/opt/spark/work-dir/naive_bayes_model"
lr_loaded = LogisticRegressionModel.load(lr_model_path)
nb_loaded = NaiveBayesModel.load(nb_model_path)



# Prédictions avec la régression logistique
lr_test_predictions = lr_loaded.transform(test_data_assembled)
lr_test_predictions.select("survived", "probability", "prediction").show(10)

# Prédictions avec Naive Bayes
nb_test_predictions = nb_loaded.transform(test_data_assembled)
nb_test_predictions.select("survived", "probability", "prediction").show(10)



from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
# Accuracy pour classification
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="survived",
    predictionCol="prediction",
    metricName="accuracy"
)

# RMSE pour régression logistique
rmse_evaluator = RegressionEvaluator(
    labelCol="survived",
    predictionCol="prediction",
    metricName="rmse"
)



# Sur l'ensemble d'entraînement
lr_train_predictions = lr_loaded.transform(train_data_assembled)
lr_train_accuracy = accuracy_evaluator.evaluate(lr_train_predictions)
lr_train_rmse = rmse_evaluator.evaluate(lr_train_predictions)

# Sur l'ensemble de test
lr_test_predictions = lr_loaded.transform(test_data_assembled)
lr_test_accuracy = accuracy_evaluator.evaluate(lr_test_predictions)
lr_test_rmse = rmse_evaluator.evaluate(lr_test_predictions)

print("=== Régression Logistique ===")
print(f"Accuracy train: {lr_train_accuracy:.4f}, RMSE train: {lr_train_rmse:.4f}")
print(f"Accuracy test : {lr_test_accuracy:.4f}, RMSE test : {lr_test_rmse:.4f}")





# Sur l'ensemble d'entraînement
nb_train_predictions = nb_loaded.transform(train_data_assembled)
nb_train_accuracy = accuracy_evaluator.evaluate(nb_train_predictions)

# Sur l'ensemble de test
nb_test_predictions = nb_loaded.transform(test_data_assembled)
nb_test_accuracy = accuracy_evaluator.evaluate(nb_test_predictions)

print("=== Naive Bayes ===")
print(f"Accuracy train: {nb_train_accuracy:.4f}")
print(f"Accuracy test : {nb_test_accuracy:.4f}")













