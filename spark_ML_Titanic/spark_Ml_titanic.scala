import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder()
  .appName("TitanicAnalysis")
  .getOrCreate()

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("/opt/spark/data/titanic.csv")

println(s"Nombre de lignes : ${df.count()}")
println(s"Nombre de colonnes : ${df.columns.length}")

df.printSchema()

df.groupBy("sex").count().show()
df.groupBy("pclass", "survived").count().show()

// Comptage des valeurs nulles
df.select(df.columns.map(c => count(when(col(c).isNull, c)).alias(c)): _*).show()

// Suppression de certaines colonnes
val df_selected = df.drop("survived", "pclass", "sex", "age", "fare", "embarked")
df_selected.show(5)


// ================= CLEANING ET ENCODING =================

val df2 = df.select("survived", "pclass", "sex", "age", "fare", "embarked", "class")

// Remplir les valeurs manquantes dans age par moyenne
val ageMean = df2.select(mean("age")).first().getDouble(0)
val dfFilled = df2.na.fill(Map("age" -> ageMean))

// Encoder sexe
val dfSexEncoded = dfFilled.withColumn("sex_int",
  when(col("sex") === "male", 0)
    .when(col("sex") === "female", 1)
    .otherwise(3)
)

// Renommer class
val dfRenamed = dfSexEncoded.withColumnRenamed("class", "class_str")

// Encoder class
val dfClassEncoded = dfRenamed.withColumn("class_int",
  when(col("class_str") === "First", 1)
    .when(col("class_str") === "Second", 2)
    .when(col("class_str") === "Third", 3)
    .otherwise(4)
)

// Vérification
dfClassEncoded.select("class_str", "class_int", "sex", "sex_int", "age").show(10)

// Encoder embarked
val dfEmbarked = dfClassEncoded.withColumn("embarked_int",
  when(col("embarked") === "S", 1)
    .when(col("embarked") === "C", 2)
    .when(col("embarked") === "Q", 3)
    .otherwise(4)
)

// Supprimer colonnes inutiles
val dfFinal = dfEmbarked.drop("pclass", "sex", "embarked", "class_str")
dfFinal.show(5)


// ================= SPLIT TRAIN/TEST =================

val Array(trainData, testData) = dfFinal.randomSplit(Array(0.8, 0.2), seed = 42)


// ================= Vector Assembler =================

import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler()
  .setInputCols(Array("age", "fare", "sex_int", "class_int", "embarked_int"))
  .setOutputCol("features")

val trainAssembled = assembler.transform(trainData)
val testAssembled = assembler.transform(testData)


// ================= Logistic Regression =================

import org.apache.spark.ml.classification.LogisticRegression
val lr = new LogisticRegression()
  .setLabelCol("survived")
  .setFeaturesCol("features")

val lrModel = lr.fit(trainAssembled)
val lrPredictions = lrModel.transform(testAssembled)

lrPredictions.select("survived", "probability", "prediction").show(10)

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("survived")

val lrAuc = evaluator.evaluate(lrPredictions)
println(f"AUC Régression Logistique = $lrAuc%.4f")


// ================= Naive Bayes =================

import org.apache.spark.ml.classification.NaiveBayes
val nb = new NaiveBayes()
  .setLabelCol("survived")
  .setFeaturesCol("features")
  .setModelType("multinomial")

val nbModel = nb.fit(trainAssembled)
val nbPredictions = nbModel.transform(testAssembled)

nbPredictions.select("survived", "probability", "prediction").show(10)

val nbAuc = evaluator.evaluate(nbPredictions)
println(f"AUC Naive Bayes = $nbAuc%.4f")


// ================= SAVE MODELS =================

val lrModelPath = "/opt/spark/work-dir/logistic_regression_model"
val nbModelPath = "/opt/spark/work-dir/naive_bayes_model"

lrModel.write.overwrite().save(lrModelPath)
nbModel.write.overwrite().save(nbModelPath)


// ================= LOAD MODELS =================

import org.apache.spark.ml.classification.{ LogisticRegressionModel, NaiveBayesModel }

val lrLoaded = LogisticRegressionModel.load(lrModelPath)
val nbLoaded = NaiveBayesModel.load(nbModelPath)

val lrTestPred = lrLoaded.transform(testAssembled)
lrTestPred.select("survived", "probability", "prediction").show(10)

val nbTestPred = nbLoaded.transform(testAssembled)
nbTestPred.select("survived", "probability", "prediction").show(10)


// ================= Accuracy + RMSE =================

import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}

val accuracyEval = new MulticlassClassificationEvaluator()
  .setLabelCol("survived")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val rmseEval = new RegressionEvaluator()
  .setLabelCol("survived")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// LR Train/Test
val lrTrainPred = lrLoaded.transform(trainAssembled)
val lrTrainAcc = accuracyEval.evaluate(lrTrainPred)
val lrTrainRmse = rmseEval.evaluate(lrTrainPred)

val lrTestAcc = accuracyEval.evaluate(lrTestPred)
val lrTestRmse = rmseEval.evaluate(lrTestPred)

println("=== Régression Logistique ===")
println(f"Accuracy train: $lrTrainAcc%.4f, RMSE train: $lrTrainRmse%.4f")
println(f"Accuracy test : $lrTestAcc%.4f, RMSE test : $lrTestRmse%.4f")

// NB Train/Test
val nbTrainPred = nbLoaded.transform(trainAssembled)
val nbTrainAcc = accuracyEval.evaluate(nbTrainPred)

val nbTestAcc = accuracyEval.evaluate(nbTestPred)

println("=== Naive Bayes ===")
println(f"Accuracy train: $nbTrainAcc%.4f")
println(f"Accuracy test : $nbTestAcc%.4f")
