import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder()
  .appName("Titanic_ML_Updated")
  .getOrCreate()

// ================= LOAD DATA =================
val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("/opt/spark/data/titanic.csv")

println(s"Rows: ${df.count()} | Columns: ${df.columns.length}")
df.printSchema()

df.groupBy("sex").count().show()
df.groupBy("pclass", "survived").count().show()

df.select(df.columns.map(c => count(when(col(c).isNull, c)).alias(c)):_*).show()

// ================= CLEANING & ENCODING =================
val df2 = df.select("survived", "pclass", "sex", "age", "fare", "embarked", "class")

// Replace missing age with mean
val ageMean = df2.select(mean("age")).first().getDouble(0)
val dfFilled = df2.na.fill(Map("age" -> ageMean))

val dfEnc = dfFilled
  .withColumn("sex_int", when(col("sex") === "male", 0).otherwise(1))
  .withColumn("class_int",
    when(col("class") === "First", 1)
      .when(col("class") === "Second", 2)
      .when(col("class") === "Third", 3)
      .otherwise(4)
  )
  .withColumn("embarked_int",
    when(col("embarked") === "S", 1)
      .when(col("embarked") === "C", 2)
      .when(col("embarked") === "Q", 3)
      .otherwise(4)
  )
  .drop("pclass", "sex", "embarked", "class")

val dfFinal = dfEnc
dfFinal.show(5)

// ================= TRAIN/TEST SPLIT =================
val Array(trainData, testData) = dfFinal.randomSplit(Array(0.8, 0.2), seed = 42)

// ================= VECTOR ASSEMBLER =================
import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler()
  .setInputCols(Array("age", "fare", "sex_int", "class_int", "embarked_int"))
  .setOutputCol("features")

// ================= PIPELINE: Logistic Regression =================
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline

val lr = new LogisticRegression()
  .setLabelCol("survived")
  .setFeaturesCol("features")

val lrPipeline = new Pipeline().setStages(Array(assembler, lr))
val lrModel = lrPipeline.fit(trainData)

val lrPred = lrModel.transform(testData)
lrPred.select("survived", "probability", "prediction").show(10)

// ================= METRICS =================
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}

val aucEval = new BinaryClassificationEvaluator()
  .setLabelCol("survived")

val accEval = new MulticlassClassificationEvaluator()
  .setLabelCol("survived")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val f1Eval = new MulticlassClassificationEvaluator()
  .setLabelCol("survived")
  .setPredictionCol("prediction")
  .setMetricName("f1")

println("==== Logistic Regression ====")
println(f"AUC = ${aucEval.evaluate(lrPred)}%.4f")
println(f"Accuracy = ${accEval.evaluate(lrPred)}%.4f")
println(f"F1-score = ${f1Eval.evaluate(lrPred)}%.4f")

// ================= CONFUSION MATRIX =================
val cm = lrPred.groupBy("survived", "prediction").count()
println("=== Confusion Matrix ===")
cm.show()

// ================= RANDOM FOREST =================
import org.apache.spark.ml.classification.RandomForestClassifier

val rf = new RandomForestClassifier()
  .setLabelCol("survived")
  .setFeaturesCol("features")
  .setNumTrees(100)

val rfPipeline = new Pipeline().setStages(Array(assembler, rf))
val rfModel = rfPipeline.fit(trainData)

val rfPred = rfModel.transform(testData)

println("==== Random Forest ====")
println(f"AUC = ${aucEval.evaluate(rfPred)}%.4f")
println(f"Accuracy = ${accEval.evaluate(rfPred)}%.4f")
println(f"F1-score = ${f1Eval.evaluate(rfPred)}%.4f")

rfPred.groupBy("survived", "prediction").count().show()

// ================= DECISION TREE =================
import org.apache.spark.ml.classification.DecisionTreeClassifier

val dt = new DecisionTreeClassifier()
  .setLabelCol("survived")
  .setFeaturesCol("features")

val dtPipeline = new Pipeline().setStages(Array(assembler, dt))
val dtModel = dtPipeline.fit(trainData)

val dtPred = dtModel.transform(testData)

println("==== Decision Tree ====")
println(f"AUC = ${aucEval.evaluate(dtPred)}%.4f")
println(f"Accuracy = ${accEval.evaluate(dtPred)}%.4f")
println(f"F1-score = ${f1Eval.evaluate(dtPred)}%.4f")

dtPred.groupBy("survived", "prediction").count().show()

// ================= SAVE MODELS =================
lrModel.write.overwrite().save("/opt/spark/work-dir/log_reg_pipeline")
rfModel.write.overwrite().save("/opt/spark/work-dir/random_forest_pipeline")
dtModel.write.overwrite().save("/opt/spark/work-dir/decision_tree_pipeline")

println("✅ Models saved successfully!")

