# Databricks notebook source
import sklearn

print("scikit-learn version:", sklearn.__version__)


# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, model.predict(X_train))
signature

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema(
    [
        ColSpec("double", "sepal length (cm)"),
        ColSpec("double", "sepal width (cm)"),
        ColSpec("double", "petal length (cm)"),
        ColSpec("double", "petal width (cm)")
    ]
)
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
signature

# COMMAND ----------

mlflow.sklearn.log_model(model, "iris_sig", signature=signature)

# COMMAND ----------

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "iris_classifier")

# COMMAND ----------

import mlflow
logged_model = 'runs:/537801acbd254986991181acde3c57be/iris_sig'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

import pandas as pd

data = {
    'sepal length (cm)': [5.1, 4.9, 4.7, 4.6, 5.0],
    'sepal width (cm)': [3.5, 3.0, 3.2, 3.1, 3.6],
    'petal length (cm)': [1.4, 1.4, 1.3, 1.5, 1.4],
    'petal width (cm)': [0.2, 0.2, 0.2, 0.2, 0.2]
}

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct, col
import pandas as pd
logged_model = 'runs:/537801acbd254986991181acde3c57be/iris_sig'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

df = [[5.1, 3.5, 1.4, 0.2], [6.3, 2.9, 5.6, 1.8]]  # Example input data

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))

# COMMAND ----------


loaded_model = mlflow.sklearn.load_model(logged_model)


# COMMAND ----------

new_data = [[5.1, 3.5, 1.4, 0.2], [6.3, 2.9, 5.6, 1.8]]  # Example input data
# predictions = loaded_model.predict(new_data)
# print("Predictions:", predictions)


# COMMAND ----------


