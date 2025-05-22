import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorSlicer
from pyspark.ml import Pipeline

def process_gold_table_labels(snapshot_date_str, silver_directory, gold_directory, spark, dpd, mob):
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to silver table
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name

    # Skip if file does not exist
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found for snapshot {snapshot_date_str}: {filepath}")
        return None

    # Read parquet
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # Get customer at 'mob'
    df = df.filter(col("mob") == mob)

    # Get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # Select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # Save gold table
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)
    
    return df

def process_gold_table_features(silver_attributes_directory, silver_financials_directory, gold_directory, spark):
    # Connect to silver tables
    attributes_path = silver_attributes_directory + "silver_features_attributes.parquet"
    financials_path = silver_financials_directory + "silver_features_financials.parquet"
    attributes_df = spark.read.parquet(attributes_path)
    financials_df = spark.read.parquet(financials_path)

    # Join on Customer_ID
    df = attributes_df.join(financials_df, on="Customer_ID", how="inner")

    # Select and cast features based on their relevance to loan defaults and the presence of minimal or manageable outliers
    df = df.select(
        col("Customer_ID").cast(StringType()),
        col("Age").cast(IntegerType()),
        col("Occupation").cast(StringType()),
        col("Monthly_Inhand_Salary").cast(FloatType()),
        col("Num_of_Loan").cast(IntegerType()),
        col("Delay_from_due_date").cast(IntegerType()),
        col("Credit_Mix").cast(StringType()),
        col("Outstanding_Debt").cast(FloatType()),
        col("Credit_Utilization_Ratio").cast(FloatType()),
        col("Credit_History_Total_Months").alias("Credit_History_Age").cast(IntegerType()),
        col("Payment_of_Min_Amount").cast(StringType()),
        col("Total_EMI_per_month").cast(FloatType()),
        col("Amount_invested_monthly").cast(FloatType()),
        col("Payment_Behaviour").cast(StringType()),
        col("Num_Bank_Accounts").cast(IntegerType()),
        col("Num_Credit_Card").cast(IntegerType()),
        col("Interest_Rate").cast(IntegerType()),
        col("Num_of_Delayed_Payment").cast(IntegerType()),
        col("Num_Credit_Inquiries").cast(IntegerType())
    )

    # Convert to Pandas to make one-hot encoding process more straightforward
    pdf = df.toPandas()

    # One-hot encode and drop 'Unemployed'/'Unknown' as base categories
    categorical_cols = {
        "Occupation": "Unemployed",
        "Credit_Mix": "Unknown",
        "Payment_of_Min_Amount": "Unknown",
        "Payment_Behaviour": "Unknown"
    }

    for col_name, base_category in categorical_cols.items():
        dummies = pd.get_dummies(pdf[col_name], prefix=col_name)
        if f"{col_name}_{base_category}" in dummies.columns:
            dummies.drop(f"{col_name}_{base_category}", axis=1, inplace=True)
        pdf = pd.concat([pdf.drop(columns=[col_name]), dummies], axis=1)

    # List of boolean columns to convert to int
    bool_cols = [
        'Occupation_Accountant', 'Occupation_Architect', 'Occupation_Developer',
        'Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Entrepreneur',
        'Occupation_Journalist', 'Occupation_Lawyer', 'Occupation_Manager',
        'Occupation_Mechanic', 'Occupation_Media_Manager', 'Occupation_Musician',
        'Occupation_Scientist', 'Occupation_Teacher', 'Occupation_Writer',
        'Credit_Mix_Bad', 'Credit_Mix_Good', 'Credit_Mix_Standard',
        'Payment_of_Min_Amount_No', 'Payment_of_Min_Amount_Yes',
        'Payment_Behaviour_High_spent_Large_value_payments',
        'Payment_Behaviour_High_spent_Medium_value_payments',
        'Payment_Behaviour_High_spent_Small_value_payments',
        'Payment_Behaviour_Low_spent_Large_value_payments',
        'Payment_Behaviour_Low_spent_Medium_value_payments',
        'Payment_Behaviour_Low_spent_Small_value_payments'
    ]
    
    # Convert to integer
    pdf[bool_cols] = pdf[bool_cols].astype(int)
    
    # Convert back to PySpark DataFrame
    final_df = spark.createDataFrame(pdf)

    # Save as parquet
    output_name = "gold_feature_store.parquet"
    output_path = gold_directory + output_name
    final_df.write.mode("overwrite").parquet(output_path)
    print("Saved to:", output_path)

    return final_df