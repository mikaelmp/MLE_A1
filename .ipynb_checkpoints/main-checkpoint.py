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

from pyspark.sql.functions import col, regexp_replace, ceil, split, size, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorSlicer
from pyspark.ml import Pipeline

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# ===== Set Up PySpark Session =====

spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# ===== Set Up Config =====

snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2025-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# ===== Build Bronze Tables =====

# lms_loan_daily.csv
# create bronze datalake for loan dataset
bronze_loan_directory = "datamart/bronze/lms_loan_daily/"
loan_csv_file_path = "data/lms_loan_daily.csv"

if not os.path.exists(bronze_loan_directory):
    os.makedirs(bronze_loan_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table_partition(date_str, loan_csv_file_path, bronze_loan_directory, spark)

# feature_clickstream.csv
# create bronze datalake for clickstream csv
bronze_clickstream_directory = "datamart/bronze/feature_clickstream/"
clickstream_csv_file_path = "data/feature_clickstream.csv"

if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table_partition(date_str, clickstream_csv_file_path, bronze_clickstream_directory, spark)

# features_attributes.csv
# create bronze datalake for attributes csv
bronze_attributes_directory = "datamart/bronze/features_attributes/"
attributes_csv_file_path = "data/features_attributes.csv"

if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

# run bronze backfill
utils.data_processing_bronze_table.process_bronze_table_simple(attributes_csv_file_path, bronze_attributes_directory, spark)

# features_financials.csv
# create bronze datalake for financials csv
bronze_financials_directory = "datamart/bronze/features_financials/"
financials_csv_file_path = "data/features_financials.csv"

if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)

# run bronze backfill
utils.data_processing_bronze_table.process_bronze_table_simple(financials_csv_file_path, bronze_financials_directory, spark)

# ===== Build Silver Tables =====

# lms_loan_daily.csv
# create silver datalake for loan dataset
silver_loan_daily_directory = "datamart/silver/lms_loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table_loan(date_str, bronze_loan_directory, silver_loan_daily_directory, spark)

# feature_clickstream.csv
# create silver datalake for clickstream csv
silver_clickstream_directory = "datamart/silver/feature_clickstream/"

if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table_clickstream(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)

# features_attributes.csv
# create silver datalake for attributes csv
silver_attributes_directory = "datamart/silver/features_attributes/"

if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)

# run silver backfill
utils.data_processing_silver_table.process_silver_table_attributes(bronze_attributes_directory, silver_attributes_directory, spark)

# features_financials.csv
# create silver datalake for financials csv
silver_financials_directory = "datamart/silver/features_financials/"

if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

# run silver backfill
utils.data_processing_silver_table.process_silver_table_financials(bronze_financials_directory, silver_financials_directory, spark)

# ===== Build Gold Tables =====

# Label Store
# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_gold_table_labels(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

# Feature Store
# create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
utils.data_processing_gold_table.process_gold_table_features(silver_attributes_directory, silver_financials_directory, gold_feature_store_directory, spark)

# Inspect Label Store
folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()