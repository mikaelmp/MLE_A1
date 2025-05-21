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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table_partition(snapshot_date_str, csv_file_path, bronze_directory, spark):
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # load data and filter
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    df = df.filter(col('snapshot_date') == snapshot_date)
    row_count = df.count()
    print(snapshot_date_str + ' row count:', row_count)

    if row_count == 0:
        print('No data for snapshot_date:', snapshot_date_str)
        return df
    
    # Build file name
    base_filename = os.path.basename(csv_file_path)
    no_ext_filename = os.path.splitext(base_filename)[0]
    partition_name = "bronze_" + no_ext_filename + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name

    # Save bronze table to datamart
    df.toPandas().to_csv(filepath, index=False)
    print('Saved to:', filepath)

    return df

def process_bronze_table_simple(csv_file_path, bronze_directory, spark):
    # load data and filter
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Count rows
    row_count = df.count()
    print('Total row count:', row_count)

    # Build file name
    base_filename = os.path.basename(csv_file_path)
    no_ext_filename = os.path.splitext(base_filename)[0]
    partition_name = "bronze_" + no_ext_filename + '.csv'
    filepath = bronze_directory + partition_name

    # Save bronze table to datamart
    df.toPandas().to_csv(filepath, index=False)
    print('Saved to:', filepath)

    return df