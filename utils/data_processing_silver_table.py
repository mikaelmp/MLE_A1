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

from pyspark.sql.functions import col, regexp_replace, ceil, split, size, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table_loan(snapshot_date_str, bronze_directory, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name

    # skip if file does not exist
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found for snapshot {snapshot_date_str}: {filepath}")
        return None

    # read csv file
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, 'row count:', df.count())
    
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)
    
    return df

def process_silver_table_clickstream(snapshot_date_str, bronze_directory, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name

    # skip if file does not exist
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found for snapshot {snapshot_date_str}: {filepath}")
        return None

    # read csv file
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # save silver table
    partition_name = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df

from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, IntegerType, DateType
import os
from datetime import datetime

def process_silver_table_attributes(bronze_directory, silver_directory, spark):
    # connect to bronze table
    file_name = "bronze_features_attributes.csv"
    filepath = bronze_directory + file_name

    # skip if file does not exist
    if not os.path.exists(filepath):
        print('[SKIP] File not found')
        return None

    # read csv file
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("Name", col("Name").cast(StringType()))

    # Clean Age: remove underscores, cast to int, flag impossible values
    df = df.withColumn("Age", col("Age").cast(StringType()))
    df = df.withColumn("Age", F.regexp_replace("Age", "_", ""))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age_Anomaly_Flag", when((col("Age") < 0) | (col("Age") > 100), 1).otherwise(0))

    # Clean SSN: format must be ###-##-####, flag missing or malformed ones
    ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
    df = df.withColumn("SSN", col("SSN").cast(StringType()))
    df = df.withColumn("SSN_Flag", when(F.col("SSN").rlike(ssn_pattern), 0).otherwise(1))
    df = df.withColumn("SSN", when(F.col("SSN_Flag") == 1, None).otherwise(F.col("SSN")))

    # Clean Occupation: replace '_______' with 'Unemployed'
    df = df.withColumn("Occupation", when(col("Occupation") == "_______", "Unemployed").otherwise(col("Occupation")))
    df = df.withColumn("Occupation", col("Occupation").cast(StringType()))

    # Format snapshot_date
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # write silver table
    partition_name = "silver_features_attributes.parquet"
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df

def process_silver_table_financials(bronze_directory, silver_directory, spark):
    file_name = "bronze_features_financials.csv"
    filepath = bronze_directory + file_name

    if not os.path.exists(filepath):
        print('[SKIP] File not found:', filepath)
        return None

    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, '| Row count:', df.count())

    # Customer_ID
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))

    # Annual_Income
    df = df.withColumn("Annual_Income", regexp_replace("Annual_Income", "_", "").cast(FloatType()))

    # Monthly_Inhand_Salary
    df = df.withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType()))

    # Num_Bank_Accounts - cap outliers at 11, largest value below IQR upper bound of 13
    df = df.withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast(IntegerType()))
    df = df.withColumn("Num_Bank_Accounts", when(col("Num_Bank_Accounts") < 0, 0)
                       .when(col("Num_Bank_Accounts") > 11, 11)
                       .otherwise(col("Num_Bank_Accounts")))

    # Num_Credit_Card - cap outliers at 11, largest value below IQR upper bound of 11.5
    df = df.withColumn("Num_Credit_Card", col("Num_Credit_Card").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Card", when(col("Num_Credit_Card") < 0, 0)
                       .when(col("Num_Credit_Card") > 11, 11)
                       .otherwise(col("Num_Credit_Card")))

    # Interest_Rate - cap outliers at 34, largest value below IQR upper bound of 38
    df = df.withColumn("Interest_Rate", col("Interest_Rate").cast(IntegerType()))
    df = df.withColumn("Interest_Rate", when(col("Interest_Rate") > 34, 34)
                       .otherwise(col("Interest_Rate")))

    # Delay from due date
    df = df.withColumn("Delay_from_due_date", col("Delay_from_due_date").cast(IntegerType()))

    # Num_of_Loan derived from Type_of_Loan
    df = df.withColumn("Num_of_Loan", when(col("Type_of_Loan").isNull() | (col("Type_of_Loan") == ""), 0)
                       .otherwise(size(split(col("Type_of_Loan"), ","))).cast(IntegerType()))

    # Num_of_Delayed_Payment - cap outliers at 28, largest value below IQR upper bound of 31.5
    df = df.withColumn("Num_of_Delayed_Payment", regexp_replace("Num_of_Delayed_Payment", "_", "").cast(IntegerType()))
    df = df.withColumn("Num_of_Delayed_Payment", when(col("Num_of_Delayed_Payment") > 28, 28)
                       .otherwise(col("Num_of_Delayed_Payment")))

    # Changed_Credit_Limit
    df = df.withColumn("Changed_Credit_Limit", when(col("Changed_Credit_Limit") == "_", "0")
                       .otherwise(col("Changed_Credit_Limit")).cast(FloatType()))

    # Num_Credit_Inquiries - cap outlier at 17, largest value below IQR upper bound of 19
    df = df.withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Inquiries", when(col("Num_Credit_Inquiries") > 17, 17)
                       .otherwise(col("Num_Credit_Inquiries")))

    # Credit_Mix
    df = df.withColumn("Credit_Mix", when(col("Credit_Mix") == "_", "Unknown").otherwise(col("Credit_Mix")).cast(StringType()))

    # Outstanding_Debt
    df = df.withColumn("Outstanding_Debt", regexp_replace("Outstanding_Debt", "_", "").cast(FloatType()))

    # Credit Utilization Ratio
    df = df.withColumn("Credit_Utilization_Ratio", col("Credit_Utilization_Ratio").cast(FloatType()))

    # Credit_History_Age → convert to total months
    df = df.withColumn("Years", regexp_replace(col("Credit_History_Age"), " Years.*", "").cast(IntegerType()))
    df = df.withColumn("Months", regexp_replace(regexp_replace(col("Credit_History_Age"), ".*and ", ""), " Months", "").cast(IntegerType()))
    df = df.withColumn("Credit_History_Total_Months", (col("Years") * 12 + col("Months")).cast(IntegerType()))

    # Payment_of_Min_Amount
    df = df.withColumn("Payment_of_Min_Amount", when(col("Payment_of_Min_Amount") == "NM", "Unknown")
                       .otherwise(col("Payment_of_Min_Amount")).cast(StringType()))

    # Total EMI per month
    df = df.withColumn("Total_EMI_per_month", col("Total_EMI_per_month").cast(FloatType()))

    # Amount Invested Monthly
    df = df.withColumn("Amount_invested_monthly", regexp_replace("Amount_invested_monthly", "_", "").cast(FloatType()))

    # Payment Behaviour
    df = df.withColumn("Payment_Behaviour", when(col("Payment_Behaviour") == "!@9#%8", "Unknown")
                       .otherwise(col("Payment_Behaviour")).cast(StringType()))

    # Monthly Balance
    df = df.withColumn("Monthly_Balance", regexp_replace("Monthly_Balance", "_", "").cast(FloatType()))

    # Snapshot Date
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # Save to Silver
    output_name = "silver_features_financials.parquet"
    output_path = silver_directory + output_name
    df.write.mode("overwrite").parquet(output_path)
    print("Saved to:", output_path)

    return df