try:
    import os
    from datetime import datetime
    from datetime import timedelta

    import numpy as np
    import pandas as pd
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator

    from evidently import ColumnMapping
    from evidently.test_suite import TestSuite
    from evidently.tests import *

except Exception as e:
    print("Error  {} ".format(e))

dir_path = "reports"
file_path = "data_quality_test_suite.html"


def load_data_execute(**context):
    data_frame = pd.read_csv('https://filedn.eu/lxdTmTrxaGWQdk0ko1ihbxp/DataSet/Toxic/train.csv')
    context["ti"].xcom_push(key="data_frame", value=data_frame)

    # target and prediction
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # reference and current data
    reference_data = data_frame[target_cols]
    current_data = data_frame[target_cols]

    context["ti"].xcom_push(key="reference", value=reference_data)
    context["ti"].xcom_push(key="current", value=current_data)


def data_quality_tests_execute(**context):
    data_quality_suite = TestSuite(
        tests=[
            TestShareOfDriftedColumns(),
            TestHighlyCorrelatedColumns(),
            TestNumberOfColumns(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestTargetFeaturesCorrelations(),
        ]
    )

    reference_data = context.get("ti").xcom_pull(key="reference")
    current_data = context.get("ti").xcom_pull(key="current")

    data_quality_suite.run(reference_data=reference_data, current_data=current_data)
    suite_results = data_quality_suite.as_dict()
    if not suite_results["summary"]["all_passed"]:
        context["ti"].xcom_push(key="test_suite", value=data_quality_suite)
        return "create_html"


def test_suite_html_execute(**context):
    reference_data = context.get("ti").xcom_pull(key="reference")
    current_data = context.get("ti").xcom_pull(key="current")

    data_quality_suite = context.get("ti").xcom_pull(key="test_suite")
    data_quality_suite.run(reference_data=reference_data, current_data=current_data)

    try:
        os.mkdir(dir_path)
    except OSError:
        print("Creation of the directory {} failed".format(dir_path))

    data_quality_suite.save_html(os.path.join(dir_path, file_path))


default_args = {
    "start_date": datetime(2022, 7, 10),
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="evidently_test_suite",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    tags=["evidently"]
) as dag:

    load_data = PythonOperator(
        task_id="load_data",
        python_callable=load_data_execute,
        provide_context=True,
    )

    data_quality_tests = PythonOperator(
        task_id="data_quality_tests",
        python_callable=data_quality_tests_execute,
        provide_context=True,
    )

    test_suite_html = PythonOperator(
        task_id="test_suite_html", provide_context=True, python_callable=test_suite_html_execute
    )

load_data >> data_quality_tests >> [test_suite_html]
