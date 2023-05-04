from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import BranchPythonOperator
from script import load_data, train_model, validate_model, push_model, do_not_validate_model
from airflow.models import Variable
import wandb

wandb_api_key = Variable.get("WandB_API_KEY")
wandb.login(key=wandb_api_key)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'train_ml_pipeline',
    default_args=default_args,
    description='A pipeline to train a machine learning model and push metrics to WandB',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(2),
    tags=['machine-learning'],
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

validate_model_task = BranchPythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

do_not_validate_model = PythonOperator(
    task_id='do_not_validate_model',
    python_callable=do_not_validate_model,
    dag=dag
)

push_model_task = PythonOperator(
    task_id='push_model',
    python_callable=push_model,
    dag=dag,
)

load_data_task >> train_model_task >> validate_model_task
validate_model_task >> [do_not_validate_model, push_model_task]
