
import pandas as pd
from datetime import datetime
import tensorflow as tf
import pickle
import wandb
import string
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
import re
import os
import requests
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def custom_standardization(sentence):
    sample = tf.strings.lower(sentence)
    sample = tf.strings.regex_replace(sample, '\W', ' ')
    sample = tf.strings.regex_replace(sample, '\d', ' ')
    return tf.strings.regex_replace(sample,
                         '[%s]'%re.escape(string.punctuation), '')

max_features = 10000
sequence_length = 250

vectorize_layer = tf.keras.layers.TextVectorization(
                        standardize=custom_standardization,
                        split='whitespace',
                        max_tokens=max_features,
                        output_mode='int',
                        output_sequence_length=sequence_length,
                        encoding='utf-8')

def load_data(**context):
    #load data from csv
    data_frame = pd.read_csv('https://filedn.eu/lxdTmTrxaGWQdk0ko1ihbxp/DataSet/Toxic/train.csv')
    data_frame = data_frame.sample(n=30)
    context["ti"].xcom_push(key="data_frame", value=data_frame)

def train_model(**context):
    data_frame = context["ti"].xcom_pull(key="data_frame")
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run = wandb.init(name=current_time+'_model', project='MichelLaPolice', entity='0xasept')

    url = 'https://filedn.eu/lxdTmTrxaGWQdk0ko1ihbxp/Model/vectorize_layer_vocab.pkl'

    response = requests.get(url)

    if response.status_code == 200:
        vocab = pickle.loads(response.content)
        # utilisez le vocabulaire chargé ici
    else:
        # gestion de l'erreur en cas d'échec de téléchargement du fichier
        print("Erreur lors du téléchargement du fichier.")
    print("vocab loaded")
    #train vectorize layer
    vectorize_layer.set_vocabulary(vocab)
    print("vectorize loaded")
    #split data
    x = data_frame.iloc[:,1:2]
    X = vectorize_layer(x).numpy()
    y = np.array(data_frame.iloc[:,2:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    print("train splited")
    artifact_toxic = run.use_artifact('0xasept/MichelLaPolice/Matraque:latest', type='model')
    artifact_dir_toxic = artifact_toxic.download()
    toxic_path = os.path.join(artifact_dir_toxic, 'model.h5')
    print("toxic is downloaded")
    print (toxic_path)
    print("Before loading model")
    with tf.device('/cpu:0'):
        model = load_model(toxic_path)
    print("After loading model")
    #train model an log it on wandb
    print("model trained")
    model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))
    print ("model is fit")
    history = model.history.history
    run.log({'history': history})
    
    #evaluate model and log it on wandb

    run.log({'model_score': model.evaluate(X_test, y_test)})
    model_score = model.evaluate(X_test, y_test)
    model.save(toxic_path)
    new_model_artifact = wandb.Artifact('Matraque', type='model', metadata={'name': 'Matraque'})
    new_model_artifact.add_file(toxic_path)
    run.log_artifact(new_model_artifact)

    context["ti"].xcom_push(key="model", value=model)
    context["ti"].xcom_push(key="run", value=run)
    context["ti"].xcom_push(key="artifact", value=new_model_artifact)
    context["ti"].xcom_push(key="model_score", value=model_score)
    context["ti"].xcom_push(key="toxic_path", value=toxic_path)
    pass
   
    
def validate_model(**context):
    threshold = 0.5
    model_score = context['ti'].xcom_pull(task_ids='train_model', key='model_score')
    
    if model_score >= threshold:
        return 'push_model'
    else:
        return 'do_not_validate_model'


def push_model(**context):
    #push model to wandb
    run = context["ti"].xcom_pull(task_ids='train_model', key="run")
    artifact = context["ti"].xcom_pull(task_ids='train_model', key="artifact")
    model = context['ti'].xcom_pull(task_ids='train_model', key='model')
    toxic_path = context['ti'].xcom_pull(task_ids='train_model', key='toxic_path')
    model.save(toxic_path)
    run.log_artifact(artifact, aliases=['PROD'])
    run.finish()
    context['ti'].xcom_push(key='model', value=model)

    print ("Model is valid and deployed")

def do_not_validate_model(**context):
    run = context["ti"].xcom_pull(task_ids='train_model',   key="run")
    print("Model is not valid")
    run.finish()