import zipfile
from _weakref import ref
import logging
import os
import json
import flask
import logging

from google.cloud import storage



from flask import request
from flask import send_from_directory
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import numpy as np
import io as io
from zipfile import ZipFile



from tensorflow.python.training import saver

'''import boto3'''
import pickle as pickle


##se necesita un /upload
##               /isModelUpdated
##              /getGlobalModel
# hay que usar una firebase de google. buckets,blob etc



app = flask.Flask(__name__)

#variables
numero = 1100
modelctr = 1100
numero2 = 2100
modelctr2 = 2100
numero3 = 3100
modelctr3 = 3100
numero4 = 4100
modelctr4 = 4100
modelctrstr = str(modelctr)

NumDescargasModelo_A = 0
NumDescargasModelo_B = 0
NumDescargasModelo_C = 0
NumDescargasModelo_D = 0

NOMBRE_CHECKPOINTS = "checkpoints_name_"+ str(modelctr)
NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
NOMBRE_CHECKPOINTS3 = "checkpoints_name3_"+ str(modelctr3)
NOMBRE_ZIP3 = "Zip_Name3_" + str(modelctr3)
NOMBRE_CHECKPOINTS2 = "checkpoints_name2_"+ str(modelctr2)
NOMBRE_ZIP2 = "Zip_Name2_" + str(modelctr2)
NOMBRE_CHECKPOINTS4 = "checkpoints_name4_"+ str(modelctr4)
NOMBRE_ZIP4 = "Zip_Name4_" + str(modelctr4)
nombre_fichero_descarga = ""
nombre_fichero_descarga1 = ""
nombre_fichero_descarga2 = ""
nombre_fichero_descarga3 = ""

NOMBRE_CHECKPOINTS3_2 = "checkpoints_name3_"+ str(modelctr3 -1)
NOMBRE_CHECKPOINTS3_3 = "checkpoints_name3_"+ str(modelctr3 -2)

NOMBRE_CHECKPOINTS2_2 = "checkpoints_name3_"+ str(modelctr2 -1)
NOMBRE_CHECKPOINTS2_3 = "checkpoints_name3_"+ str(modelctr2 -2)

NOMBRE_CHECKPOINTS1_2 = "checkpoints_name3_"+ str(modelctr -1)
NOMBRE_CHECKPOINTS1_3 = "checkpoints_name3_"+ str(modelctr -2)

NOMBRE_CHECKPOINTS4_2 = "checkpoints_name4_"+ str(modelctr4 -1)
NOMBRE_CHECKPOINTS4_3 = "checkpoints_name4_"+ str(modelctr4 -2)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


@app.route('/', methods=['GET'])

def home():
    return "Hello, this is the home page"


@app.route('/prueba', methods=['GET'])

def prueba():
    modelo()
    print("modelo")
    return "Hello, this is the prueba tab"

@app.route('/modelo', methods=['GET'])

def modelo():
    modelctr =contador + 1
    modeloSencillo()
    print("modelo sencillo")
    return "Hello, this is modelo sencillo"


@app.route('/crear_weight', methods=['GET'])

def crear_weight():

    averageWeights()
    print("se han usado los pesos y se ha crado un modleoa actualizado")
    return "se han usado los pesos y se ha crado un modleoa actualizado"

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]


@app.route("/upload2", methods = ['GET'])
def u():
    configurabucket()
    return "es una prueba u"
@app.route("/upload_download", methods = ['GET'])     #funciona!!!! descargar un fichero de la bucket de google y lo manda a la app.
def u3():
    if flask.request.method == "GET":
     storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
     BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
     bucket = storage_client.get_bucket(BUCKET_NAME)

     blob = bucket.blob('checkpoint_name.ckpt.meta')

     blob.download_to_filename("ejemplo")

     app.config["FICHERO_CHECK"] = ""

     return send_from_directory(app.config["FICHERO_CHECK"], filename="ejemplo", as_attachment=True)  #PROBAR

 #   configurabucket2()

@app.route("/contador", methods = ['GET'])  #perfecto
def u4():
     global numero
     numero = numero + 1
     global modelctr
     modelctr = modelctr +1
     global NOMBRE_CHECKPOINTS
     NOMBRE_CHECKPOINTS = "checkpoints_name_"+ str(modelctr)
     global NOMBRE_ZIP
     NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
     ChangeIsModelUpdated()

     return "numero= " + str(numero) + " , nombre= " + NOMBRE_CHECKPOINTS + "isModelUpdated= " +str(isModelUpdated) + "Zip_name= " + str(NOMBRE_ZIP)

@app.route("/numDescargas", methods = ['GET'])  #perfecto
def numDescargas():


     return "Numero descargas A = " + str(NumDescargasModelo_A) + "\n" + "Numero descargas B = " + str(NumDescargasModelo_B) + "\n"   + "Numero descargas C = " + str(NumDescargasModelo_C) + "\n"   + "Numero descargas D = " + str(NumDescargasModelo_D)





@app.route("/contador2", methods = ['GET'])  #perfecto
def uCCONT2():
     Contador()
     ChangeIsModelUpdated()

     return "numero= " + str(numero) + " , nombre= " + NOMBRE_CHECKPOINTS + "isModelUpdated= " +str(isModelUpdated) + "Zip_name= " + str(NOMBRE_ZIP)


def Contador():
    global numero
    numero = numero + 1
    global modelctr
    modelctr = modelctr + 1
    global NOMBRE_CHECKPOINTS
    NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
    global NOMBRE_ZIP
    NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
    return

def ChangeIsModelUpdated():
    global isModelUpdated
    if (isModelUpdated):
     isModelUpdated = False
     return
    else:
     isModelUpdated = True
     return

def configurabucket():
#poner el path al fichero key json
 storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
 #BUCKET_NAME = 'bucket-alejandro-aguilera' # Nombre del bucket que he creado en el google-cloud
 #bucket = storage_client.get_bucket(BUCKET_NAME)
 bucket_nombre = 'prueba'
 bucket = storage_client.create_bucket(bucket_nombre)
 print("bucket lista pra usarse")

def configurabucket2():
#poner el path al fichero key json
 storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
 BUCKET_NAME = 'bucket-alejandro-aguilera' # Nombre del bucket que he creado en el google-cloud
 bucket = storage_client.get_bucket(BUCKET_NAME)

 print("bucket lista pra usarse")

def subirficheroALmacenado():
#para subir
 blob = bucket.blob("nombrefichero.txt")
 blob.upload_from_filename("nombrefichero.txt")
 print("file uploaded")
 return
def descargarfichero():
#para descargar
 blob = bucket.blob("nombrefichero.txt")
 blob.download_from_filename("nombrefichero.txt")
 print("file downloaded")

def createbucket():
#si quiero crear una bucket
 bucket_Creada = storage_client.create_bucket("nombre bucket")

modelUpdated = True


@app.route("/isModelUpdated", methods = ['GET'])
def isModelUpdated():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #mandar un mensaje : no se haclo, igual con send(foto
  if modelUpdated:
      return "YES"
  else:
      return "NO"


@app.route("/empresa?ip=test", methods = ['GET'])
def getDirIp():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #mandar un mensaje : no se haclo, igual con send(foto
  x = {
  "name": "John",
  "age": 20,
  "dirIp": "test",
  "city": "New York"
   }
  y= json.dumps(x)
  return y


@app.route("/json", methods = ['GET'])
def getDirIptest():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #mandar un mensaje : no se haclo, igual con send(foto
  x = {
  "name": "John",
  "age": 20,
  "serr": "test",
  "city": "New York",
  "company": [
  {
         "name": "",
         "UUID": "",
         "Permalink": "",
         "category": "Advertising Server",
         "region": "",
         "city": "",
         "countryCode": "",
         "sateCode": "",
         "URL": ""
         }
         ]
   }
  y= json.dumps(x)
  return y





@app.route("/getGlobalModel", methods = ['GET'])   #funciona subiendo todos los ficheros y zip al bucket. Hay que ver el tema de los nombres
def getGlobalModel():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints
    Contador()
    modeloSencillo()
    Contador()
    #fileaux = modeloSencillo()
    #return fileaux

    print("se ha mandado el modelo")
    return "OK"

  else: # actualizo el modelo
    return

@app.route("/getModelPrueba", methods = ['GET'])
def getModelPrueba():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints
    #modeloSencillo()
    #fileaux = modeloSencillo()
    #return fileaux


    print("se ha mandado el modelo")
    return


  else: # actualizo el modelo
    return


@app.route("/descargar1", methods = ['GET']) #FUNCIONA: DESCARGAS EL FICHERO
def descargar1():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints

      global numero
      numero = numero + 1
      global modelctr
      modelctr = modelctr + 1
      global NOMBRE_CHECKPOINTS
      NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
      global NOMBRE_ZIP
      NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
      ################################
      peso_w = 0
      peso_b = 0

      storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
      BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
      bucket = storage_client.get_bucket(BUCKET_NAME)

      blob = bucket.blob('fichero_pesos')

      blob.download_to_filename("ejemplo_pesos")
      print("descargado fichero: fichero_pesos")

      f = open('ejemplo_pesos')
      string_list = f.readlines()
      f.close()

      sub_string_w_aux = string_list[2] #la tercera linea del fichero
      sub_string_b_aux = string_list[3]

      sub_string_w = sub_string_w_aux[0] + sub_string_w_aux[1] + sub_string_w_aux[2] + sub_string_w_aux[3] + sub_string_w_aux[4] + sub_string_w_aux[5]
      sub_string_b = sub_string_b_aux[0] + sub_string_b_aux[1] + sub_string_b_aux[2] + sub_string_b_aux[3] + sub_string_b_aux[4] + sub_string_b_aux[5]

      peso_w = float(sub_string_w)
      peso_b = float(sub_string_b)

      ## EXTRA POR SI FUNCIONA
      peso_w = np.float32(peso_w)
      peso_b = np.float32(peso_b)

      ################################

      x = tf.placeholder(tf.float32, name='input')
      y_ = tf.placeholder(tf.float32, name='target')

     # W = tf.Variable(5., name='W')
     # b = tf.Variable(3., name='b')
      W = tf.Variable(peso_w, name='W')
      b = tf.Variable(peso_b, name='b')

     # y = x * W + b
      y = tf.add(tf.multiply(x, W), b)
      y = tf.identity(y, name='output')

      loss = tf.reduce_mean(tf.square(y - y_))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
      train_op = optimizer.minimize(loss, name='train')

      init = tf.global_variables_initializer()

      # Creating a tf.train.Saver adds operations to the graph to save and
      # restore variables from checkpoints.

      saver_def = tf.train.Saver().as_saver_def
      ###NO VA EL SAVER DE GRAH DEF##########################
      # with open('graph.pb', 'wb') as f:
      #     f.write(tf.get_default_graph().as_graph_def())

      saver = tf.train.Saver()
      # Training
      # saver.save(sess, your_path + "/checkpoint_name.ckpt")
      # TensorFlow session
     #### sess = tf.Session()
      sess = tf.keras.backend.get_session()
      sess.run(init)


      ### saver.save(sess, "+"+ NOMBRE_CHECKPOINTS + "-.ckpt")
      save_path = saver.save(sess, "+"+ NOMBRE_CHECKPOINTS + "-.ckpt")

      app.config["FICHERO_CHECK"] = ""
    #  zipf = zipfile.ZipFile(NOMBRE_CHECKPOINTS + '.ckpt.meta.zip','w'. zipfile.ZIP_DEFLATED)
   #   zipObj = ZipFile("ficherotestnoviembre.zip", 'w')
   #   zipObj.write(NOMBRE_CHECKPOINTS + ".ckpt.meta")
   #   zipObj.close()
   ##   return send_from_directory(app.config["FICHERO_CHECK"], filename='ficherotestnoviembre.zip', as_attachment=True)
      zipObj = ZipFile('test.zip', 'w')
      zipObj.write("+"+NOMBRE_CHECKPOINTS + "-.ckpt.meta")
      zipObj.close()
      return send_from_directory(app.config["FICHERO_CHECK"], filename='test.zip', as_attachment=True)
  #    return flask.send_file('test.zip', mimetype = 'zip',attachment_filename= 'test.zip', as_attachment= True)

  else: # actualizo el modelo
    return

@app.route("/descargar_graf_pesos_test", methods = ['GET']) # PARA MANDAR UN GRAPH NEUVO TRAS RECIBIR LOS PESOS. PRUEBA
def descargar_graph_pesos_test():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints

      global numero
      numero = numero + 1
      global modelctr
      modelctr = modelctr + 1
      global NOMBRE_CHECKPOINTS
      NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
      global NOMBRE_ZIP
      NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
      ################################
      peso_w = 0
      peso_b = 0

      storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
      BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
      bucket = storage_client.get_bucket(BUCKET_NAME)

      blob = bucket.blob('fichero_pesos')

      blob.download_to_filename("ejemplo_pesos")
      print("descargado fichero: fichero_pesos")

      f = open('ejemplo_pesos')
      string_list = f.readlines()
      f.close()

      sub_string_w_aux = string_list[2] #la tercera linea del fichero
      sub_string_b_aux = string_list[3]

      sub_string_w = sub_string_w_aux[0] + sub_string_w_aux[1] + sub_string_w_aux[2] + sub_string_w_aux[3] + sub_string_w_aux[4] + sub_string_w_aux[5]
      sub_string_b = sub_string_b_aux[0] + sub_string_b_aux[1] + sub_string_b_aux[2] + sub_string_b_aux[3] + sub_string_b_aux[4] + sub_string_b_aux[5]

      peso_w = float(sub_string_w)
      peso_b = float(sub_string_b)

      ## EXTRA POR SI FUNCIONA
      peso_w = np.float32(peso_w)
      peso_b = np.float32(peso_b)

      ################################

      x = tf.placeholder(tf.float32, name='input')
      y_ = tf.placeholder(tf.float32, name='target')

     # W = tf.Variable(5., name='W')
     # b = tf.Variable(3., name='b')
      W = tf.Variable(peso_w, name='W')
      b = tf.Variable(peso_b, name='b')

     # y = x * W + b
      y = tf.add(tf.multiply(x, W), b)
      y = tf.identity(y, name='output')

      loss = tf.reduce_mean(tf.square(y - y_))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
      train_op = optimizer.minimize(loss, name='train')

      init = tf.global_variables_initializer()

      # Creating a tf.train.Saver adds operations to the graph to save and
      # restore variables from checkpoints.

      saver_def = tf.train.Saver().as_saver_def
      ###NO VA EL SAVER DE GRAH DEF##########################
      # with open('graph.pb', 'wb') as f:
      #     f.write(tf.get_default_graph().as_graph_def())

      saver = tf.train.Saver()
      # Training
      # saver.save(sess, your_path + "/checkpoint_name.ckpt")
      # TensorFlow session
     #### sess = tf.Session()
      sess = tf.keras.backend.get_session()
      sess.run(init)


      ### saver.save(sess, "+"+ NOMBRE_CHECKPOINTS + "-.ckpt")
      save_path = saver.save(sess, "+"+ NOMBRE_CHECKPOINTS + "-.ckpt")
      with open('graph_pesos.pb', 'wb') as f:
           f.write(tf.get_default_graph().as_graph_def().SerializeToString())

      app.config["FICHERO_CHECK"] = ""
    #  zipf = zipfile.ZipFile(NOMBRE_CHECKPOINTS + '.ckpt.meta.zip','w'. zipfile.ZIP_DEFLATED)
   #   zipObj = ZipFile("ficherotestnoviembre.zip", 'w')
   #   zipObj.write(NOMBRE_CHECKPOINTS + ".ckpt.meta")
   #   zipObj.close()
   ##   return send_from_directory(app.config["FICHERO_CHECK"], filename='ficherotestnoviembre.zip', as_attachment=True)
      zipObj = ZipFile('test.zip', 'w')
      zipObj.write("graph_pesos.pb")
      zipObj.close()
      return send_from_directory(app.config["FICHERO_CHECK"], filename='test.zip', as_attachment=True)
  #    return flask.send_file('test.zip', mimetype = 'zip',attachment_filename= 'test.zip', as_attachment= True)

  else: # actualizo el modelo
    return



@app.route("/descargar_A", methods = ['GET'])
def descargar_A():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints

     # global numero
     # numero = numero + 1
     # global modelctr
     # modelctr = modelctr + 1
 #     global NOMBRE_CHECKPOINTS
 #     NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
  #    global NOMBRE_ZIP
  #    NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
  #    global modelctrstr
  #    modelctrstr = str(modelctr)
  #    ################################
      peso_w = 0
      peso_b = 0

      storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
      BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
      bucket = storage_client.get_bucket(BUCKET_NAME)



      global NumDescargasModelo_A
      NumDescargasModelo_A = NumDescargasModelo_A + 1



   #   blob = bucket.blob('fichero_pesos')
      global nombre_fichero_descarga

      print("fichero descargado deberia ser " + nombre_fichero_descarga)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
      blob = bucket.blob(nombre_fichero_descarga)
      print("Descargando fichero:  "+nombre_fichero_descarga)
      blob.download_to_filename("ejemplo_pesos")
      print("descargado fichero: fichero_pesos")

      f = open('ejemplo_pesos')
      string_list = f.readlines()
      f.close()

      sub_string_Plocation_aux = string_list[2]  # la tercera linea del fichero
      sub_string_Pemail_aux = string_list[3]
      sub_string_Pdevice_aux = string_list[4]
      sub_string_Pimei_aux = string_list[5]
      sub_string_Pserial_aux = string_list[6]
      sub_string_Pmac_aux = string_list[7]
      sub_string_Padvertiser_aux = string_list[8]
      sub_string_verde_aux = string_list[9]
      sub_string_naranja_aux = string_list[10]
      sub_string_rojo_aux = string_list[11]
      #destinos servidor
      sub_string_Pinternal_dst_aux = string_list[12]
      sub_string_Pads_dst_aux = string_list[13]
      sub_string_Panalytics_dst_aux = string_list[14]
      sub_string_Psns_dst_aux = string_list[15]
      sub_string_Pdevelop_dst_aux = string_list[16]


      
      try:
       sub_string_Plocation = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
      except:
       sub_string_Plocation = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
      try:
       sub_string_Pemail = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
      except:
       sub_string_Pemail = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
      try:
       sub_string_Pdevice = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
      except:
       sub_string_Pdevice = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] 
      try:
       sub_string_Pimei = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
      except:
       sub_string_Pimei = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] 
      try:
       sub_string_Pserial = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
      except:
       sub_string_Pserial = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
      try:
       sub_string_Pmac = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
      except:
       sub_string_Pmac = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
      try:
       sub_string_Padvertiser = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
      except:
       sub_string_Padvertiser = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
      try:
       sub_string_verde = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
      except:
       sub_string_verde = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
      try:
       sub_string_naranja = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
      except:
       sub_string_naranja = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] 
      try:
       sub_string_rojo = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
      except:
       sub_string_rojo = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
      try:
       sub_string_Pinternal_dst = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
      except:
       sub_string_Pinternal_dst = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
      try:
       sub_string_Pads_dst = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
      except:
       sub_string_Pads_dst = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
      try:
       sub_string_Panalytics_dst = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
      except:
       sub_string_Panalytics_dst = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
      try:
       sub_string_Psns_dst = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
      except:
       sub_string_Psns_dst = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
      try:
       sub_string_Pdevelop_dst = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
      except:
       sub_string_Pdevelop_dst = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]



      peso_Plocation = float(sub_string_Plocation)
      peso_Pemail = float(sub_string_Pemail)
      peso_Pdevice = float(sub_string_Pdevice)
      peso_Pimei = float(sub_string_Pimei)
      peso_Pserial = float(sub_string_Pserial)
      peso_Pmac = float(sub_string_Pmac)
      peso_Padvertiser = float(sub_string_Padvertiser)

      umbral_verde = float(sub_string_verde)
      umbral_naranja = float(sub_string_naranja)
      umbral_rojo = float(sub_string_rojo)

      peso_Pinternal_dst = float(sub_string_Pinternal_dst)
      peso_Pads_dst = float(sub_string_Pads_dst)
      peso_Panalytics_dst = float(sub_string_Panalytics_dst)
      peso_Psns_dst = float(sub_string_Psns_dst)
      peso_Pdevelop_dst = float(sub_string_Pdevelop_dst)
      
      try:
      #2    
       global nombre_fichero_descarga1

       print("fichero descargado deberia ser " + nombre_fichero_descarga1)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
       blob = bucket.blob(nombre_fichero_descarga1)
       print("Descargando fichero:  "+nombre_fichero_descarga1)
       blob.download_to_filename("ejemplo_pesos")
       print("descargado fichero: fichero_pesos")

       f = open('ejemplo_pesos1')
       string_list1 = f.readlines()
       f.close()

       sub_string_Plocation_aux = string_list1[2]  # la tercera linea del fichero
       sub_string_Pemail_aux = string_list1[3]
       sub_string_Pdevice_aux = string_list1[4]
       sub_string_Pimei_aux = string_list1[5]
       sub_string_Pserial_aux = string_list1[6]
       sub_string_Pmac_aux = string_list1[7]
       sub_string_Padvertiser_aux = string_list1[8]
       sub_string_verde_aux = string_list1[9]
       sub_string_naranja_aux = string_list1[10]
       sub_string_rojo_aux = string_list1[11]
       #destinos servidor
       sub_string_Pinternal_dst_aux = string_list1[12]
       sub_string_Pads_dst_aux = string_list1[13]
       sub_string_Panalytics_dst_aux = string_list1[14]
       sub_string_Psns_dst_aux = string_list1[15]
       sub_string_Pdevelop_dst_aux = string_list1[16]

     
       try:
        sub_string_Plocation1 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
       except:
        sub_string_Plocation1 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
       try:
        sub_string_Pemail1 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
       except:
        sub_string_Pemail1 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
       try:
        sub_string_Pdevice1 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
       except:
        sub_string_Pdevice1 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] 
       try:
        sub_string_Pimei1 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
       except:
        sub_string_Pimei1 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] 
       try:
        sub_string_Pserial1 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
       except:
        sub_string_Pserial1 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
       try:
        sub_string_Pmac1 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
       except:
        sub_string_Pmac1 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
       try:
        sub_string_Padvertiser1 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
       except:
        sub_string_Padvertiser1 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
       try:
        sub_string_verde1 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
       except:
        sub_string_verde1 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
       try:
        sub_string_naranja1 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
       except:
        sub_string_naranja1 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] 
       try:
        sub_string_rojo1 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
       except:
        sub_string_rojo1 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]   
       #destinations
       try:
        sub_string_Pinternal_dst1 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
       except:
        sub_string_Pinternal_dst1 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
       try:
        sub_string_Pads_dst1 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
       except:
        sub_string_Pads_dst1 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
       try:
        sub_string_Panalytics_dst1 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
       except:
        sub_string_Panalytics_dst1 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
       try:
        sub_string_Psns_dst1 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
       except:
        sub_string_Psns_dst1 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
       try:
        sub_string_Pdevelop_dst1 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
       except:
        sub_string_Pdevelop_dst1 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]
      

       peso_Plocation1 = float(sub_string_Plocation1)
       peso_Pemail1 = float(sub_string_Pemail1)
       peso_Pdevice1 = float(sub_string_Pdevice1)
       peso_Pimei1 = float(sub_string_Pimei1)
       peso_Pserial1 = float(sub_string_Pserial1)
       peso_Pmac1 = float(sub_string_Pmac1)
       peso_Padvertiser1 = float(sub_string_Padvertiser1)

       umbral_verde1 = float(sub_string_verde1)
       umbral_naranja1 = float(sub_string_naranja1)
       umbral_rojo1 = float(sub_string_rojo1)

       peso_Pinternal_dst1 = float(sub_string_Pinternal_dst1)
       peso_Pads_dst1 = float(sub_string_Pads_dst1)
       peso_Panalytics_dst1 = float(sub_string_Panalytics_dst1)
       peso_Psns_dst1 = float(sub_string_Psns_dst1)
       peso_Pdevelop_dst1 = float(sub_string_Pdevelop_dst1)
      except:
       print("mal") 

      try:
      #3    
       global nombre_fichero_descarga2

       print("fichero descargado deberia ser " + nombre_fichero_descarga2)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
       blob = bucket.blob(nombre_fichero_descarga2)
       print("Descargando fichero:  "+nombre_fichero_descarga2)
       blob.download_to_filename("ejemplo_pesos")
       print("descargado fichero: fichero_pesos")

       f = open('ejemplo_pesos2')
       string_list2 = f.readlines()
       f.close()

       sub_string_Plocation_aux = string_list2[2]  # la tercera linea del fichero
       sub_string_Pemail_aux = string_list2[3]
       sub_string_Pdevice_aux = string_list2[4]
       sub_string_Pimei_aux = string_list2[5]
       sub_string_Pserial_aux = string_list2[6]
       sub_string_Pmac_aux = string_list2[7]
       sub_string_Padvertiser_aux = string_list2[8]
       sub_string_verde_aux = string_list2[9]
       sub_string_naranja_aux = string_list2[10]
       sub_string_rojo_aux = string_list2[11]
       #destinos servidor
       sub_string_Pinternal_dst_aux = string_list2[12]
       sub_string_Pads_dst_aux = string_list2[13]
       sub_string_Panalytics_dst_aux = string_list2[14]
       sub_string_Psns_dst_aux = string_list2[15]
       sub_string_Pdevelop_dst_aux = string_list2[16]

       
       try:
        sub_string_Plocation2 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
       except:
        sub_string_Plocation2 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
       try:
        sub_string_Pemail2 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
       except:
        sub_string_Pemail2 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
       try:
        sub_string_Pdevice2 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
       except:
        sub_string_Pdevice2 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] 
       try:
        sub_string_Pime2 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
       except:
        sub_string_Pimei2 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] 
       try:
        sub_string_Pserial2 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
       except:
        sub_string_Pserial2 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
       try:
        sub_string_Pmac2 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
       except:
        sub_string_Pmac2 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
       try:
        sub_string_Padvertiser2 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
       except:
        sub_string_Padvertiser2 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
       try:
        sub_string_verde2 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
       except:
        sub_string_verde2 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
       try:
        sub_string_naranja2 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
       except:
        sub_string_naranja2 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] 
       try:
        sub_string_rojo2 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
       except:
        sub_string_rojo2 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
       try:
        sub_string_Pinternal_dst2 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
       except:
        sub_string_Pinternal_dst2 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
       try:
        sub_string_Pads_dst2 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
       except:
        sub_string_Pads_dst2 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
       try:
        sub_string_Panalytics_dst2 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
       except:
        sub_string_Panalytics_dst2 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
       try:
        sub_string_Psns_dst2 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
       except:
        sub_string_Psns_dst2 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
       try:
        sub_string_Pdevelop_dst2 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
       except:
        sub_string_Pdevelop_dst2 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]



      

       peso_Plocation2 = float(sub_string_Plocation2)
       peso_Pemail2 = float(sub_string_Pemail2)
       peso_Pdevice2 = float(sub_string_Pdevice2)
       peso_Pimei2 = float(sub_string_Pimei2)
       peso_Pserial2 = float(sub_string_Pserial2)
       peso_Pmac2 = float(sub_string_Pmac2)
       peso_Padvertiser2 = float(sub_string_Padvertiser2)

       umbral_verde2 = float(sub_string_verde2)
       umbral_naranja2 = float(sub_string_naranja2)
       umbral_rojo2 = float(sub_string_rojo2)

       peso_Pinternal_dst2 = float(sub_string_Pinternal_dst2)
       peso_Pads_dst2 = float(sub_string_Pads_dst2)
       peso_Panalytics_dst2 = float(sub_string_Panalytics_dst2)
       peso_Psns_dst2 = float(sub_string_Psns_dst2)
       peso_Pdevelop_dst2 = float(sub_string_Pdevelop_dst2)
      except:
       print("mal2")

      #se hace la media: 
      
      peso_Plocation= peso_Plocation 
      peso_Pemail = peso_Pemail
      peso_Pdevice = peso_Pdevice
      peso_Pimei = peso_Pimei
      peso_Pserial = peso_Pserial
      peso_Pmac = peso_Pmac
      peso_Padvertiser = peso_Padvertiser
      umbral_verde = umbral_verde
      umbral_naranja = umbral_naranja
      umbral_rojo = umbral_rojo

      peso_Pinternal_dst = peso_Pinternal_dst
      peso_Pads_dst = peso_Pads_dst
      peso_Panalytics_dst = peso_Panalytics_dst
      peso_Psns_dst = peso_Psns_dst
      peso_Pdevelop_dst = peso_Pdevelop_dst

      try: 
       peso_Plocation= peso_Plocation + peso_Plocation1 + peso_Plocation2
       peso_Pemail = peso_Pemail + peso_Pemail1 + peso_Pemail2
       peso_Pdevice = peso_Pdevice + peso_Pdevice1 + peso_Pdevice2
       peso_Pimei = peso_Pimei + peso_Pimei1 + peso_Pimei2
       peso_Pserial = peso_Pserial + peso_Pserial1 + peso_Pserial2
       peso_Pmac = peso_Pmac + peso_Pmac1 + peso_Pmac2
       peso_Padvertiser = peso_Padvertiser + peso_Padvertiser1 + peso_Padvertiser2
       umbral_verde = umbral_verde + umbral_verde1 + umbral_verde2
       umbral_naranja = umbral_naranja + umbral_naranja1 + umbral_naranja2
       umbral_rojo = umbral_rojo + umbral_rojo1 + umbral_rojo2

       peso_Pinternal_dst = peso_Pinternal_dst + peso_Pinternal_dst1 + peso_Pinternal_dst2
       peso_Pads_dst = peso_Pads_dst + peso_Pads_dst1 + peso_Pads_dst2
       peso_Panalytics_dst = peso_Panalytics_dst + peso_Panalytics_dst1 + peso_Panalytics_dst2
       peso_Psns_dst = peso_Psns_dst + peso_Psns_dst1 + peso_Psns_dst2
       peso_Pdevelop_dst = peso_Pdevelop_dst + peso_Pdevelop_dst1 + peso_Pdevelop_dst2

       peso_Plocation= peso_Plocation / 3 
       peso_Pemail = peso_Pemail / 3 
       peso_Pdevice = peso_Pdevice / 3 
       peso_Pimei = peso_Pimei / 3 
       peso_Pserial = peso_Pserial / 3 
       peso_Pmac = peso_Pmac / 3 
       peso_Padvertiser = peso_Padvertiser / 3 
       umbral_verde = umbral_verde / 3 
       umbral_naranja = umbral_naranja / 3 
       umbral_rojo = umbral_rojo  / 3

       peso_Pinternal_dst = peso_Pinternal_dst  / 3
       peso_Pads_dst = peso_Pads_dst  / 3
       peso_Panalytics_dst = peso_Panalytics_dst  / 3
       peso_Psns_dst = peso_Psns_dst  / 3
       peso_Pdevelop_dst = peso_Pdevelop_dst  / 3

      except: 
       print("mal3")

      try: 
       peso_Plocation= peso_Plocation + peso_Plocation1 
       peso_Pemail = peso_Pemail + peso_Pemail1 
       peso_Pdevice = peso_Pdevice + peso_Pdevice1 
       peso_Pimei = peso_Pimei + peso_Pimei1 
       peso_Pserial = peso_Pserial + peso_Pserial1 
       peso_Pmac = peso_Pmac + peso_Pmac1 + peso_Pmac2
       peso_Padvertiser = peso_Padvertiser + peso_Padvertiser1 
       umbral_verde = umbral_verde + umbral_verde1 
       umbral_naranja = umbral_naranja + umbral_naranja1 
       umbral_rojo = umbral_rojo + umbral_rojo1

       peso_Pinternal_dst = peso_Pinternal_dst + peso_Pinternal_dst1
       peso_Pads_dst = peso_Pads_dst + peso_Pads_dst1
       peso_Panalytics_dst = peso_Panalytics_dst + peso_Panalytics_dst1
       peso_Psns_dst = peso_Psns_dst + peso_Psns_dst1
       peso_Pdevelop_dst = peso_Pdevelop_dst + peso_Pdevelop_dst1

       peso_Plocation= peso_Plocation / 2 
       peso_Pemail = peso_Pemail / 2 
       peso_Pdevice = peso_Pdevice / 2 
       peso_Pimei = peso_Pimei / 2 
       peso_Pserial = peso_Pserial / 2 
       peso_Pmac = peso_Pmac / 2 
       peso_Padvertiser = peso_Padvertiser / 2 
       umbral_verde = umbral_verde / 2 
       umbral_naranja = umbral_naranja / 2 
       umbral_rojo = umbral_rojo  / 2

       peso_Pinternal_dst = peso_Pinternal_dst  / 2
       peso_Pads_dst = peso_Pads_dst  / 2
       peso_Panalytics_dst = peso_Panalytics_dst  / 2
       peso_Psns_dst = peso_Psns_dst  / 2
       peso_Pdevelop_dst = peso_Pdevelop_dst  / 2

      except: 
       print("mal4")   
      
      ###fichero con pesos y umbrales: 


      ################################
     # my_file = open("fichero_datos"+modelctrstr, "w")
      my_file = open("fichero_datos"+"global", "w")
      #my_file.write("Pesos actualizados modelo Ant"+"\n"+"\n"+sub_string_Plocation+"\n"+sub_string_Pemail+"\n"+sub_string_Pdevice+"\n"+sub_string_Pimei+"\n"+sub_string_Pserial+"\n"+sub_string_Pmac+"\n"+sub_string_Padvertiser+"\n"+sub_string_verde+"\n"+sub_string_naranja+"\n"+sub_string_rojo)
      my_file.write("Pesos actualizados modelo Ant"+"\n"+"\n"+str(peso_Plocation)+"\n"+str(peso_Pemail)+"\n"+str(peso_Pdevice)+"\n"+str(peso_Pimei)+"\n"+str(peso_Pserial)+"\n"+str(peso_Pmac)+"\n"+str(peso_Padvertiser)+"\n"+str(umbral_verde)+"\n"+str(umbral_naranja)+"\n"+str(umbral_rojo))

      my_file.close()

     # Ejemplo para 4 datos: Location, Email, IMEI y Device ID:
      filtracion_location = tf.placeholder(tf.float32, name='location_input')
      filtracion_email = tf.placeholder(tf.float32, name='email_input')
      filtracion_imei = tf.placeholder(tf.float32, name='imei_input')
      filtracion_device = tf.placeholder(tf.float32, name='device_input')
      filtracion_serialnumber = tf.placeholder(tf.float32, name='serialnumber_input')
      filtracion_macaddress = tf.placeholder(tf.float32, name='macaddress_input')
      filtracion_advertiser = tf.placeholder(tf.float32, name='advertiser_input')


      #destinos: 5 posibles--> Internal, Ads, Analytics, Sns, Develop               1 o 0 si se manda ahi o no
      destino_internal = tf.placeholder(tf.float32, name='internal_dst_input')
      destino_ads = tf.placeholder(tf.float32, name='ads_dst_input')
      destino_analytics = tf.placeholder(tf.float32, name='analytics_dst_input')
      destino_sns = tf.placeholder(tf.float32, name='sns_dst_input')
      destino_develop = tf.placeholder(tf.float32, name='develop_dst_input')

  # objetivo: answer: 1 o 0
      y_ = tf.placeholder(tf.float32, name='target')

      #distintos pesos para cada dato: nombre= Px
      Plocation = tf.Variable(peso_Plocation, name='Plocation') #9
      Pemail = tf.Variable(peso_Pemail, name='Pemail') #8
      Pimei = tf.Variable(peso_Pimei, name='Pimei') #3
      Pdevice = tf.Variable(peso_Pdevice, name='Pdevice') #2

      #Pdni = tf.placeholder(tf.float32, name='Pdni') #10
      #Pphone = tf.placeholder(tf.float32, name='Pphone') #8
      Pserialnumber = tf.Variable(peso_Pserial, name='Pserialnumber') #3
      Pmacaddress = tf.Variable(peso_Pmac, name='Pmacaddress') #1
      Padvertiser = tf.Variable(peso_Padvertiser, name='Padvertiser') #5

      #Pesos para cada destino:
      Pinternal_dst = tf.Variable(peso_Pinternal_dst, name='Pinternal_dst') #5
      Pads_dst = tf.Variable(peso_Pads_dst, name='Pads_dst') #5
      Panalytics_dst = tf.Variable(peso_Panalytics_dst, name='Panalytics_dst') #5
      Psns_dst = tf.Variable(peso_Psns_dst, name='Psns_dst') #5
      Pdevelop_dst = tf.Variable(peso_Pdevelop_dst, name='Pdevelop_dst') #5

      

  # umbral de decision
  # umbral = tf.constant(10., name='umbral')
      #umbral = tf.placeholder(tf.float32, name='umbral')

    #  aux = tf.Variable(0., name='Vaux')
     #salida 0 o 1

      #salida del modelo
      y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), 0.0)
      #y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), aux)
      y_aux2 = tf.add(tf.multiply(filtracion_email, Pemail), y_aux1)
      y_aux3 = tf.add(tf.multiply(filtracion_imei, Pimei), y_aux2)
      y_aux4 = tf.add(tf.multiply(filtracion_device, Pdevice), y_aux3)


      #y_aux11 = tf.add(tf.multiply(filtracion_dni, Pdni), y_aux4)
      #y_aux12 = tf.add(tf.multiply(filtracion_phone, Pphone), y_aux11)
      y_aux13 = tf.add(tf.multiply(filtracion_serialnumber, Pserialnumber), y_aux4)
      y_aux14 = tf.add(tf.multiply(filtracion_macaddress, Pmacaddress), y_aux13)
      #y_aux14 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

      #destinos
      y_aux1_dst = tf.add(tf.multiply(destino_internal, Pinternal_dst), 0.0)
      y_aux2_dst = tf.add(tf.multiply(destino_ads, Pads_dst), y_aux1_dst)
      y_aux3_dst = tf.add(tf.multiply(destino_analytics, Panalytics_dst), y_aux2_dst)
      y_aux4_dst = tf.add(tf.multiply(destino_sns, Psns_dst), y_aux3_dst)
      y_aux5_dst = tf.add(tf.multiply(destino_develop, Pdevelop_dst), y_aux4_dst)
      #

      y_aux_final1 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

      y = tf.multiply(y_aux_final1, y_aux5_dst)

      y = tf.identity(y, name='output')

      loss = tf.reduce_mean(tf.square(y - y_))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
      train_op = optimizer.minimize(loss, name='train')

      init = tf.global_variables_initializer()

      # Creating a tf.train.Saver adds operations to the graph to save and
      # restore variables from checkpoints.

      saver_def = tf.train.Saver().as_saver_def
      ###NO VA EL SAVER DE GRAH DEF##########################
      # with open('graph.pb', 'wb') as f:
      #     f.write(tf.get_default_graph().as_graph_def())

      saver = tf.train.Saver()
      # Training
      # saver.save(sess, your_path + "/checkpoint_name.ckpt")
      # TensorFlow session
      sess = tf.Session()
      sess.run(init)
     # saver.save(sess, "-"+ NOMBRE_CHECKPOINTS + "-.ckpt")  "checkpoint_actualizado.ckpt"
     # saver.save(sess, "checkpoint_actualizado_"+modelctrstr+".ckpt")
      saver.save(sess, "checkpoint_actualizado_"+"global"+".ckpt")
      print("se han creado fichero con nombre ... "+ modelctrstr)

      #
      app.config["FICHERO_CHECK"] = ""
    #  zipf = zipfile.ZipFile(NOMBRE_CHECKPOINTS + '.ckpt.meta.zip','w'. zipfile.ZIP_DEFLATED)
   #   zipObj = ZipFile("ficherotestnoviembre.zip", 'w')
   #   zipObj.write(NOMBRE_CHECKPOINTS + ".ckpt.meta")
   #   zipObj.close()
   ##   return send_from_directory(app.config["FICHERO_CHECK"], filename='ficherotestnoviembre.zip', as_attachment=True)
      zipObj = ZipFile('test.zip', 'w')
      zipObj.write("checkpoint")
    #  zipObj.write("-"+ NOMBRE_CHECKPOINTS + "-.ckpt"+'.index')
    #  zipObj.write("-" + NOMBRE_CHECKPOINTS + "-.ckpt" + '.data-00000-of-00001')
    #  zipObj.write("-" + NOMBRE_CHECKPOINTS + "-.ckpt.meta") # no se si igual .meta
   #   zipObj.write("checkpoint_actualizado_"+modelctrstr+".ckpt" + '.index')
   #   zipObj.write("checkpoint_actualizado_"+modelctrstr+".ckpt" + '.data-00000-of-00001')
  #    zipObj.write("checkpoint_actualizado_"+modelctrstr+".ckpt.meta") # no se si igual .meta
      #zipObj.write("fichero_datos"+modelctrstr)
      zipObj.write("checkpoint_actualizado_"+"global"+".ckpt" + '.index')  #si
      zipObj.write("checkpoint_actualizado_"+"global"+".ckpt" + '.data-00000-of-00001')
    
      zipObj.write("fichero_datos"+"global")
      zipObj.close()
      return send_from_directory(app.config["FICHERO_CHECK"], filename='test.zip', as_attachment=True)
  #    return flask.send_file('test.zip', mimetype = 'zip',attachment_filename= 'test.zip', as_attachment= True)

  else: # actualizo el modelo
    return
##


@app.route("/descargar_B", methods = ['GET'])
def descargar_B():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints

     # global numero
     # numero = numero + 1
     # global modelctr
     # modelctr = modelctr + 1
 #     global NOMBRE_CHECKPOINTS
 #     NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
  #    global NOMBRE_ZIP
  #    NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
  #    global modelctrstr
  #    modelctrstr = str(modelctr)
  #    ################################
      peso_w = 0
      peso_b = 0

      storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
      BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
      bucket = storage_client.get_bucket(BUCKET_NAME)


      global NumDescargasModelo_B
      NumDescargasModelo_B = NumDescargasModelo_B + 1


   #   blob = bucket.blob('fichero_pesos')
      global nombre_fichero_descarga

      print("fichero descargado deberia ser " + nombre_fichero_descarga)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
      blob = bucket.blob(nombre_fichero_descarga)
      print("Descargando fichero:  "+nombre_fichero_descarga)
      blob.download_to_filename("ejemplo_pesos")
      print("descargado fichero: fichero_pesos")

      f = open('ejemplo_pesos')
      string_list = f.readlines()
      f.close()

      sub_string_Plocation_aux = string_list[2]  # la tercera linea del fichero
      sub_string_Pemail_aux = string_list[3]
      sub_string_Pdevice_aux = string_list[4]
      sub_string_Pimei_aux = string_list[5]
      sub_string_Pserial_aux = string_list[6]
      sub_string_Pmac_aux = string_list[7]
      sub_string_Padvertiser_aux = string_list[8]
      sub_string_verde_aux = string_list[9]
      sub_string_naranja_aux = string_list[10]
      sub_string_rojo_aux = string_list[11]
      #destinos servidor
      sub_string_Pinternal_dst_aux = string_list[12]
      sub_string_Pads_dst_aux = string_list[13]
      sub_string_Panalytics_dst_aux = string_list[14]
      sub_string_Psns_dst_aux = string_list[15]
      sub_string_Pdevelop_dst_aux = string_list[16]

       
      try:
       sub_string_Plocation = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
      except:
       sub_string_Plocation = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
      try:
       sub_string_Pemail = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
      except:
       sub_string_Pemail = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
      try:
       sub_string_Pdevice = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
      except:
       sub_string_Pdevice = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] 
      try:
       sub_string_Pimei = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
      except:
       sub_string_Pimei = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] 
      try:
       sub_string_Pserial = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
      except:
       sub_string_Pserial = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
      try:
       sub_string_Pmac = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
      except:
       sub_string_Pmac = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
      try:
       sub_string_Padvertiser = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
      except:
       sub_string_Padvertiser = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
      try:
       sub_string_verde = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
      except:
       sub_string_verde = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
      try:
       sub_string_naranja = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
      except:
       sub_string_naranja = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] 
      try:
       sub_string_rojo = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
      except:
       sub_string_rojo = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
      try:
       sub_string_Pinternal_dst = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
      except:
       sub_string_Pinternal_dst = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
      try:
       sub_string_Pads_dst = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
      except:
       sub_string_Pads_dst = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
      try:
       sub_string_Panalytics_dst = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
      except:
       sub_string_Panalytics_dst = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
      try:
       sub_string_Psns_dst = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
      except:
       sub_string_Psns_dst = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
      try:
       sub_string_Pdevelop_dst = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
      except:
       sub_string_Pdevelop_dst = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]



      peso_Plocation = float(sub_string_Plocation)
      peso_Pemail = float(sub_string_Pemail)
      peso_Pdevice = float(sub_string_Pdevice)
      peso_Pimei = float(sub_string_Pimei)
      peso_Pserial = float(sub_string_Pserial)
      peso_Pmac = float(sub_string_Pmac)
      peso_Padvertiser = float(sub_string_Padvertiser)

      umbral_verde = float(sub_string_verde)
      umbral_naranja = float(sub_string_naranja)
      umbral_rojo = float(sub_string_rojo)

      peso_Pinternal_dst = float(sub_string_Pinternal_dst)
      peso_Pads_dst = float(sub_string_Pads_dst)
      peso_Panalytics_dst = float(sub_string_Panalytics_dst)
      peso_Psns_dst = float(sub_string_Psns_dst)
      peso_Pdevelop_dst = float(sub_string_Pdevelop_dst)
      
      try:
      #2    
       global nombre_fichero_descarga1

       print("fichero descargado deberia ser " + nombre_fichero_descarga1)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
       blob = bucket.blob(nombre_fichero_descarga1)
       print("Descargando fichero:  "+nombre_fichero_descarga1)
       blob.download_to_filename("ejemplo_pesos")
       print("descargado fichero: fichero_pesos")

       f = open('ejemplo_pesos1')
       string_list1 = f.readlines()
       f.close()

       sub_string_Plocation_aux = string_list1[2]  # la tercera linea del fichero
       sub_string_Pemail_aux = string_list1[3]
       sub_string_Pdevice_aux = string_list1[4]
       sub_string_Pimei_aux = string_list1[5]
       sub_string_Pserial_aux = string_list1[6]
       sub_string_Pmac_aux = string_list1[7]
       sub_string_Padvertiser_aux = string_list1[8]
       sub_string_verde_aux = string_list1[9]
       sub_string_naranja_aux = string_list1[10]
       sub_string_rojo_aux = string_list1[11]
       #destinos servidor
       sub_string_Pinternal_dst_aux = string_list1[12]
       sub_string_Pads_dst_aux = string_list1[13]
       sub_string_Panalytics_dst_aux = string_list1[14]
       sub_string_Psns_dst_aux = string_list1[15]
       sub_string_Pdevelop_dst_aux = string_list1[16]

       
       try:
        sub_string_Plocation1 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
       except:
        sub_string_Plocation1 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
       try:
        sub_string_Pemail1 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
       except:
        sub_string_Pemail1 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
       try:
        sub_string_Pdevice1 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
       except:
        sub_string_Pdevice1 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2]
       try:
        sub_string_Pimei1 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
       except:
        sub_string_Pimei1 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2]
       try:
        sub_string_Pserial1 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
       except:
        sub_string_Pserial1 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
       try:
        sub_string_Pmac1 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
       except:
        sub_string_Pmac1 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
       try:
        sub_string_Padvertiser1 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
       except:
        sub_string_Padvertiser1 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
       try:
        sub_string_verde1 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
       except:
        sub_string_verde1 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
       try:
        sub_string_naranja1 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
       except:
        sub_string_naranja1 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2]
       try:
        sub_string_rojo1 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
       except:
        sub_string_rojo1 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
       try:
        sub_string_Pinternal_dst1 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
       except:
        sub_string_Pinternal_dst1 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
       try:
        sub_string_Pads_dst1 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
       except:
        sub_string_Pads_dst1 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
       try:
        sub_string_Panalytics_dst1 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
       except:
        sub_string_Panalytics_dst1 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
       try:
        sub_string_Psns_dst1 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
       except:
        sub_string_Psns_dst1 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
       try:
        sub_string_Pdevelop_dst1 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
       except:
        sub_string_Pdevelop_dst1 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]


       peso_Plocation1 = float(sub_string_Plocation1)
       peso_Pemail1 = float(sub_string_Pemail1)
       peso_Pdevice1 = float(sub_string_Pdevice1)
       peso_Pimei1 = float(sub_string_Pimei1)
       peso_Pserial1 = float(sub_string_Pserial1)
       peso_Pmac1 = float(sub_string_Pmac1)
       peso_Padvertiser1 = float(sub_string_Padvertiser1)

       umbral_verde1 = float(sub_string_verde1)
       umbral_naranja1 = float(sub_string_naranja1)
       umbral_rojo1 = float(sub_string_rojo1)

       peso_Pinternal_dst1 = float(sub_string_Pinternal_dst1)
       peso_Pads_dst1 = float(sub_string_Pads_dst1)
       peso_Panalytics_dst1 = float(sub_string_Panalytics_dst1)
       peso_Psns_dst1 = float(sub_string_Psns_dst1)
       peso_Pdevelop_dst1 = float(sub_string_Pdevelop_dst1)
      except:
       print("mal") 

      try:
      #3    
       global nombre_fichero_descarga2

       print("fichero descargado deberia ser " + nombre_fichero_descarga2)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
       blob = bucket.blob(nombre_fichero_descarga2)
       print("Descargando fichero:  "+nombre_fichero_descarga2)
       blob.download_to_filename("ejemplo_pesos")
       print("descargado fichero: fichero_pesos")

       f = open('ejemplo_pesos2')
       string_list2 = f.readlines()
       f.close()

       sub_string_Plocation_aux = string_list2[2]  # la tercera linea del fichero
       sub_string_Pemail_aux = string_list2[3]
       sub_string_Pdevice_aux = string_list2[4]
       sub_string_Pimei_aux = string_list2[5]
       sub_string_Pserial_aux = string_list2[6]
       sub_string_Pmac_aux = string_list2[7]
       sub_string_Padvertiser_aux = string_list2[8]
       sub_string_verde_aux = string_list2[9]
       sub_string_naranja_aux = string_list2[10]
       sub_string_rojo_aux = string_list2[11]
       #destinos servidor
       sub_string_Pinternal_dst_aux = string_list2[12]
       sub_string_Pads_dst_aux = string_list2[13]
       sub_string_Panalytics_dst_aux = string_list2[14]
       sub_string_Psns_dst_aux = string_list2[15]
       sub_string_Pdevelop_dst_aux = string_list2[16]

       
       try:
        sub_string_Plocation2 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
       except:
        sub_string_Plocation2 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
       try:
        sub_string_Pemail2 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
       except:
        sub_string_Pemail2 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
       try:
        sub_string_Pdevice2 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
       except:
        sub_string_Pdevice2 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2]
       try:
        sub_string_Pime2 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
       except:
        sub_string_Pimei2 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2]
       try:
        sub_string_Pserial2 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
       except:
        sub_string_Pserial2 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
       try:
        sub_string_Pmac2 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
       except:
        sub_string_Pmac2 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
       try:
        sub_string_Padvertiser2 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
       except:
        sub_string_Padvertiser2 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
       try:
        sub_string_verde2 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
       except:
        sub_string_verde2 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
       try:
        sub_string_naranja2 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
       except:
        sub_string_naranja2 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2]
       try:
        sub_string_rojo2 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
       except:
        sub_string_rojo2 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
       try:
        sub_string_Pinternal_dst2 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
       except:
        sub_string_Pinternal_dst2 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
       try:
        sub_string_Pads_dst2 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
       except:
        sub_string_Pads_dst2 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
       try:
        sub_string_Panalytics_dst2 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
       except:
        sub_string_Panalytics_dst2 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
       try:
        sub_string_Psns_dst2 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
       except:
        sub_string_Psns_dst2 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
       try:
        sub_string_Pdevelop_dst2 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
       except:
        sub_string_Pdevelop_dst2 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]





       peso_Plocation2 = float(sub_string_Plocation2)
       peso_Pemail2 = float(sub_string_Pemail2)
       peso_Pdevice2 = float(sub_string_Pdevice2)
       peso_Pimei2 = float(sub_string_Pimei2)
       peso_Pserial2 = float(sub_string_Pserial2)
       peso_Pmac2 = float(sub_string_Pmac2)
       peso_Padvertiser2 = float(sub_string_Padvertiser2)

       umbral_verde2 = float(sub_string_verde2)
       umbral_naranja2 = float(sub_string_naranja2)
       umbral_rojo2 = float(sub_string_rojo2)

       peso_Pinternal_dst2 = float(sub_string_Pinternal_dst2)
       peso_Pads_dst2 = float(sub_string_Pads_dst2)
       peso_Panalytics_dst2 = float(sub_string_Panalytics_dst2)
       peso_Psns_dst2 = float(sub_string_Psns_dst2)
       peso_Pdevelop_dst2 = float(sub_string_Pdevelop_dst2)
      except:
       print("mal2")

      #se hace la media: 
      
      peso_Plocation= peso_Plocation 
      peso_Pemail = peso_Pemail
      peso_Pdevice = peso_Pdevice
      peso_Pimei = peso_Pimei
      peso_Pserial = peso_Pserial
      peso_Pmac = peso_Pmac
      peso_Padvertiser = peso_Padvertiser
      umbral_verde = umbral_verde
      umbral_naranja = umbral_naranja
      umbral_rojo = umbral_rojo

      peso_Pinternal_dst = peso_Pinternal_dst
      peso_Pads_dst = peso_Pads_dst
      peso_Panalytics_dst = peso_Panalytics_dst
      peso_Psns_dst = peso_Psns_dst
      peso_Pdevelop_dst = peso_Pdevelop_dst

      try:
       peso_Plocation= peso_Plocation + peso_Plocation1 + peso_Plocation2
       peso_Pemail = peso_Pemail + peso_Pemail1 + peso_Pemail2
       peso_Pdevice = peso_Pdevice + peso_Pdevice1 + peso_Pdevice2
       peso_Pimei = peso_Pimei + peso_Pimei1 + peso_Pimei2
       peso_Pserial = peso_Pserial + peso_Pserial1 + peso_Pserial2
       peso_Pmac = peso_Pmac + peso_Pmac1 + peso_Pmac2
       peso_Padvertiser = peso_Padvertiser + peso_Padvertiser1 + peso_Padvertiser2
       umbral_verde = umbral_verde + umbral_verde1 + umbral_verde2
       umbral_naranja = umbral_naranja + umbral_naranja1 + umbral_naranja2
       umbral_rojo = umbral_rojo + umbral_rojo1 + umbral_rojo2

       peso_Pinternal_dst = peso_Pinternal_dst + peso_Pinternal_dst1 + peso_Pinternal_dst2
       peso_Pads_dst = peso_Pads_dst + peso_Pads_dst1 + peso_Pads_dst2
       peso_Panalytics_dst = peso_Panalytics_dst + peso_Panalytics_dst1 + peso_Panalytics_dst2
       peso_Psns_dst = peso_Psns_dst + peso_Psns_dst1 + peso_Psns_dst2
       peso_Pdevelop_dst = peso_Pdevelop_dst + peso_Pdevelop_dst1 + peso_Pdevelop_dst2

       peso_Plocation= peso_Plocation / 3
       peso_Pemail = peso_Pemail / 3
       peso_Pdevice = peso_Pdevice / 3
       peso_Pimei = peso_Pimei / 3
       peso_Pserial = peso_Pserial / 3
       peso_Pmac = peso_Pmac / 3
       peso_Padvertiser = peso_Padvertiser / 3
       umbral_verde = umbral_verde / 3
       umbral_naranja = umbral_naranja / 3
       umbral_rojo = umbral_rojo  / 3

       peso_Pinternal_dst = peso_Pinternal_dst  / 3
       peso_Pads_dst = peso_Pads_dst  / 3
       peso_Panalytics_dst = peso_Panalytics_dst  / 3
       peso_Psns_dst = peso_Psns_dst  / 3
       peso_Pdevelop_dst = peso_Pdevelop_dst  / 3

      except:
       print("mal3")

      try:
       peso_Plocation= peso_Plocation + peso_Plocation1
       peso_Pemail = peso_Pemail + peso_Pemail1
       peso_Pdevice = peso_Pdevice + peso_Pdevice1
       peso_Pimei = peso_Pimei + peso_Pimei1
       peso_Pserial = peso_Pserial + peso_Pserial1
       peso_Pmac = peso_Pmac + peso_Pmac1 + peso_Pmac2
       peso_Padvertiser = peso_Padvertiser + peso_Padvertiser1
       umbral_verde = umbral_verde + umbral_verde1
       umbral_naranja = umbral_naranja + umbral_naranja1
       umbral_rojo = umbral_rojo + umbral_rojo1

       peso_Pinternal_dst = peso_Pinternal_dst + peso_Pinternal_dst1
       peso_Pads_dst = peso_Pads_dst + peso_Pads_dst1
       peso_Panalytics_dst = peso_Panalytics_dst + peso_Panalytics_dst1
       peso_Psns_dst = peso_Psns_dst + peso_Psns_dst1
       peso_Pdevelop_dst = peso_Pdevelop_dst + peso_Pdevelop_dst1

       peso_Plocation= peso_Plocation / 2
       peso_Pemail = peso_Pemail / 2
       peso_Pdevice = peso_Pdevice / 2
       peso_Pimei = peso_Pimei / 2
       peso_Pserial = peso_Pserial / 2
       peso_Pmac = peso_Pmac / 2
       peso_Padvertiser = peso_Padvertiser / 2
       umbral_verde = umbral_verde / 2
       umbral_naranja = umbral_naranja / 2
       umbral_rojo = umbral_rojo  / 2

       peso_Pinternal_dst = peso_Pinternal_dst  / 2
       peso_Pads_dst = peso_Pads_dst  / 2
       peso_Panalytics_dst = peso_Panalytics_dst  / 2
       peso_Psns_dst = peso_Psns_dst  / 2
       peso_Pdevelop_dst = peso_Pdevelop_dst  / 2

      except: 
       print("mal4")   
      
      ###fichero con pesos y umbrales: 


      ################################
      #my_file = open("fichero_datos"+modelctrstr2, "w")
      my_file = open("fichero_datos"+"global", "w")
      #my_file.write("Pesos actualizados modelo Ant"+"\n"+"\n"+sub_string_Plocation+"\n"+sub_string_Pemail+"\n"+sub_string_Pdevice+"\n"+sub_string_Pimei+"\n"+sub_string_Pserial+"\n"+sub_string_Pmac+"\n"+sub_string_Padvertiser+"\n"+sub_string_verde+"\n"+sub_string_naranja+"\n"+sub_string_rojo)
      my_file.write("Pesos actualizados modelo Ant"+"\n"+"\n"+str(peso_Plocation)+"\n"+str(peso_Pemail)+"\n"+str(peso_Pdevice)+"\n"+str(peso_Pimei)+"\n"+str(peso_Pserial)+"\n"+str(peso_Pmac)+"\n"+str(peso_Padvertiser)+"\n"+str(umbral_verde)+"\n"+str(umbral_naranja)+"\n"+str(umbral_rojo))

      my_file.close()

     # Ejemplo para 4 datos: Location, Email, IMEI y Device ID:
      filtracion_location = tf.placeholder(tf.float32, name='location_input')
      filtracion_email = tf.placeholder(tf.float32, name='email_input')
      filtracion_imei = tf.placeholder(tf.float32, name='imei_input')
      filtracion_device = tf.placeholder(tf.float32, name='device_input')
      filtracion_serialnumber = tf.placeholder(tf.float32, name='serialnumber_input')
      filtracion_macaddress = tf.placeholder(tf.float32, name='macaddress_input')
      filtracion_advertiser = tf.placeholder(tf.float32, name='advertiser_input')


      #destinos: 5 posibles--> Internal, Ads, Analytics, Sns, Develop               1 o 0 si se manda ahi o no
      destino_internal = tf.placeholder(tf.float32, name='internal_dst_input')
      destino_ads = tf.placeholder(tf.float32, name='ads_dst_input')
      destino_analytics = tf.placeholder(tf.float32, name='analytics_dst_input')
      destino_sns = tf.placeholder(tf.float32, name='sns_dst_input')
      destino_develop = tf.placeholder(tf.float32, name='develop_dst_input')

  # objetivo: answer: 1 o 0
      y_ = tf.placeholder(tf.float32, name='target')

      #distintos pesos para cada dato: nombre= Px
      Plocation = tf.Variable(peso_Plocation, name='Plocation') #9
      Pemail = tf.Variable(peso_Pemail, name='Pemail') #8
      Pimei = tf.Variable(peso_Pimei, name='Pimei') #3
      Pdevice = tf.Variable(peso_Pdevice, name='Pdevice') #2

      #Pdni = tf.placeholder(tf.float32, name='Pdni') #10
      #Pphone = tf.placeholder(tf.float32, name='Pphone') #8
      Pserialnumber = tf.Variable(peso_Pserial, name='Pserialnumber') #3
      Pmacaddress = tf.Variable(peso_Pmac, name='Pmacaddress') #1
      Padvertiser = tf.Variable(peso_Padvertiser, name='Padvertiser') #5

      #Pesos para cada destino:
      Pinternal_dst = tf.Variable(peso_Pinternal_dst, name='Pinternal_dst') #5
      Pads_dst = tf.Variable(peso_Pads_dst, name='Pads_dst') #5
      Panalytics_dst = tf.Variable(peso_Panalytics_dst, name='Panalytics_dst') #5
      Psns_dst = tf.Variable(peso_Psns_dst, name='Psns_dst') #5
      Pdevelop_dst = tf.Variable(peso_Pdevelop_dst, name='Pdevelop_dst') #5



  # umbral de dec
  # umbral = tf.constant(10., name='umbral')
      #umbral = tf.placeholder(tf.float32, name='umbral')

    #  aux = tf.Variable(0., name='Vaux')
     #salida 0 o 1

      #salida del modelo
      y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), 0.0)
      #y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), aux)
      y_aux2 = tf.add(tf.multiply(filtracion_email, Pemail), y_aux1)
      y_aux3 = tf.add(tf.multiply(filtracion_imei, Pimei), y_aux2)
      y_aux4 = tf.add(tf.multiply(filtracion_device, Pdevice), y_aux3)


      #y_aux11 = tf.add(tf.multiply(filtracion_dni, Pdni), y_aux4)
      #y_aux12 = tf.add(tf.multiply(filtracion_phone, Pphone), y_aux11)
      y_aux13 = tf.add(tf.multiply(filtracion_serialnumber, Pserialnumber), y_aux4)
      y_aux14 = tf.add(tf.multiply(filtracion_macaddress, Pmacaddress), y_aux13)
      #y_aux14 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

      #destinos
      y_aux1_dst = tf.add(tf.multiply(destino_internal, Pinternal_dst), 0.0)
      y_aux2_dst = tf.add(tf.multiply(destino_ads, Pads_dst), y_aux1_dst)
      y_aux3_dst = tf.add(tf.multiply(destino_analytics, Panalytics_dst), y_aux2_dst)
      y_aux4_dst = tf.add(tf.multiply(destino_sns, Psns_dst), y_aux3_dst)
      y_aux5_dst = tf.add(tf.multiply(destino_develop, Pdevelop_dst), y_aux4_dst)
      #

      y_aux_final1 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

      y = tf.multiply(y_aux_final1, y_aux5_dst)

      y = tf.identity(y, name='output')

      loss = tf.reduce_mean(tf.square(y - y_))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
      train_op = optimizer.minimize(loss, name='train')

      init = tf.global_variables_initializer()

      # Creating a tf.train.Saver adds operations to the graph to save and
      # restore variables from checkpoints.

      saver_def = tf.train.Saver().as_saver_def
      ###NO VA EL SAVER DE GRAH DEF##########################
      # with open('graph.pb', 'wb') as f:
      #     f.write(tf.get_default_graph().as_graph_def())

      saver = tf.train.Saver()
      # Training
      # saver.save(sess, your_path + "/checkpoint_name.ckpt")
      # TensorFlow session
      sess = tf.Session()
      sess.run(init)
     # saver.save(sess, "-"+ NOMBRE_CHECKPOINTS + "-.ckpt")  "checkpoint_actualizado.ckpt"
     # saver.save(sess, "checkpoint_actualizado_"+modelctrstr2+".ckpt")
     # saver.save(sess, "checkpoint_actualizado_"+modelctrstr+".ckpt")
      saver.save(sess, "checkpoint_actualizado_"+"global"+".ckpt")
      print("se han creado fichero con nombre ... "+ modelctrstr2)

      #
      app.config["FICHERO_CHECK"] = ""
    #  zipf = zipfile.ZipFile(NOMBRE_CHECKPOINTS + '.ckpt.meta.zip','w'. zipfile.ZIP_DEFLATED)
   #   zipObj = ZipFile("ficherotestnoviembre.zip", 'w')
   #   zipObj.write(NOMBRE_CHECKPOINTS + ".ckpt.meta")
   #   zipObj.close()
   ##   return send_from_directory(app.config["FICHERO_CHECK"], filename='ficherotestnoviembre.zip', as_attachment=True)
      zipObj = ZipFile('test.zip', 'w')
      zipObj.write("checkpoint")
    #  zipObj.write("-"+ NOMBRE_CHECKPOINTS + "-.ckpt"+'.index')
    #  zipObj.write("-" + NOMBRE_CHECKPOINTS + "-.ckpt" + '.data-00000-of-00001')
    #  zipObj.write("-" + NOMBRE_CHECKPOINTS + "-.ckpt.meta") # no se si igual .meta
     # zipObj.write("checkpoint_actualizado_"+modelctrstr2+".ckpt" + '.index')  #si
     # zipObj.write("checkpoint_actualizado_"+modelctrstr2+".ckpt" + '.data-00000-of-00001') #si
  #    zipObj.write("checkpoint_actualizado_"+modelctrstr+".ckpt.meta") # no se si igual .meta
      zipObj.write("checkpoint_actualizado_"+"global"+".ckpt" + '.index')  #si
      zipObj.write("checkpoint_actualizado_"+"global"+".ckpt" + '.data-00000-of-00001')
     # zipObj.write("fichero_datos"+modelctrstr2)
      zipObj.write("fichero_datos"+"global")
      zipObj.close()
      return send_from_directory(app.config["FICHERO_CHECK"], filename='test.zip', as_attachment=True)
  #    return flask.send_file('test.zip', mimetype = 'zip',attachment_filename= 'test.zip', as_attachment= True)

  else: # actualizo el modelo
    return




@app.route("/descargar_C", methods = ['GET'])
def descargar_C():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints

     # global numero
     # numero = numero + 1
     # global modelctr
     # modelctr = modelctr + 1
 #     global NOMBRE_CHECKPOINTS
 #     NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
  #    global NOMBRE_ZIP
  #    NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
  #    global modelctrstr
  #    modelctrstr = str(modelctr)
  #    ################################
      peso_w = 0
      peso_b = 0

      storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
      BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
      bucket = storage_client.get_bucket(BUCKET_NAME)

      global NumDescargasModelo_C
      NumDescargasModelo_C = NumDescargasModelo_C + 1


   #   blob = bucket.blob('fichero_pesos')
      global nombre_fichero_descarga

      print("fichero descargado deberia ser " + nombre_fichero_descarga)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
      blob = bucket.blob(nombre_fichero_descarga)
      print("Descargando fichero:  "+nombre_fichero_descarga)
      blob.download_to_filename("ejemplo_pesos")
      print("descargado fichero: fichero_pesos")

      f = open('ejemplo_pesos')
      string_list = f.readlines()
      f.close()

      sub_string_Plocation_aux = string_list[2]  # la tercera linea del fichero
      sub_string_Pemail_aux = string_list[3]
      sub_string_Pdevice_aux = string_list[4]
      sub_string_Pimei_aux = string_list[5]
      sub_string_Pserial_aux = string_list[6]
      sub_string_Pmac_aux = string_list[7]
      sub_string_Padvertiser_aux = string_list[8]
      sub_string_verde_aux = string_list[9]
      sub_string_naranja_aux = string_list[10]
      sub_string_rojo_aux = string_list[11]
      #destinos servidor
      sub_string_Pinternal_dst_aux = string_list[12]
      sub_string_Pads_dst_aux = string_list[13]
      sub_string_Panalytics_dst_aux = string_list[14]
      sub_string_Psns_dst_aux = string_list[15]
      sub_string_Pdevelop_dst_aux = string_list[16]

       #p
      try:
       sub_string_Plocation = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
      except:
       sub_string_Plocation = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
      try:
       sub_string_Pemail = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
      except:
       sub_string_Pemail = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
      try:
       sub_string_Pdevice = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
      except:
       sub_string_Pdevice = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2]
      try:
       sub_string_Pimei = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
      except:
       sub_string_Pimei = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2]
      try:
       sub_string_Pserial = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
      except:
       sub_string_Pserial = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
      try:
       sub_string_Pmac = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
      except:
       sub_string_Pmac = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
      try:
       sub_string_Padvertiser = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
      except:
       sub_string_Padvertiser = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
      try:
       sub_string_verde = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
      except:
       sub_string_verde = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
      try:
       sub_string_naranja = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
      except:
       sub_string_naranja = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2]
      try:
       sub_string_rojo = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
      except:
       sub_string_rojo = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
      try:
       sub_string_Pinternal_dst = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
      except:
       sub_string_Pinternal_dst = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
      try:
       sub_string_Pads_dst = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
      except:
       sub_string_Pads_dst = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
      try:
       sub_string_Panalytics_dst = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
      except:
       sub_string_Panalytics_dst = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
      try:
       sub_string_Psns_dst = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
      except:
       sub_string_Psns_dst = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
      try:
       sub_string_Pdevelop_dst = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
      except:
       sub_string_Pdevelop_dst = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]



      peso_Plocation = float(sub_string_Plocation)
      peso_Pemail = float(sub_string_Pemail)
      peso_Pdevice = float(sub_string_Pdevice)
      peso_Pimei = float(sub_string_Pimei)
      peso_Pserial = float(sub_string_Pserial)
      peso_Pmac = float(sub_string_Pmac)
      peso_Padvertiser = float(sub_string_Padvertiser)

      umbral_verde = float(sub_string_verde)
      umbral_naranja = float(sub_string_naranja)
      umbral_rojo = float(sub_string_rojo)

      peso_Pinternal_dst = float(sub_string_Pinternal_dst)
      peso_Pads_dst = float(sub_string_Pads_dst)
      peso_Panalytics_dst = float(sub_string_Panalytics_dst)
      peso_Psns_dst = float(sub_string_Psns_dst)
      peso_Pdevelop_dst = float(sub_string_Pdevelop_dst)
      
      try:
      #2    
       global nombre_fichero_descarga1

       print("fichero descargado deberia ser " + nombre_fichero_descarga1)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
       blob = bucket.blob(nombre_fichero_descarga1)
       print("Descargando fichero:  "+nombre_fichero_descarga1)
       blob.download_to_filename("ejemplo_pesos")
       print("descargado fichero: fichero_pesos")

       f = open('ejemplo_pesos1')
       string_list1 = f.readlines()
       f.close()

       sub_string_Plocation_aux = string_list1[2]  # la tercera linea del fichero
       sub_string_Pemail_aux = string_list1[3]
       sub_string_Pdevice_aux = string_list1[4]
       sub_string_Pimei_aux = string_list1[5]
       sub_string_Pserial_aux = string_list1[6]
       sub_string_Pmac_aux = string_list1[7]
       sub_string_Padvertiser_aux = string_list1[8]
       sub_string_verde_aux = string_list1[9]
       sub_string_naranja_aux = string_list1[10]
       sub_string_rojo_aux = string_list1[11]
       #destinos servidor
       sub_string_Pinternal_dst_aux = string_list1[12]
       sub_string_Pads_dst_aux = string_list1[13]
       sub_string_Panalytics_dst_aux = string_list1[14]
       sub_string_Psns_dst_aux = string_list1[15]
       sub_string_Pdevelop_dst_aux = string_list1[16]

       #
       try:
        sub_string_Plocation1 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
       except:
        sub_string_Plocation1 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
       try:
        sub_string_Pemail1 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
       except:
        sub_string_Pemail1 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
       try:
        sub_string_Pdevice1 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
       except:
        sub_string_Pdevice1 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2]
       try:
        sub_string_Pimei1 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
       except:
        sub_string_Pimei1 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2]
       try:
        sub_string_Pserial1 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
       except:
        sub_string_Pserial1 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
       try:
        sub_string_Pmac1 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
       except:
        sub_string_Pmac1 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
       try:
        sub_string_Padvertiser1 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
       except:
        sub_string_Padvertiser1 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
       try:
        sub_string_verde1 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
       except:
        sub_string_verde1 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
       try:
        sub_string_naranja1 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
       except:
        sub_string_naranja1 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2]
       try:
        sub_string_rojo1 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
       except:
        sub_string_rojo1 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
       try:
        sub_string_Pinternal_dst1 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
       except:
        sub_string_Pinternal_dst1 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
       try:
        sub_string_Pads_dst1 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
       except:
        sub_string_Pads_dst1 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
       try:
        sub_string_Panalytics_dst1 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
       except:
        sub_string_Panalytics_dst1 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
       try:
        sub_string_Psns_dst1 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
       except:
        sub_string_Psns_dst1 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
       try:
        sub_string_Pdevelop_dst1 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
       except:
        sub_string_Pdevelop_dst1 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]


       peso_Plocation1 = float(sub_string_Plocation1)
       peso_Pemail1 = float(sub_string_Pemail1)
       peso_Pdevice1 = float(sub_string_Pdevice1)
       peso_Pimei1 = float(sub_string_Pimei1)
       peso_Pserial1 = float(sub_string_Pserial1)
       peso_Pmac1 = float(sub_string_Pmac1)
       peso_Padvertiser1 = float(sub_string_Padvertiser1)

       umbral_verde1 = float(sub_string_verde1)
       umbral_naranja1 = float(sub_string_naranja1)
       umbral_rojo1 = float(sub_string_rojo1)

       peso_Pinternal_dst1 = float(sub_string_Pinternal_dst1)
       peso_Pads_dst1 = float(sub_string_Pads_dst1)
       peso_Panalytics_dst1 = float(sub_string_Panalytics_dst1)
       peso_Psns_dst1 = float(sub_string_Psns_dst1)
       peso_Pdevelop_dst1 = float(sub_string_Pdevelop_dst1)
      except:
       print("mal") 

      try:
      #3    
       global nombre_fichero_descarga2

       print("fichero descargado deberia ser " + nombre_fichero_descarga2)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
       blob = bucket.blob(nombre_fichero_descarga2)
       print("Descargando fichero:  "+nombre_fichero_descarga2)
       blob.download_to_filename("ejemplo_pesos")
       print("descargado fichero: fichero_pesos")

       f = open('ejemplo_pesos2')
       string_list2 = f.readlines()
       f.close()

       sub_string_Plocation_aux = string_list2[2]  # la tercera linea del fichero
       sub_string_Pemail_aux = string_list2[3]
       sub_string_Pdevice_aux = string_list2[4]
       sub_string_Pimei_aux = string_list2[5]
       sub_string_Pserial_aux = string_list2[6]
       sub_string_Pmac_aux = string_list2[7]
       sub_string_Padvertiser_aux = string_list2[8]
       sub_string_verde_aux = string_list2[9]
       sub_string_naranja_aux = string_list2[10]
       sub_string_rojo_aux = string_list2[11]
       #destinos servidor
       sub_string_Pinternal_dst_aux = string_list2[12]
       sub_string_Pads_dst_aux = string_list2[13]
       sub_string_Panalytics_dst_aux = string_list2[14]
       sub_string_Psns_dst_aux = string_list2[15]
       sub_string_Pdevelop_dst_aux = string_list2[16]

       #
       try:
        sub_string_Plocation2 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
       except:
        sub_string_Plocation2 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
       try:
        sub_string_Pemail2 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
       except:
        sub_string_Pemail2 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
       try:
        sub_string_Pdevice2 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
       except:
        sub_string_Pdevice2 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2]
       try:
        sub_string_Pime2 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
       except:
        sub_string_Pimei2 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2]
       try:
        sub_string_Pserial2 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
       except:
        sub_string_Pserial2 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
       try:
        sub_string_Pmac2 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
       except:
        sub_string_Pmac2 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
       try:
        sub_string_Padvertiser2 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
       except:
        sub_string_Padvertiser2 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
       try:
        sub_string_verde2 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
       except:
        sub_string_verde2 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
       try:
        sub_string_naranja2 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
       except:
        sub_string_naranja2 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2]
       try:
        sub_string_rojo2 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
       except:
        sub_string_rojo2 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
       try:
        sub_string_Pinternal_dst2 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
       except:
        sub_string_Pinternal_dst2 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
       try:
        sub_string_Pads_dst2 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
       except:
        sub_string_Pads_dst2 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
       try:
        sub_string_Panalytics_dst2 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
       except:
        sub_string_Panalytics_dst2 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
       try:
        sub_string_Psns_dst2 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
       except:
        sub_string_Psns_dst2 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
       try:
        sub_string_Pdevelop_dst2 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
       except:
        sub_string_Pdevelop_dst2 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]





       peso_Plocation2 = float(sub_string_Plocation2)
       peso_Pemail2 = float(sub_string_Pemail2)
       peso_Pdevice2 = float(sub_string_Pdevice2)
       peso_Pimei2 = float(sub_string_Pimei2)
       peso_Pserial2 = float(sub_string_Pserial2)
       peso_Pmac2 = float(sub_string_Pmac2)
       peso_Padvertiser2 = float(sub_string_Padvertiser2)

       umbral_verde2 = float(sub_string_verde2)
       umbral_naranja2 = float(sub_string_naranja2)
       umbral_rojo2 = float(sub_string_rojo2)

       peso_Pinternal_dst2 = float(sub_string_Pinternal_dst2)
       peso_Pads_dst2 = float(sub_string_Pads_dst2)
       peso_Panalytics_dst2 = float(sub_string_Panalytics_dst2)
       peso_Psns_dst2 = float(sub_string_Psns_dst2)
       peso_Pdevelop_dst2 = float(sub_string_Pdevelop_dst2)
      except:
       print("mal2")

      #se hace la media: 
      
      peso_Plocation= peso_Plocation 
      peso_Pemail = peso_Pemail
      peso_Pdevice = peso_Pdevice
      peso_Pimei = peso_Pimei
      peso_Pserial = peso_Pserial
      peso_Pmac = peso_Pmac
      peso_Padvertiser = peso_Padvertiser
      umbral_verde = umbral_verde
      umbral_naranja = umbral_naranja
      umbral_rojo = umbral_rojo

      peso_Pinternal_dst = peso_Pinternal_dst
      peso_Pads_dst = peso_Pads_dst
      peso_Panalytics_dst = peso_Panalytics_dst
      peso_Psns_dst = peso_Psns_dst
      peso_Pdevelop_dst = peso_Pdevelop_dst

      try:
       peso_Plocation= peso_Plocation + peso_Plocation1 + peso_Plocation2
       peso_Pemail = peso_Pemail + peso_Pemail1 + peso_Pemail2
       peso_Pdevice = peso_Pdevice + peso_Pdevice1 + peso_Pdevice2
       peso_Pimei = peso_Pimei + peso_Pimei1 + peso_Pimei2
       peso_Pserial = peso_Pserial + peso_Pserial1 + peso_Pserial2
       peso_Pmac = peso_Pmac + peso_Pmac1 + peso_Pmac2
       peso_Padvertiser = peso_Padvertiser + peso_Padvertiser1 + peso_Padvertiser2
       umbral_verde = umbral_verde + umbral_verde1 + umbral_verde2
       umbral_naranja = umbral_naranja + umbral_naranja1 + umbral_naranja2
       umbral_rojo = umbral_rojo + umbral_rojo1 + umbral_rojo2

       peso_Pinternal_dst = peso_Pinternal_dst + peso_Pinternal_dst1 + peso_Pinternal_dst2
       peso_Pads_dst = peso_Pads_dst + peso_Pads_dst1 + peso_Pads_dst2
       peso_Panalytics_dst = peso_Panalytics_dst + peso_Panalytics_dst1 + peso_Panalytics_dst2
       peso_Psns_dst = peso_Psns_dst + peso_Psns_dst1 + peso_Psns_dst2
       peso_Pdevelop_dst = peso_Pdevelop_dst + peso_Pdevelop_dst1 + peso_Pdevelop_dst2

       peso_Plocation= peso_Plocation / 3
       peso_Pemail = peso_Pemail / 3
       peso_Pdevice = peso_Pdevice / 3
       peso_Pimei = peso_Pimei / 3
       peso_Pserial = peso_Pserial / 3
       peso_Pmac = peso_Pmac / 3
       peso_Padvertiser = peso_Padvertiser / 3
       umbral_verde = umbral_verde / 3
       umbral_naranja = umbral_naranja / 3
       umbral_rojo = umbral_rojo  / 3

       peso_Pinternal_dst = peso_Pinternal_dst  / 3
       peso_Pads_dst = peso_Pads_dst  / 3
       peso_Panalytics_dst = peso_Panalytics_dst  / 3
       peso_Psns_dst = peso_Psns_dst  / 3
       peso_Pdevelop_dst = peso_Pdevelop_dst  / 3

      except:
       print("mal3")

      try:
       peso_Plocation= peso_Plocation + peso_Plocation1
       peso_Pemail = peso_Pemail + peso_Pemail1
       peso_Pdevice = peso_Pdevice + peso_Pdevice1
       peso_Pimei = peso_Pimei + peso_Pimei1
       peso_Pserial = peso_Pserial + peso_Pserial1
       peso_Pmac = peso_Pmac + peso_Pmac1 + peso_Pmac2
       peso_Padvertiser = peso_Padvertiser + peso_Padvertiser1
       umbral_verde = umbral_verde + umbral_verde1
       umbral_naranja = umbral_naranja + umbral_naranja1
       umbral_rojo = umbral_rojo + umbral_rojo1

       peso_Pinternal_dst = peso_Pinternal_dst + peso_Pinternal_dst1
       peso_Pads_dst = peso_Pads_dst + peso_Pads_dst1
       peso_Panalytics_dst = peso_Panalytics_dst + peso_Panalytics_dst1
       peso_Psns_dst = peso_Psns_dst + peso_Psns_dst1
       peso_Pdevelop_dst = peso_Pdevelop_dst + peso_Pdevelop_dst1

       peso_Plocation= peso_Plocation / 2
       peso_Pemail = peso_Pemail / 2
       peso_Pdevice = peso_Pdevice / 2
       peso_Pimei = peso_Pimei / 2
       peso_Pserial = peso_Pserial / 2
       peso_Pmac = peso_Pmac / 2
       peso_Padvertiser = peso_Padvertiser / 2
       umbral_verde = umbral_verde / 2
       umbral_naranja = umbral_naranja / 2
       umbral_rojo = umbral_rojo  / 2

       peso_Pinternal_dst = peso_Pinternal_dst  / 2
       peso_Pads_dst = peso_Pads_dst  / 2
       peso_Panalytics_dst = peso_Panalytics_dst  / 2
       peso_Psns_dst = peso_Psns_dst  / 2
       peso_Pdevelop_dst = peso_Pdevelop_dst  / 2

      except: 
       print("mal4")   
      
      ###fichero con pesos y umbrales:  


      ################################
     # my_file = open("fichero_datos"+modelctrstr3, "w")
      my_file = open("fichero_datos"+"global", "w")
      #my_file.write("Pesos actualizados modelo Ant"+"\n"+"\n"+sub_string_Plocation+"\n"+sub_string_Pemail+"\n"+sub_string_Pdevice+"\n"+sub_string_Pimei+"\n"+sub_string_Pserial+"\n"+sub_string_Pmac+"\n"+sub_string_Padvertiser+"\n"+sub_string_verde+"\n"+sub_string_naranja+"\n"+sub_string_rojo)
      my_file.write("Pesos actualizados modelo Ant"+"\n"+"\n"+str(peso_Plocation)+"\n"+str(peso_Pemail)+"\n"+str(peso_Pdevice)+"\n"+str(peso_Pimei)+"\n"+str(peso_Pserial)+"\n"+str(peso_Pmac)+"\n"+str(peso_Padvertiser)+"\n"+str(umbral_verde)+"\n"+str(umbral_naranja)+"\n"+str(umbral_rojo))

      my_file.close()

     # Ejemplo para 4 datos: Location, Email, IMEI y Device ID:
      filtracion_location = tf.placeholder(tf.float32, name='location_input')
      filtracion_email = tf.placeholder(tf.float32, name='email_input')
      filtracion_imei = tf.placeholder(tf.float32, name='imei_input')
      filtracion_device = tf.placeholder(tf.float32, name='device_input')
      filtracion_serialnumber = tf.placeholder(tf.float32, name='serialnumber_input')
      filtracion_macaddress = tf.placeholder(tf.float32, name='macaddress_input')
      filtracion_advertiser = tf.placeholder(tf.float32, name='advertiser_input')


      #destinos: 5 posibles--> Internal, Ads, Analytics, Sns, Develop               1 o 0 si se manda ahi o no
      destino_internal = tf.placeholder(tf.float32, name='internal_dst_input')
      destino_ads = tf.placeholder(tf.float32, name='ads_dst_input')
      destino_analytics = tf.placeholder(tf.float32, name='analytics_dst_input')
      destino_sns = tf.placeholder(tf.float32, name='sns_dst_input')
      destino_develop = tf.placeholder(tf.float32, name='develop_dst_input')

  # objetivo: answer: 1 o 0
      y_ = tf.placeholder(tf.float32, name='target')

      #distintos pesos para cada dato: nombre= Px
      Plocation = tf.Variable(peso_Plocation, name='Plocation') #9
      Pemail = tf.Variable(peso_Pemail, name='Pemail') #8
      Pimei = tf.Variable(peso_Pimei, name='Pimei') #3
      Pdevice = tf.Variable(peso_Pdevice, name='Pdevice') #2

      #Pdni = tf.placeholder(tf.float32, name='Pdni') #10
      #Pphone = tf.placeholder(tf.float32, name='Pphone') #8
      Pserialnumber = tf.Variable(peso_Pserial, name='Pserialnumber') #3
      Pmacaddress = tf.Variable(peso_Pmac, name='Pmacaddress') #1
      Padvertiser = tf.Variable(peso_Padvertiser, name='Padvertiser') #5

      #Pesos para cada destino:
      Pinternal_dst = tf.Variable(peso_Pinternal_dst, name='Pinternal_dst') #5
      Pads_dst = tf.Variable(peso_Pads_dst, name='Pads_dst') #5
      Panalytics_dst = tf.Variable(peso_Panalytics_dst, name='Panalytics_dst') #5
      Psns_dst = tf.Variable(peso_Psns_dst, name='Psns_dst') #5
      Pdevelop_dst = tf.Variable(peso_Pdevelop_dst, name='Pdevelop_dst') #5



  # umbral de decisin
  # umbral = tf.constant(10., name='umbral')
      #umbral = tf.placeholder(tf.float32, name='umbral')

    #  aux = tf.Variable(0., name='Vaux')
     #salida 0 o 1

      #salida del modelo
      y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), 0.0)
      #y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), aux)
      y_aux2 = tf.add(tf.multiply(filtracion_email, Pemail), y_aux1)
      y_aux3 = tf.add(tf.multiply(filtracion_imei, Pimei), y_aux2)
      y_aux4 = tf.add(tf.multiply(filtracion_device, Pdevice), y_aux3)


      #y_aux11 = tf.add(tf.multiply(filtracion_dni, Pdni), y_aux4)
      #y_aux12 = tf.add(tf.multiply(filtracion_phone, Pphone), y_aux11)
      y_aux13 = tf.add(tf.multiply(filtracion_serialnumber, Pserialnumber), y_aux4)
      y_aux14 = tf.add(tf.multiply(filtracion_macaddress, Pmacaddress), y_aux13)
      #y_aux14 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

      #destinos
      y_aux1_dst = tf.add(tf.multiply(destino_internal, Pinternal_dst), 0.0)
      y_aux2_dst = tf.add(tf.multiply(destino_ads, Pads_dst), y_aux1_dst)
      y_aux3_dst = tf.add(tf.multiply(destino_analytics, Panalytics_dst), y_aux2_dst)
      y_aux4_dst = tf.add(tf.multiply(destino_sns, Psns_dst), y_aux3_dst)
      y_aux5_dst = tf.add(tf.multiply(destino_develop, Pdevelop_dst), y_aux4_dst)
      #

      y_aux_final1 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

      y = tf.multiply(y_aux_final1, y_aux5_dst)

      y = tf.identity(y, name='output')

      loss = tf.reduce_mean(tf.square(y - y_))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
      train_op = optimizer.minimize(loss, name='train')

      init = tf.global_variables_initializer()

      # Creating a tf.train.Saver adds operations to the graph to save and
      # restore variables from checkpoints.

      saver_def = tf.train.Saver().as_saver_def
      ###NO VA EL SAVER DE GRAH DEF##########################
      # with open('graph.pb', 'wb') as f:
      #     f.write(tf.get_default_graph().as_graph_def())

      saver = tf.train.Saver()
      # Training
      # saver.save(sess, your_path + "/checkpoint_name.ckpt")
      # TensorFlow session
      sess = tf.Session()
      sess.run(init)
     # saver.save(sess, "-"+ NOMBRE_CHECKPOINTS + "-.ckpt")  "checkpoint_actualizado.ckpt"
      #saver.save(sess, "checkpoint_actualizado_"+modelctrstr3+".ckpt")
     # saver.save(sess, "checkpoint_actualizado_"+modelctrstr+".ckpt")
      saver.save(sess, "checkpoint_actualizado_"+"global"+".ckpt")
      print("se han creado fichero con nombre ... "+ modelctrstr3)

      #
      app.config["FICHERO_CHECK"] = ""
    #  zipf = zipfile.ZipFile(NOMBRE_CHECKPOINTS + '.ckpt.meta.zip','w'. zipfile.ZIP_DEFLATED)
   #   zipObj = ZipFile("ficherotestnoviembre.zip", 'w')
   #   zipObj.write(NOMBRE_CHECKPOINTS + ".ckpt.meta")
   #   zipObj.close()
   ##   return send_from_directory(app.config["FICHERO_CHECK"], filename='ficherotestnoviembre.zip', as_attachment=True)
      zipObj = ZipFile('test.zip', 'w')
      zipObj.write("checkpoint")
    #  zipObj.write("-"+ NOMBRE_CHECKPOINTS + "-.ckpt"+'.index')
    #  zipObj.write("-" + NOMBRE_CHECKPOINTS + "-.ckpt" + '.data-00000-of-00001')
    #  zipObj.write("-" + NOMBRE_CHECKPOINTS + "-.ckpt.meta") # no se si igual .meta
    #  zipObj.write("checkpoint_actualizado_"+modelctrstr3+".ckpt" + '.index')
    #  zipObj.write("checkpoint_actualizado_"+modelctrstr3+".ckpt" + '.data-00000-of-00001')
  #    zipObj.write("checkpoint_actualizado_"+modelctrstr+".ckpt.meta") # no se si igual .meta
     # zipObj.write("fichero_datos"+modelctrstr3)
      zipObj.write("checkpoint_actualizado_"+"global"+".ckpt" + '.index')  #si
      zipObj.write("checkpoint_actualizado_"+"global"+".ckpt" + '.data-00000-of-00001')
     
      zipObj.write("fichero_datos"+"global")
      zipObj.close()
      return send_from_directory(app.config["FICHERO_CHECK"], filename='test.zip', as_attachment=True)
  #    return flask.send_file('test.zip', mimetype = 'zip',attachment_filename= 'test.zip', as_attachment= True)

  else: # actualizo el modelo
    return





##

@app.route("/descargar_D", methods = ['GET'])
def descargar_D():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints

     # global numero
     # numero = numero + 1
     # global modelctr
     # modelctr = modelctr + 1
 #     global NOMBRE_CHECKPOINTS
 #     NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
  #    global NOMBRE_ZIP
  #    NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
  #    global modelctrstr
  #    modelctrstr = str(modelctr)
  #    ################################
      peso_w = 0
      peso_b = 0

      storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
      BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
      bucket = storage_client.get_bucket(BUCKET_NAME)


      global NumDescargasModelo_D
      NumDescargasModelo_D = NumDescargasModelo_D + 1
   #   blob = bucket.blob('fichero_pesos')
      global nombre_fichero_descarga

      print("fichero descargado deberia ser " + nombre_fichero_descarga)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
      blob = bucket.blob(nombre_fichero_descarga)
      print("Descargando fichero:  "+nombre_fichero_descarga)
      blob.download_to_filename("ejemplo_pesos")
      print("descargado fichero: fichero_pesos")

      f = open('ejemplo_pesos')
      string_list = f.readlines()
      f.close()

      sub_string_Plocation_aux = string_list[2]  # la tercera linea del fichero
      sub_string_Pemail_aux = string_list[3]
      sub_string_Pdevice_aux = string_list[4]
      sub_string_Pimei_aux = string_list[5]
      sub_string_Pserial_aux = string_list[6]
      sub_string_Pmac_aux = string_list[7]
      sub_string_Padvertiser_aux = string_list[8]
      sub_string_verde_aux = string_list[9]
      sub_string_naranja_aux = string_list[10]
      sub_string_rojo_aux = string_list[11]
      #destinos servidor
      sub_string_Pinternal_dst_aux = string_list[12]
      sub_string_Pads_dst_aux = string_list[13]
      sub_string_Panalytics_dst_aux = string_list[14]
      sub_string_Psns_dst_aux = string_list[15]
      sub_string_Pdevelop_dst_aux = string_list[16]

       #posible error si se busca y no hay nmero Ejemplo: 5.0, no existe [3] pero en 10.0 
      try:
       sub_string_Plocation = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
      except:
       sub_string_Plocation = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
      try:
       sub_string_Pemail = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
      except:
       sub_string_Pemail = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
      try:
       sub_string_Pdevice = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
      except:
       sub_string_Pdevice = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2]
      try:
       sub_string_Pimei = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
      except:
       sub_string_Pimei = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2]
      try:
       sub_string_Pserial = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
      except:
       sub_string_Pserial = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
      try:
       sub_string_Pmac = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
      except:
       sub_string_Pmac = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
      try:
       sub_string_Padvertiser = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
      except:
       sub_string_Padvertiser = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
      try:
       sub_string_verde = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
      except:
       sub_string_verde = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
      try:
       sub_string_naranja = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
      except:
       sub_string_naranja = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2]
      try:
       sub_string_rojo = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
      except:
       sub_string_rojo = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
      try:
       sub_string_Pinternal_dst = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
      except:
       sub_string_Pinternal_dst = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
      try:
       sub_string_Pads_dst = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
      except:
       sub_string_Pads_dst = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
      try:
       sub_string_Panalytics_dst = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
      except:
       sub_string_Panalytics_dst = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
      try:
       sub_string_Psns_dst = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
      except:
       sub_string_Psns_dst = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
      try:
       sub_string_Pdevelop_dst = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
      except:
       sub_string_Pdevelop_dst = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]



      peso_Plocation = float(sub_string_Plocation)
      peso_Pemail = float(sub_string_Pemail)
      peso_Pdevice = float(sub_string_Pdevice)
      peso_Pimei = float(sub_string_Pimei)
      peso_Pserial = float(sub_string_Pserial)
      peso_Pmac = float(sub_string_Pmac)
      peso_Padvertiser = float(sub_string_Padvertiser)

      umbral_verde = float(sub_string_verde)
      umbral_naranja = float(sub_string_naranja)
      umbral_rojo = float(sub_string_rojo)

      peso_Pinternal_dst = float(sub_string_Pinternal_dst)
      peso_Pads_dst = float(sub_string_Pads_dst)
      peso_Panalytics_dst = float(sub_string_Panalytics_dst)
      peso_Psns_dst = float(sub_string_Psns_dst)
      peso_Pdevelop_dst = float(sub_string_Pdevelop_dst)

      try:
      #2
       global nombre_fichero_descarga1

       print("fichero descargado deberia ser " + nombre_fichero_descarga1)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
       blob = bucket.blob(nombre_fichero_descarga1)
       print("Descargando fichero:  "+nombre_fichero_descarga1)
       blob.download_to_filename("ejemplo_pesos")
       print("descargado fichero: fichero_pesos")

       f = open('ejemplo_pesos1')
       string_list1 = f.readlines()
       f.close()

       sub_string_Plocation_aux = string_list1[2]  # la tercera linea del fichero
       sub_string_Pemail_aux = string_list1[3]
       sub_string_Pdevice_aux = string_list1[4]
       sub_string_Pimei_aux = string_list1[5]
       sub_string_Pserial_aux = string_list1[6]
       sub_string_Pmac_aux = string_list1[7]
       sub_string_Padvertiser_aux = string_list1[8]
       sub_string_verde_aux = string_list1[9]
       sub_string_naranja_aux = string_list1[10]
       sub_string_rojo_aux = string_list1[11]
       #destinos servidor
       sub_string_Pinternal_dst_aux = string_list1[12]
       sub_string_Pads_dst_aux = string_list1[13]
       sub_string_Panalytics_dst_aux = string_list1[14]
       sub_string_Psns_dst_aux = string_list1[15]
       sub_string_Pdevelop_dst_aux = string_list1[16]

       #posible error si se busca y no hay nmero Ejemplo: 5.0, no existe [3] pero en 10.0 
       try:
        sub_string_Plocation1 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
       except:
        sub_string_Plocation1 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
       try:
        sub_string_Pemail1 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
       except:
        sub_string_Pemail1 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
       try:
        sub_string_Pdevice1 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
       except:
        sub_string_Pdevice1 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2]
       try:
        sub_string_Pimei1 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
       except:
        sub_string_Pimei1 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2]
       try:
        sub_string_Pserial1 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
       except:
        sub_string_Pserial1 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
       try:
        sub_string_Pmac1 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
       except:
        sub_string_Pmac1 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
       try:
        sub_string_Padvertiser1 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
       except:
        sub_string_Padvertiser1 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
       try:
        sub_string_verde1 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
       except:
        sub_string_verde1 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
       try:
        sub_string_naranja1 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
       except:
        sub_string_naranja1 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2]
       try:
        sub_string_rojo1 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
       except:
        sub_string_rojo1 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
       try:
        sub_string_Pinternal_dst1 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
       except:
        sub_string_Pinternal_dst1 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
       try:
        sub_string_Pads_dst1 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
       except:
        sub_string_Pads_dst1 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
       try:
        sub_string_Panalytics_dst1 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
       except:
        sub_string_Panalytics_dst1 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
       try:
        sub_string_Psns_dst1 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
       except:
        sub_string_Psns_dst1 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
       try:
        sub_string_Pdevelop_dst1 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
       except:
        sub_string_Pdevelop_dst1 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]


       peso_Plocation1 = float(sub_string_Plocation1)
       peso_Pemail1 = float(sub_string_Pemail1)
       peso_Pdevice1 = float(sub_string_Pdevice1)
       peso_Pimei1 = float(sub_string_Pimei1)
       peso_Pserial1 = float(sub_string_Pserial1)
       peso_Pmac1 = float(sub_string_Pmac1)
       peso_Padvertiser1 = float(sub_string_Padvertiser1)

       umbral_verde1 = float(sub_string_verde1)
       umbral_naranja1 = float(sub_string_naranja1)
       umbral_rojo1 = float(sub_string_rojo1)

       peso_Pinternal_dst1 = float(sub_string_Pinternal_dst1)
       peso_Pads_dst1 = float(sub_string_Pads_dst1)
       peso_Panalytics_dst1 = float(sub_string_Panalytics_dst1)
       peso_Psns_dst1 = float(sub_string_Psns_dst1)
       peso_Pdevelop_dst1 = float(sub_string_Pdevelop_dst1)
      except:
       print("mal")

      try:
      #3
       global nombre_fichero_descarga2

       print("fichero descargado deberia ser " + nombre_fichero_descarga2)

     # blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
       blob = bucket.blob(nombre_fichero_descarga2)
       print("Descargando fichero:  "+nombre_fichero_descarga2)
       blob.download_to_filename("ejemplo_pesos")
       print("descargado fichero: fichero_pesos")

       f = open('ejemplo_pesos2')
       string_list2 = f.readlines()
       f.close()

       sub_string_Plocation_aux = string_list2[2]  # la tercera linea del fichero
       sub_string_Pemail_aux = string_list2[3]
       sub_string_Pdevice_aux = string_list2[4]
       sub_string_Pimei_aux = string_list2[5]
       sub_string_Pserial_aux = string_list2[6]
       sub_string_Pmac_aux = string_list2[7]
       sub_string_Padvertiser_aux = string_list2[8]
       sub_string_verde_aux = string_list2[9]
       sub_string_naranja_aux = string_list2[10]
       sub_string_rojo_aux = string_list2[11]
       #destinos servidor
       sub_string_Pinternal_dst_aux = string_list2[12]
       sub_string_Pads_dst_aux = string_list2[13]
       sub_string_Panalytics_dst_aux = string_list2[14]
       sub_string_Psns_dst_aux = string_list2[15]
       sub_string_Pdevelop_dst_aux = string_list2[16]

       #posible error si se busca y no hay nmero Ejemplo: 5.0, no existe [3] pero en 10.0 
       try:
        sub_string_Plocation2 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2] + sub_string_Plocation_aux[3]
       except:
        sub_string_Plocation2 = sub_string_Plocation_aux[0] + sub_string_Plocation_aux[1] + sub_string_Plocation_aux[2]
       try:
        sub_string_Pemail2 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2] + sub_string_Pemail_aux[3]
       except:
        sub_string_Pemail2 = sub_string_Pemail_aux[0] + sub_string_Pemail_aux[1] + sub_string_Pemail_aux[2]
       try:
        sub_string_Pdevice2 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2] + sub_string_Pdevice_aux[3]
       except:
        sub_string_Pdevice2 = sub_string_Pdevice_aux[0] + sub_string_Pdevice_aux[1] + sub_string_Pdevice_aux[2]
       try:
        sub_string_Pime2 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2] + sub_string_Pimei_aux[3]
       except:
        sub_string_Pimei2 = sub_string_Pimei_aux[0] + sub_string_Pimei_aux[1] + sub_string_Pimei_aux[2]
       try:
        sub_string_Pserial2 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2] + sub_string_Pserial_aux[3]
       except:
        sub_string_Pserial2 = sub_string_Pserial_aux[0] + sub_string_Pserial_aux[1] + sub_string_Pserial_aux[2]
       try:
        sub_string_Pmac2 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2] + sub_string_Pmac_aux[3]
       except:
        sub_string_Pmac2 = sub_string_Pmac_aux[0] + sub_string_Pmac_aux[1] + sub_string_Pmac_aux[2]
       try:
        sub_string_Padvertiser2 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2] + sub_string_Padvertiser_aux[3]
       except:
        sub_string_Padvertiser2 = sub_string_Padvertiser_aux[0] + sub_string_Padvertiser_aux[1] + sub_string_Padvertiser_aux[2]
       try:
        sub_string_verde2 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2] + sub_string_verde_aux[3]
       except:
        sub_string_verde2 = sub_string_verde_aux[0] + sub_string_verde_aux[1] + sub_string_verde_aux[2]
       try:
        sub_string_naranja2 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2] + sub_string_naranja_aux[3]
       except:
        sub_string_naranja2 = sub_string_naranja_aux[0] + sub_string_naranja_aux[1] + sub_string_naranja_aux[2]
       try:
        sub_string_rojo2 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2] + sub_string_rojo_aux[3]
       except:
        sub_string_rojo2 = sub_string_rojo_aux[0] + sub_string_rojo_aux[1] + sub_string_rojo_aux[2]
       #destinations
       try:
        sub_string_Pinternal_dst2 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2] + sub_string_Pinternal_dst_aux[3]
       except:
        sub_string_Pinternal_dst2 = sub_string_Pinternal_dst_aux[0] + sub_string_Pinternal_dst_aux[1] + sub_string_Pinternal_dst_aux[2]
       try:
        sub_string_Pads_dst2 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2] + sub_string_Pads_dst_aux[3]
       except:
        sub_string_Pads_dst2 = sub_string_Pads_dst_aux[0] + sub_string_Pads_dst_aux[1] + sub_string_Pads_dst_aux[2]
       try:
        sub_string_Panalytics_dst2 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2] + sub_string_Panalytics_dst_aux[3]
       except:
        sub_string_Panalytics_dst2 = sub_string_Panalytics_dst_aux[0] + sub_string_Panalytics_dst_aux[1] + sub_string_Panalytics_dst_aux[2]
       try:
        sub_string_Psns_dst2 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2] + sub_string_Psns_dst_aux[3]
       except:
        sub_string_Psns_dst2 = sub_string_Psns_dst_aux[0] + sub_string_Psns_dst_aux[1] + sub_string_Psns_dst_aux[2]
       try:
        sub_string_Pdevelop_dst2 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2] + sub_string_Pdevelop_dst_aux[3]
       except:
        sub_string_Pdevelop_dst2 = sub_string_Pdevelop_dst_aux[0] + sub_string_Pdevelop_dst_aux[1] + sub_string_Pdevelop_dst_aux[2]





       peso_Plocation2 = float(sub_string_Plocation2)
       peso_Pemail2 = float(sub_string_Pemail2)
       peso_Pdevice2 = float(sub_string_Pdevice2)
       peso_Pimei2 = float(sub_string_Pimei2)
       peso_Pserial2 = float(sub_string_Pserial2)
       peso_Pmac2 = float(sub_string_Pmac2)
       peso_Padvertiser2 = float(sub_string_Padvertiser2)

       umbral_verde2 = float(sub_string_verde2)
       umbral_naranja2 = float(sub_string_naranja2)
       umbral_rojo2 = float(sub_string_rojo2)

       peso_Pinternal_dst2 = float(sub_string_Pinternal_dst2)
       peso_Pads_dst2 = float(sub_string_Pads_dst2)
       peso_Panalytics_dst2 = float(sub_string_Panalytics_dst2)
       peso_Psns_dst2 = float(sub_string_Psns_dst2)
       peso_Pdevelop_dst2 = float(sub_string_Pdevelop_dst2)
      except:
       print("mal2")

      #se hace la media:

      peso_Plocation= peso_Plocation
      peso_Pemail = peso_Pemail
      peso_Pdevice = peso_Pdevice
      peso_Pimei = peso_Pimei
      peso_Pserial = peso_Pserial
      peso_Pmac = peso_Pmac
      peso_Padvertiser = peso_Padvertiser
      umbral_verde = umbral_verde
      umbral_naranja = umbral_naranja
      umbral_rojo = umbral_rojo

      peso_Pinternal_dst = peso_Pinternal_dst
      peso_Pads_dst = peso_Pads_dst
      peso_Panalytics_dst = peso_Panalytics_dst
      peso_Psns_dst = peso_Psns_dst
      peso_Pdevelop_dst = peso_Pdevelop_dst

      try:
       peso_Plocation= peso_Plocation + peso_Plocation1 + peso_Plocation2
       peso_Pemail = peso_Pemail + peso_Pemail1 + peso_Pemail2
       peso_Pdevice = peso_Pdevice + peso_Pdevice1 + peso_Pdevice2
       peso_Pimei = peso_Pimei + peso_Pimei1 + peso_Pimei2
       peso_Pserial = peso_Pserial + peso_Pserial1 + peso_Pserial2
       peso_Pmac = peso_Pmac + peso_Pmac1 + peso_Pmac2
       peso_Padvertiser = peso_Padvertiser + peso_Padvertiser1 + peso_Padvertiser2
       umbral_verde = umbral_verde + umbral_verde1 + umbral_verde2
       umbral_naranja = umbral_naranja + umbral_naranja1 + umbral_naranja2
       umbral_rojo = umbral_rojo + umbral_rojo1 + umbral_rojo2

       peso_Pinternal_dst = peso_Pinternal_dst + peso_Pinternal_dst1 + peso_Pinternal_dst2
       peso_Pads_dst = peso_Pads_dst + peso_Pads_dst1 + peso_Pads_dst2
       peso_Panalytics_dst = peso_Panalytics_dst + peso_Panalytics_dst1 + peso_Panalytics_dst2
       peso_Psns_dst = peso_Psns_dst + peso_Psns_dst1 + peso_Psns_dst2
       peso_Pdevelop_dst = peso_Pdevelop_dst + peso_Pdevelop_dst1 + peso_Pdevelop_dst2

       peso_Plocation= peso_Plocation / 3
       peso_Pemail = peso_Pemail / 3
       peso_Pdevice = peso_Pdevice / 3
       peso_Pimei = peso_Pimei / 3
       peso_Pserial = peso_Pserial / 3
       peso_Pmac = peso_Pmac / 3
       peso_Padvertiser = peso_Padvertiser / 3
       umbral_verde = umbral_verde / 3
       umbral_naranja = umbral_naranja / 3
       umbral_rojo = umbral_rojo  / 3

       peso_Pinternal_dst = peso_Pinternal_dst  / 3
       peso_Pads_dst = peso_Pads_dst  / 3
       peso_Panalytics_dst = peso_Panalytics_dst  / 3
       peso_Psns_dst = peso_Psns_dst  / 3
       peso_Pdevelop_dst = peso_Pdevelop_dst  / 3

      except:
       print("mal3")

      try:
       peso_Plocation= peso_Plocation + peso_Plocation1
       peso_Pemail = peso_Pemail + peso_Pemail1
       peso_Pdevice = peso_Pdevice + peso_Pdevice1
       peso_Pimei = peso_Pimei + peso_Pimei1
       peso_Pserial = peso_Pserial + peso_Pserial1
       peso_Pmac = peso_Pmac + peso_Pmac1 + peso_Pmac2
       peso_Padvertiser = peso_Padvertiser + peso_Padvertiser1
       umbral_verde = umbral_verde + umbral_verde1
       umbral_naranja = umbral_naranja + umbral_naranja1
       umbral_rojo = umbral_rojo + umbral_rojo1

       peso_Pinternal_dst = peso_Pinternal_dst + peso_Pinternal_dst1
       peso_Pads_dst = peso_Pads_dst + peso_Pads_dst1
       peso_Panalytics_dst = peso_Panalytics_dst + peso_Panalytics_dst1
       peso_Psns_dst = peso_Psns_dst + peso_Psns_dst1
       peso_Pdevelop_dst = peso_Pdevelop_dst + peso_Pdevelop_dst1

       peso_Plocation= peso_Plocation / 2
       peso_Pemail = peso_Pemail / 2
       peso_Pdevice = peso_Pdevice / 2
       peso_Pimei = peso_Pimei / 2
       peso_Pserial = peso_Pserial / 2
       peso_Pmac = peso_Pmac / 2
       peso_Padvertiser = peso_Padvertiser / 2
       umbral_verde = umbral_verde / 2
       umbral_naranja = umbral_naranja / 2
       umbral_rojo = umbral_rojo  / 2

       peso_Pinternal_dst = peso_Pinternal_dst  / 2
       peso_Pads_dst = peso_Pads_dst  / 2
       peso_Panalytics_dst = peso_Panalytics_dst  / 2
       peso_Psns_dst = peso_Psns_dst  / 2
       peso_Pdevelop_dst = peso_Pdevelop_dst  / 2

      except:
       print("mal4")

      ###fichero con pesos y umbrales:


      ################################
     # my_file = open("fichero_datos"+modelctrstr3, "w")
      my_file = open("fichero_datos"+"global", "w")
      #my_file.write("Pesos actualizados modelo Ant"+"\n"+"\n"+sub_string_Plocation+"\n"+sub_string_Pemail+"\n"+sub_string_Pdevice+"\n"+sub_string_Pimei+"\n"+sub_string_Pserial+"\n"+sub_string_Pmac+"\n"+sub_string_Padvertiser+"\n"+sub_string_verde+"\n"+sub_string_naranja+"\n"+sub_string_rojo)
      my_file.write("Pesos actualizados modelo Ant"+"\n"+"\n"+str(peso_Plocation)+"\n"+str(peso_Pemail)+"\n"+str(peso_Pdevice)+"\n"+str(peso_Pimei)+"\n"+str(peso_Pserial)+"\n"+str(peso_Pmac)+"\n"+str(peso_Padvertiser)+"\n"+str(umbral_verde)+"\n"+str(umbral_naranja)+"\n"+str(umbral_rojo))

      my_file.close()

     # Ejemplo para 4 datos: Location, Email, IMEI y Device ID:
      filtracion_location = tf.placeholder(tf.float32, name='location_input')
      filtracion_email = tf.placeholder(tf.float32, name='email_input')
      filtracion_imei = tf.placeholder(tf.float32, name='imei_input')
      filtracion_device = tf.placeholder(tf.float32, name='device_input')
      filtracion_serialnumber = tf.placeholder(tf.float32, name='serialnumber_input')
      filtracion_macaddress = tf.placeholder(tf.float32, name='macaddress_input')
      filtracion_advertiser = tf.placeholder(tf.float32, name='advertiser_input')


      #destinos: 5 posibles--> Internal, Ads, Analytics, Sns, Develop               1 o 0 si se manda ahi o no
      destino_internal = tf.placeholder(tf.float32, name='internal_dst_input')
      destino_ads = tf.placeholder(tf.float32, name='ads_dst_input')
      destino_analytics = tf.placeholder(tf.float32, name='analytics_dst_input')
      destino_sns = tf.placeholder(tf.float32, name='sns_dst_input')
      destino_develop = tf.placeholder(tf.float32, name='develop_dst_input')

  # objetivo: answer: 1 o 0
      y_ = tf.placeholder(tf.float32, name='target')

      #distintos pesos para cada dato: nombre= Px
      Plocation = tf.Variable(peso_Plocation, name='Plocation') #9
      Pemail = tf.Variable(peso_Pemail, name='Pemail') #8
      Pimei = tf.Variable(peso_Pimei, name='Pimei') #3
      Pdevice = tf.Variable(peso_Pdevice, name='Pdevice') #2

      #Pdni = tf.placeholder(tf.float32, name='Pdni') #10
      #Pphone = tf.placeholder(tf.float32, name='Pphone') #8
      Pserialnumber = tf.Variable(peso_Pserial, name='Pserialnumber') #3
      Pmacaddress = tf.Variable(peso_Pmac, name='Pmacaddress') #1
      Padvertiser = tf.Variable(peso_Padvertiser, name='Padvertiser') #5

      #Pesos para cada destino:
      Pinternal_dst = tf.Variable(peso_Pinternal_dst, name='Pinternal_dst') #5
      Pads_dst = tf.Variable(peso_Pads_dst, name='Pads_dst') #5
      Panalytics_dst = tf.Variable(peso_Panalytics_dst, name='Panalytics_dst') #5
      Psns_dst = tf.Variable(peso_Psns_dst, name='Psns_dst') #5
      Pdevelop_dst = tf.Variable(peso_Pdevelop_dst, name='Pdevelop_dst') #5



  # umbral de decisi
  # umbral = tf.constant(10., name='umbral')
      #umbral = tf.placeholder(tf.float32, name='umbral')

    #  aux = tf.Variable(0., name='Vaux')
     #salida 0 o 1

      #salida del modelo
      y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), 0.0)
      #y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), aux)
      y_aux2 = tf.add(tf.multiply(filtracion_email, Pemail), y_aux1)
      y_aux3 = tf.add(tf.multiply(filtracion_imei, Pimei), y_aux2)
      y_aux4 = tf.add(tf.multiply(filtracion_device, Pdevice), y_aux3)


      #y_aux11 = tf.add(tf.multiply(filtracion_dni, Pdni), y_aux4)
      #y_aux12 = tf.add(tf.multiply(filtracion_phone, Pphone), y_aux11)
      y_aux13 = tf.add(tf.multiply(filtracion_serialnumber, Pserialnumber), y_aux4)
      y_aux14 = tf.add(tf.multiply(filtracion_macaddress, Pmacaddress), y_aux13)
      #y_aux14 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

      #destinos
      y_aux1_dst = tf.add(tf.multiply(destino_internal, Pinternal_dst), 0.0)
      y_aux2_dst = tf.add(tf.multiply(destino_ads, Pads_dst), y_aux1_dst)
      y_aux3_dst = tf.add(tf.multiply(destino_analytics, Panalytics_dst), y_aux2_dst)
      y_aux4_dst = tf.add(tf.multiply(destino_sns, Psns_dst), y_aux3_dst)
      y_aux5_dst = tf.add(tf.multiply(destino_develop, Pdevelop_dst), y_aux4_dst)
      #

      y_aux_final1 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

      y = tf.multiply(y_aux_final1, y_aux5_dst)

      y = tf.identity(y, name='output')

      loss = tf.reduce_mean(tf.square(y - y_))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
      train_op = optimizer.minimize(loss, name='train')

      init = tf.global_variables_initializer()

      # Creating a tf.train.Saver adds operations to the graph to save and
      # restore variables from checkpoints.

      saver_def = tf.train.Saver().as_saver_def
      ###NO VA EL SAVER DE GRAH DEF##########################
      # with open('graph.pb', 'wb') as f:
      #     f.write(tf.get_default_graph().as_graph_def())

      saver = tf.train.Saver()
      # Training
      # saver.save(sess, your_path + "/checkpoint_name.ckpt")
      # TensorFlow session
      sess = tf.Session()
      sess.run(init)
     # saver.save(sess, "-"+ NOMBRE_CHECKPOINTS + "-.ckpt")  "checkpoint_actualizado.ckpt"
      #saver.save(sess, "checkpoint_actualizado_"+modelctrstr3+".ckpt")
     # saver.save(sess, "checkpoint_actualizado_"+modelctrstr+".ckpt")
      saver.save(sess, "checkpoint_actualizado_"+"global"+".ckpt")
      print("se han creado fichero con nombre ... "+ modelctrstr4)

      #
      app.config["FICHERO_CHECK"] = ""
    #  zipf = zipfile.ZipFile(NOMBRE_CHECKPOINTS + '.ckpt.meta.zip','w'. zipfile.ZIP_DEFLATED)
   #   zipObj = ZipFile("ficherotestnoviembre.zip", 'w')
   #   zipObj.write(NOMBRE_CHECKPOINTS + ".ckpt.meta")
   #   zipObj.close()
   ##   return send_from_directory(app.config["FICHERO_CHECK"], filename='ficherotestnoviembre.zip', as_attachment=True)
      zipObj = ZipFile('test.zip', 'w')
      zipObj.write("checkpoint")
    #  zipObj.write("-"+ NOMBRE_CHECKPOINTS + "-.ckpt"+'.index')
    #  zipObj.write("-" + NOMBRE_CHECKPOINTS + "-.ckpt" + '.data-00000-of-00001')
    #  zipObj.write("-" + NOMBRE_CHECKPOINTS + "-.ckpt.meta") # no se si igual .meta
    #  zipObj.write("checkpoint_actualizado_"+modelctrstr3+".ckpt" + '.index')
    #  zipObj.write("checkpoint_actualizado_"+modelctrstr3+".ckpt" + '.data-00000-of-00001')
  #    zipObj.write("checkpoint_actualizado_"+modelctrstr+".ckpt.meta") # no se si igual .meta
     # zipObj.write("fichero_datos"+modelctrstr3)
      zipObj.write("checkpoint_actualizado_"+"global"+".ckpt" + '.index')  #si
      zipObj.write("checkpoint_actualizado_"+"global"+".ckpt" + '.data-00000-of-00001')

      zipObj.write("fichero_datos"+"global")
      zipObj.close()
      return send_from_directory(app.config["FICHERO_CHECK"], filename='test.zip', as_attachment=True)
  #    return flask.send_file('test.zip', mimetype = 'zip',attachment_filename= 'test.zip', as_attachment= True)

  else: # actualizo el modelo
    return





##



@app.route("/descargar_graph", methods = ['GET']) #FUNCIONA: DESCARGAS EL FICHERO
def descargar_graph():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints

      global numero
      numero = numero + 1
      global modelctr
      modelctr = modelctr + 1
      global NOMBRE_CHECKPOINTS
      NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
      global NOMBRE_ZIP
      NOMBRE_ZIP = "Zip_Name_" + str(modelctr)

      x = tf.placeholder(tf.float32, name='input')
      y_ = tf.placeholder(tf.float32, name='target')

      W = tf.Variable(5., name='W')
      b = tf.Variable(3., name='b')

      y = tf.add(tf.multiply(x, W), b)
      y = tf.identity(y, name='output')

      loss = tf.reduce_mean(tf.square(y - y_))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
      train_op = optimizer.minimize(loss, name='train')

      init = tf.global_variables_initializer()

      # Creating a tf.train.Saver adds operations to the graph to save and
      # restore variables from checkpoints.

      saver_def = tf.train.Saver().as_saver_def()
      ###NO VA EL SAVER DE GRAH DEF##########################
      with open('graph.pb', 'wb') as f:
           f.write(tf.get_default_graph().as_graph_def().SerializeToString())



      app.config["FICHERO_CHECK"] = ""


      return send_from_directory(app.config["FICHERO_CHECK"], filename='graph.pb', as_attachment=True)
  else: # actualizo el modelo
    return


@app.route("/descargar_graph_dense", methods = ['GET']) #FUNCIONA: DESCARGAS EL FICHERO
def descargar_graph_dense():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints

      global numero
      numero = numero + 1
      global modelctr
      modelctr = modelctr + 1
      global NOMBRE_CHECKPOINTS
      NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
      global NOMBRE_ZIP
      NOMBRE_ZIP = "Zip_Name_" + str(modelctr)

      model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,)),
      tf.keras.layers.Dense(25, activation=tf.keras.activations.relu),
      tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)])

      model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mean_squared_error)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

      init = tf.global_variables_initializer()

      saver_def = tf.train.Saver().as_saver_def()
      ###NO VA EL SAVER DE GRAH DEF##########################
      with open('graph.pb', 'wb') as f:
           f.write(tf.get_default_graph().as_graph_def().SerializeToString())



      app.config["FICHERO_CHECK"] = ""


      return send_from_directory(app.config["FICHERO_CHECK"], filename='graph.pb', as_attachment=True)
  else: # actualizo el modelo
    return
"""
    zipObj = ZipFile(NOMBRE_ZIP + "_PRUEBA_" + str(modelctr) + '.zip', 'w')
    zipObj.write('checkpoint')
    zipObj.write(NOMBRE_CHECKPOINTS + ".ckpt.meta")
    zipObj.close()
    return send_from_directory(app.config["FICHERO_CHECK"], filename= NOMBRE_ZIP + "_PRUEBA_" + str(modelctr) + '.zip', as_attachment=True)
"""


@app.route("/descargar_checkpoint", methods = ['GET']) #FUNCIONA: DESCARGAS EL FICHERO ****si
def descargar_checkpoint():
 if flask.request.method == "GET":
  print("mirando si esta updated")
  #habra que mirar una variable o algo
  if modelUpdated: # mando el fichero checkpoints
    app.config["FICHERO_TEXTO1"] = ""

    x = tf.placeholder(tf.float32, name='input')
    y_ = tf.placeholder(tf.float32, name='target')

    W = tf.Variable(5., name='W')
    b = tf.Variable(3., name='b')

    y = x * W + b
    y = tf.identity(y, name='output')

    loss = tf.reduce_mean(tf.square(y - y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, name='train')

    init = tf.global_variables_initializer()

    # Creating a tf.train.Saver adds operations to the graph to save and
    # restore variables from checkpoints.

    saver_def = tf.train.Saver().as_saver_def
    ###NO VA EL SAVER DE GRAH DEF##########################
    # with open('graph.pb', 'wb') as f:
    #     f.write(tf.get_default_graph().as_graph_def())

    saver = tf.train.Saver()
    # Training
    # saver.save(sess, your_path + "/checkpoint_name.ckpt")
    # TensorFlow session
    sess = tf.Session()
    sess.run(init)
    saver.save(sess, "checkpoint_name.ckpt")


    #igual hay que ponerlo como zip y mandarlo

    return send_from_directory(app.config["FICHERO_TEXTO1"], filename= "checkpoint_name.ckpt.meta", as_attachment=True)


  else: # actualizo el modelo
    return

 @app.route("/descargar4", methods=['GET'])  # NO FUNCIONA
 def descargar4():
        if flask.request.method == "GET":
            print("mirando si esta updated")
            # habra que mirar una variable o algo
            if modelUpdated:  # mando el fichero checkpoints
                storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
                BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
                bucket = storage_client.get_bucket(BUCKET_NAME)

                blob = bucket.blob("checkpoint")
                blob.download_to_filename("checkpoint1")
                print('Blop {} downloaded to {}.' .format(
                    checkpoint,
                    checkpoint1 ))


                app.config["FICHERO_CHECK"] = ""
                return send_from_directory(app.config["FICHERO_CHECK"], filename="checkpoint1", as_attachment= True)


        else:  # actualizo el modelo
            return


##empieza federated parasp
##The following function is responsible for receiving the weights and temporarily storing them for averaging:

@app.route("/upload", methods = ['POST'])   #PONER BIEN LOS NOMBRES
def upload():
 global numero
 numero = numero + 1
 global modelctr
 modelctr = modelctr + 1
 global NOMBRE_CHECKPOINTS
 NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
 global NOMBRE_ZIP
 NOMBRE_ZIP = "Zip_Name_" + str(modelctr)

 if flask.request.method == "POST":
  modelUpdated = False # ahora el modelo no esta actualizado. hay que sacar los pesos y evaluarlos para volver a entrenar el modelo y sacar el fichero checkpoints
  print("Uploading File")
  if flask.request.files["file"]:                       #se coge la infomracion de file y se pone en weights
    weights = flask.request.files["file"].read()
    weights_stream = io.BytesIO(weights)
    #bucket = storage.bucket()

    storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
    BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
    bucket = storage_client.get_bucket(BUCKET_NAME)


    #Uploading Files to Firebase
    print("Saving at Server")
    with open("delta.bin", "wb") as f:             #se crea un fichero delta.bin con los weigths descargados que llegan desde la app
      f.write(weights_stream.read())
    print("Starting upload to Firebase")                        ####*************esto es para usbirlo a la firebase queno es mio. O uso una o lo creo en una direccion local
    with open("delta.bin", "rb") as upload:
      byte_w = upload.read()
      #Preprocessing data before upload. File to be sent to Firebase is named "Weights.bin"
    with open("Weights.bin", "wb") as f: #nombre del fichero que se envia desde la app
      pickle.dump(weights, f)
    with open("Weights.bin", "rb") as f:    #manda a google-cloud el fichero Weights.bin
      blob = bucket.blob('weight__'+ str(modelctr))
      blob.upload_from_file(f)
      print("File Successfully Uploaded to Firebase")
      return "File Uploaded\n"
  else:
    print("File not found")


@app.route("/upload_A", methods = ['POST'])   #PONER BIEN LOS NOMBRES
def upload_A():
 global numero
 numero = numero + 1
 global modelctr
 modelctr = modelctr + 1
 global NOMBRE_CHECKPOINTS
 NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
 global NOMBRE_ZIP
 NOMBRE_ZIP = "Zip_Name_" + str(modelctr)
 global modelctrstr
 modelctrstr = str(modelctr)
 global NOMBRE_CHECKPOINTS1_2
 NOMBRE_CHECKPOINTS1_2 = "checkpoints_name_" + str(modelctr -1)
 global NOMBRE_CHECKPOINTS1_3
 NOMBRE_CHECKPOINTS1_3 = "checkpoints_name_" + str(modelctr -2)

 

 if flask.request.method == "POST":
    modelUpdated = False # ahora el modelo no  actualizado. hay que sacar los pesos y evaluarlos para volver a entrenar el modelo y sacar el fichero checkpoints
    print("Uploading File")
                      #se coge la infomracion de file y se pone en weights
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)


        storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
        BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
        bucket = storage_client.get_bucket(BUCKET_NAME)


    #Uploading Files to Firebase
        print("Saving at Server")

        with open(uploaded_file.filename, "rb") as f:    #manda a google-cloud el fichero Weights.bin
       #  blob = bucket.blob(uploaded_file.filename + str(modelctr))
         blob = bucket.blob(NOMBRE_CHECKPOINTS+'_fichero_pesos_')
         blob.upload_from_file(f)
         print("File Successfully Uploaded to Firebase")
         print("subido el fichero "+NOMBRE_CHECKPOINTS+'_fichero_pesos_')
         global nombre_fichero_descarga
         nombre_fichero_descarga = NOMBRE_CHECKPOINTS+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga)

         global nombre_fichero_descarga1
         nombre_fichero_descarga1 = NOMBRE_CHECKPOINTS1_2+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga1)

         global nombre_fichero_descarga2
         nombre_fichero_descarga2 = NOMBRE_CHECKPOINTS1_3+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga2)


        return "File Uploaded\n"
    else:
        print("File not found")

@app.route("/upload_B", methods = ['POST'])   #PONER BIEN LOS NOMBRES
def upload_B():
 global numero2
 numero2 = numero2 + 1
 global modelctr2
 modelctr2 = modelctr2 + 1
 global NOMBRE_CHECKPOINTS2
 NOMBRE_CHECKPOINTS2 = "checkpoints_name_" + str(modelctr2)
 global NOMBRE_ZIP2
 NOMBRE_ZIP2 = "Zip_Name2_" + str(modelctr2)
 global modelctrstr2
 modelctrstr2 = str(modelctr2)
 global NOMBRE_CHECKPOINTS2_2
 NOMBRE_CHECKPOINTS2_2 = "checkpoints_name_" + str(modelctr2 -1)
 global NOMBRE_CHECKPOINTS2_3
 NOMBRE_CHECKPOINTS2_3 = "checkpoints_name_" + str(modelctr2 -2)

 

 if flask.request.method == "POST":
    modelUpdated = False # ahora el modelo no  actualizado. hay que sacar los pesos y evaluarlos para volver a entrenar el modelo y sacar el fichero checkpoints
    print("Uploading File")
                      #se coge la infomracion de file y se pone en weights
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)


        storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
        BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
        bucket = storage_client.get_bucket(BUCKET_NAME)


    #Uploading Files to Firebase
        print("Saving at Server")

        with open(uploaded_file.filename, "rb") as f:    #manda a google-cloud el fichero Weights.bin
       #  blob = bucket.blob(uploaded_file.filename + str(modelctr))
         blob = bucket.blob(NOMBRE_CHECKPOINTS2+'_fichero_pesos_')
         blob.upload_from_file(f)
         print("File Successfully Uploaded to Firebase")
         print("subido el fichero "+NOMBRE_CHECKPOINTS2+'_fichero_pesos_')
         global nombre_fichero_descarga
         nombre_fichero_descarga = NOMBRE_CHECKPOINTS2+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga)

         global nombre_fichero_descarga1
         nombre_fichero_descarga1 = NOMBRE_CHECKPOINTS2_2+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga1)

         global nombre_fichero_descarga2
         nombre_fichero_descarga2 = NOMBRE_CHECKPOINTS2_3+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga2)


        return "File Uploaded\n"
    else:
        print("File not found")

@app.route("/upload_C", methods = ['POST'])   #PONER BIEN LOS NOMBRES
def upload_C():
 global numero3
 numero3 = numero3 + 1
 global modelctr3
 modelctr3 = modelctr3 + 1
 global NOMBRE_CHECKPOINTS3
 NOMBRE_CHECKPOINTS3 = "checkpoints_name_" + str(modelctr3)
 global NOMBRE_ZIP3
 NOMBRE_ZIP3 = "Zip_Name3_" + str(modelctr3)
 global modelctrstr3
 modelctrstr3 = str(modelctr3)
 global NOMBRE_CHECKPOINTS3_2
 NOMBRE_CHECKPOINTS3_2 = "checkpoints_name_" + str(modelctr3 -1)
 global NOMBRE_CHECKPOINTS3_3
 NOMBRE_CHECKPOINTS3_3 = "checkpoints_name_" + str(modelctr3 -2)

 

 if flask.request.method == "POST":
    modelUpdated = False # ahora el modelo no  actualizado. hay que sacar los pesos y evaluarlos para volver a entrenar el modelo y sacar el fichero checkpoints
    print("Uploading File")
                      #se coge la infomracion de file y se pone en weights
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)


        storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
        BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
        bucket = storage_client.get_bucket(BUCKET_NAME)


    #Uploading Files to Firebase
        print("Saving at Server")

        with open(uploaded_file.filename, "rb") as f:    #manda a google-cloud el fichero Weights.bin
       #  blob = bucket.blob(uploaded_file.filename + str(modelctr))
         blob = bucket.blob(NOMBRE_CHECKPOINTS3+'_fichero_pesos_')
         blob.upload_from_file(f)
         print("File Successfully Uploaded to Firebase")
         print("subido el fichero "+NOMBRE_CHECKPOINTS3+'_fichero_pesos_')
         global nombre_fichero_descarga
         nombre_fichero_descarga = NOMBRE_CHECKPOINTS3+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga)

         global nombre_fichero_descarga1
         nombre_fichero_descarga1 = NOMBRE_CHECKPOINTS3_2+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga1)

         global nombre_fichero_descarga2
         nombre_fichero_descarga2 = NOMBRE_CHECKPOINTS3_3+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga2)


        return "File Uploaded\n"
    else:
        print("File not found")


@app.route("/upload_D", methods = ['POST'])   #PONER BIEN LOS NOMBRES
def upload_D():
 global numero4
 numero4 = numero4 + 1
 global modelctr4
 modelctr4 = modelctr4 + 1
 global NOMBRE_CHECKPOINTS4
 NOMBRE_CHECKPOINTS4 = "checkpoints_name_" + str(modelctr4)
 global NOMBRE_ZIP4
 NOMBRE_ZIP4 = "Zip_Name4_" + str(modelctr4)
 global modelctrstr4
 modelctrstr4 = str(modelctr4)
 global NOMBRE_CHECKPOINTS4_2
 NOMBRE_CHECKPOINTS4_2 = "checkpoints_name_" + str(modelctr4 -1)
 global NOMBRE_CHECKPOINTS4_3
 NOMBRE_CHECKPOINTS4_3 = "checkpoints_name_" + str(modelctr4 -2)



 if flask.request.method == "POST":
    modelUpdated = False # ahora el modelo no  actualizado. hay que sacar los pesos y evaluarlos para volver a entrenar el modelo y sacar el fichero checkpoints
    print("Uploading File")
                      #se coge la infomracion de file y se pone en weights
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)


        storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
        BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
        bucket = storage_client.get_bucket(BUCKET_NAME)


    #Uploading Files to Firebase
        print("Saving at Server")

        with open(uploaded_file.filename, "rb") as f:    #manda a google-cloud el fichero Weights.bin
       #  blob = bucket.blob(uploaded_file.filename + str(modelctr))
         blob = bucket.blob(NOMBRE_CHECKPOINTS4+'_fichero_pesos_')
         blob.upload_from_file(f)
         print("File Successfully Uploaded to Firebase")
         print("subido el fichero "+NOMBRE_CHECKPOINTS4+'_fichero_pesos_')
         global nombre_fichero_descarga
         nombre_fichero_descarga = NOMBRE_CHECKPOINTS4+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga)

         global nombre_fichero_descarga1
         nombre_fichero_descarga1 = NOMBRE_CHECKPOINTS4_2+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga1)

         global nombre_fichero_descarga2
         nombre_fichero_descarga2 = NOMBRE_CHECKPOINTS4_3+'_fichero_pesos_'
         print("fichero con nombre " + nombre_fichero_descarga2)


        return "File Uploaded\n"
    else:
        print("File not found")

"""
####
####
### Now, the weights are averaged on the server
##
##


def averageweights():


    return

def modelo():
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(15, activation=tf.keras.activations.relu, input_shape=(10,)),
         tf.keras.layers.Dense(25, activation=tf.keras.activations.relu),
         tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mean_squared_error)

    saver = tf.train.Saver()
    sess = tf.keras.backend.get_session()
    save_path = saver.save(sess, "model.ckpt")
  #  model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))
  #  model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss=tf.keras.losses.mean_squared_error)

   # model.set_weights(model_weights)
   # saver = tf.train.Saver()
   # sess = tf.keras.backend.get_session()
   # save_path = saver.save(sess, "model.ckpt")

"""
def modeloSencillo():
    global numero
    numero = numero + 1
    global modelctr
    modelctr = modelctr + 1
    global NOMBRE_CHECKPOINTS
    NOMBRE_CHECKPOINTS = "checkpoints_name_" + str(modelctr)
    global NOMBRE_ZIP
    NOMBRE_ZIP = "Zip_Name_" + str(modelctr)

    x = tf.placeholder(tf.float32, name='input')
    y_ = tf.placeholder(tf.float32, name='target')

    W = tf.Variable(5., name='W')
    b = tf.Variable(3., name='b')

    y = x * W + b
    y = tf.identity(y, name='output')

    loss = tf.reduce_mean(tf.square(y - y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, name='train')

    init = tf.global_variables_initializer()

    # Creating a tf.train.Saver adds operations to the graph to save and
    # restore variables from checkpoints.

    saver_def = tf.train.Saver().as_saver_def
###NO VA EL SAVER DE GRAH DEF##########################
   # with open('graph.pb', 'wb') as f:
   #     f.write(tf.get_default_graph().as_graph_def())

    saver = tf.train.Saver()
    # Training
    # saver.save(sess, your_path + "/checkpoint_name.ckpt")
    # TensorFlow session
    sess = tf.Session()
    sess.run(init)
    saver.save(sess,  NOMBRE_CHECKPOINTS+".ckpt")

    storage_client = storage.Client.from_service_account_json("tftalejandroaguilera-8712e9c215d2.json")
    BUCKET_NAME = 'bucket-alejandro-aguilera'  # Nombre del bucket que he creado en el google-cloud
    bucket = storage_client.get_bucket(BUCKET_NAME)

  #  with open("checkpoint", "rb") as f:
  #      blob = bucket.blob('checkpoint'+str(modelctr))               #mando a google-cloud el fichero chechpoint
  #      blob.upload_from_file(f)
    with open(NOMBRE_CHECKPOINTS+".ckpt.data-00000-of-00001", "rb") as f:            #mando a google-cloud el fichero FINAL_GRAPH.ckpt.data
        blob = bucket.blob(NOMBRE_CHECKPOINTS+'.ckpt.data-00000-of-00001')
        blob.upload_from_file(f)
    with open(NOMBRE_CHECKPOINTS+".ckpt.index", "rb") as f:                  #mando a google-cloud el fichero final_graph.ckpt.index
        blob = bucket.blob(NOMBRE_CHECKPOINTS+'.ckpt.index')
        blob.upload_from_file(f)
    with open(NOMBRE_CHECKPOINTS+".ckpt.meta", "rb") as f:              #mando a google-cloud el fichero final_graph.ckpt.meta
        blob = bucket.blob(NOMBRE_CHECKPOINTS+".ckpt.meta")
        blob.upload_from_file(f)

    print("Files Uploaded")
    print("Global Model Updated")
    zipf = zipfile.ZipFile('model' + str(modelctr) + '.zip', 'w', zipfile.ZIP_DEFLATED)
    # bucket = storage.bucket(app=fireapp)

    #movido de sitio
    zipObj = ZipFile(NOMBRE_ZIP + "_"+ str(modelctr) + '.zip', 'w')
    zipObj.write('checkpoint')
    zipObj.write(NOMBRE_CHECKPOINTS+".ckpt.meta")
    zipObj.close()
    print("zip created")


    with open(NOMBRE_ZIP + "_"+ str(modelctr) + '.zip', 'rb') as f:
       blob = bucket.blob(NOMBRE_ZIP+ "_"+ str(modelctr) + '.zip')
       blob.upload_from_file(f)

    global isModelUpdated
    isModelUpdated = True
    return


def mandar_fichero():
    app.config["FICHERO_TEXTO"] = ".idea/almacen" #igual poner ruta completa
    return send_from_directory(app.config["FICHERO_TEXTO"], filename = ficheromandar.txt, as_attachment= True )

# return checkpoint_name.ckpt.meta
"""
    blobs = bucket.list_blobs()  # igaul no esta bien
    for blob in blobs:
     #   if blob.name == 'checkpoint' or 'FINAL_GRAPH' in blob.name:
        if blob.name == 'checkpoint' or NOMBRE_CHECKPOINTS in blob.name:
            blob.download_to_filename(blob.name)
            zipf.write(blob.name)
"""


def averageWeights():
    # Average the weights

    ##memWeights es par diferenciar los wights leiods en stream vs los de memoria

    w1 = []
    b1 = []

    w1.append(8.5)
    w1.append(10)
    w1.append(9.3)

    b1.append(2)
    b1.append(1.6)
    b1.append(1.8)

    n_w = 3
    sumatorio_b1 =0
    sumatorio_w1 =0

    #w1 /= n_w
    #b1 /= n_w

    for i in range (1,n_w):
      #  global sumatorio_b1
        sumatorio_b1 = b1[i]
      #  global sumatorio_w1
        sumatorio_w1 = w1[i]


    w1 = [sumatorio_w1/n_w]
    b1 = [sumatorio_b1 / n_w]

    print("W1: ")
    print(w1)
    print("B1: ")
    print(b1)


    model_weights = [w1, b1]

    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(1,)),
         # tf.keras.layers.Dense(25, activation=tf.keras.activations.relu),
         tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss=tf.keras.losses.mean_squared_error)

    print(model_weights)
    model.set_weights(model_weights)
    saver = tf.train.Saver()
    sess = tf.keras.backend.get_session()
    save_path = saver.save(sess, "model_linear.ckpt")



"""
    model = tf.keras.models.Sequential(
    #    [tf.keras.layers.Dense(1, input_shape=(features_shape,))])    #features shape ?? not defined
         [tf.keras.layers.Dense(1, input_shape=(1))])  # features shape ?? not defined
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss=tf.keras.losses.mean_squared_error)
"""



"""
def averageWeights():
 #Average the weights

 ##memWeights es par diferenciar los wights leiods en stream vs los de memoria
   


    w1 = []
    h1 = []
    w2 = []
    h2 = []
    w3 = []
    h3 = []
    n_w = np.shape(memWeights)[0]
    print(np.shape(memWeights)) # (n_bins = 5, n_weights = 5)
    w1 = np.ones((150, ))
    w1 = memWeights[0][0].flatten()
    for i in range(1, n_w):
      w1 = np.add(w1, memWeights[i][0].flatten())
    w1 /= n_w
    w1 = np.reshape(w1, (10, 15))
    print("W1: ")
    print(w1)


    h1 = np.ones((15, ))
    for i in range(np.shape(memWeights)[0]):
      h1 += ((memWeights[i][1]).flatten())
    h1 /= np.shape(memWeights)[0]
    h1 = np.reshape(h1, (15,))
    print("H1: ")
    print(h1)


    w2 = np.ones((375,))
    for i in range(1, np.shape(memWeights)[0]):
      w2 = np.add(w2, memWeights[i][2].flatten())
    w2 /= (np.shape(memWeights)[0])
    w2 = np.reshape(w2, (15, 25))
    print("W2: ")
    print(w2)

    h2 = np.ones((25,))
    for i in range(np.shape(memWeights)[0]):
      h2 += ((memWeights[i][3]).flatten())
    h2 /= (np.shape(memWeights)[0])
    h2 = np.reshape(h2, (25, ))
    print("H2: ")
    print(h2)


    w3 = np.ones((25,))
    for i in range(np.shape(memWeights)[0]):
      w3 += ((memWeights[i][4]).flatten())
    w3 /= (np.shape(memWeights)[0])
    w3 = np.reshape(w3, (25, 1))
    print("W3: ")
    print(w3)


    h3 = np.ones((1,))
    for i in range(np.shape(memWeights)[0]):
      h3 += ((memWeights[i][5]).flatten())
    h3 /= (np.shape(memWeights)[0])
    h3 = np.reshape(h3, (1, ))
    print("H3: ")
    print(h3)


    model_weights = [w1, h1, w2, h2, w3, h3]


    model = tf.keras.models.Sequential([tf.keras.layers.Dense(15, activation=tf.keras.activations.relu, input_shape=(10, )),
    tf.keras.layers.Dense(25, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mean_squared_error)
    print(model_weights)
    model.set_weights(model_weights)
    saver = tf.train.Saver()
    sess = tf.keras.backend.get_session()
    save_path = saver.save(sess, "model.ckpt")
    ##para que se pueda usar bucket ESTO LO HE PUESTO YO
    bucket = storage.bucket()
    with open("checkpoint", "rb") as f:
        blob = bucket.blob('checkpoint')
        blob.upload_from_file(f)
    with open("FINAL_GRAPH.ckpt.data-00000-of-00001", "rb") as f:
        blob = bucket.blob('FINAL_GRAPH.ckpt.data-00000-of-00001')
        blob.upload_from_file(f)
    with open("FINAL_GRAPH.ckpt.index", "rb") as f:
        blob = bucket.blob('FINAL_GRAPH.ckpt.index')
        blob.upload_from_file(f)
    with open("FINAL_GRAPH.ckpt.meta", "rb") as f:
        blob = bucket.blob('FINAL_GRAPH.ckpt.meta')
        blob.upload_from_file(f)
    print("Files Uploaded")
    print("Global Model Updated")
    zipf = zipfile.ZipFile('model' + str(modelctr) + '.zip','w', zipfile.ZIP_DEFLATED)
    bucket = storage.bucket(app = fireapp)
    blobs = bucket.list_blobs()
    for blob in blobs:
      if blob.name == 'checkpoint' or 'FINAL_GRAPH' in blob.name:
        blob.download_to_filename(blob.name)
        zipf.write(blob.name)
    blob = bucket.blob('checkpoint')
    with open('model' + str(modelctr) + '.zip', 'rb') as f:
      blob.upload_from_file(f)
    ref.update({'isUpdated': 'True'})
    ref.update({'modelctr': (modelctr + 1)})
    with open('x_test.bin', 'rb') as f:
      x_test = pickle.load(f)
    print(np.shape(x_test))
    print(np.shape(x_test[0]))
    with open('y_test.bin', 'rb') as f:
      y_test = pickle.load(f)
    x_test = np.asarray(x_test, dtype = np.float32)
    y_test = np.asarray(y_test, dtype = np.float32)
    print((y_test.shape))
"""
