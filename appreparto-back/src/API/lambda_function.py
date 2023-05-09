import sys
import os
import logging
import json
import psycopg2
from psycopg2 import DatabaseError
from Cliente import Cliente
from Administrador import Administrador
from Comercio import Comercio
from Franja_horaria import Franja_horaria
from Pedido import Pedido
from Repartidor import Repartidor
from Usuario import Usuario

from database.custom_encoder import CustomEncoder

# El logger para ver los logs en Cloud Watch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Definición de los métodos.
getMethod = 'GET'
postMethod = 'POST'
patchMethod = 'PATCH'
putMethod = 'PUT'
deleteMethod = 'DELETE'

# Definición de las rutas.
clientesPath = '/clientes'
administradoresPath = '/administradores'
comerciosPath = '/comercios'
franjas_horariasPath = '/franjas_horarias'
pedidosPath = '/pedidos'
repartidoresPath = '/repartidores'
usuariosPath = '/usuarios'

# Establecer conexión
try:
    print('Conectando a bd')
    connection = psycopg2.connect(
        user= os.environ['username'], 
        password= os.environ['password'], 
        host=os.environ['host'], 
        database= "repartos_test"
        )
except DatabaseError as ex:
    logger.error("ERROR: Error inesperado. No se pudo conectar con la instancia de la base de datos de PostgreSQL.")
    logger.error(ex)
    sys.exit()
logger.info("SUCCESS: Conectado correctamente a la instancia de RDS PostgreSQL.")

# Lógica de la API
def lambda_handler(event, context):
    logger.info(event)
    logger.info("Evento recibido: " + json.dumps(event))
    httpMethod = event['httpMethod']
    path = event['path']
    # Lógica para el uso de una función u otra dependiendo del path y el method.
    if httpMethod == getMethod and path == clientesPath:
        response = getClientes()
    elif httpMethod == postMethod and path == clientesPath:
        response = postClientes(json.loads(event['body']))
    elif httpMethod == getMethod and path == administradoresPath:
        response = getAdministradores()
    elif httpMethod == getMethod and path == comerciosPath:
        response = getComercios()
    elif httpMethod == getMethod and path == franjas_horariasPath:
        response = getFranjas_Horarias()
    elif httpMethod == getMethod and path == pedidosPath:
        response = getPedidos()
    elif httpMethod == postMethod and path == pedidosPath:
        response = postPedidos(json.loads(event['body']))
    elif httpMethod == putMethod and path == pedidosPath:
        response = putPedidos(json.loads(event['body']))
    elif httpMethod == getMethod and path == repartidoresPath:
        response = getRepartidores()
    elif httpMethod == putMethod and path == repartidoresPath:
        response = putRepartidores(json.loads(event['body']))
    elif httpMethod == getMethod and path == usuariosPath:
        response = getUsuarios()
    return response

## Visualizar datos de todos los clientes.
def getClientes():
    logger.info("Iniciando consulta")
    clientes = []
    #Conexión a la base de datos
    with connection.cursor() as cursor:
                cursor.execute("SELECT id, nombre, email, direccion, latitud, longitud FROM clientes ORDER BY nombre ASC")
                resultset = cursor.fetchall()
                for row in resultset:
                    
                    cliente = Cliente(row[0], row[1], row[2], row[3], row[4], row[5])
                    #Añadir en la lista clientes un json con los datos
                    clientes.append(cliente.to_JSON())
                    body = clientes                    
    logger.info(body)
    return buildResponse(200, body)

def postClientes(requestBody):

    new_client = requestBody
    
    # Realizar la operación INSERT en la base de datos
    with connection.cursor() as cursor:
        query = "INSERT INTO clientes (nombre, email, direccion, latitud, longitud) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (new_client['nombre'], new_client['email'], new_client['direccion'], new_client['latitud'], new_client['longitud']))
    connection.commit()

    # Ejecutar la consulta SELECT para obtener la lista actualizada de clientes
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, nombre, email, direccion, latitud, longitud FROM clientes ORDER BY nombre ASC")
        resultset = cursor.fetchall()

    # Crear una lista de objetos Cliente a partir de los resultados
    clientes = []
    for row in resultset:
        cliente = Cliente(row[0], row[1], row[2], row[3], row[4], row[5])
        clientes.append(cliente.to_JSON())

    # Devolver una respuesta HTTP JSON con la lista de clientes actualizada
        body = clientes                 
    logger.info(body)
    return buildResponse(200, body)


def getAdministradores():
    logger.info("Iniciando consulta")
    administradores = []
    #Conexión a la base de datos
    with connection.cursor() as cursor:
                cursor.execute("SELECT id, nombre, email FROM administradores ORDER BY nombre ASC")
                resultset = cursor.fetchall()
                for row in resultset:
                    
                    administrador = Administrador(row[0], row[1], row[2])
                    #Añadir en la lista administradores un json con los datos
                    administradores.append(administrador.to_JSON())
                    body = administradores                   
    logger.info(body)
    return buildResponse(200, body)

def getComercios():
    logger.info("Iniciando consulta")
    comercios = []
    #Conexión a la base de datos
    with connection.cursor() as cursor:
                cursor.execute("SELECT id, nombre, zona, direccion, tipo, latitud, longitud FROM comercios ORDER BY id ASC")
                resultset = cursor.fetchall()
                for row in resultset:
                    
                    comercio = Comercio(row[0], row[1], row[2], row[3], row[4], row[5], row[6])
                    #Añadir en la lista comercios un json con los datos
                    comercios.append(comercio.to_JSON())
                    body = comercios                  
    logger.info(body)
    return buildResponse(200, body)

def getFranjas_Horarias():
    logger.info("Iniciando consulta")
    franjas_horarias = []
    #Conexión a la base de datos
    with connection.cursor() as cursor:
                cursor.execute("SELECT id, h_inicio, h_final FROM franjas_horarias ORDER BY id ASC")
                resultset = cursor.fetchall()
                for row in resultset:
                    
                    franja_horaria = Franja_horaria(row[0], str(row[1]), str(row[2]))
                    #Añadir en la lista franjas_horarias un json con los datos
                    franjas_horarias.append(franja_horaria.to_JSON())
                    body = franjas_horarias                  
    logger.info(body)
    return buildResponse(200, body)

def getPedidos():
    logger.info("Iniciando consulta")
    pedidos = []
    #Conexión a la base de datos
    with connection.cursor() as cursor:
                cursor.execute("SELECT id_pedido, id_cliente, id_comercio, id_repartidor, latitud, longitud, datetime_pedido, tamaño, status, direccion FROM pedidos ORDER BY id_pedido ASC")
                resultset = cursor.fetchall()
                for row in resultset:
                    
                    pedido = Pedido(row[0], row[1], row[2], row[3], row[4], row[5], str(row[6]), row[7], row[8], row[9])
                    #Añadir en la lista pedidos un json con los datos
                    pedidos.append(pedido.to_JSON())
                    body = pedidos                  
    logger.info(body)
    return buildResponse(200, body)

def putPedidos(requestBody):
    # Obtener los datos del nuevo estado de la solicitud HTTP PUT
    updated_pedidos = requestBody
    
    # Realizar la operación UPDATE en la base de datos
    with connection.cursor() as cursor:
        query = "UPDATE pedidos SET id_cliente = %s, id_comercio = %s, id_repartidor = %s, latitud = %s, longitud= %s, datetime_pedido = %s, tamaño = %s, status = %s, direccion = %s WHERE id_pedido = %s"
        cursor.execute(query, (updated_pedidos['id_cliente'], updated_pedidos['id_comercio'], updated_pedidos['id_repartidor'], updated_pedidos['latitud'], 
                               updated_pedidos['longitud'], updated_pedidos['datetime_pedido'], updated_pedidos['tamaño'], updated_pedidos['status'], updated_pedidos['direccion'], updated_pedidos['id_pedido']))
    connection.commit()

    # Ejecutar la consulta SELECT para obtener la lista actualizada de pedidos
    with connection.cursor() as cursor:
        cursor.execute("SELECT id_pedido, id_cliente, id_comercio, id_repartidor, latitud, longitud, datetime_pedido, tamaño, status, direccion FROM pedidos ORDER BY id_pedido ASC")
        resultset = cursor.fetchall()

    # Crear una lista de objetos Pedidos a partir de los resultados
    pedidos = []
    for row in resultset:
        pedido = Pedido(row[0], row[1], row[2], row[3], row[4], row[5], str(row[6]), row[7], row[8], row[9])
        pedidos.append(pedido.to_JSON())

    # Devolver una respuesta HTTP JSON con la lista de pedidos actualizada
        body = pedidos                    
    logger.info(body)
    return buildResponse(200, body)

def postPedidos(requestBody):

    new_pedido = requestBody
    
    # Realizar la operación INSERT en la base de datos
    with connection.cursor() as cursor:
        query = "INSERT INTO pedidos (id_pedido, id_cliente, id_comercio, id_repartidor, latitud, longitud, datetime_pedido, tamaño, status, direccion) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(query, (new_pedido['id_pedido'], new_pedido['id_cliente'], new_pedido['id_comercio'], new_pedido['id_repartidor'],new_pedido['latitud'], new_pedido['longitud'],
                               new_pedido['datetime_pedido'] , new_pedido['tamaño'], new_pedido['status'], new_pedido['direccion']))
    connection.commit()

    # Ejecutar la consulta SELECT para obtener la lista actualizada de clientes
    with connection.cursor() as cursor:
        cursor.execute("SELECT id_pedido, id_cliente, id_comercio, id_repartidor, latitud, longitud, datetime_pedido, tamaño, status, direccion FROM pedidos ORDER BY id_pedido ASC")
        resultset = cursor.fetchall()

    # Crear una lista de objetos Cliente a partir de los resultados
    pedidos = []
    for row in resultset:
        pedido = Pedido(row[0], row[1], row[2], row[3], row[4], row[5], str(row[6]), row[7], row[8], row[9])
        pedidos.append(pedido.to_JSON())

    # Devolver una respuesta HTTP JSON con la lista de clientes actualizada
        body = pedidos                 
    logger.info(body)
    return buildResponse(200, body)

def getRepartidores():
    logger.info("Iniciando consulta")
    repartidores = []
    with connection.cursor() as cursor:
                cursor.execute("SELECT id, nombre, email, status, vehiculo, ocupado, zona, latitud, longitud, datetime_ult_act FROM repartidores ORDER BY id ASC")
                resultset = cursor.fetchall()
                for row in resultset:
                    
                    repartidor = Repartidor(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
                    #Añadir en la lista repartidores un json con los datos
                    repartidores.append(repartidor.to_JSON())
                    body = repartidores                    
    logger.info(body)
    return buildResponse(200, body)

def putRepartidores(requestBody):
    # Obtener los datos del nuevo estado de la solicitud HTTP PUT
    updated_repartidor = requestBody
    
    # Realizar la operación UPDATE en la base de datos
    with connection.cursor() as cursor:
        query = "UPDATE repartidores SET status = %s, ocupado = %s, latitud = %s, longitud = %s WHERE id = %s"
        cursor.execute(query, (updated_repartidor['status'], updated_repartidor['ocupado'], updated_repartidor['latitud'], updated_repartidor['longitud'], updated_repartidor['id']))
    connection.commit()

    # Ejecutar la consulta SELECT para obtener la lista actualizada de repartidores
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, nombre, email, status, vehiculo, ocupado, zona, latitud, longitud, datetime_ult_act FROM repartidores ORDER BY id ASC")
        resultset = cursor.fetchall()

    # Crear una lista de objetos Repartidor a partir de los resultados
    repartidores = []
    for row in resultset:
        repartidor = Repartidor(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
        repartidores.append(repartidor.to_JSON())

    # Devolver una respuesta HTTP JSON con la lista de repartidores actualizada
        body = repartidores                    
    logger.info(body)
    return buildResponse(200, body)



def getUsuarios():
    logger.info("Iniciando consulta")
    usuarios = []
    #Conexión a la base de datos
    with connection.cursor() as cursor:
                cursor.execute("SELECT id, username, id_cliente, id_administrador, id_repartidor, tipo, contraseña FROM usuarios ORDER BY username ASC")
                resultset = cursor.fetchall()
                for row in resultset:
                    
                    usuario = Usuario(row[0], row[1], row[2], row[3], row[4], row[5], row[6])
                    #Añadir en la lista usuarios un json con los datos
                    usuarios.append(usuario.to_JSON())
                    body = usuarios                    
    logger.info(body)
    return buildResponse(200, body)

## Definir respuesta.
def buildResponse (statusCode, body=None):
    response = {
        'statusCode': statusCode,
        'headers': {
            'Content-Type': 'application/json',
        }
    }
    if body is not None:
        response['body'] = json.dumps(body, cls=CustomEncoder)
    print(response)
    return response

