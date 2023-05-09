import psycopg2
import params as params
import numpy as np
import math
import random
from faker import Faker
from geopy.geocoders import Nominatim


def generador_direccion():
    """
    Función que genera una dirección aleatoriamente en un radio de 10km centrado
    en las coords del centro de Madrid. Para ello calcula un punto con la fórmula
    de distancias de lat y long gracias a la librería math y usa la librería 
    geocoders para obtener direcciones y recalcular coordenadas en base a esta.
    """

    geolocator = Nominatim(user_agent="durumkebab")

    centro = (40.4168, -3.7038)
    lat_madrid = math.radians(centro[0])
    lon_madrid = math.radians(centro[1])

    direccion_flag = True
    while direccion_flag:
        d = np.random.uniform(0, 10)
        ang = np.random.uniform(0, 2*np.pi)
        R = 6371
        lat = math.asin(math.sin(lat_madrid) * math.cos(d/R) + 
                        math.cos(lat_madrid) * math.sin(d/R) * math.cos(ang))
        lon = lon_madrid + \
            math.atan2(math.sin(ang) * math.sin(d/R) * math.cos(lat_madrid), 
                       math.cos(d/R) - math.sin(lat_madrid) * math.sin(lat))
        lat = math.degrees(lat)
        lon = math.degrees(lon)

        direccion = geolocator.reverse((lat, lon))
        try:
            int(direccion[0].split(",")[0])
            try:
                int(direccion[0].split(",")[1])
            except:
                latitud = direccion[1][0]
                longitud = direccion[1][1]
                direccion = "{}, {}, Madrid" \
                    .format(direccion[0].split(",")[1].strip(), 
                        direccion[0].split(",")[0])
                direccion_flag = False
        except:
            pass
    
    return direccion, latitud, longitud


def generador_clientes(n_clientes_nuevos):
    """
    Función que genera una los datos de un nuevo cliente, constando de nombre
    completo, email, dirección, y coordenadas de esta. Para ello se hace uso
    de la librería Faker, instanciando con datos de España, y la función
    'generador_direccion' desarrollada. El perfil generado es insertado tanto
    en la tabla clientes de la base de datos como en la de usuarios (leyendo
    previamente el id asignado en clientes) con una contraseña predefinida.
    """
    try:
        conn = psycopg2.connect(
            host = params.host, 
            dbname = params.dbname, 
            user = params.user, 
            password = params.password, 
            port = params.port)
        cur = conn.cursor()
    except (Exception, psycopg2.Error) as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        return
    
    f = Faker('es_ES')
    i = 0
    n_errores = 0
    while i < n_clientes_nuevos:
        cliente = {}
        cliente["nombre"] = f.name()
        cliente["email"] = '{}@mail.com' \
            .format(cliente["nombre"].lower() \
            .replace(" ", ".")
            .replace("á", "a") \
            .replace("é", "e") \
            .replace("í", "i") \
            .replace("ó", "o") \
            .replace("ú", "u") \
            .replace("ü", "u"))
        
        cliente["direccion"], cliente["latitud"], \
            cliente["longitud"] = generador_direccion()

        try:
            sql_query = """
                INSERT INTO 
                    clientes(nombre, email, direccion, latitud, longitud)
                VALUES
                    (%s, %s, %s, %s, %s);"""
            args = (cliente["nombre"], 
                    cliente["email"], 
                    cliente["direccion"], 
                    cliente["latitud"], 
                    cliente["longitud"])
            cur.execute(sql_query, args)
            conn.commit()

            sql_query = """
                    SELECT id FROM clientes WHERE nombre = %s;
                    """
            cur.execute(sql_query, (cliente["nombre"],))
            id = cur.fetchone()[0]

            sql_query = """
                INSERT INTO
                    usuarios (username, id_cliente, tipo, contraseña)
                VALUES
                    (%s, %s, %s, %s);"""
            args = (cliente["email"], id, 'cliente', '1234')
            cur.execute(sql_query, args)
            conn.commit()

            i += 1

        except Exception:
            print("Error:")
            print(cliente["nombre"], 
                cliente["email"], 
                cliente["direccion"], 
                cliente["latitud"], 
                cliente["longitud"])
            print(Exception)
            conn.rollback()
            n_errores += 1
  
    cur.close()
    conn.close()

    return "{} errores en la inserción".format(n_errores)


def generador_repartidores(n_repartidores_nuevos):
    """
    Función que genera una los datos de un nuevo repartidor, constando de nombre
    completo, email, el status (por defecto en False) y vehículo. Para ello se 
    hace uso de la librería Faker, instanciando con datos de España. El perfil 
    generado es insertado tanto en la tabla repartidores de la base de datos 
    como en la de usuarios (leyendo previamente el id asignado en repartidores)
    con una contraseña predefinida.
    """
    try:
        conn = psycopg2.connect(
            host = params.host, 
            dbname = params.dbname, 
            user = params.user, 
            password = params.password, 
            port = params.port)
        cur = conn.cursor()
    except (Exception, psycopg2.Error) as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        return
    
    f = Faker('es_ES')

    i = 0
    n_errores = 0
    while i < n_repartidores_nuevos:
        repartidor = {}
        repartidor["nombre"] = f.name()
        repartidor["email"] = '{}@mail.com' \
            .format(repartidor["nombre"].lower() \
            .replace(" ", ".") \
            .replace("á", "a") \
            .replace("é", "e") \
            .replace("í", "i") \
            .replace("ó", "o") \
            .replace("ú", "u") \
            .replace("ü", "u"))
        
        status_lst = [True, False]
        repartidor["status"] = random.choices(status_lst, weights = [0, 100])[0]

        vehiculos_lst = ["Bicicleta", "Motocicleta", "Patín"]
        repartidor["vehiculo"] = random.choices(vehiculos_lst, weights = [35, 55, 10])[0]

        try:
            sql_query = """
                INSERT INTO 
                    repartidores(nombre, email, status, vehiculo)
                VALUES
                    (%s, %s, %s, %s);"""
            args = (repartidor["nombre"], 
                    repartidor["email"], 
                    repartidor["status"], 
                    repartidor["vehiculo"])
            cur.execute(sql_query, args)
            conn.commit()

            sql_query = """
                    SELECT id FROM repartidores WHERE nombre = %s;
                    """
            cur.execute(sql_query, (repartidor["nombre"],))
            id = cur.fetchone()[0]

            sql_query = """
                INSERT INTO
                    usuarios (username, id_repartidor, tipo, contraseña)
                VALUES
                    (%s, %s, %s, %s);"""
            args = (repartidor["email"], id, 'repartidor', 'abcd')
            cur.execute(sql_query, args)
            conn.commit()

            i += 1

        except Exception:
            print("Error:")
            print(cliente["nombre"], 
                cliente["email"], 
                cliente["direccion"], 
                cliente["latitud"], 
                cliente["longitud"])
            print(Exception)
            conn.rollback()
            n_errores += 1

    cur.close()
    conn.close()

    return "{} errores en la inserción".format(n_errores)


def generador_admins(n_admins_nuevos):

    try:
        conn = psycopg2.connect(
            host = params.host, 
            dbname = params.dbname, 
            user = params.user, 
            password = params.password, 
            port = params.port)
        cur = conn.cursor()
    except (Exception, psycopg2.Error) as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        return
    
    f = Faker('es_ES')

    i = 0
    n_errores = 0
    while i < n_admins_nuevos:
        admin = {}
        admin["nombre"] = f.name()
        admin["email"] = '{}@mail.com' \
            .format(admin["nombre"].lower() \
            .replace(" ", ".")
            .replace("á", "a") \
            .replace("é", "e") \
            .replace("í", "i") \
            .replace("ó", "o") \
            .replace("ú", "u") \
            .replace("ü", "u"))

        try:
            sql_query = """
                INSERT INTO 
                    administradores(nombre, email)
                VALUES
                    (%s, %s);"""
            args = (admin["nombre"], 
                    admin["email"])
            cur.execute(sql_query, args)
            conn.commit()
            
            sql_query = """
                    SELECT id FROM administradores WHERE nombre = %s;
                    """
            cur.execute(sql_query, (admin["nombre"],))
            id = cur.fetchone()[0]

            sql_query = """
                INSERT INTO
                    usuarios (username, id_administrador, tipo, contraseña)
                VALUES
                    (%s, %s, %s, %s);"""
            args = (admin["email"], id, 'admin', 'qwerty')
            cur.execute(sql_query, args)
            conn.commit()

            i += 1

        except Exception:
            print("Error:")
            print(admin["nombre"], 
                admin["email"])
            print(Exception)
            conn.rollback()
            n_errores += 1
    
    cur.close()
    conn.close()

    return "{} errores en la inserción".format(n_errores)