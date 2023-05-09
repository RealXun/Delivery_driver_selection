import psycopg2
import params as params

try:
    conn = psycopg2.connect(
        host = params.host, 
        dbname = params.dbname, 
        user = params.user, 
        password = params.password, 
        port = params.port)
    conn.set_session(autocommit=True)
    cur = conn.cursor()

except (Exception, psycopg2.Error) as error:
    print("Error en la apertura de base de datos: \n\t{}".format(error))
    conn = psycopg2.connect(
        host = params.host, 
        dbname = 'postgres', 
        user = params.user, 
        password = params.password, 
        port = params.port)
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    
    cur.execute("CREATE DATABASE repartos;")

    cur.close()
    conn.close()

    conn = psycopg2.connect(
        host = params.host, 
        dbname = params.dbname, 
        user = params.user, 
        password = params.password, 
        port = params.port)
    conn.set_session(autocommit=True)
    cur = conn.cursor()

try:
    cur.execute("""CREATE TABLE IF NOT EXISTS clientes(
        id SERIAL PRIMARY KEY,
        nombre VARCHAR(50) NOT NULL,
        email VARCHAR(50) NOT NULL UNIQUE,
        direccion VARCHAR(250) NOT NULL,
        latitud NUMERIC(9,6) NOT NULL,
        longitud NUMERIC(9,6) NOT NULL);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS comercios(
        id SMALLSERIAL PRIMARY KEY,
        nombre VARCHAR(50) NOT NULL,
        email VARCHAR(50) NOT NULL UNIQUE,
        zona SMALLINT NOT NULL,
        direccion VARCHAR(250) NOT NULL,
        tipo VARCHAR(50) NOT NULL,
        latitud NUMERIC(9,6) NOT NULL,
        longitud NUMERIC(9,6) NOT NULL);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS repartidores(
        id SMALLSERIAL PRIMARY KEY NOT NULL,
        nombre VARCHAR(50) NOT NULL,
        email VARCHAR(50) NOT NULL UNIQUE,
        status BOOLEAN NOT NULL,
        vehiculo VARCHAR(50) NOT NULL,
        ocupado BOOLEAN,
        zona SMALLINT,
        latitud NUMERIC(9,6),
        longitud NUMERIC(9,6),
        datetime_ult_act TIMESTAMP);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS administradores(
        id SERIAL PRIMARY KEY NOT NULL,
        nombre VARCHAR(50) NOT NULL,
        email VARCHAR(50) NOT NULL UNIQUE);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS usuarios(
        id SERIAL PRIMARY KEY NOT NULL,
        username VARCHAR(50) NOT NULL UNIQUE,
        id_cliente INTEGER,
        id_administrador SMALLINT,
        id_repartidor SMALLINT,
        id_comercio SMALLINT,
        tipo VARCHAR(50) NOT NULL,
        contraseña VARCHAR(20) NOT NULL,
        FOREIGN KEY (id_cliente) REFERENCES clientes(id),
        FOREIGN KEY (id_administrador) REFERENCES administradores(id),
        FOREIGN KEY (id_repartidor) REFERENCES repartidores(id),
        FOREIGN KEY (id_comercio) REFERENCES comercios(id));""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS pedidos(
        id_pedido SERIAL PRIMARY KEY NOT NULL,
        status VARCHAR(50) NOT NULL,
        id_cliente INTEGER NOT NULL,
        id_comercio SMALLINT NOT NULL,
        id_repartidor SMALLINT,
        direccion VARCHAR(250) NOT NULL,
        datetime_pedido TIMESTAMP NOT NULL,
        fecha_entrega DATE,
        hora_entrega TIME,
        FOREIGN KEY (id_cliente) REFERENCES clientes(id),
        FOREIGN KEY (id_comercio) REFERENCES comercios(id),
        FOREIGN KEY (id_repartidor) REFERENCES repartidores(id));""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS pedidos_agrupados(
        fecha TIME NOT NULL,
        zona SMALLINT NOT NULL,
        franja_horaria SMALLINT NOT NULL,
        total SMALLINT NOT NULL);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS pedidos_prediccion(
        fecha TIME NOT NULL,
        zona SMALLINT NOT NULL,
        franja_horaria SMALLINT NOT NULL,
        total SMALLINT NOT NULL);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS franjas_horarias(
        id SMALLINT PRIMARY KEY,
        h_inicio TIME NOT NULL,
        h_final TIME NOT NULL);""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS climatologia(
        fecha DATE NOT NULL, 
        hora TIME NOT NULL,
        temperatura REAL,
        lluvia REAL,
        descripcion VARCHAR(50));""")

except Exception:
    conn.rollback()
    raise Exception

finally:
    cur.close()
    conn.close()