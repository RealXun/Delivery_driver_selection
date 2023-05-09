import pandas as pd
import psycopg2
import datetime as dt
import sys
import params as params

# Apertura de la base de datos
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
    sys.exit()

# Carga de la tabla de franjas horarias para establecer rango de horas 
# a procesar
franjas_horarias = pd.read_sql("SELECT * FROM franjas_horarias;", con = conn)
franjas_horarias = franjas_horarias.rename(columns={"id": "franja_horaria"})

fecha_actual = dt.date.today()
hora_actual = dt.datetime.now().time().strftime('%H:%M:%S')
hora_actual = dt.datetime.strptime(hora_actual, '%H:%M:%S').time()
for row_franjas in franjas_horarias.itertuples():
        if (hora_actual >= row_franjas[2]) & \
            ((hora_actual < row_franjas[3])):
            franja_actual = row_franjas[1]
            break
# ASIGNACIÓN TEMPORAL DE FECHA Y FRANJA PARA DEMO       
# franja_actual = 7        
# fecha_actual = "2022-03-31"
# fecha_actual = dt.datetime.strptime(fecha_actual, "%Y-%m-%d").date()
if franja_actual == franjas_horarias["franja_horaria"].min():
    franja_anterior = franjas_horarias["franja_horaria"].max()
    fecha_franja_anterior = fecha_actual - dt.timedelta(days=1)
else: 
    franja_anterior = franja_actual - 1
    fecha_franja_anterior = fecha_actual

# Carga de pedidos comprendidos en el rango de fecha y hora establecido
cur.execute("""SELECT id_pedido, id_comercio 
            FROM pedidos_modelo 
            WHERE status = 'Entregado'
                AND fecha_entrega = %s 
                AND hora_entrega BETWEEN %s AND %s;""",
            (fecha_franja_anterior,
            franjas_horarias[franjas_horarias["franja_horaria"] == franja_anterior]["h_inicio"].iloc[0],
            franjas_horarias[franjas_horarias["franja_horaria"] == franja_anterior]["h_final"].iloc[0]))
pedidos_franja = pd.DataFrame(cur.fetchall(), columns = ["id_pedido", "id_comercio"]) 

# Carga de la tabla de comercios, unión con los pedidos para asociar una zona a
# estos y agrupación por zonas (la fecha y franja horaria son iguales en todos 
# los pedidos por definición)
comercios = pd.read_sql("SELECT * FROM comercios;", con = conn)
pedidos_franja = pedidos_franja.merge(comercios[["id", "zona"]], 
                                      left_on = "id_comercio", 
                                      right_on = "id").drop(columns=["id_comercio", "id"])
pedidos_grouped = pedidos_franja.groupby(by = ["zona"]).count() \
                                    .reset_index() \
                                    .rename(columns = {"id_pedido": "Total"})
pedidos_grouped["franja_horaria"] = franja_anterior
pedidos_grouped["fecha_entrega"] = fecha_franja_anterior

# Volcado de los pedidos agrupados por fecha, zona y franja horaria en la BBDD
for row in pedidos_grouped.itertuples():
    try:
        cur.execute("""INSERT INTO 
                        pedidos_agrupados(fecha, zona, franja_horaria, total)
                    VALUES
                        (%s, %s, %s, %s);""",
            (row[4], row[1], row[3], row[2]))
        conn.commit()
    except Exception as error:
        print("Error: ", error)
        conn.rollback()

cur.close()
conn.close()