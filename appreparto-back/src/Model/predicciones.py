import pandas as pd
import psycopg2
import datetime as dt
import pickle
import json
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
# sobre el que se hará la predicción
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
# franja_actual = 5        
# fecha_actual = "2022-03-31"
# fecha_actual = dt.datetime.strptime(fecha_actual, "%Y-%m-%d").date()
if franja_actual == franjas_horarias["franja_horaria"].max():
    franja_pred = franjas_horarias["franja_horaria"].min()
    fecha_pred = fecha_actual + dt.timedelta(days=1)
else: 
    franja_pred = franja_actual + 1
    fecha_pred = fecha_actual

# Carga de la tabla de comercios para generar el dataframe base para las 
# predicciones caracterizadas por las zonas.
comercios = pd.read_sql("SELECT * FROM comercios;", con = conn)
zonas = comercios.sort_values(by=["zona"])["zona"].unique()
pred_df = pd.DataFrame(zonas, columns = ["zona"])
pred_df["franja_horaria"] = franja_pred

# Comparativa con la lista de festivos del año y asignación numérica
# del día de la semana.
with open('Festivos_2022.json', 'r') as festivos_2022:
    festivos = json.load(festivos_2022)
pred_df["festivo"] = fecha_pred.strftime("%Y-%m-%d") in festivos["Festivos"]
pred_df["isoweekday"] = fecha_pred.isoweekday()

# Carga de los datos históricos conteniendo el número de pedidos asociados
# a la zona: en el mismo día dos franjas horarias antes de la predicción;
# el día anterior durante la misma franja horaria; la semana anterior, en el 
# mismo día y misma franja horaria. 
try:    
    if franja_pred == franjas_horarias["franja_horaria"].min():
        cur.execute("""SELECT zona, total 
                    FROM pedidos_agrupados 
                    WHERE fecha = %s
                        AND franja_horaria = %s;""",
                        ((fecha_pred - dt.timedelta(days=1)),
                        franjas_horarias["franja_horaria"].max().item() - 1))
        grouped_franja = pd.DataFrame(cur.fetchall(), columns = ["zona", "total_franja"])
    elif franja_pred == (franjas_horarias["franja_horaria"].min() + 1):
        cur.execute("""SELECT zona, total 
                    FROM pedidos_agrupados 
                    WHERE fecha = %s
                        AND franja_horaria = %s;""",
                        ((fecha_pred - dt.timedelta(days=1)),
                        franjas_horarias["franja_horaria"].max().item())) 
        grouped_franja = pd.DataFrame(cur.fetchall(), columns = ["zona", "total_franja"])
    else:
        cur.execute("""SELECT zona, total 
                    FROM pedidos_agrupados 
                    WHERE fecha = %s
                        AND franja_horaria = %s;""",
                        ((fecha_pred),
                        franja_pred - 2))
        grouped_franja = pd.DataFrame(cur.fetchall(), columns = ["zona", "total_franja"])
except Exception as error:
    print(error)
    conn.rollback()
pred_df = pd.merge(pred_df, grouped_franja, on = "zona", how = "left").fillna(0)

try:
    cur.execute("""SELECT zona, total 
            FROM pedidos_agrupados 
            WHERE fecha = %s
                AND franja_horaria = %s;""",
                ((fecha_pred - dt.timedelta(days=1)),
                franja_pred))
    grouped_dia = pd.DataFrame(cur.fetchall(), columns = ["zona", "total_dia"])
except Exception as error:
    print(error)
    conn.rollback()

pred_df = pd.merge(pred_df, grouped_dia, on = "zona", how = "left").fillna(0)

try:
    cur.execute("""SELECT zona, total 
            FROM pedidos_agrupados 
            WHERE fecha = %s
                AND franja_horaria = %s;""",
                ((fecha_pred - dt.timedelta(days=7)),
                franja_pred))
    grouped_semana = pd.DataFrame(cur.fetchall(), columns = ["zona", "total_semana"])
except Exception as error:
    print(error)
    conn.rollback()

pred_df = pd.merge(pred_df, grouped_semana, on = "zona", how = "left").fillna(0)

# Carga del modelo, encoders y scalers (unificados a través de una misma pipeline)
with open('model.model', "rb") as archivo_entrada:
    model = pickle.load(archivo_entrada)

# Predicciones y redondeo al número entero más cercano. En caso de resultar una
# predicción negativa, se asigna el valor cero.
predicciones = model.predict(pred_df)
predicciones_round = [round(prediccion) if prediccion > 0 else 0 for prediccion in predicciones]
pred_df["predicciones"] = predicciones_round

# Volcado de los pedidos predichos por fecha, zona y franja horaria en la BBDD
for i,row in pred_df.iterrows():
    try:
        cur.execute("""INSERT INTO 
                        pedidos_prediccion(fecha, zona, franja_horaria, total)
                    VALUES
                        (%s, %s, %s, %s);""",
            (fecha_pred, row["zona"], franja_pred, row["predicciones"]))
        conn.commit()
    except Exception as error:
        print("Error: ", error)
        conn.rollback()


cur.close()
conn.close()
