import pandas as pd
import numpy as np
import psycopg2
import datetime as dt
import sys
import params as params
from sklearn.metrics import mean_squared_error, r2_score, \
                            mean_absolute_percentage_error

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

# Obtención de la fecha de ejecución para cálculo de las fechas para el informe
# En este caso, primer y último día de la semana anterior a la semana en curso.
fecha_actual = dt.date.today()
final_semanal = fecha_actual - dt.timedelta(days = fecha_actual.isoweekday())
inicio_semanal = final_semanal - dt.timedelta(days = 6)

# Carga de la tabla con los datos para el rango de fechas indicado con la zona, 
# franja, valor agrupado (real) y valor predicho.
try:
    cur.execute("""SELECT p.zona, p.franja_horaria, a.total, p.total
            FROM pedidos_prediccion as p
            LEFT JOIN pedidos_agrupados as a
            ON p.zona = a.zona
                AND p.franja_horaria = a.franja_horaria
            WHERE p.fecha >= %s AND p.fecha <= %s;""",
                (inicio_semanal, final_semanal))
    informe = pd.DataFrame(cur.fetchall(), 
                               columns = ["zona", 
                                          "franja_horaria", 
                                          "total_agrupado", 
                                          "total_prediccion"]).fillna(0)
    informe[["total_agrupado", "total_prediccion"]] = informe[["total_agrupado", 
                                                               "total_prediccion"]].astype(int)
except Exception as error:
    print(error)
    conn.rollback()

# Cálculo de las métricas y exportado a la base de datos.
y_real = informe["total_agrupado"]
y_pred = informe["total_prediccion"]

r2 = r2_score(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
mape = mean_absolute_percentage_error(y_real, y_pred)

try:
    cur.execute("""INSERT INTO informe_metricas (tipo, fecha_informe, r2, rmse, mape)
                        VALUES %s, %s, %s, %s, %s;""",
                ("semanal", fecha_actual, r2, rmse, mape))
    conn.commit()
except Exception as error:
    print(error)
    conn.rollback()

cur.close()
conn.close()