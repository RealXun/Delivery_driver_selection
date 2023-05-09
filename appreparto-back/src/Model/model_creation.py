import pandas as pd
import numpy as np
import psycopg2
import datetime as dt
import requests
import json
import sys
import params as params

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_percentage_error
import xgboost as xgb
import pickle

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

# Carga de los datos de pedidos y filtrado por status para coger solo aquellos
# entregados
pedidos = pd.read_sql("SELECT * FROM pedidos_modelo WHERE status = 'Entregado';", 
                      con = conn)
pedidos["fecha_pedido"] = pedidos["datetime_pedido"].apply(lambda x: x.date())
pedidos["hora_pedido"] = pedidos["datetime_pedido"].apply(lambda x: x.time())
pedidos.drop(columns = ["datetime_pedido"], inplace = True)

# Carga de las tablas de comercios, para extraer sus zonas asignadas, franjas
# horarias y creación de rango de fechas. Unión mediante cross merge para crear
# el dataframe base de entrenamiento.
comercios = pd.read_sql("SELECT * FROM comercios;", con = conn)
zonas = comercios.sort_values(by=["zona"])["zona"].unique()
zonas_df = pd.DataFrame(zonas, columns = ["zona"])
zonas_df.head()

franjas_horarias = pd.read_sql("SELECT * FROM franjas_horarias;", con = conn)
franjas_horarias = franjas_horarias.rename(columns={"id": "franja_horaria"})

fechas_range = pd.date_range(start=pedidos["fecha_pedido"].min(),
                             end=pedidos["fecha_pedido"].max())
fechas_df = pd.DataFrame(fechas_range, columns = ["fecha_pedido"])

base_df = pd.merge(zonas_df, franjas_horarias[["franja_horaria"]], how = "cross") \
            .merge(fechas_df, how = "cross")

# Asignación de la franja horaria al pedido
franjas_list = []
for row_pedidos in pedidos.itertuples():
    for row_franjas in franjas_horarias.itertuples():
        if (row_pedidos[8] >= row_franjas[2]) & \
            ((row_pedidos[8] < row_franjas[3])):
            franjas_list.append(row_franjas[1]) 
            break    
pedidos["franja_horaria"] = franjas_list

# Asignación de la zona al pedido
pedidos = pedidos.merge(comercios[["id", "zona"]], left_on = "id_comercio", right_on = "id")

# Agrupación de los pedidos por fecha, zona y franja horaria
pedidos_grouped = pedidos.groupby(
        by=["fecha_pedido", 
            "zona", 
            "franja_horaria"])[["id_pedido"]].count() \
                .reset_index() \
                .rename(columns = {"id_pedido": "Total"})

# Mergeo de los dataframes de pedidos con el dataframe base
pedidos_grouped["fecha_pedido"] = pd.to_datetime(pedidos_grouped["fecha_pedido"])
full_pedidos = pd.merge(base_df, pedidos_grouped, \
    right_on = ["fecha_pedido", "zona", "franja_horaria"], \
    left_on = ["fecha_pedido", "zona", "franja_horaria"], how = "left")
full_pedidos["Total"] = full_pedidos["Total"].fillna(0).astype("int")

# Creación de los campos Festivo e Isoweekday (L-D -> 1-7)
with open('Festivos_2022.json', 'r') as festivos_2022:
    festivos = json.load(festivos_2022)
full_pedidos["festivo"] = full_pedidos["fecha_pedido"].isin(festivos["Festivos"])
full_pedidos["isoweekday"] = full_pedidos["fecha_pedido"] \
    .apply(lambda x: x.isoweekday())

# Asignación de los pedidos totales en la franja, día y día de la semana anterior
franja_anterior = []
dia_anterior = []
semana_anterior = [] 
for i,row in full_pedidos.iterrows():
    franja = full_pedidos[(full_pedidos["zona"] == row["zona"]) &
        (full_pedidos["fecha_pedido"] == row["fecha_pedido"]) &
        (full_pedidos["franja_horaria"] == row["franja_horaria"] - 1)]["Total"]
    if franja.shape[0] == 0:
        franja_anterior.append(np.nan)
    else:
        franja_anterior.append(franja.values[0])
    dia = full_pedidos[(full_pedidos["zona"] == row["zona"]) &
        (full_pedidos["franja_horaria"] == row["franja_horaria"]) &
        (full_pedidos["fecha_pedido"] == 
            (row["fecha_pedido"] - dt.timedelta(days = 1)))]["Total"]
    if dia.shape[0] == 0:
        dia_anterior.append(np.nan)
    else:
        dia_anterior.append(dia.values[0])
    semana = full_pedidos[(full_pedidos["zona"] == row["zona"]) &
        (full_pedidos["franja_horaria"] == row["franja_horaria"]) &
        (full_pedidos["fecha_pedido"] == 
            (row["fecha_pedido"] - dt.timedelta(days = 7)))]["Total"]
    if semana.shape[0] == 0:
        semana_anterior.append(np.nan)
    else:
        semana_anterior.append(semana.values[0])
full_pedidos["Total_franja"] = franja_anterior
full_pedidos["Total_dia"] = dia_anterior
full_pedidos["Total_semana"] = semana_anterior

# Borrado de los registros que no contengan todos los datos de acumulación de pedidos
# (esto es a efectos prácticos deshacerse de la primera semana)
full_pedidos = full_pedidos.dropna()

# Asignación de los datos climatológicos
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzdXNhbmFydWl6bW9sZXJvQGdtYWlsLmNvbSIsImp0aSI6ImQ2OWY0ZDYwLTQwYmItNGNjYS04YzZkLTcwY2ZmZjQxNTUwMCIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNjc5Njc2Njg4LCJ1c2VySWQiOiJkNjlmNGQ2MC00MGJiLTRjY2EtOGM2ZC03MGNmZmY0MTU1MDAiLCJyb2xlIjoiIn0.wZdHR5OvKkjhqTGx3RsWl-vDkZHXdQ3z6dxAdUwcGCQ"
url= "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/2022-02-11T00:00:00UTC/fechafin/2022-04-06T23:59:59UTC/estacion/3195/?api_key=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzdXNhbmFydWl6bW9sZXJvQGdtYWlsLmNvbSIsImp0aSI6ImQ2OWY0ZDYwLTQwYmItNGNjYS04YzZkLTcwY2ZmZjQxNTUwMCIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNjc5Njc2Njg4LCJ1c2VySWQiOiJkNjlmNGQ2MC00MGJiLTRjY2EtOGM2ZC03MGNmZmY0MTU1MDAiLCJyb2xlIjoiIn0.wZdHR5OvKkjhqTGx3RsWl-vDkZHXdQ3z6dxAdUwcGCQ"
response = requests.get(url)
data = response.json()["datos"]
data = requests.get(data).json()
clima_df = pd.DataFrame.from_dict(data)
clima_df = clima_df.rename(columns={'tmed': 'temperatura', 
                                    'prec': 'precipitaciones'})
clima_df["fecha"] = pd.to_datetime(clima_df["fecha"])
full_pedidos = full_pedidos.merge(clima_df[["fecha", 
                                            "temperatura", 
                                            "precipitaciones"]], 
                                  left_on = "fecha_pedido", 
                                  right_on = "fecha").drop(columns=["fecha"])
full_pedidos["temperatura"] = full_pedidos["temperatura"] \
                                .apply(lambda x: float(x.replace(",",".")))
full_pedidos["precipitaciones"] = full_pedidos["precipitaciones"] \
                                .apply(lambda x: float(x.replace(",",".")))

# Reajustes del dataframe
full_pedidos[["Total_franja", "Total_dia", "Total_semana"]] = \
    full_pedidos[["Total_franja", "Total_dia", "Total_semana"]].astype("int")
full_pedidos["zona"] = full_pedidos["zona"].apply(lambda x: str(x)[2:])
full_pedidos = full_pedidos.drop(columns = ["fecha_pedido"])

# Construcción del modelo
y = full_pedidos["Total"]
X = full_pedidos.drop(columns = ["Total"])
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state=42)

num_attr = ["Total_franja", "Total_dia", "Total_semana", "temperatura", "precipitaciones"]
cat_attr = ["zona", "franja_horaria", "isoweekday"]

numeric_transformer = Pipeline([('std_scaler', StandardScaler())])
categorical_transformer = Pipeline([ ('onehotencoder', OneHotEncoder())])

preprocessor = ColumnTransformer(transformers = [('num', numeric_transformer, num_attr),
                                                 ('cat', categorical_transformer, cat_attr)])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', xgb.XGBRegressor())])
model.fit(X_train, y_train)

with open('finished_model.model', "wb") as archivo_salida:
    pickle.dump(model, archivo_salida)