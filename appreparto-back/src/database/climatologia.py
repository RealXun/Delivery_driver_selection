import pandas as pd
import psycopg2
import datetime as dt
from pyowm import OWM
from pyowm.utils.config import get_default_config
import sys
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
except Exception as error:
    print(error)
    sys.exit()

estado='ES'
ciudad='Madrid'

API_key = params.owm_key
config_dict = get_default_config()
config_dict['language'] = 'es'
owm = OWM(API_key, config_dict)
mgr = owm.weather_manager()
observation = mgr.forecast_at_place("{},{}".format(ciudad,estado),'3h').forecast
weathers = observation.weathers
for weather in weathers:
    try:
        temp = weather.temperature('celsius')["temp"]
        if weather.rain:
            rain = weather.rain["3h"]
        else:
            rain = 0
        detailed_status = weather.detailed_status
        datetime_forecast = dt.datetime.fromtimestamp(weather.to_dict()["reference_time"])
        sql = """INSERT INTO climatologia (fecha, hora, temperatura, lluvia, descripcion) VALUES (%s, %s, %s, %s, %s)"""
        args = (datetime_forecast.date(), 
                datetime_forecast.time(), 
                temp,
                rain,
                detailed_status)
        cur.execute(sql, args)
    except Exception as error:
        print(error)
        print(dt.datetime.fromtimestamp(weather.to_dict()["reference_time"]))
        conn.rollback()

cur.close()
conn.close()