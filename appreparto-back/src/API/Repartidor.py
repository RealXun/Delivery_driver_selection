class Repartidor():

    def __init__(self, id, nombre=None, email=None, status=None, vehiculo=None, 
                 ocupado=None, zona=None, latitud=None, longitud=None, datetime_ult_act=None) -> None:
        self.id = id
        self.nombre = nombre
        self.email= email
        self.status = status
        self.vehiculo = vehiculo
        self.ocupado = ocupado
        self.zona = zona
        self.latitud = latitud
        self.longitud = longitud
        self.datetime_ult_act = datetime_ult_act
        

    def to_JSON(self):
        return {
            'id': self.id,
            'unombre': self.nombre,
            'email': self.email,
            'status':self.status,
            'vehiculo': self.vehiculo,
            'ocupado': self.ocupado,
            'zona': self.zona,
            'latitud': self.latitud,
            'longitud':self.longitud,
            'datetime_ult_act': self.datetime_ult_act 
        }