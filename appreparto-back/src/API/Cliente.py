
class Cliente():

    def __init__(self, id, nombre=None, email=None, direccion=None, latitud=None, longitud=None) -> None:
        self.id = id
        self.nombre = nombre
        self.email = email
        self.direccion = direccion
        self.latitud= latitud
        self.longitud= longitud

    def to_JSON(self):
        return {
            'id': self.id,
            'nombre': self.nombre,
            'email': self.email,
            'direccion': self.direccion,
            'latitud': self.latitud,
            'longitud': self.longitud
        }

