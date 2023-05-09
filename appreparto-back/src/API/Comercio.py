class Comercio():

    def __init__(self, id, nombre=None, zona=None, 
                 direccion=None, tipo=None,
                latitud=None, longitud=None) -> None:
        self.id= id
        self.nombre = nombre
        self.zona = zona
        self.direccion = direccion
        self.tipo=tipo
        self.latitud = latitud
        self.longitud = longitud
        
    def to_JSON(self):
        return {
            'id': self.id,
            'nombre': self.nombre,
            'zona': self.zona,
            'direccion':self.direccion,
            'tipo':self.tipo,
            'latitud': self.latitud,
            'longitud':self.longitud
        }