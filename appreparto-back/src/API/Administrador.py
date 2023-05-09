class Administrador():

    def __init__(self, id, nombre=None, email=None) -> None:
        self.id = id
        self.nombre = nombre
        self.email = email
        

    def to_JSON(self):
        return {
            'id': self.id,
            'nombre': self.nombre,
            'email': self.email,
            
        }