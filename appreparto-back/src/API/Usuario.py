class Usuario():

    def __init__(self, id, username=None, id_cliente=None, id_administrador=None, id_repartidor=None, tipo=None, contraseña=None) -> None:
        self.id = id
        self.username = username
        self.id_cliente = id_cliente
        self.id_administrador = id_administrador
        self.id_repartidor= id_repartidor
        self.tipo= tipo
        self.contraseña=contraseña

    def to_JSON(self):
        return {
            'id': self.id,
            'username': self.username,
            'id_cliente': self.id_cliente,
            'id_administrador': self.id_administrador,
            'id_repartidor': self.id_repartidor,
            'tipo': self.tipo,
            'contraseña': self.contraseña
        }