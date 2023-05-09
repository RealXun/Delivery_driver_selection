class Pedido():

    def __init__(self, id_pedido, id_cliente=None, id_comercio=None, 
                 id_repartidor=None, latitud=None, longitud=None, datetime_pedido=None, 
                 tamaño=None, status=None, direccion=None) -> None:
        self.id_pedido= id_pedido
        self.id_cliente = id_cliente
        self.id_comercio = id_comercio
        self.id_repartidor = id_repartidor
        self.latitud = latitud
        self.longitud = longitud
        self.datetime_pedido = datetime_pedido
        self.tamaño = tamaño
        self.status = status
        self.direccion = direccion

    def to_JSON(self):
        return {
            'id_pedido': self.id_pedido,
            'id_cliente': self.id_cliente,
            'id_comercio': self.id_comercio,
            'id_repartidor': self.id_repartidor,
            'latitud': self.latitud,
            'longitud':self.longitud,
            'datetime_pedido': self.datetime_pedido,
            'tamaño':self.tamaño,
            'status':self.status,
            'direccion':self.direccion
        }