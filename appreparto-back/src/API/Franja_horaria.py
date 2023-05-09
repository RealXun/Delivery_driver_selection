class Franja_horaria():

    def __init__(self, id, h_inicio=None, h_final=None) -> None:
        self.id = id
        self.h_inicio = h_inicio
        self.h_final = h_final

    def to_JSON(self):
        return {
            'id': self.id,
            'h_inicio': self.h_inicio,
            'h_final': self.h_final,
        }