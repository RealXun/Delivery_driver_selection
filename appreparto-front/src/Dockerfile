# Especifica la imagen base de Docker
FROM python:3.9-slim-buster

# Copia los archivos requeridos para la aplicación de Streamlit
COPY requirements.txt /app/requirements.txt
COPY . ./app

# Establece el directorio de trabajo
WORKDIR /app

# Instalar herramientas de compilación
RUN apt-get update && apt-get install -y build-essential

# Actualiza pip
RUN python -m pip install --upgrade pip

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto en el que la aplicación de Streamlit se ejecuta
EXPOSE 8501

# Inicia la aplicación de Streamlit
CMD ["streamlit", "run", "app2.py"]