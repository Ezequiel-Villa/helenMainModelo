# Helen - Instrucciones para ejecutar el proyecto

## 1. Clonar el repositorio
```bash
git clone https://github.com/ChrisUBS/HelenProyecto
cd HelenProyecto
```

## 2. Configurar las diferentes entidades del proyecto (Modelo - API - Interfaz)
Se deben instalar las dependencias de cada división de forma individual.
- [Instalación de librerías del Modelo](#id1)
- [Instalación de librerías del API](#id2)
- [Instalación de dependencias del interfaz de usuario](#id3)

## Instalación de Librerías

### Modelo <a id="id1" name="id1"></a>
**Requisito:** Python 3.7 requerido.

#### Para Linux/macOS:
1. Crear Entorno Virtual:
   ```bash
   cd Hellen_model_RN
   python3.7 -m venv venv
   ```
2. Instalación de dependencias:
   ```bash
   source venv/bin/activate
   python3.7 -m pip install -r requirements.txt
   ```

#### Para Windows:
1. Crear Entorno Virtual:
   ```cmd
   cd Hellen_model_RN
   python3.7 -m venv venv
   ```
2. Instalación de dependencias:
   ```cmd
   venv\Scripts\activate
   python3.7 -m pip install -r requirements.txt
   ```

### API <a id="id2" name="id2"></a>
**Requisito:** Python 3.10 o 3.12 recomendado.

#### Para Linux/macOS:
1. Crear Entorno Virtual:
   ```bash
   cd backendHelen
   python3 -m venv venv
   ```
2. Instalación de dependencias:
   ```bash
   source venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

#### Para Windows:
1. Crear Entorno Virtual:
   ```cmd
   cd backendHelen
   python3 -m venv venv
   ```
2. Instalación de dependencias:
   ```cmd
   venv\Scripts\activate
   python3 -m pip install -r requirements.txt
   ```

### Interfaz <a id="id3" name="id3"></a>
1. Instalación de dependencias:
   ```bash
   cd helen
   npm install
   ```

## Ejecución de Helen
Helen funciona con tres programas ejecutándose simultáneamente. Se requiere una consola por cada uno.

### 1. Modelo
#### Para Linux/macOS:
```bash
cd Hellen_model_RN
source venv/bin/activate
python3.7 inference_classifier.py
```

#### Para Windows:
```cmd
cd Hellen_model_RN
venv\Scripts\activate
python3.7 inference_classifier.py
```

### 2. API
#### Para Linux/macOS:
```bash
cd backendHelen
source venv/bin/activate
python3 server.py
```

#### Para Windows:
```cmd
cd backendHelen
venv\Scripts\activate
python3 server.py
```

### 3. Interfaz
#### Para Linux/macOS:
```bash
cd helen
npm start
```

#### Para Windows:
```cmd
cd helen
npm start
```

---
Este flujo asegura que el modelo, la API y la interfaz se ejecuten correctamente.