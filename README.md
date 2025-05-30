# 🏗️ Sistema de Recomendaciones Inteligentes para Corona

## 📋 Descripción del Proyecto

Este proyecto desarrolla una solución analítica integral de sistemas de recomendaciones personalizadas para Corona, empresa líder en materiales de construcción. La solución implementa algoritmos de machine learning especializados para los segmentos B2C y B2B, optimizando la experiencia del cliente y maximizando las oportunidades de ventas cruzadas.

### 🎯 Objetivos Principales

- **Personalización Avanzada**: Desarrollar recomendaciones precisas adaptadas a cada segmento de cliente
- **Optimización de Ventas**: Maximizar oportunidades de cross-selling y up-selling
- **Eficiencia Operativa**: Automatizar el proceso de recomendación de productos
- **Escalabilidad**: Crear una solución robusta y escalable para el ecosistema Corona

### 🔬 Metodología

El proyecto implementa un **sistema híbrido multi-algoritmo** que combina múltiples técnicas de machine learning:

- **Filtrado Colaborativo**: Análisis de patrones de comportamiento entre usuarios similares
- **Filtrado Basado en Contenido**: Recomendaciones por similitud de productos
- **Análisis Geográfico**: Cross-selling basado en patrones zonales
- **Co-ocurrencia Inteligente**: Productos frecuentemente comprados juntos
- **Reciprocal Rank Fusion (RRF)**: Fusión optimizada de múltiples modelos

## 🏗️ Estructura del Proyecto

```
CoronaReto Ultimo intento/
├── 📊 Datos/                              # Datasets originales (archivos .txt)
├── 🔧 Ingenieria de Caracteres y Analisis de datos/  # Procesamiento y transformación
├── 🤖 Algoritmos/                         # Algoritmos de recomendación
│   ├── algoritmo_B2B.ipynb              # Algoritmo B2B (producción)
│   ├── algoritmo_B2B_train.ipynb        # Algoritmo B2B (entrenamiento/métricas)
│   ├── algoritmo_B2C_hibrido.ipynb      # Algoritmo B2C personalizado (producción)
│   ├── algoritmo_B2C_hibrido_train.ipynb # Algoritmo B2C personalizado (entrenamiento)
│   ├── algoritmo_B2C_historial.ipynb    # Algoritmo B2C por producto (producción)
│   └── algoritmo_B2C_historial_train.ipynb # Algoritmo B2C por producto (entrenamiento)
├── 🎯 Modelos/                           # Modelos entrenados
│   ├── ModelosCompletos/                 # Modelos entrenados con datos completos
│   └── ModelosTrain/                     # Modelos para validación y métricas
├── 📈 Metricas de Algoritmos/            # Validación y evaluación
│   ├── validacion_B2C_hibrido.ipynb     # Métricas del algoritmo personalizado
│   └── validacion_B2C_historico.ipynb   # Métricas del algoritmo por producto
├── 📑 documento.tex                      # Documentación técnica completa
└── 📖 README.md                          # Esta documentación
```

## 🚀 Algoritmos Implementados

### 1. 🏢 **Algoritmo B2B Empresarial**
- **Input**: ID de cliente B2B
- **Técnicas**: Filtrado por valor, predicción de reposición, cross-selling geográfico, co-ocurrencia
- **Características Especiales**:
  - Pesos adaptativos según segmento de cliente (Alto Valor, Nuevos, etc.)
  - Boost del 20% para productos estratégicos (alineación > 0.8)
  - Ajustes estacionales dinámicos
  - Optimización para grandes volúmenes y valor transaccional

### 2. 👤 **Algoritmo B2C Personalizado**
- **Input**: ID de cliente B2C
- **Técnicas**: Filtrado colaborativo, análisis demográfico, patrones geográficos
- **Características**: Recomendaciones basadas en historial completo del cliente

### 3. 🛍️ **Algoritmo B2C por Producto**
- **Input**: Producto específico
- **Técnicas**: Similitud de contenido, co-compra, popularidad categórica
- **Uso**: Ideal para asesores de venta y cross-selling en punto de venta

## 📊 Datasets Utilizados

| Dataset | Registros | Clientes | Productos | Valor Total |
|---------|-----------|----------|-----------|-------------|
| **Transacciones B2C** | 2,099,836 | 419,226 | 7,280 | $83,975,996 |
| **Cotizaciones B2C** | 180,387 | 57,184 | 2,735 | $6,657,250 |
| **Transacciones B2B** | 25,866 | 6 | 2,564 | $39,729,642 |

## 🛠️ Instalación y Configuración

### Prerrequisitos
```bash

python == 3.11 
# Librerías principales para análisis de datos
pandas==2.2.3
numpy==2.0.2
scipy==1.15.1

# Machine Learning y modelado
scikit-learn==1.6.1
statsmodels==0.14.4

# Visualización y análisis exploratorio
matplotlib==3.10.0
seaborn==0.13.2
plotly==6.0.0

# Jupyter y entorno de desarrollo
ipykernel==6.29.5
ipython==8.31.0
jupyter_client==8.6.3
jupyter_core==5.7.2

# Procesamiento y utilidades
openpyxl==3.1.5  # Para archivos Excel
tqdm==4.67.1     # Barras de progreso
joblib==1.4.2    # Paralelización y serialización

# Análisis de texto (si se utiliza)
nltk==3.9.1
spacy==3.8.4

# Perfilado de datos (opcional)
ydata-profiling==4.12.2
```

### Instalación Rápida
```bash
# Instalar todas las dependencias principales
pip install pandas==2.2.3 numpy==2.0.2 scipy==1.15.1 scikit-learn==1.6.1 matplotlib==3.10.0 seaborn==0.13.2 jupyter
```

### Estructura de Datos Requerida
1. Coloque los archivos de datos (.txt) en la carpeta `Datos/`
2. Los datasets deben tener las siguientes estructuras mínimas:
   - **B2C Transacciones**: cliente_id, producto, fecha, valor, municipio, zona, categoría
   - **B2C Cotizaciones**: cliente_id, producto, fecha, valor, estado
   - **B2B Transacciones**: id_b2b, producto, fecha, valor_total, zona, alineación_estratégica

## 🔄 Flujo de Ejecución

### 📋 Pasos Detallados

1. **📥 Preparación de Datos**
   ```bash
   # Ejecutar ingeniería de características
   jupyter notebook "Ingenieria de Caracteres y Analisis de datos/ingenieria_de_Caracteres_B2C.ipynb"
   ```

2. **🎯 Entrenamiento de Modelos**
   ```bash
   # Para algoritmos B2C (ejecutar en orden)
   jupyter notebook "Algoritmos/algoritmo_B2C_hibrido_train.ipynb"
   jupyter notebook "Algoritmos/algoritmo_B2C_historial_train.ipynb"
   
   # Para algoritmo B2B
   jupyter notebook "Algoritmos/algoritmo_B2B_train.ipynb"
   ```

3. **📊 Validación y Métricas**
   ```bash
   # Métricas B2C (después del entrenamiento)
   jupyter notebook "Metricas de Algoritmos/validacion_B2C_hibrido.ipynb"
   jupyter notebook "Metricas de Algoritmos/validacion_B2C_historico.ipynb"
   
   # Métricas B2B (incluidas en el mismo notebook)
   # Ver algoritmo_B2B.ipynb
   ```

4. **🚀 Producción**
   ```bash
   # Algoritmos listos para producción
   jupyter notebook "Algoritmos/algoritmo_B2C_hibrido.ipynb"      # Recomendaciones personalizadas
   jupyter notebook "Algoritmos/algoritmo_B2C_historial.ipynb"    # Recomendaciones por producto
   jupyter notebook "Algoritmos/algoritmo_B2B.ipynb"             # Recomendaciones B2B
   ```

## 📈 Características Técnicas Avanzadas

### 🔄 **Sistema Híbrido RRF (Reciprocal Rank Fusion)**
- Combina múltiples algoritmos con pesos adaptativos
- Optimización automática según contexto del cliente
- Fusión inteligente de rankings individuales

### 🎛️ **Personalización Dinámica**
- **Clientes Alto Valor**: Priorización de productos rentables (w_v = 0.4)
- **Clientes Nuevos**: Activación exclusiva de componente geográfico (w_g = 1.0)
- **Productos Estratégicos**: Boost automático del 20%
- **Ajustes Estacionales**: Factores dinámicos por patrones temporales

### 📊 **Métricas de Evaluación**
- **Precision@K**: Relevancia de recomendaciones top-K
- **Recall@K**: Cobertura de productos relevantes
- **Coverage**: Diversidad del catálogo recomendado
- **Serendipity**: Capacidad de descubrimiento
- **Business Impact**: Métricas de valor comercial

## 👥 Autores

- **Andrés Chaparro Díaz** - Desarrollo de algoritmos y metodología
- **Juan Andrés Bernal** - Ingeniería de características y validación

**Institución**: Universidad de los Andes - Departamento de Ingeniería Industrial

## 📄 Documentación Técnica

Para documentación detallada de la metodología, resultados y análisis técnico, consulte:
- **Documento técnico completo**: `documento.tex` (compilar con LaTeX)
- **Notebooks individuales**: Cada algoritmo incluye documentación inline detallada

## 🔒 Confidencialidad

Este proyecto utiliza datos anonimizados y transformados proporcionados por Corona para proteger la confidencialidad empresarial. Las relaciones estructurales y patrones de comportamiento se mantienen intactos para permitir el desarrollo efectivo de los algoritmos.

