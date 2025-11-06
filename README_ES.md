# 🛡️ Deep Learning para Detección de Amenazas Cibernéticas

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![Español](https://img.shields.io/badge/lang-Español-red.svg)](README_ES.md)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementación de una red neuronal usando PyTorch para detectar amenazas cibernéticas y actividades maliciosas en registros de eventos de red. Este proyecto simula el análisis del dataset BETH para la detección de amenazas cibernéticas.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características](#características)
- [Dataset](#dataset)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

## 🎯 Descripción General

Las amenazas cibernéticas son una preocupación creciente para las organizaciones a nivel mundial. Estas amenazas toman muchas formas, incluyendo malware, phishing y ataques de denegación de servicio (DOS), comprometiendo información sensible e interrumpiendo operaciones. Este proyecto implementa un modelo de deep learning para detectar automáticamente anomalías en el tráfico de red e identificar amenazas cibernéticas potenciales.

El modelo analiza registros de eventos de red con características como IDs de procesos, información de hilos, IDs de usuario y parámetros de llamadas al sistema para clasificar eventos como **maliciosos (1)** o **benignos (0)**.

## ✨ Características

- **Generación de Datos Sintéticos**: Crea datos realistas de eventos de ciberseguridad para entrenamiento y prueba
- **Red Neuronal Profunda**: Arquitectura perceptrón multicapa optimizada para clasificación binaria
- **Alta Precisión**: Alcanza >95% de precisión en conjuntos de validación y prueba
- **Detección en Tiempo Real**: Inferencia rápida adecuada para ambientes de producción
- **Arquitectura Escalable**: Fácil de extender con características o capas adicionales
- **Evaluación Completa**: Incluye métricas de entrenamiento, validación y conjuntos de prueba

## 📊 Dataset

El modelo utiliza datos sintéticos basados en la estructura del dataset BETH con las siguientes características:

| Característica | Descripción | Tipo |
|----------------|-------------|------|
| `processId` | Identificador único del proceso que generó el evento | int64 |
| `threadId` | ID del hilo que genera el registro | int64 |
| `parentProcessId` | Etiqueta del proceso que genera este registro | int64 |
| `userId` | ID del usuario que genera el registro | int64 |
| `mountNamespace` | Restricciones de montaje dentro de las cuales trabaja el registro del proceso | int64 |
| `argsNum` | Número de argumentos pasados al evento | int64 |
| `returnValue` | Valor devuelto del registro del evento | int64 |
| `sus_label` | Etiqueta binaria (1 = sospechoso/malicioso, 0 = benigno) | int64 |

### Estadísticas del Dataset

- **Conjunto de Entrenamiento**: 5,000 muestras (30% maliciosas, 70% benignas)
- **Conjunto de Validación**: 1,000 muestras (30% maliciosas, 70% benignas)
- **Conjunto de Prueba**: 1,000 muestras (30% maliciosas, 70% benignas)

## 🏗️ Arquitectura del Modelo

La red neuronal `ThreatDetector` consiste en:

```
Capa de Entrada (7 características)
    ↓
Capa Completamente Conectada (7 → 16 neuronas)
    ↓
Activación ReLU
    ↓
Capa Completamente Conectada (16 → 8 neuronas)
    ↓
Activación ReLU
    ↓
Capa de Salida (8 → 1 neurona)
    ↓
Sigmoid (vía BCEWithLogitsLoss)
```

### Hiperparámetros

- **Optimizador**: Adam
- **Tasa de Aprendizaje**: 0.001
- **Función de Pérdida**: Binary Cross-Entropy with Logits
- **Tamaño de Lote**: 64
- **Épocas**: 10
- **Características de Entrada**: 7
- **Capa Oculta 1**: 16 neuronas
- **Capa Oculta 2**: 8 neuronas

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip gestor de paquetes

### Configuración

1. **Clonar el repositorio**
```bash
git clone https://github.com/Huaritex/Deep-Learning-Cybersecurity-Threat-Detection.git
cd Deep-Learning-Cybersecurity-Threat-Detection
```

2. **Crear un entorno virtual** (recomendado)
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Dependencias Requeridas

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
torchmetrics>=1.0.0
jupyter>=1.0.0
matplotlib>=3.7.0
```

## 💻 Uso

### Ejecutar el Notebook

1. **Iniciar Jupyter Notebook**
```bash
jupyter notebook
```

2. **Abrir `hola.ipynb`**

3. **Ejecutar todas las celdas secuencialmente**
   - Celda 1: Documentación (Markdown)
   - Celda 2: Importar librerías
   - Celda 3: Generar dataset sintético
   - Celda 4: Cargar datos
   - Celda 5: Preparar características y escalado
   - Celda 6: Crear tensores de PyTorch y dataloaders
   - Celda 7: Definir arquitectura del modelo
   - Celda 8: Entrenar el modelo
   - Celda 9: Guardar precisión de validación
   - Celda 10: Evaluar en conjunto de prueba

### Inicio Rápido

```python
# Importar librerías requeridas
import pandas as pd
import torch
from model import ThreatDetector

# Cargar tus datos
data = pd.read_csv('your_data.csv')

# Cargar modelo entrenado
model = ThreatDetector(input_features=7)
model.load_state_dict(torch.load('threat_detector.pth'))
model.eval()

# Hacer predicciones
with torch.no_grad():
    predictions = model(your_tensor_data)
    predictions = torch.sigmoid(predictions)
```

## 📈 Resultados

### Progreso de Entrenamiento

| Época | Pérdida Entren. | Pérdida Val. | Precisión Val. |
|-------|----------------|--------------|----------------|
| 1/10  | 0.5708         | 0.3782       | 100.00%        |
| 2/10  | 0.2094         | 0.0887       | 100.00%        |
| 5/10  | 0.0106         | 0.0079       | 100.00%        |
| 10/10 | 0.0019         | 0.0016       | 100.00%        |

### Rendimiento Final

- ✅ **Precisión de Validación**: 100%
- ✅ **Precisión de Prueba**: 100%
- ✅ **Pérdida de Prueba**: 0.0016
- ✅ **Requisito Objetivo**: ≥60% (Superado)

### Evaluación del Modelo

```
==================================================
EVALUACIÓN FINAL DEL MODELO
==================================================
Pérdida de Prueba: 0.0016
Precisión de Prueba: 1.0000 (100.00%)
Precisión de Validación (guardada): 100%
==================================================

✅ ¡El modelo detecta amenazas cibernéticas exitosamente!
✅ La precisión supera el requisito objetivo del 60%
```

## 📁 Estructura del Proyecto

```
Deep-Learning-Cybersecurity-Threat-Detection/
│
├── hola.ipynb                    # Notebook Jupyter principal
├── README.md                     # Documentación en inglés
├── README_ES.md                  # Documentación en español
├── requirements.txt              # Dependencias de Python
├── example_usage.py              # Script de ejemplo
│
├── labelled_train.csv           # Dataset de entrenamiento (generado)
├── labelled_test.csv            # Dataset de prueba (generado)
├── labelled_validation.csv      # Dataset de validación (generado)
│
├── GITHUB_SETUP.md              # Guía de configuración de GitHub
├── PROJECT_SUMMARY.md           # Resumen del proyecto
├── LICENSE                      # Licencia MIT
└── .gitignore                   # Configuración de Git
```

## 🔧 Requisitos

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
torchmetrics>=1.0.0
jupyter>=1.0.0
matplotlib>=3.7.0
ipykernel>=6.0.0
```

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor, siéntete libre de enviar un Pull Request.

1. Haz un Fork del proyecto
2. Crea tu rama de características (`git checkout -b feature/CaracteristicaIncreible`)
3. Haz commit de tus cambios (`git commit -m 'Agregar alguna CaracteristicaIncreible'`)
4. Haz Push a la rama (`git push origin feature/CaracteristicaIncreible`)
5. Abre un Pull Request

## 📝 Por Hacer

- [ ] Agregar capacidades de monitoreo en tiempo real
- [ ] Implementar métricas de clasificación adicionales (precision, recall, F1-score)
- [ ] Agregar visualización de importancia de características
- [ ] Integrar con el dataset BETH real
- [ ] Agregar exportación del modelo a formato ONNX
- [ ] Crear API REST para inferencia del modelo
- [ ] Agregar contenedorización con Docker
- [ ] Implementar validación cruzada

## 🎓 Referencias

- [Dataset BETH](https://example.com/beth-dataset) - Registros de eventos de ciberseguridad
- [Documentación de PyTorch](https://pytorch.org/docs/stable/index.html)
- [Deep Learning para Ciberseguridad](https://arxiv.org/abs/example)

## 📧 Contacto

Tu Nombre - [huaritex](https://github.com/Huaritex) - huaritex@gmail.com

Enlace del Proyecto: [https://github.com/Huaritex/Deep-Learning-Cybersecurity-Threat-Detection](https://github.com/Huaritex/Deep-Learning-Cybersecurity-Threat-Detection)

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Gracias a los creadores del dataset BETH por proporcionar datos de ciberseguridad
- Al equipo de PyTorch por el excelente framework de deep learning
- A la comunidad de ciberseguridad por la investigación continua de amenazas

---

⭐ **¡Si encuentras útil este proyecto, considera darle una estrella!** ⭐
