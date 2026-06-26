# LLM QLoRA Finetuner

Un pipeline modular y eficiente para realizar el ajuste fino (fine-tuning) de modelos de lenguaje mediante Adaptacion de Bajo Rango (LoRA) y cuantizacion simulada de 4 bits (QLoRA) en PyTorch, orientado a optimizar el uso de memoria en entornos con recursos limitados.

Este modulo implementa de forma artesanal y nativa la inyeccion de capas adaptadoras de bajo rango y la cuantizacion bloque a bloque bajo el formato NormalFloat4 (NF4) propuesto por Tim Dettmers et al. (2023), permitiendo emular el comportamiento de bitsandbytes sin necesidad de compilaciones locales complejas de CUDA.

## Concepto Matematico y Arquitectura

### 1. Adaptacion de Bajo Rango (LoRA)
Durante el ajuste fino tradicional de modelos de lenguaje, los pesos originales del modelo $W_0 \in \mathbb{R}^{d \times k}$ se modifican. LoRA propone congelar la matriz original $W_0$ e inyectar dos matrices de bajo rango entrenables:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A$$

Donde:
*   $A \in \mathbb{R}^{r \times d}$ es la matriz de proyeccion hacia abajo, inicializada con una distribucion normal (Kaiming Uniform).
*   $B \in \mathbb{R}^{k \times r}$ es la matriz de proyeccion hacia arriba, inicializada con ceros para asegurar que $\Delta W = 0$ al inicio del entrenamiento.
*   $r$ es el rango intrinseco configurado (generalmente $r \ll d$).
*   $\alpha$ es un factor de escala constante que evita tener que reajustar los hiperparametros al cambiar el rango.

### 2. Cuantizacion NormalFloat4 (NF4)
QLoRA mejora LoRA cuantizando los pesos de la red original $W_0$ a 4 bits usando NormalFloat4. NF4 es un tipo de datos optimizado para pesos inicializados con una distribucion normal de media cero y desviacion estandar constante. 
Este modulo divide la matriz de pesos en bloques (por defecto de tamano 64) y escala cada bloque por su maximo absoluto (absmax) para mapear los pesos en el rango de los 16 cuantiles predefinidos de NF4, reduciendo la memoria del modelo base en un 75% aproximadamente. Durante la propagacion hacia adelante (forward pass), los pesos se decuantizan dinamicamente a precision completa.

## Requisitos de Instalacion

*   Python 3.8 o superior
*   PyTorch 2.0.0 o superior
*   Numpy

Para instalar las dependencias locales en su entorno virtual, ejecute:

```bash
pip install -r requirements.txt
```

## Pruebas y Validacion

1.  **Ejecutar la suite de pruebas unitarias:**
    ```bash
    python -m unittest test_finetuner.py
    ```
    Esto validara la precision matematica de la cuantizacion NF4, la inyeccion de capas y el flujo correcto de gradientes en el grafo de PyTorch.

2.  **Ejecutar el script de demostracion:**
    ```bash
    python example.py
    ```
    El script demostrara de forma practica:
    *   La inyeccion automatica de adaptadores LoRA sobre un modelo base MLP.
    *   La congelacion del modelo base cuantizado en NF4.
    *   Un bucle de entrenamiento AdamW optimizando unicamente los pesos LoRA sobre un dataset sintetico, mostrando la reduccion del error cuadratico medio (MSE).
    *   La exportacion de los adaptadores (PEFT weights) y su posterior acoplamiento exitoso en un modelo limpio para inferencia.
