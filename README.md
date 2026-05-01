# Prototipo de Red Neuronal: Reconocimiento de Figuras Geométricas 🤖📐

Este proyecto consiste en el diseño y desarrollo de una **Red Neuronal Artificial (RNA)** capaz de clasificar imágenes de figuras geométricas planas (círculos, cuadrados y triángulos). Desarrollado como parte del curso de **Sistemas Expertos**, el modelo utiliza técnicas de aprendizaje profundo para procesar patrones visuales en una cuadrícula de 28x28 píxeles.

## 🚀 Características Técnicas
*   **Arquitectura:** Red Neuronal Densa (Multilayer Perceptron).
*   **Capas Ocultas:** Implementación de funciones de activación **ReLU** para el procesamiento de datos no lineales.
*   **Capa de Salida:** Uso de activación **Softmax** para clasificación multiclase.
*   **Optimización:** Inclusión de una capa de **Dropout (0.2)** para prevenir el sobreajuste y mejorar la generalización.
*   **Dataset:** Generación sintética de imágenes en escala de grises mediante código.

## 🛠️ Requisitos e Instalación
Este proyecto fue desarrollado y probado en **Python 3.11** utilizando **Visual Studio Code**.

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/nombre-del-repo.git](https://github.com/tu-usuario/nombre-del-repo.git)
    ```
2.  **Instalar las dependencias necesarias:**
    ```bash
    pip install tensorflow numpy matplotlib
    ```

## 💻 Uso del Prototipo
Para ejecutar el entrenamiento y ver la predicción en tiempo real, abre una terminal en la carpeta del proyecto y ejecuta:
```bash
python Proyecto_Semana_08.py
