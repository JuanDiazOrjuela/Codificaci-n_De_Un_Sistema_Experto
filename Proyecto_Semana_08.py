import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1. DEFINICIÓN DEL PROBLEMA Y GENERACIÓN DE DATOS
def generar_figuras(cantidad=400):
    imagenes = []
    etiquetas = []
    for _ in range(cantidad):
        img = np.zeros((28, 28))
        tipo = np.random.randint(0, 3) # 0: Cuadrado, 1: Triángulo, 2: Círculo
        
        if tipo == 0: # Cuadrado
            img[5:22, 5:22] = 1.0
        elif tipo == 1: # Triángulo
            for i in range(5, 23):
                img[i, 14-(i-5)//2 : 14+(i-5)//2] = 1.0
        elif tipo == 2: # Círculo
            y, x = np.ogrid[:28, :28]
            mascara = (x - 14)**2 + (y - 14)**2 <= 8**2
            img[mascara] = 1.0
            
        imagenes.append(img)
        etiquetas.append(tipo)
    return np.array(imagenes), np.array(etiquetas)

# Dividir en entrenamiento y prueba
x_train, y_train = generar_figuras(1200)
x_test, y_test = generar_figuras(300)

# 2. DISEÑO DE LA RED NEURONAL (Arquitectura solicitada)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # Entrada: 784 neuronas
    layers.Dense(128, activation='relu'),          # Capa oculta 1 con ReLU
    layers.Dropout(0.2),                           # Optimización: Regularización Dropout
    layers.Dense(64, activation='relu'),           # Capa oculta 2
    layers.Dense(3, activation='softmax')          # Salida: 3 neuronas (clases)
])

# 3. COMPILACIÓN Y ENTRENAMIENTO
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("Entrenando la red neuronal...")
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# 4. VALIDACIÓN DE RENDIMIENTO
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nPrecisión final del prototipo: {test_acc*100:.2f}%')

# 5. RESULTADOS VISUALES
def mostrar_prediccion(n):
    pred = model.predict(x_test[n:n+1])
    nombres = ['Cuadrado', 'Triángulo', 'Círculo']
    clase_predicha = nombres[np.argmax(pred)]
    
    plt.figure(figsize=(4,4))
    plt.imshow(x_test[n], cmap='gray')
    plt.title(f"Figura: {nombres[y_test[n]]} | Predicción: {clase_predicha}")
    plt.axis('off')
    plt.show()

# Probar con una imagen aleatoria
mostrar_prediccion(np.random.randint(0, 300))
