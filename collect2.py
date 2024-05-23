import os
import cv2

def capture_images(data_dir, class_name, dataset_size=100):
    # Crear directorio para la clase si no existe
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Recolectando datos para la clase {class_name}')

    # Captura de video
    cap = cv2.VideoCapture(0)

    # Esperar hasta que se presione 'q' para comenzar la recolección de datos
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mostrar texto en la pantalla
        cv2.putText(frame, 'Presiona q para iniciar', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Dibujar un rectángulo guía para asegurar que ambas manos estén en la vista
        height, width, _ = frame.shape
        cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (255, 0, 0), 2)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capturar imágenes
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mostrar el frame y dibujar el mismo rectángulo guía
        cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        
        # Guardar la imagen
        cv2.imwrite(os.path.join(class_dir, f'{class_name}_{counter}.jpg'), frame)
        
        counter += 1
        cv2.waitKey(25)

    # Liberar la captura y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_dir = input("Ingrese el directorio donde desea guardar las imágenes: ")
    class_name = input("Ingrese el nombre de la clase: ")
    dataset_size = int(input("Ingrese el tamaño del conjunto de datos: "))
    
    capture_images(data_dir, class_name, dataset_size)
