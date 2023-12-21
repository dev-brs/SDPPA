import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
from keras.models import load_model
import os
from screeninfo import get_monitors
import cv2
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the Keras model
model = load_model("modelo_atualizado_v2.h5", compile=False)

# Load the labels
class_names = ['Imagem sem plástico', 'Imagem com plástico']

# Function to open and classify an image
def open_and_classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Imagens", ".png .jpg .gif")])
    if file_path:
        imagem = cv2.imread(file_path)
        imagem_nome = os.path.basename(file_path)  # Obtém o nome da imagem

        altura_imagem, largura_imagem = imagem.shape[:2]

        # Pré-processamento da imagem
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (11, 11), 0)

        # Aplicação de um filtro de detecção de bordas (Sobel)
        bordas = cv2.Canny(imagem_suavizada, 50, 150)

        # Aplicar uma operação de dilatação para melhorar a conectividade dos contornos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bordas = cv2.dilate(bordas, kernel, iterations=5)

        # Segmentação de objetos
        contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhar caixas delimitadoras ao redor dos objetos e salvar as imagens
        imagem_com_caixas = imagem.copy()
        margem_de_erro = 30  # Defina a margem de erro desejada
        count = 0
        num_total_caixas=0

        caixas_com_plastico = []

        # Função para processar caixas em paralelo
        def processar_caixas_paralelo(caixas):
            nonlocal count
            nonlocal num_total_caixas
            for caixa in caixas:
                x, y, largura, altura = caixa

                # Faça o processamento da caixa aqui
                caixa_recortada = imagem[y:y + altura, x:x + largura]

                # Verificar se a imagem recortada não é vazia antes de redimensionar
                if caixa_recortada.shape[0] > 0 and caixa_recortada.shape[1] > 0 and (
                        altura > largura * 0.2 and largura > altura * 0.2 and altura * largura > altura_imagem * largura_imagem * 0.001):
                    num_total_caixas += 1
                    # Redimensionar a caixa para 224x224 pixels
                    caixa_redimensionada = cv2.resize(caixa_recortada, (224, 224))
                    # Resize the image to fit the model input shape
                    image_array = np.asarray(caixa_redimensionada)
                    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

                    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                    data[0] = normalized_image_array
                    prediction = model.predict(data)
                    index = np.argmax(prediction)
                    class_name = class_names[index].strip()
                    confidence_score = prediction[0][index]

                    if index == 1:
                        # Verificar a interseção com caixas existentes
                        caixa_atual = (x, y, x + largura, y + altura)
                        unir_caixas = False

                        for i, caixa_existente in enumerate(caixas_com_plastico):
                            if intersecta(caixa_atual, caixa_existente):
                                x1, y1, x2, y2 = caixa_existente
                                x1 = min(x1, x)
                                y1 = min(y1, y)
                                x2 = max(x2, x + largura)
                                y2 = max(y2, y + altura)
                                caixas_com_plastico[i] = (x1, y1, x2, y2)
                                unir_caixas = True
                                break

                        if not unir_caixas:
                            caixas_com_plastico.append(caixa_atual)

                        count += 1

        # Processar os contornos para obter as coordenadas x, y, largura e altura
        caixas_formatadas = [cv2.boundingRect(contorno) for contorno in contornos]
        # Dividir as caixas em partes para processamento em paralelo
        num_threads = 16  # Especifique o número de threads desejado
        caixas_divididas = divide_caixas(caixas_formatadas, num_threads)

        # Criar e iniciar as threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=processar_caixas_paralelo, args=(caixas_divididas[i],))
            threads.append(thread)
            thread.start()

        # Aguardar até que todas as threads terminem
        for thread in threads:
            thread.join()

        # Desenhar as caixas agrupadas com plástico na imagem original
        for caixa in caixas_com_plastico:
            x1, y1, x2, y2 = caixa
            cv2.rectangle(imagem_com_caixas, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Update the image display
        print(count, 'caixas com plástico encontradas de', num_total_caixas, "caixas")
        cv2.imwrite("imagem_com_caixas.jpg", imagem_com_caixas)
        display_image(imagem_com_caixas)

# Função para exibir uma imagem usando o Tkinter
def display_image(image):
    new_width = int(screen_width / 2)
    new_height = int(screen_height / 1.5)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    image = image.resize((new_width, new_height), Image.BILINEAR)
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo

# Função para verificar a interseção de caixas
def intersecta(caixa1, caixa2):
    x1a, y1a, x2a, y2a = caixa1
    x1b, y1b, x2b, y2b = caixa2

    # Verifica a interseção das caixas
    return not (x2a < x1b or x1a > x2b or y2a < y1b or y1a > y2b)

# Função para dividir caixas em partes para processamento em paralelo
def divide_caixas(caixas, num_threads):
    caixas_divididas = [[] for _ in range(num_threads)]
    for i, caixa in enumerate(caixas):
        caixas_divididas[i % num_threads].append(caixa)
    return caixas_divididas

monitors = get_monitors()
if monitors:
    primary_monitor = monitors[0]
    screen_width = primary_monitor.width
    screen_height = primary_monitor.height
else:
    screen_width = 800
    screen_height = 600

root = tk.Tk()
root.title("Classificador")

root.geometry(f"{screen_width}x{screen_height}")
root.state("zoomed")

style = ttk.Style()
root.configure(bg="#262626")
style.theme_use("clam")

label = tk.Label(root, bg="#262626")
label.pack()

result_label = tk.Label(root, bg="#262626", fg="white", font=("Helvetica", 12), justify="left")
result_label.pack()

button_frame = tk.Frame(root, bg="#262626")
button_frame.pack(fill="both", expand=True, padx=10, pady=10, anchor="center")

button_select_image = tk.Button(button_frame, text="Selecione a Imagem", command=open_and_classify_image, bg="#a31018", fg="white",
                                font=("Helvetica", 12), width=20, height=3)
button_select_image.pack(side="bottom", pady=50)

root.mainloop()
