# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

#IMPORTAÇÕES DAS BIBLIOTECAS

from tkinter import Button, Tk, Label #usada pra interfces
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
#redes neurais
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #usada para construção da rede, camadas
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class Interface():
    
    def Captura(self): #abre uma janela do SO que retorna o path de um determinado arquivo
        self.filename = askopenfilename()
        self.image = Image.open(self.filename) #abrir imagem selecionada pelo path
        self.photo = ImageTk.PhotoImage(self.image)
        #criando label do formato que ocupa 3 linhas que recebe a foto escolhida self.photo..
        Label(self.root, image=self.photo).grid(row=1, column = 0, padx = 15, pady = 5, rowspan = 3)
    
    def Treinamento(self):
        self.rede = Sequential()
        #operador de convoluçao, numero de kernels, tamanho do kernel 3x3, conversao da imagem 64 altura 64 largura e 3 canais RGB + funcao de ativacao relu
        self.rede.add(Conv2D(64, (3,3), 
                      input_shape=(64,64,3),
                      activation='relu'))
        
        self.rede.add(BatchNormalization()) #diminuir a quantidade de processamento q seria necessario e aumentar velocidade do trienamento
        #depois da multiplicaçõa da imagem original pelo kernel, resulta em outra matriz, fazendo a etapa de MaxPooling
        self.rede.add(MaxPooling2D(pool_size=(2,2)))
        ##etapa de Flatten = transformar tudo isso em um vetor para servir de entrada para as camadas ocultas
        self.rede.add(Flatten())
        
        #camadas ocultas
        self.rede.add(Dense(units=100, activation='relu'))
        self.rede.add(Dense(units=100, activation='relu'))
        #camada de saida
        self.rede.add(Dense(units=1, activation='sigmoid')) #so tem 2 classes, 0 ou 1. Faz uma regressao logistica e retorna a probabilidade de ser 0(Deserto) ou 1(Floresta)
        self.rede.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        #processo de Data Aumentation simples, ter maior variedade de imagens com operações
        #para cada imagem vai fazer varias operações com ela: rotação, zoom, etc..
        gerador_treinamento = ImageDataGenerator(rescale=1. / 255, 
                                                 rotation_range = 7, 
                                                 horizontal_flip=True, 
                                                 shear_range=0.2, 
                                                 height_shift_range = 0.07, 
                                                 zoom_range = 0.2)
        
        gerador_teste = ImageDataGenerator(rescale = 1./ 255)
        
        #selecionar pasta com fotos para treinamento
        base_treinamento = gerador_treinamento.flow_from_directory(r'C:\Users\preda\Treinamento', target_size=(64,64), batch_size = 32, class_mode ='binary')
        base_teste = gerador_teste.flow_from_directory(r'C:\Users\preda\Teste', target_size=(64,64), batch_size = 32, class_mode ='binary')
        
        #utilizar a funcao fit para treinar nossa rede
        self.rede.fit_generator(base_treinamento, 
                                steps_per_epoch = 180, #imagens utilizadas em cada epoch
                                epochs=7, 
                                validation_data = base_teste, #já vai treinando e validando ao mesmo tempo
                                validation_steps=180)
        
    def ClassificarImagens(self):
        
        imagem_teste = image.load_img(self.filename, target_size=(64,64))
        
        imagem_teste = image.img_to_array(imagem_teste)
        
        imagem_teste = imagem_teste / 255  #divide tudo por 255 para ter valores entre 0 e 1
        
        imagem_teste = np.expand_dims(imagem_teste, axis=0) #criando nova coluna
        
        previsao = self.rede.predict(imagem_teste) #aqui pela funcao de ativacao da ultima camada, sigmoid, vai retornar uma probabilidade da IMAGEM ESCOLHIDA ser de uma classe ou outra
        
        #checar retorno para classificacao
        if previsao > 0.5:
            print('A imagem é de uma Floresta')
            Label(self.root, text='A Imagem é de uma Floresta').grid(row=1, column=0)
            
        elif previsao < 0.5:
            print('A imagem é de um Deserto')
            Label(self.root, text='A Imagem é de um Deserto').grid(row=1, column=0)
            
    
    def __init__(self): 
        self.root = Tk()
        self.root.title('Classificador de Imagens')
        
        Button(self.root, text = 'Seleciona a Imagem', command = self.Captura).grid(row=0, column=0, pady=5) #botão de seleção de imagem no sistema. Ao clicar no botao, chama o comando Capturar para selecionar imagem
        
        Button(self.root, text = 'Treinar a Rede', command = self.Treinamento, width=10, height=2).grid(row=0, column=1)
        
        Button(self.root, text = 'Classificar', command = self.ClassificarImagens, width=10, height=2).grid(row=1, column=1)
        self.root.mainloop()
        
Interface()