import random   #biblioteca pra sacar numeros random
import numpy as np #biblioteca pra matrizes e cenas


#variaveis e merdas...
#não é necessário declarar o tipo de variável (int, float, etc.) e as variaveis podem até mudar de tipo depois de serem atribuidas 
#no entanto podem ser atribuidas usando os comandos ("str()", "int()" e "float()")
a =10
b =5.2
c=5j #numeros complexos escrevem-se com "j" na parte imaginária
texto = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.""" # """ para meter texto mais longo e/ou com espaços numa variavel (string)
var ="Cenas"        #'texto pra ficar como String' 
#strings podem ser declaradas tanto com '' ou com ""
#da pra checkar o tipo de variavel com o comando "type()"
bol = True          #"Booleano"
bolf = False
#pra avaliar uma variavel (true / false) usa-se "bool()" ex: print(bool(a)) = class "int"
"""
comentarios
mais
longos
"""

print("sup")
print(type(a))
print(bool(a))
print(bool(bolf))
print(random.randrange(1, 10)) #gera (e da print) num numero random dentro do range especificado
print(var)
print(texto)
print(texto[4]) # da print no caracter nº4 (quinto) da string
print(texto[:5]) #"slice" da print até ao caracter 5
print(texto[100:])#da print apartir do caracter 100
print(texto[-5:-2])#da print apartir de caracter a começar do final até ao 2º a contar do fim (nao incluido)
print(var.upper()) #da print em maiusculas
print(var.lower()) #da print em minusculas
print(var.replace("C", "J")) #da replace no caracter especificado
#fininho
for l in "banana":
  print(l)
print(len(texto)) #da print no tamanho (lenght) de uma string
txt = "The best things in life are free!" 
if "free" in txt:
  print("Yes, 'free' is present.")
else: print("No, 'free' is not present.") #se escrever freeee ele vai continuar a dizer que está presente...
print("expensive" not in txt) #faz o mesmo mas sem if's (mas este só dá print a true ou false conforme)
