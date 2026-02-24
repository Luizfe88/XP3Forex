<<<<<<< HEAD
from google import genai

client = genai.Client(api_key="AIzaSyA_fWKKRoxWSLdsaVLMTm75VriDMH_HfMc")

print("--- LISTA DE MODELOS DISPONÍVEIS ---")
try:
    for model in client.models.list():
        # Vamos imprimir apenas o nome para evitar erros de atributos
        print(f"Nome: {model.name}")
except Exception as e:
=======
from google import genai

client = genai.Client(api_key="AIzaSyA_fWKKRoxWSLdsaVLMTm75VriDMH_HfMc")

print("--- LISTA DE MODELOS DISPONÍVEIS ---")
try:
    for model in client.models.list():
        # Vamos imprimir apenas o nome para evitar erros de atributos
        print(f"Nome: {model.name}")
except Exception as e:
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
    print(f"Erro crítico: {e}")