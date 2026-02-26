from google import genai

client = genai.Client(api_key="AIzaSyA_fWKKRoxWSLdsaVLMTm75VriDMH_HfMc")

print("--- LISTA DE MODELOS DISPONÍVEIS ---")
try:
    for model in client.models.list():
        # Vamos imprimir apenas o nome para evitar erros de atributos
        print(f"Nome: {model.name}")
except Exception as e:
    print(f"Erro crítico: {e}")