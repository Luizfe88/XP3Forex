<<<<<<< HEAD
import os
import json
from google import genai
from google.genai import types

# --- GESTOR DE TAREFAS (TASK MANAGER) ---
def salvar_tarefa(descricao: str, prioridade: str = "Média"):
    """
    Adiciona uma nova tarefa ao arquivo tasks.json.
    Args:
        descricao: O que deve ser feito.
        prioridade: Alta, Média ou Baixa.
    """
    file_path = 'tasks.json'
    tarefas = []
    
    # Cria o arquivo se não existir ou lê se existir
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tarefas = json.load(f)
        except json.JSONDecodeError:
            tarefas = [] # Se o arquivo estiver corrompido, inicia vazio
    
    nova_tarefa = {
        "id": len(tarefas) + 1, 
        "tarefa": descricao, 
        "prioridade": prioridade, 
        "status": "pendente"
    }
    tarefas.append(nova_tarefa)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(tarefas, f, indent=4, ensure_ascii=False)
    
    return f"SUCESSO: Tarefa salva no arquivo tasks.json com ID {nova_tarefa['id']}."

# --- CONFIGURAÇÃO DO CLIENTE ---
# Substitua pela sua API KEY
client = genai.Client(api_key="AIzaSyA_fWKKRoxWSLdsaVLMTm75VriDMH_HfMc")

def ler_arquivos_locais():
    contexto = ""
    # Pastas que o script deve IGNORAR para não gastar tokens
    ignorar = ['venv', '.git', '__pycache__', 'build', 'dist']
    
    for raiz, diretorios, arquivos in os.walk('.'):
        # Remove pastas ignoradas da busca
        diretorios[:] = [d for d in diretorios if d not in ignorar]
        
        for arquivo in arquivos:
            if arquivo.endswith('.py') and arquivo != 'gemini_assistant.py':
                caminho_completo = os.path.join(raiz, arquivo)
                try:
                    with open(caminho_completo, 'r', encoding='utf-8') as f:
                        conteudo = f.read()
                        # Só adiciona se o arquivo não for gigantesco
                        if len(conteudo) < 50000: 
                            contexto += f"\n--- ARQUIVO: {arquivo} ---\n{conteudo}\n"
                except Exception:
                    continue
    return contexto

# --- LOOP DO CLI ---
def chat():
    print("🤖 Assistente Xp3 Forex Online (SDK google-genai v1.0)")
    print("Digite 'sair' para encerrar.\n")
    
    # 1. Configuração correta para o novo SDK
    # Removemos o parâmetro inválido e passamos apenas as tools
    config = types.GenerateContentConfig(
        system_instruction="Você é um Analista Quantitativo Sênior e Engenheiro de Software. Seu foco é o projeto Xp3Forex. Analise o código e use a ferramenta 'salvar_tarefa' para registrar pendências.",
        tools=[salvar_tarefa], 
        temperature=0.5 # Mais preciso para código
    )

    # 2. Iniciamos o chat (que gerencia o histórico automaticamente)
    chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=config
    )

    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ['sair', 'exit']: break
        
        # Lê o código atualizado a cada interação
        codigo = ler_arquivos_locais()
        mensagem_completa = f"CONTEXTO DO PROJETO ATUAL:\n{codigo}\n\nMINHA SOLICITAÇÃO: {pergunta}"
        
        try:
            # O parâmetro message substitui o contents no modo chat
            response = chat_session.send_message(message=mensagem_completa)
            
            # Verifica se houve execução de ferramenta
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                         print(f"⚡ Executando ferramenta: {part.function_call.name}...")
            
            print(f"\nGemini: {response.text}")
            
        except Exception as e:
            print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
=======
import os
import json
from google import genai
from google.genai import types

# --- GESTOR DE TAREFAS (TASK MANAGER) ---
def salvar_tarefa(descricao: str, prioridade: str = "Média"):
    """
    Adiciona uma nova tarefa ao arquivo tasks.json.
    Args:
        descricao: O que deve ser feito.
        prioridade: Alta, Média ou Baixa.
    """
    file_path = 'tasks.json'
    tarefas = []
    
    # Cria o arquivo se não existir ou lê se existir
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tarefas = json.load(f)
        except json.JSONDecodeError:
            tarefas = [] # Se o arquivo estiver corrompido, inicia vazio
    
    nova_tarefa = {
        "id": len(tarefas) + 1, 
        "tarefa": descricao, 
        "prioridade": prioridade, 
        "status": "pendente"
    }
    tarefas.append(nova_tarefa)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(tarefas, f, indent=4, ensure_ascii=False)
    
    return f"SUCESSO: Tarefa salva no arquivo tasks.json com ID {nova_tarefa['id']}."

# --- CONFIGURAÇÃO DO CLIENTE ---
# Substitua pela sua API KEY
client = genai.Client(api_key="AIzaSyA_fWKKRoxWSLdsaVLMTm75VriDMH_HfMc")

def ler_arquivos_locais():
    contexto = ""
    # Pastas que o script deve IGNORAR para não gastar tokens
    ignorar = ['venv', '.git', '__pycache__', 'build', 'dist']
    
    for raiz, diretorios, arquivos in os.walk('.'):
        # Remove pastas ignoradas da busca
        diretorios[:] = [d for d in diretorios if d not in ignorar]
        
        for arquivo in arquivos:
            if arquivo.endswith('.py') and arquivo != 'gemini_assistant.py':
                caminho_completo = os.path.join(raiz, arquivo)
                try:
                    with open(caminho_completo, 'r', encoding='utf-8') as f:
                        conteudo = f.read()
                        # Só adiciona se o arquivo não for gigantesco
                        if len(conteudo) < 50000: 
                            contexto += f"\n--- ARQUIVO: {arquivo} ---\n{conteudo}\n"
                except Exception:
                    continue
    return contexto

# --- LOOP DO CLI ---
def chat():
    print("🤖 Assistente Xp3 Forex Online (SDK google-genai v1.0)")
    print("Digite 'sair' para encerrar.\n")
    
    # 1. Configuração correta para o novo SDK
    # Removemos o parâmetro inválido e passamos apenas as tools
    config = types.GenerateContentConfig(
        system_instruction="Você é um Analista Quantitativo Sênior e Engenheiro de Software. Seu foco é o projeto Xp3Forex. Analise o código e use a ferramenta 'salvar_tarefa' para registrar pendências.",
        tools=[salvar_tarefa], 
        temperature=0.5 # Mais preciso para código
    )

    # 2. Iniciamos o chat (que gerencia o histórico automaticamente)
    chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=config
    )

    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ['sair', 'exit']: break
        
        # Lê o código atualizado a cada interação
        codigo = ler_arquivos_locais()
        mensagem_completa = f"CONTEXTO DO PROJETO ATUAL:\n{codigo}\n\nMINHA SOLICITAÇÃO: {pergunta}"
        
        try:
            # O parâmetro message substitui o contents no modo chat
            response = chat_session.send_message(message=mensagem_completa)
            
            # Verifica se houve execução de ferramenta
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                         print(f"⚡ Executando ferramenta: {part.function_call.name}...")
            
            print(f"\nGemini: {response.text}")
            
        except Exception as e:
            print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
    chat()