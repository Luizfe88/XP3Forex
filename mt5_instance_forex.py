"""
MT5 INSTANCE MANAGER - Detecta e conecta na instância correta do MetaTrader 5
"""
import MetaTrader5 as mt5
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("mt5_manager")


class MT5Instance:
    """Representa uma instância do MT5"""
    def __init__(self, path: str, terminal_path: str):
        self.path = path  # Caminho do executável
        self.terminal_path = terminal_path  # Caminho da pasta terminal64.exe
        self.account_number = None
        self.account_name = None
        self.broker = None
        self.is_demo = None
        self.is_forex = None
    
    def __repr__(self):
        return (f"MT5Instance(account={self.account_number}, "
                f"broker={self.broker}, forex={self.is_forex})")


def find_mt5_instances() -> List[MT5Instance]:
    """
    Encontra todas as instâncias do MT5 instaladas no PC
    """
    instances = []
    
    # Locais comuns de instalação
    possible_paths = [
        Path("C:/Program Files/MetaTrader 5"),
        Path("C:/Program Files (x86)/MetaTrader 5"),
        Path(os.path.expanduser("~/AppData/Roaming/MetaQuotes/Terminal")),
        Path("D:/Program Files/MetaTrader 5"),
        Path("C:/MT5"),
        Path("D:/MT5"),
    ]
    
    # Procura por múltiplas instâncias
    for base_path in possible_paths:
        if not base_path.exists():
            continue
        
        # Procura terminal64.exe
        for root, dirs, files in os.walk(str(base_path)):
            if "terminal64.exe" in files:
                terminal_path = Path(root) / "terminal64.exe"
                
                instance = MT5Instance(
                    path=str(terminal_path),
                    terminal_path=str(root)
                )
                instances.append(instance)
                logger.info(f"🔍 MT5 encontrado: {terminal_path}")
    
    return instances


def get_instance_info(instance: MT5Instance) -> bool:
    """
    Conecta temporariamente na instância para obter informações
    """
    try:
        # Inicializa com caminho específico
        if not mt5.initialize(path=instance.path):
            logger.warning(f"⚠️ Falha ao conectar: {instance.path}")
            return False
        
        # Obtém informações da conta
        account = mt5.account_info()
        if not account:
            mt5.shutdown()
            return False
        
        instance.account_number = account.login
        instance.account_name = account.name
        instance.broker = account.company
        instance.is_demo = "demo" in account.server.lower()
        
        instance.is_forex = True
        
        logger.info(
            f"✅ {instance.broker} | Conta: {instance.account_number} | "
            f"FOREX | "
            f"{'DEMO' if instance.is_demo else 'REAL'}"
        )
        
        mt5.shutdown()
        return True
    
    except Exception as e:
        logger.error(f"❌ Erro ao obter info: {e}")
        mt5.shutdown()
        return False


def select_instance_interactive(instances: List[MT5Instance]) -> Optional[MT5Instance]:
    """
    Permite o usuário selecionar a instância correta
    """
    print("\n" + "="*80)
    print("🔍 MÚLTIPLAS INSTÂNCIAS MT5 DETECTADAS")
    print("="*80)
    
    valid_instances = []
    
    for i, inst in enumerate(instances, 1):
        if inst.account_number:
            valid_instances.append(inst)
            
            market_type = "FOREX"
            account_type = "DEMO" if inst.is_demo else "REAL"
            
            print(f"\n[{i}] {inst.broker}")
            print(f"    Conta: {inst.account_number} ({inst.account_name})")
            print(f"    Tipo: {market_type} | {account_type}")
            print(f"    Caminho: {inst.terminal_path}")
    
    if not valid_instances:
        print("\n❌ Nenhuma instância válida encontrada!")
        return None
    
    print("\n" + "="*80)
    
    while True:
        try:
            choice = input(f"\nSelecione a instância [1-{len(valid_instances)}]: ")
            idx = int(choice) - 1
            
            if 0 <= idx < len(valid_instances):
                selected = valid_instances[idx]
                
                # Confirmação
                market = "FOREX"
                confirm = input(
                    f"\n✅ Confirma instância {market} "
                    f"(Conta {selected.account_number})? (S/N): "
                )
                
                if confirm.upper() == "S":
                    return selected
                else:
                    print("❌ Seleção cancelada. Tente novamente.")
            else:
                print("❌ Opção inválida!")
        
        except ValueError:
            print("❌ Digite um número válido!")
        except KeyboardInterrupt:
            print("\n\n❌ Processo cancelado pelo usuário")
            return None


def select_instance_by_market(
    instances: List[MT5Instance], 
    prefer_forex: bool = True
) -> Optional[MT5Instance]:
    """
    Seleciona automaticamente a instância correta baseado no mercado
    
    Args:
        instances: Lista de instâncias encontradas
        prefer_forex: True = prioriza Forex
    
    Returns:
        Instância selecionada ou None
    """
    # Filtra por tipo de mercado
    target_instances = [
        inst for inst in instances 
        if inst.account_number and inst.is_forex == prefer_forex
    ]
    
    if not target_instances:
        logger.warning("⚠️ Nenhuma instância Forex encontrada!")
        return None
    
    # Prioriza conta real sobre demo
    real_accounts = [inst for inst in target_instances if not inst.is_demo]
    
    if real_accounts:
        selected = real_accounts[0]
        logger.info(
            f"✅ Instância selecionada automaticamente: "
            f"{selected.broker} | Conta {selected.account_number} (REAL)"
        )
        return selected
    
    # Se só tiver demo, usa a primeira
    selected = target_instances[0]
    logger.info(
        f"✅ Instância selecionada (DEMO): "
        f"{selected.broker} | Conta {selected.account_number}"
    )
    return selected


def save_instance_preference(instance: MT5Instance):
    """
    Salva a preferência do usuário para não perguntar novamente
    """
    try:
        with open("mt5_instance.cfg", "w") as f:
            f.write(f"path={instance.path}\n")
            f.write(f"account={instance.account_number}\n")
            f.write(f"broker={instance.broker}\n")
        f.write(f"forex=True\n")
        
        logger.info("✅ Preferência salva em mt5_instance.cfg")
    except Exception as e:
        logger.warning(f"⚠️ Não foi possível salvar preferência: {e}")


def load_instance_preference() -> Optional[Dict[str, str]]:
    """
    Carrega a preferência salva anteriormente
    """
    try:
        if not os.path.exists("mt5_instance.cfg"):
            return None
        
        config = {}
        with open("mt5_instance.cfg", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key] = value
        
        logger.info(f"📋 Preferência carregada: {config.get('broker')}")
        return config
    
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar preferência: {e}")
        return None


def initialize_mt5_smart(prefer_forex: bool = True, force_select: bool = False) -> bool:
    """
    Inicializa MT5 de forma inteligente:
    1. Tenta usar preferência salva
    2. Se não existir ou force_select=True, procura instâncias
    3. Se múltiplas, permite seleção manual ou automática
    
    Args:
        prefer_forex: Se True, prioriza conta Forex
        force_select: Se True, ignora preferência e força seleção
    
    Returns:
        True se conectou com sucesso
    """
    print("\n" + "="*80)
    print("🚀 INICIALIZANDO METATRADER 5")
    print("="*80)
    
    # 1. Tenta usar preferência salva
    if not force_select:
        preference = load_instance_preference()
        
        if preference:
            path = preference.get("path")
            
            print(f"\n📋 Usando instância salva:")
            print(f"   Broker: {preference.get('broker')}")
            print(f"   Conta: {preference.get('account')}")
        print(f"   Tipo: FOREX")
            
            if mt5.initialize(path=path):
                print("✅ Conectado com sucesso!")
                return True
            else:
                print("⚠️ Falha ao conectar na instância salva. Procurando alternativas...")
    
    # 2. Procura todas as instâncias
    print("\n🔍 Procurando instâncias do MT5...")
    instances = find_mt5_instances()
    
    if not instances:
        print("❌ Nenhuma instância do MT5 encontrada!")
        print("\nVerifique se o MetaTrader 5 está instalado corretamente.")
        return False
    
    print(f"✅ {len(instances)} instância(s) encontrada(s)")
    
    # 3. Obtém informações de cada instância
    print("\n📊 Obtendo informações das contas...")
    valid_instances = []
    
    for inst in instances:
        if get_instance_info(inst):
            valid_instances.append(inst)
    
    if not valid_instances:
        print("❌ Nenhuma instância válida (com conta ativa) encontrada!")
        return False
    
    # 4. Seleção
    if len(valid_instances) == 1:
        # Só uma instância, usa ela
        selected = valid_instances[0]
        market = "FOREX"
        
        print(f"\n✅ Única instância encontrada:")
        print(f"   {selected.broker} | Conta {selected.account_number}")
        print(f"   Tipo: {market}")
        
        confirm = input("\nUsar esta instância? (S/N): ")
        if confirm.upper() != "S":
            print("❌ Conexão cancelada pelo usuário")
            return False
    
    else:
        # Múltiplas instâncias
        # Tenta seleção automática primeiro
        selected = select_instance_by_market(valid_instances, prefer_forex)
        
        if not selected:
            # Se não achou do tipo desejado, pergunta ao usuário
            print(f"\n⚠️ Nenhuma instância Forex encontrada automaticamente.")
            selected = select_instance_interactive(valid_instances)
        
        else:
            # Achou automaticamente, mas pergunta se está OK
            market = "FOREX"
            
            print(f"\n🎯 Instância {market} detectada automaticamente:")
            print(f"   {selected.broker} | Conta {selected.account_number}")
            
            confirm = input("\nUsar esta instância? (S/N, ou 'L' para listar todas): ")
            
            if confirm.upper() == "L":
                selected = select_instance_interactive(valid_instances)
            elif confirm.upper() != "S":
                print("❌ Conexão cancelada pelo usuário")
                return False
    
    if not selected:
        print("❌ Nenhuma instância selecionada")
        return False
    
    # 5. Conecta na instância selecionada
    print(f"\n🔌 Conectando em {selected.broker}...")
    
    if mt5.initialize(path=selected.path):
        print("✅ Conexão estabelecida com sucesso!")
        
        # Salva preferência
        save_instance_preference(selected)
        
        # Exibe informações finais
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        
        print("\n" + "="*80)
        print("📊 INFORMAÇÕES DA CONTA")
        print("="*80)
        print(f"Broker: {account.company}")
        print(f"Servidor: {account.server}")
        print(f"Conta: {account.login} ({account.name})")
        print(f"Tipo: {'DEMO' if selected.is_demo else 'REAL'}")
        print(f"Mercado: FOREX")
        print(f"Balance: ${account.balance:,.2f}")
        print(f"Equity: ${account.equity:,.2f}")
        print(f"Margem Livre: ${account.margin_free:,.2f}")
        print(f"Alavancagem: 1:{account.leverage}")
        print(f"Trading Permitido: {'SIM' if terminal.trade_allowed else 'NÃO'}")
        print("="*80 + "\n")
        
        return True
    
    else:
        error = mt5.last_error()
        print(f"❌ Falha na conexão: {error}")
        return False


# ============================================
# FUNÇÕES AUXILIARES PARA O BOT
# ============================================

def ensure_mt5_connection_smart(prefer_forex: bool = True) -> bool:
    """
    Garante conexão ativa, reconecta se necessário
    """
    terminal_info = None
    try:
        terminal_info = mt5.terminal_info()
    except:
        pass
    
    # Se já está conectado e operável, retorna True
    if terminal_info and terminal_info.connected:
        return True
    
    # Senão, tenta inicializar
    logger.warning("🔄 Reconectando ao MT5...")
    return initialize_mt5_smart(prefer_forex=prefer_forex, force_select=False)


def get_current_market_type() -> str:
    try:
        return "FOREX"
    except:
        return "FOREX"


# ============================================
# EXEMPLO DE USO NO MAIN() DO BOT
# ============================================

def example_usage():
    """
    Exemplo de como usar no bot.py main()
    """
    
    # Modo 1: Seleção automática (prioriza Forex)
    if not initialize_mt5_smart(prefer_forex=True):
        print("❌ Falha ao conectar no MT5")
        return
    
    # Modo 2: Forçar seleção manual
    # if not initialize_mt5_smart(prefer_forex=True, force_select=True):
    #     return
    
    # Modo 3: Verificar tipo de mercado conectado
    market_type = get_current_market_type()
    print(f"📊 Mercado conectado: {market_type}")
    
    # Agora pode usar mt5 normalmente
    account = mt5.account_info()
    print(f"Balance: ${account.balance:,.2f}")


if __name__ == "__main__":
    # Teste standalone
    example_usage()
