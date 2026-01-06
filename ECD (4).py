"""
Sistema ECD - Escrituração Contábil Digital
Análise Completa de Demonstrações Contábeis e Fiscalização Inteligente
Receita Estadual de Santa Catarina
Versão 2.0 - Dashboard Streamlit com Machine Learning
"""

# =============================================================================
# 1. IMPORTS E CONFIGURAÇÕES INICIAIS
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import warnings
import ssl
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Configurações SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings('ignore')

def limpar_dataframe_para_exibicao(df):
    """Remove valores None/NaN de DataFrames antes de formatar."""
    df = df.copy()
    
    # Primeiro: tentar converter colunas 'object' que deveriam ser numéricas
    for col in df.columns:
        if df[col].dtype == 'object':
            # Tentar converter para numérico
            converted = pd.to_numeric(df[col], errors='ignore')
            # Se a conversão funcionou (resultado é numérico), use ela
            if converted.dtype in ['float64', 'int64']:
                df[col] = converted
    
    # Colunas numéricas: substituir None por 0
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Colunas de texto: substituir None por string vazia
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].fillna('')
    
    return df

# =============================================================================
# 2. SENHA DE ACESSO
# =============================================================================

SENHA = "ecd2025"

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("<div style='text-align: center; padding: 50px;'><h1>🔐 Sistema ECD - Acesso Restrito</h1></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Escrituração Contábil Digital")
            st.markdown("**Receita Estadual de Santa Catarina**")
            senha_input = st.text_input("Digite a senha:", type="password", key="pwd_input")
            if st.button("Entrar", use_container_width=True):
                if senha_input == SENHA:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Senha incorreta")
        st.stop()

check_password()

# =============================================================================
# 3. CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="ECD - Análise Contábil Digital",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Forçar sidebar sempre aberta
st.markdown("""
<style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 300px;
        max-width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"]{
        min-width: 300px;
        max-width: 300px;
        margin-left: -300px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 4. ESTILOS CSS CUSTOMIZADOS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563eb;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-red {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-yellow {
        background: linear-gradient(135deg, #ffa751 0%, #ffe259 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-critico {
        background-color: #fee;
        border-left: 5px solid #e53e3e;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-alto {
        background-color: #fff5e6;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-medio {
        background-color: #fffbeb;
        border-left: 5px solid #fbbf24;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-positivo {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .balanço-ativo {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
    }
    .balanço-passivo {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
    }
    .dre-receita {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }
    .dre-despesa {
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
    .stDataFrame {
        font-size: 0.85rem;
    }
    .sidebar-info {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0284c7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 5. FUNÇÕES DE CONEXÃO COM BANCO DE DADOS
# =============================================================================

# Configurações do Impala (ajuste conforme necessário)
IMPALA_HOST = 'bdaworkernode02.sef.sc.gov.br'
IMPALA_PORT = 21050
DATABASE = 'teste'

# Credenciais (use st.secrets em produção)
try:
    IMPALA_USER = st.secrets["impala_credentials"]["user"]
    IMPALA_PASSWORD = st.secrets["impala_credentials"]["password"]
except:
    # Fallback para desenvolvimento local
    IMPALA_USER = "seu_usuario"
    IMPALA_PASSWORD = "sua_senha"

@st.cache_resource
def get_impala_engine():
    """Cria e retorna engine Impala (compartilhado entre sessões)."""
    try:
        engine = create_engine(
            f'impala://{IMPALA_HOST}:{IMPALA_PORT}/{DATABASE}',
            connect_args={
                'user': IMPALA_USER,
                'password': IMPALA_PASSWORD,
                'auth_mechanism': 'LDAP',
                'use_ssl': True
            }
        )
        return engine
    except Exception as e:
        st.error(f"❌ Erro ao criar engine Impala: {e}")
        return None

# =============================================================================
# 6. FUNÇÕES DE CARREGAMENTO DE DADOS (COM CACHE OTIMIZADO)
# =============================================================================

@st.cache_data(ttl=3600)
def carregar_resumo_geral(_engine):
    """Carrega resumo agregado para carregamento inicial rápido."""
    if _engine is None:
        return None
    
    query = f"""
    SELECT 
        COUNT(DISTINCT cnpj) as total_empresas,
        COUNT(DISTINCT ano_referencia) as total_anos,
        MAX(ano_referencia) as ano_mais_recente,
        COUNT(DISTINCT cnae_divisao) as total_setores,
        COUNT(DISTINCT cd_uf) as total_estados
    FROM {DATABASE}.ecd_empresas_cadastro
    WHERE ano_referencia > 0
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Erro ao carregar resumo geral: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_indicadores_agregados(_engine, ano=None):
    """Carrega indicadores financeiros agregados por setor."""
    if _engine is None:
        return None

    ano_filter = f"AND ind.ano_referencia = {ano}" if ano else ""

    query = f"""
    SELECT
        COALESCE(ec.cnae_divisao_descricao, ec.de_cnae, 'Não Classificado') as setor,
        COUNT(DISTINCT ec.cnpj) as qtd_empresas,
        ROUND(AVG(ind.ativo_total) / 1000000, 2) as media_ativo_milhoes,
        ROUND(AVG(ind.receita_liquida) / 1000000, 2) as media_receita_milhoes,
        ROUND(AVG(ind.liquidez_corrente), 2) as media_liquidez,
        ROUND(AVG(ind.endividamento_geral), 2) as media_endividamento,
        ROUND(AVG(ind.margem_liquida_perc), 2) as media_margem_liquida,
        ROUND(AVG(ind.roa_retorno_ativo_perc), 2) as media_roa,
        ROUND(AVG(ind.roe_retorno_patrimonio_perc), 2) as media_roe
    FROM {DATABASE}.ecd_empresas_cadastro ec
    INNER JOIN {DATABASE}.ecd_indicadores_financeiros ind
        ON ec.cnpj = ind.cnpj
        AND CAST(ec.ano_referencia / 100 AS INT) = ind.ano_referencia
    WHERE 1=1
        {ano_filter}
    GROUP BY COALESCE(ec.cnae_divisao_descricao, ec.de_cnae, 'Não Classificado')
    HAVING COUNT(DISTINCT ec.cnpj) > 0
    ORDER BY qtd_empresas DESC
    LIMIT 50
    """

    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar indicadores agregados: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_empresas_por_setor(_engine, setor, ano=None):
    """Carrega lista de empresas de um setor específico."""
    if _engine is None:
        return None

    ano_filter = f"AND ind.ano_referencia = {ano}" if ano else ""

    query = f"""
    SELECT
        ec.cnpj,
        ec.nm_razao_social,
        ec.nm_fantasia,
        ec.cd_uf,
        ec.empresa_grande_porte,
        ROUND(ind.ativo_total / 1000000, 2) as ativo_milhoes,
        ROUND(ind.receita_liquida / 1000000, 2) as receita_milhoes,
        ROUND(ind.liquidez_corrente, 2) as liquidez,
        ROUND(ind.margem_liquida_perc, 2) as margem_liquida,
        sr.score_risco_total,
        sr.classificacao_risco
    FROM {DATABASE}.ecd_empresas_cadastro ec
    INNER JOIN {DATABASE}.ecd_indicadores_financeiros ind
        ON ec.cnpj = ind.cnpj
        AND CAST(ec.ano_referencia / 100 AS INT) = ind.ano_referencia
    LEFT JOIN {DATABASE}.ecd_score_risco_consolidado sr
        ON ec.cnpj = sr.cnpj
        AND CAST(ec.ano_referencia / 100 AS INT) = sr.ano_referencia
    WHERE COALESCE(ec.cnae_divisao_descricao, ec.de_cnae) = '{setor}'
        {ano_filter}
    ORDER BY ind.ativo_total DESC
    LIMIT 200
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar empresas: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_dados_empresa(_engine, cnpj):
    """Carrega dados completos de uma empresa específica (sob demanda)."""
    if _engine is None:
        return {}
    
    dados = {}
    
    # 1. Dados cadastrais
    query_cadastro = f"""
    SELECT *
    FROM {DATABASE}.ecd_empresas_cadastro
    WHERE cnpj = '{cnpj}'
    ORDER BY ano_referencia DESC
    LIMIT 1
    """
    
    # 2. Indicadores financeiros
    query_indicadores = f"""
    SELECT *
    FROM {DATABASE}.ecd_indicadores_financeiros
    WHERE cnpj = '{cnpj}'
    ORDER BY ano_referencia DESC
    """
    
    # 3. Balanço Patrimonial (tabela já agregada)
    query_balanco = f"""
    SELECT *
    FROM {DATABASE}.ecd_balanco_patrimonial
    WHERE cnpj = '{cnpj}'
    ORDER BY ano_referencia DESC, data_fim_periodo DESC
    """
    
    # 4. DRE - Demonstração do Resultado (tabela já agregada)
    query_dre = f"""
    SELECT *
    FROM {DATABASE}.ecd_dre
    WHERE cnpj = '{cnpj}'
    ORDER BY ano_referencia DESC, data_fim_periodo DESC
    """
    
    # 5. Score de risco
    query_risco = f"""
    SELECT *
    FROM {DATABASE}.ecd_score_risco_consolidado
    WHERE cnpj = '{cnpj}'
    ORDER BY ano_referencia DESC
    """
    
    try:
        dados['cadastro'] = pd.read_sql(query_cadastro, _engine)
        dados['indicadores'] = pd.read_sql(query_indicadores, _engine)
        dados['balanco'] = pd.read_sql(query_balanco, _engine)
        dados['dre'] = pd.read_sql(query_dre, _engine)
        dados['risco'] = pd.read_sql(query_risco, _engine)
        return dados
    except Exception as e:
        st.error(f"Erro ao carregar dados da empresa: {e}")
        return {}

@st.cache_data(ttl=3600)
def carregar_empresas_alto_risco(_engine, limite=500):
    """Carrega empresas com alto score de risco para fiscalização."""
    if _engine is None:
        return None
    
    query = f"""
    SELECT 
        ec.cnpj,
        ec.nm_razao_social,
        ec.nm_fantasia,
        ec.cd_uf,
        ec.cnae_divisao_descricao as setor,
        ec.empresa_grande_porte,
        sr.score_risco_total,
        sr.classificacao_risco,
        sr.score_equacao_contabil,
        sr.score_neaf,
        sr.score_risco_financeiro,
        sr.qtd_indicios_neaf,
        ROUND(ind.ativo_total / 1000000, 2) as ativo_milhoes,
        ROUND(ind.receita_liquida / 1000000, 2) as receita_milhoes,
        ROUND(ind.liquidez_corrente, 2) as liquidez,
        ROUND(ind.endividamento_geral, 2) as endividamento,
        ROUND(ind.margem_liquida_perc, 2) as margem_liquida,
        CASE 
            WHEN sr.score_risco_total >= 7 AND ind.ativo_total >= 100000000 THEN 1
            WHEN sr.score_risco_total >= 7 OR ind.ativo_total >= 100000000 THEN 2
            WHEN sr.score_risco_total >= 5 THEN 3
            WHEN sr.score_risco_total >= 3 THEN 4
            ELSE 5
        END as prioridade_fiscalizacao
    FROM {DATABASE}.ecd_score_risco_consolidado sr
    INNER JOIN {DATABASE}.ecd_empresas_cadastro ec
        ON sr.cnpj = ec.cnpj
        AND sr.ano_referencia = CAST(ec.ano_referencia / 100 AS INT)
    INNER JOIN {DATABASE}.ecd_indicadores_financeiros ind
        ON sr.cnpj = ind.cnpj
        AND sr.ano_referencia = ind.ano_referencia
    WHERE sr.score_risco_total >= 3
        AND sr.ano_referencia = (
            SELECT MAX(ano_referencia)
            FROM {DATABASE}.ecd_score_risco_consolidado
        )
    ORDER BY prioridade_fiscalizacao ASC, sr.score_risco_total DESC, ind.ativo_total DESC
    LIMIT {limite}
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar empresas de alto risco: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_plano_contas_agregado(_engine, ano=None):
    """Carrega estatísticas agregadas do plano de contas."""
    if _engine is None:
        return None
    
    # Filtros corretos para os formatos diferentes
    # plano_contas: ano_referencia = 202401 (YYYYMM)
    # saldos: ano_referencia = 2024 (YYYY)
    ano_filter_plano = f"AND pc.ano_referencia BETWEEN {ano}01 AND {ano}12" if ano else ""
    ano_filter_saldos = f"AND ano_referencia = {ano}" if ano else ""  # ⚠️ SEM 'sc.'!
    
    query = f"""
    WITH ultimo_saldo AS (
        SELECT 
            cnpj,
            cd_conta,
            saldo_final_contabil,
            ano_referencia,
            YEAR(data_fim_periodo) * 100 + MONTH(data_fim_periodo) as ano_mes,
            ROW_NUMBER() OVER (PARTITION BY cnpj, cd_conta ORDER BY data_fim_periodo DESC) as rn
        FROM {DATABASE}.ecd_saldos_contas_v2
        WHERE 1=1
            {ano_filter_saldos}
    )
    SELECT 
        pc.cd_conta,
        pc.nm_conta,
        pc.descricao_grupo_balanco,
        pc.cd_conta_referencial,
        pc.tipo_conta,
        pc.nivel_conta,
        pc.cd_conta_sint1,
        pc.nm_conta_sint1,
        
        COUNT(DISTINCT pc.cnpj) AS qtd_empresas_usam,
        ROUND(AVG(ABS(sc.saldo_final_contabil)) / 1000000, 2) AS media_saldo_milhoes,
        ROUND(SUM(ABS(sc.saldo_final_contabil)) / 1000000000, 2) AS total_saldo_bilhoes,
        ROUND(MIN(sc.saldo_final_contabil) / 1000000, 2) AS min_saldo_milhoes,
        ROUND(MAX(sc.saldo_final_contabil) / 1000000, 2) AS max_saldo_milhoes
        
    FROM {DATABASE}.ecd_plano_contas pc
    LEFT JOIN ultimo_saldo sc
        ON pc.cnpj = sc.cnpj
        AND pc.cd_conta = sc.cd_conta
        AND pc.ano_referencia = sc.ano_mes
        AND sc.rn = 1
    WHERE pc.tipo_conta = 'A'
        {ano_filter_plano}
    GROUP BY pc.cd_conta, pc.nm_conta, pc.descricao_grupo_balanco, pc.cd_conta_referencial,
             pc.tipo_conta, pc.nivel_conta, pc.cd_conta_sint1, pc.nm_conta_sint1
    HAVING COUNT(DISTINCT pc.cnpj) >= 5
        AND SUM(ABS(COALESCE(sc.saldo_final_contabil, 0))) > 100000
    ORDER BY qtd_empresas_usam DESC
    LIMIT 200
    """
    
    try:
        df = pd.read_sql(query, _engine)
        
        # Garantir que todas as colunas numéricas sejam do tipo correto
        df['total_saldo_bilhoes'] = pd.to_numeric(df['total_saldo_bilhoes'], errors='coerce').fillna(0)
        df['media_saldo_milhoes'] = pd.to_numeric(df['media_saldo_milhoes'], errors='coerce').fillna(0)
        df['min_saldo_milhoes'] = pd.to_numeric(df['min_saldo_milhoes'], errors='coerce').fillna(0)
        df['max_saldo_milhoes'] = pd.to_numeric(df['max_saldo_milhoes'], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar plano de contas: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_indicios_neaf(_engine, cnpj=None, limite=500):
    """Carrega indícios de NEAF detalhados."""
    if _engine is None:
        return None

    cnpj_filter = f"WHERE cnpj = '{cnpj}'" if cnpj else ""

    query = f"""
    SELECT
        cnpj,
        descricao_indicio,
        complemento_indicio,
        COUNT(*) as qtd_ocorrencias
    FROM {DATABASE}.ecd_neaf_indicios
    {cnpj_filter}
    GROUP BY cnpj, descricao_indicio, complemento_indicio
    ORDER BY qtd_ocorrencias DESC
    LIMIT {limite}
    """

    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar indícios NEAF: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_score_neaf(_engine, limite=500):
    """Carrega scores de risco NEAF."""
    if _engine is None:
        return None

    query = f"""
    SELECT
        ns.cnpj,
        ec.nm_razao_social,
        ec.cnae_divisao_descricao as setor,
        ec.cd_uf,
        ns.qtd_total_indicios,
        ns.qtd_tipos_indicios_distintos,
        ns.score_risco_neaf,
        ns.classificacao_risco_neaf
    FROM {DATABASE}.ecd_neaf_score_risco ns
    INNER JOIN {DATABASE}.ecd_empresas_cadastro ec
        ON ns.cnpj = ec.cnpj
    ORDER BY ns.score_risco_neaf DESC
    LIMIT {limite}
    """

    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar score NEAF: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_inconsistencias_equacao(_engine, ano=None, limite=500):
    """Carrega inconsistências na equação contábil."""
    if _engine is None:
        return None

    # ie.ano_referencia é YYYYMM, então filtramos pelo ano
    ano_filter = f"AND CAST(ie.ano_referencia / 100 AS INT) = {ano}" if ano else ""

    query = f"""
    SELECT
        ie.cnpj,
        ec.nm_razao_social,
        COALESCE(ec.cnae_divisao_descricao, ec.de_cnae, 'Não Classificado') as setor,
        ec.cd_uf,
        ie.ano_referencia,
        ie.data_fim_periodo,
        ie.ativo_total,
        ie.passivo_pl_total,
        ie.diferenca_absoluta,
        ie.percentual_diferenca,
        ie.classificacao_inconsistencia,
        ie.score_risco_equacao
    FROM {DATABASE}.ecd_inconsistencias_equacao ie
    INNER JOIN {DATABASE}.ecd_empresas_cadastro ec
        ON ie.cnpj = ec.cnpj
        AND ie.ano_referencia = ec.ano_referencia
    WHERE ie.classificacao_inconsistencia != 'OK'
        {ano_filter}
    ORDER BY ie.score_risco_equacao DESC, ABS(ie.diferenca_absoluta) DESC
    LIMIT {limite}
    """

    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar inconsistências: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_inconsistencias_variacoes(_engine, ano=None, limite=500):
    """Carrega variações anômalas de contas."""
    if _engine is None:
        return None

    ano_filter = f"AND iv.ano_referencia = {ano}" if ano else ""

    query = f"""
    SELECT
        iv.cnpj,
        ec.nm_razao_social,
        COALESCE(ec.cnae_divisao_descricao, ec.de_cnae, 'Não Classificado') as setor,
        ec.cd_uf,
        iv.cd_conta,
        iv.ano_referencia,
        iv.saldo_anterior,
        iv.saldo_atual,
        iv.variacao_absoluta,
        iv.variacao_percentual,
        iv.classificacao_variacao,
        iv.score_risco_variacao
    FROM {DATABASE}.ecd_inconsistencias_variacoes iv
    INNER JOIN {DATABASE}.ecd_empresas_cadastro ec
        ON iv.cnpj = ec.cnpj
    WHERE iv.classificacao_variacao IN ('Variação Extrema', 'Variação Muito Alta', 'Variação Alta')
        {ano_filter}
    ORDER BY iv.score_risco_variacao DESC, ABS(iv.variacao_percentual) DESC
    LIMIT {limite}
    """

    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar variações: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_benchmark_setorial(_engine, ano=None):
    """Carrega benchmark setorial por CNAE."""
    if _engine is None:
        return None

    ano_filter = f"WHERE ano_referencia = {ano}" if ano else ""

    query = f"""
    SELECT
        cd_cnae,
        de_cnae,
        cnae_secao,
        cnae_secao_descricao,
        cnae_divisao,
        cnae_divisao_descricao,
        ano_referencia,
        qtd_empresas_setor,
        media_ativo_total_setor,
        media_receita_liquida_setor,
        media_resultado_liquido_setor,
        media_liquidez_corrente_setor,
        media_endividamento_setor,
        media_margem_liquida_setor,
        media_roe_setor,
        min_liquidez_setor,
        max_liquidez_setor,
        min_margem_liquida_setor,
        max_margem_liquida_setor
    FROM {DATABASE}.ecd_benchmark_setorial
    {ano_filter}
    ORDER BY qtd_empresas_setor DESC
    """

    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar benchmark setorial: {e}")
        return None

@st.cache_data(ttl=3600)
def carregar_empresas_suspeitas_indicador(_engine, indicador, threshold_min=None, threshold_max=None, ano=None):
    """Carrega empresas suspeitas para um indicador específico."""
    if _engine is None:
        return None
    
    ano_filter = f"AND ind.ano_referencia = {ano}" if ano else ""
    
    # Mapear indicador para coluna e condições
    condicoes = {
        'Liquidez Corrente': ('liquidez_corrente', 'ind.liquidez_corrente < 0.5', 'ASC'),
        'Endividamento Geral': ('endividamento_geral', 'ind.endividamento_geral > 0.8', 'DESC'),
        'Margem Líquida': ('margem_liquida_perc', 'ind.margem_liquida_perc < -10', 'ASC'),
        'ROA': ('roa_retorno_ativo_perc', 'ind.roa_retorno_ativo_perc < -10', 'ASC'),
        'ROE': ('roe_retorno_patrimonio_perc', 'ind.roe_retorno_patrimonio_perc < -20', 'ASC')
    }
    
    if indicador not in condicoes:
        return None
    
    coluna, condicao, ordem = condicoes[indicador]
    
    query = f"""
    SELECT 
        ec.cnpj,
        ec.nm_razao_social,
        ec.cd_uf,
        COALESCE(ec.cnae_divisao_descricao, ec.de_cnae, 'Não Classificado') as setor,
        ROUND(ind.{coluna}, 2) as valor_indicador,
        ROUND(ind.ativo_total / 1000000, 2) as ativo_milhoes,
        ROUND(ind.receita_liquida / 1000000, 2) as receita_milhoes,
        sr.score_risco_total,

        -- Média do setor
        ROUND(AVG(ind.{coluna}) OVER (PARTITION BY COALESCE(ec.cnae_divisao_descricao, ec.de_cnae, 'Não Classificado')), 2) as media_setor,

        -- Desvio da média
        ROUND(ind.{coluna} - AVG(ind.{coluna}) OVER (PARTITION BY COALESCE(ec.cnae_divisao_descricao, ec.de_cnae, 'Não Classificado')), 2) as desvio_setor

    FROM {DATABASE}.ecd_indicadores_financeiros ind
    INNER JOIN {DATABASE}.ecd_empresas_cadastro ec
        ON ind.cnpj = ec.cnpj
        AND ind.ano_referencia = CAST(ec.ano_referencia / 100 AS INT)
    LEFT JOIN {DATABASE}.ecd_score_risco_consolidado sr
        ON ind.cnpj = sr.cnpj
        AND ind.ano_referencia = sr.ano_referencia
    WHERE {condicao}
        {ano_filter}
    ORDER BY valor_indicador {ordem}
    LIMIT 100
    """
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar empresas suspeitas: {e}")
        return None

# =============================================================================
# 7. FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def criar_balanco_patrimonial(dados_balanco, ano):
    """Cria visualização do Balanço Patrimonial."""
    if dados_balanco is None or dados_balanco.empty:
        st.info("Sem dados de Balanço Patrimonial disponíveis.")
        return
    
    # Filtrar por ano se especificado
    if ano:
        # Converter ano para formato YYYYMM (início e fim do ano)
        ano_inicio = int(f"{ano}01")
        ano_fim = int(f"{ano}12")
        df = dados_balanco[
            (dados_balanco['ano_referencia'] >= ano_inicio) & 
            (dados_balanco['ano_referencia'] <= ano_fim)
        ].copy()
    else:
        df = dados_balanco.copy()
    
    if df.empty:
        st.info(f"Sem dados de Balanço para o ano {ano}.")
        return
    
    # Pegar o período mais recente
    df_recente = df.nlargest(1, 'data_fim_periodo')
    
    st.markdown(f"### 📊 Balanço Patrimonial - {df_recente.iloc[0]['data_fim_periodo'].strftime('%m/%Y')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 ATIVO")
        st.markdown(f"""
        <div class='balanço-ativo'>
            <h4>Ativo Total: R$ {df_recente.iloc[0]['ativo_total']:,.2f}</h4>
            <p><strong>Ativo Circulante:</strong> R$ {df_recente.iloc[0]['ativo_circulante']:,.2f}</p>
            <p><strong>Ativo Não Circulante:</strong> R$ {df_recente.iloc[0]['ativo_nao_circulante']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📋 PASSIVO + PL")
        st.markdown(f"""
        <div class='balanço-passivo'>
            <h4>Passivo + PL: R$ {df_recente.iloc[0]['passivo_pl_total']:,.2f}</h4>
            <p><strong>Passivo Circulante:</strong> R$ {df_recente.iloc[0]['passivo_circulante']:,.2f}</p>
            <p><strong>Passivo Não Circulante:</strong> R$ {df_recente.iloc[0]['passivo_nao_circulante']:,.2f}</p>
            <p><strong>Patrimônio Líquido:</strong> R$ {df_recente.iloc[0]['patrimonio_liquido']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de evolução
    st.markdown("---")
    st.markdown("### 📈 Evolução Mensal")
    
    df_sorted = df.sort_values('data_fim_periodo')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_sorted['data_fim_periodo'],
        y=df_sorted['ativo_total'],
        name='Ativo Total',
        line=dict(color='#2196F3', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['data_fim_periodo'],
        y=df_sorted['passivo_total'],
        name='Passivo Total',
        line=dict(color='#FF9800', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['data_fim_periodo'],
        y=df_sorted['patrimonio_liquido'],
        name='Patrimônio Líquido',
        line=dict(color='#4CAF50', width=3)
    ))
    
    fig.update_layout(
        title='Evolução do Balanço Patrimonial',
        xaxis_title='Período',
        yaxis_title='Valor (R$)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela detalhada
    st.markdown("---")
    st.markdown("### 📋 Dados Mensais")
    
    df_display = df[['data_fim_periodo', 'ativo_total', 'ativo_circulante', 
                     'ativo_nao_circulante', 'passivo_total', 'passivo_circulante',
                     'passivo_nao_circulante', 'patrimonio_liquido']].copy()
    
    df_display.columns = ['Período', 'Ativo Total', 'Ativo Circ.', 
                         'Ativo Não Circ.', 'Passivo Total', 'Passivo Circ.',
                         'Passivo Não Circ.', 'Patrimônio Líquido']
    
    df_display = limpar_dataframe_para_exibicao(df_display)
    
    st.dataframe(
        df_display.style.format({
            'Ativo Total': 'R$ {:,.2f}',
            'Ativo Circ.': 'R$ {:,.2f}',
            'Ativo Não Circ.': 'R$ {:,.2f}',
            'Passivo Total': 'R$ {:,.2f}',
            'Passivo Circ.': 'R$ {:,.2f}',
            'Passivo Não Circ.': 'R$ {:,.2f}',
            'Patrimônio Líquido': 'R$ {:,.2f}'
        }),
        use_container_width=True
    )

def criar_dre(dados_dre, ano):
    """Cria visualização da DRE."""
    if dados_dre is None or dados_dre.empty:
        st.info("Sem dados de DRE disponíveis.")
        return
    
    # Filtrar por ano se especificado
    if ano:
        # Converter ano para formato YYYYMM (início e fim do ano)
        ano_inicio = int(f"{ano}01")
        ano_fim = int(f"{ano}12")
        df = dados_dre[
            (dados_dre['ano_referencia'] >= ano_inicio) & 
            (dados_dre['ano_referencia'] <= ano_fim)
        ].copy()
    else:
        df = dados_dre.copy()
    
    if df.empty:
        st.info(f"Sem dados de DRE para o ano {ano}.")
        return
    
    # Pegar o período mais recente
    df_recente = df.nlargest(1, 'data_fim_periodo')
    
    st.markdown(f"### 📊 DRE - Demonstração do Resultado - {df_recente.iloc[0]['data_fim_periodo'].strftime('%m/%Y')}")
    
    st.markdown(f"""
    <div class='dre-receita'>
        <h4>💰 Receita Bruta: R$ {df_recente.iloc[0]['receita_bruta']:,.2f}</h4>
        <p><strong>(-) Deduções:</strong> R$ {df_recente.iloc[0]['deducoes_receita']:,.2f}</p>
        <p><strong>(=) Receita Líquida:</strong> R$ {df_recente.iloc[0]['receita_liquida']:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='dre-despesa'>
        <h4>📉 Custos e Despesas</h4>
        <p><strong>Custos Totais:</strong> R$ {df_recente.iloc[0]['custos_totais']:,.2f}</p>
        <p><strong>Despesas Totais:</strong> R$ {df_recente.iloc[0]['despesas_totais']:,.2f}</p>
        <p><strong>Lucro Bruto:</strong> R$ {df_recente.iloc[0]['lucro_bruto']:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='alert-positivo' style='margin-top: 20px;'>
        <h3>🎯 Resultado Líquido: R$ {df_recente.iloc[0]['resultado_liquido']:,.2f}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico de evolução
    st.markdown("---")
    st.markdown("### 📈 Evolução Mensal")
    
    df_sorted = df.sort_values('data_fim_periodo')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_sorted['data_fim_periodo'],
        y=df_sorted['receita_liquida'],
        name='Receita Líquida',
        line=dict(color='#4CAF50', width=3),
        fill='tonexty'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['data_fim_periodo'],
        y=df_sorted['custos_totais'],
        name='Custos Totais',
        line=dict(color='#FF9800', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['data_fim_periodo'],
        y=df_sorted['despesas_totais'],
        name='Despesas Totais',
        line=dict(color='#F44336', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['data_fim_periodo'],
        y=df_sorted['resultado_liquido'],
        name='Resultado Líquido',
        line=dict(color='#2196F3', width=3)
    ))
    
    fig.update_layout(
        title='Evolução da DRE',
        xaxis_title='Período',
        yaxis_title='Valor (R$)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela detalhada
    st.markdown("---")
    st.markdown("### 📋 Dados Mensais")
    
    df_display = df[['data_fim_periodo', 'receita_bruta', 'deducoes_receita',
                     'receita_liquida', 'custos_totais', 'despesas_totais',
                     'lucro_bruto', 'resultado_liquido']].copy()
    
    df_display.columns = ['Período', 'Receita Bruta', 'Deduções',
                         'Receita Líq.', 'Custos', 'Despesas',
                         'Lucro Bruto', 'Resultado Líq.']
    
    df_display = limpar_dataframe_para_exibicao(df_display)
    
    st.dataframe(
        df_display.style.format({
            'Receita Bruta': 'R$ {:,.2f}',
            'Deduções': 'R$ {:,.2f}',
            'Receita Líq.': 'R$ {:,.2f}',
            'Custos': 'R$ {:,.2f}',
            'Despesas': 'R$ {:,.2f}',
            'Lucro Bruto': 'R$ {:,.2f}',
            'Resultado Líq.': 'R$ {:,.2f}'
        }),
        use_container_width=True
    )

def criar_graficos_indicadores(dados_indicadores):
    """Cria gráficos de evolução dos indicadores financeiros."""
    if dados_indicadores is None or dados_indicadores.empty:
        st.warning("Sem dados de indicadores disponíveis.")
        return
    
    df = dados_indicadores.copy()
    df['ano'] = df['ano_referencia'] // 100
    df = df.sort_values('ano')
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Liquidez Corrente', 'Endividamento Geral', 
                       'Margem Líquida (%)', 'ROE (%)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Liquidez
    fig.add_trace(
        go.Scatter(x=df['ano'], y=df['liquidez_corrente'], 
                  mode='lines+markers', name='Liquidez',
                  line=dict(color='#4CAF50', width=3)),
        row=1, col=1
    )
    
    # Endividamento
    fig.add_trace(
        go.Scatter(x=df['ano'], y=df['endividamento_geral'], 
                  mode='lines+markers', name='Endividamento',
                  line=dict(color='#F44336', width=3)),
        row=1, col=2
    )
    
    # Margem Líquida
    fig.add_trace(
        go.Scatter(x=df['ano'], y=df['margem_liquida_perc'], 
                  mode='lines+markers', name='Margem',
                  line=dict(color='#2196F3', width=3)),
        row=2, col=1
    )
    
    # ROE
    fig.add_trace(
        go.Scatter(x=df['ano'], y=df['roe_retorno_patrimonio_perc'], 
                  mode='lines+markers', name='ROE',
                  line=dict(color='#FF9800', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Evolução dos Indicadores Financeiros")
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 8. FUNÇÕES DE MACHINE LEARNING
# =============================================================================

def treinar_modelo_fiscalizacao(dados_empresas):
    """Treina modelo de ML para identificar empresas para fiscalização."""
    if dados_empresas is None or dados_empresas.empty:
        return None, None
    
    # Selecionar features relevantes
    features = ['score_risco_total', 'score_equacao_contabil', 'score_neaf', 
                'ativo_milhoes', 'receita_milhoes', 'liquidez', 
                'endividamento', 'margem_liquida']
    
    # Criar cópia e remover NaNs
    df_ml = dados_empresas[features].copy()
    df_ml = df_ml.dropna()
    
    if len(df_ml) < 10:
        st.warning("Dados insuficientes para treinar modelo de ML")
        return None, None
    
    # Guardar índices originais antes da normalização
    indices_originais = df_ml.index.tolist()
    
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ml)
    
    # Treinar Isolation Forest para detectar anomalias
    modelo_anomalia = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    anomalias = modelo_anomalia.fit_predict(X_scaled)
    
    # Treinar K-Means para clustering
    modelo_cluster = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = modelo_cluster.fit_predict(X_scaled)
    
    # Adicionar resultados ao DataFrame usando os índices corretos
    df_ml['anomalia'] = anomalias
    df_ml['cluster'] = clusters
    df_ml['indice'] = indices_originais
    
    return df_ml, scaler

def calcular_score_ml(dados_ml, dados_empresas):
    """Calcula score de fiscalização baseado em ML."""
    if dados_ml is None or dados_empresas is None:
        return dados_empresas
    
    # Criar score de ML
    dados_empresas_ml = dados_empresas.copy()
    
    # Adicionar pontuação de anomalia (anomalias = -1, normais = 1)
    dados_empresas_ml['score_ml_anomalia'] = 0
    dados_empresas_ml.loc[dados_ml['indice'], 'score_ml_anomalia'] = dados_ml['anomalia'].map({-1: 5, 1: 0})
    
    # Adicionar pontuação de cluster (clusters de maior risco)
    dados_empresas_ml['cluster_ml'] = -1
    dados_empresas_ml.loc[dados_ml['indice'], 'cluster_ml'] = dados_ml['cluster']
    
    # Score final de ML
    dados_empresas_ml['score_ml_total'] = (
        dados_empresas_ml['score_ml_anomalia'] + 
        dados_empresas_ml['score_risco_total']
    )
    
    # Classificação de prioridade ML
    dados_empresas_ml['prioridade_ml'] = pd.cut(
        dados_empresas_ml['score_ml_total'],
        bins=[0, 5, 8, 11, 100],
        labels=['Baixa', 'Média', 'Alta', 'Crítica']
    )
    
    return dados_empresas_ml

# =============================================================================
# 9. SIDEBAR - NAVEGAÇÃO PRINCIPAL
# =============================================================================

st.sidebar.markdown("# 📊 Sistema ECD")
st.sidebar.markdown("**Escrituração Contábil Digital**")
st.sidebar.markdown("---")

# Conectar ao banco
engine = get_impala_engine()

if engine is None:
    st.sidebar.error("❌ Não foi possível conectar ao banco de dados.")
    st.sidebar.info("Configure as credenciais em st.secrets")
    st.stop()

# Carregar resumo geral
with st.sidebar:
    with st.spinner("Carregando dados..."):
        resumo = carregar_resumo_geral(engine)
        
        if resumo:
            st.sidebar.success("✅ Conectado ao banco!")
            
            st.sidebar.markdown("### 📈 Resumo Geral")
            st.sidebar.metric("Empresas", f"{resumo['total_empresas']:,}")
            st.sidebar.metric("Anos", resumo['total_anos'])
            st.sidebar.metric("Ano Mais Recente", resumo['ano_mais_recente'])
            st.sidebar.metric("Setores", resumo['total_setores'])
            st.sidebar.markdown("---")

# Menu de navegação
st.sidebar.markdown("### 🔍 Navegação")
pagina = st.sidebar.radio(
    "",
    [
        "🏠 Visão Geral",
        "📊 Análise por Setor",
        "🏢 Detalhamento de Empresa",
        "🎯 Fiscalização Inteligente (ML)",
        "⚠️ Empresas Alto Risco",
        "📉 Indicadores Financeiros",
        "🗂️ Plano de Contas",
        "🔍 Indícios NEAF",
        "⚖️ Inconsistências Contábeis",
        "📈 Benchmark Setorial"
    ],
    label_visibility="collapsed"
)

# ✅ ADICIONAR ESTAS 3 LINHAS:
if 'force_page' in st.session_state:
    pagina = st.session_state['force_page']
    del st.session_state['force_page']

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Filtros Globais")

# Filtro de ano
anos_disponiveis = list(range(2018, 2025))
ano_selecionado = st.sidebar.selectbox("Ano de Referência", anos_disponiveis, index=len(anos_disponiveis)-1)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Dica:** Comece pela Visão Geral e depois navegue para análises específicas.")

# =============================================================================
# 10. PÁGINAS DO SISTEMA
# =============================================================================

# ---------------------------------------------------------------------------
# PÁGINA 1: VISÃO GERAL
# ---------------------------------------------------------------------------

if pagina == "🏠 Visão Geral":
    st.markdown("<h1 class='main-header'>📊 Sistema ECD - Visão Geral</h1>", unsafe_allow_html=True)
    st.markdown("### Escrituração Contábil Digital - Dashboard Executivo")
    
    # Carregar indicadores agregados
    with st.spinner("Carregando indicadores por setor..."):
        df_setores = carregar_indicadores_agregados(engine, ano_selecionado)
    
    if df_setores is not None and not df_setores.empty:
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        total_empresas = df_setores['qtd_empresas'].sum()
        media_ativo = df_setores['media_ativo_milhoes'].mean()
        media_receita = df_setores['media_receita_milhoes'].mean()
        media_liquidez = df_setores['media_liquidez'].mean()
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' title='Soma de empresas únicas de todos os setores analisados no ano {ano_selecionado}'>
                <h3>{total_empresas:,}</h3>
                <p>Empresas Analisadas ❓</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card-green' title='Média do ativo total (em milhões) calculada sobre todos os setores. Fórmula: AVG(ativo_total) de todas empresas'>
                <h3>R$ {media_ativo:,.1f}M</h3>
                <p>Ativo Médio ❓</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card-blue' title='Média da receita líquida (em milhões) de todas as empresas. Fórmula: AVG(receita_liquida)'>
                <h3>R$ {media_receita:,.1f}M</h3>
                <p>Receita Média ❓</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            cor_liquidez = 'metric-card-green' if media_liquidez > 1 else 'metric-card-red'
            status_liquidez = 'Saudável (>1.0)' if media_liquidez > 1 else 'Atenção (<1.0)'
            st.markdown(f"""
            <div class='{cor_liquidez}' title='Média da Liquidez Corrente de todas empresas. Fórmula: AVG(Ativo Circulante / Passivo Circulante). {status_liquidez}'>
                <h3>{media_liquidez:.2f}</h3>
                <p>Liquidez Corrente Média ❓</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏭 Empresas por Setor (Top 15)")
            # ✅ Garantir tipo numérico
            df_setores['qtd_empresas'] = pd.to_numeric(df_setores['qtd_empresas'], errors='coerce').fillna(0)
            df_top_setores = df_setores.nlargest(15, 'qtd_empresas')
            
            fig = px.bar(
                df_top_setores,
                x='qtd_empresas',
                y='setor',
                orientation='h',
                color='qtd_empresas',
                color_continuous_scale='Blues',
                labels={'qtd_empresas': 'Quantidade', 'setor': 'Setor'}
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Ativo Médio por Setor (Top 15)")
            # ✅ Garantir tipo numérico
            df_setores['media_ativo_milhoes'] = pd.to_numeric(df_setores['media_ativo_milhoes'], errors='coerce').fillna(0)
            df_top_ativo = df_setores.nlargest(15, 'media_ativo_milhoes')
            
            fig = px.bar(
                df_top_ativo,
                x='media_ativo_milhoes',
                y='setor',
                orientation='h',
                color='media_ativo_milhoes',
                color_continuous_scale='Greens',
                labels={'media_ativo_milhoes': 'Ativo Médio (R$ Mi)', 'setor': 'Setor'}
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Gráficos de indicadores financeiros
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Liquidez Corrente por Setor")
            # ✅ Garantir tipo numérico
            df_setores['media_liquidez'] = pd.to_numeric(df_setores['media_liquidez'], errors='coerce').fillna(0)
            df_liquidez = df_setores.nlargest(15, 'media_liquidez')
            
            fig = px.bar(
                df_liquidez,
                x='media_liquidez',
                y='setor',
                orientation='h',
                color='media_liquidez',
                color_continuous_scale='RdYlGn',
                labels={'media_liquidez': 'Liquidez Corrente', 'setor': 'Setor'}
            )
            fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Mínimo Saudável")
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 ROE Médio por Setor (%)")
            # ✅ Garantir tipo numérico
            df_setores['media_roe'] = pd.to_numeric(df_setores['media_roe'], errors='coerce').fillna(0)
            df_roe = df_setores.nlargest(15, 'media_roe')
            
            fig = px.bar(
                df_roe,
                x='media_roe',
                y='setor',
                orientation='h',
                color='media_roe',
                color_continuous_scale='Oranges',
                labels={'media_roe': 'ROE (%)', 'setor': 'Setor'}
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabela resumida
        st.markdown("---")
        st.markdown("### 📋 Resumo dos Setores")
        df_setores = limpar_dataframe_para_exibicao(df_setores)
        st.dataframe(
            df_setores.style.format({
                'qtd_empresas': '{:,.0f}',
                'media_ativo_milhoes': 'R$ {:,.2f}',
                'media_receita_milhoes': 'R$ {:,.2f}',
                'media_liquidez': '{:.2f}',
                'media_endividamento': '{:.2f}',
                'media_margem_liquida': '{:.2f}%',
                'media_roa': '{:.2f}%',
                'media_roe': '{:.2f}%'
            }).background_gradient(subset=['media_liquidez'], cmap='RdYlGn', vmin=0, vmax=2),
            use_container_width=True,
            height=400
        )
    else:
        st.error("Não foi possível carregar os dados agregados.")

# ---------------------------------------------------------------------------
# PÁGINA 2: ANÁLISE POR SETOR
# ---------------------------------------------------------------------------

elif pagina == "📊 Análise por Setor":
    st.markdown("<h1 class='main-header'>📊 Análise Detalhada por Setor</h1>", unsafe_allow_html=True)
    
    # Carregar setores
    with st.spinner("Carregando setores..."):
        df_setores = carregar_indicadores_agregados(engine, ano_selecionado)
    
    if df_setores is not None and not df_setores.empty:
        # Seletor de setor
        setor_selecionado = st.selectbox(
            "🏭 Selecione o Setor para Análise",
            df_setores['setor'].tolist()
        )
        
        # Métricas do setor
        setor_info = df_setores[df_setores['setor'] == setor_selecionado].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Empresas no Setor", f"{int(setor_info['qtd_empresas']):,}")
        with col2:
            st.metric("Ativo Médio", f"R$ {setor_info['media_ativo_milhoes']:,.2f}M")
        with col3:
            st.metric("Receita Média", f"R$ {setor_info['media_receita_milhoes']:,.2f}M")
        with col4:
            st.metric("Liquidez Média", f"{setor_info['media_liquidez']:.2f}")
        
        st.markdown("---")
        
        # Carregar empresas do setor
        with st.spinner(f"Carregando empresas do setor {setor_selecionado}..."):
            df_empresas = carregar_empresas_por_setor(engine, setor_selecionado, ano_selecionado)
        
        if df_empresas is not None and not df_empresas.empty:
            st.markdown(f"### 🏢 Empresas do Setor: {setor_selecionado}")
            st.markdown(f"**Total de empresas:** {len(df_empresas)}")
            
            # Gráficos
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribuição por Porte")
                porte_count = df_empresas['empresa_grande_porte'].value_counts()
                fig = px.pie(
                    values=porte_count.values,
                    names=porte_count.index,
                    title="Grande Porte vs Outros"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Classificação de Risco")
                risco_count = df_empresas['classificacao_risco'].value_counts()
                fig = px.pie(
                    values=risco_count.values,
                    names=risco_count.index,
                    title="Distribuição por Risco",
                    color_discrete_map={
                        'Muito Alto': '#d32f2f',
                        'Alto': '#f57c00',
                        'Médio': '#fbc02d',
                        'Baixo': '#689f38'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Top empresas por ativo
            st.markdown("#### 💰 Top 10 Empresas por Ativo")
            # ✅ Garantir tipo numérico
            df_empresas['ativo_milhoes'] = pd.to_numeric(df_empresas['ativo_milhoes'], errors='coerce').fillna(0)
            df_top = df_empresas.nlargest(10, 'ativo_milhoes')
            
            fig = px.bar(
                df_top,
                x='ativo_milhoes',
                y='nm_razao_social',
                orientation='h',
                color='liquidez',
                color_continuous_scale='RdYlGn',
                labels={'ativo_milhoes': 'Ativo (R$ Mi)', 'nm_razao_social': 'Empresa'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de empresas
            st.markdown("#### 📋 Listagem Completa de Empresas")
            
            # Formatar tabela
            df_exibir = df_empresas.copy()
            df_exibir = df_exibir[[
                'nm_razao_social', 'cd_uf', 'empresa_grande_porte',
                'ativo_milhoes', 'receita_milhoes', 'liquidez',
                'margem_liquida', 'score_risco_total', 'classificacao_risco'
            ]]

            df_exibir = limpar_dataframe_para_exibicao(df_exibir)
            st.dataframe(
                df_exibir.style.format({
                    'ativo_milhoes': 'R$ {:,.2f}',
                    'receita_milhoes': 'R$ {:,.2f}',
                    'liquidez': '{:.2f}',
                    'margem_liquida': '{:.2f}%',
                    'score_risco_total': '{:.1f}'
                }).background_gradient(subset=['liquidez'], cmap='RdYlGn', vmin=0, vmax=2)
                  .background_gradient(subset=['score_risco_total'], cmap='RdYlGn_r', vmin=0, vmax=10),
                use_container_width=True,
                height=400
            )
            
            # Ver detalhes de empresas
            st.markdown("---")
            st.markdown("### 🔍 Ver Detalhes de Empresas do Setor")
            st.info("💡 Clique no botão **👁️ Ver** para abrir os detalhes da empresa")
            
            # Top 15 empresas
            # ✅ Garantir tipo numérico
            df_empresas['ativo_milhoes'] = pd.to_numeric(df_empresas['ativo_milhoes'], errors='coerce').fillna(0)
            df_top = df_empresas.nlargest(15, 'ativo_milhoes')
            
            for idx, (index, row) in enumerate(df_top.iterrows()):
                col1, col2, col3, col4, col5 = st.columns([3, 1.2, 1.2, 1, 0.8])
                
                with col1:
                    st.markdown(f"**{row['nm_razao_social'][:45]}**")
                
                with col2:
                    st.write(f"💰 R$ {row['ativo_milhoes']:,.1f}M")
                
                with col3:
                    st.write(f"📍 {row['cd_uf']}")
                
                with col4:
                    score = row['score_risco_total']
                    if score >= 7:
                        st.markdown("🔴 **Alto**")
                    elif score >= 5:
                        st.markdown("🟡 **Médio**")
                    else:
                        st.markdown("🟢 **Baixo**")
                
                with col5:
                    btn_key = f"ver_{idx}_{row['cnpj'][-4:]}"
                    if st.button("👁️ Ver", key=btn_key):
                        st.session_state['cnpj_drill'] = row['cnpj']
                        st.session_state['force_page'] = "🏢 Detalhamento de Empresa"
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
                
                if idx < len(df_top) - 1:
                    st.markdown("<hr style='margin: 8px 0; opacity: 0.2;'>", unsafe_allow_html=True)
        else:
            st.warning("Nenhuma empresa encontrada para este setor.")
    else:
        st.error("Não foi possível carregar os setores.")
        
# ---------------------------------------------------------------------------
# PÁGINA 3: DETALHAMENTO DE EMPRESA
# ---------------------------------------------------------------------------

elif pagina == "🏢 Detalhamento de Empresa":
    st.markdown("<h1 class='main-header'>🏢 Detalhamento Completo de Empresa</h1>", unsafe_allow_html=True)
    
    # Campo de busca sempre visível
    st.markdown("### 🔍 Buscar Empresa")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Verificar se veio de drilldown
        cnpj_inicial = st.session_state.get('cnpj_drill', '')
        cnpj_busca = st.text_input(
            "Digite o CNPJ da empresa (14 dígitos)", 
            value=cnpj_inicial,
            max_chars=14,
            key='cnpj_input'
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        buscar = st.button("🔎 Buscar", use_container_width=True)
    
    # Limpar session state após usar
    if 'cnpj_drill' in st.session_state and cnpj_busca:
        del st.session_state['cnpj_drill']
    
    if cnpj_busca and len(cnpj_busca) == 14:
        with st.spinner("Carregando dados completos da empresa..."):
            dados_empresa = carregar_dados_empresa(engine, cnpj_busca)
        
        if dados_empresa and 'cadastro' in dados_empresa and not dados_empresa['cadastro'].empty:
            cadastro = dados_empresa['cadastro'].iloc[0]
            
            # Header da empresa
            st.markdown(f"## {cadastro['nm_razao_social']}")
            st.markdown(f"**CNPJ:** {cnpj_busca} | **UF:** {cadastro['cd_uf']} | **Setor:** {cadastro.get('cnae_divisao_descricao', 'N/A')}")
            
            # Abas de navegação
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📝 Dados Cadastrais",
                "📊 Indicadores Financeiros",
                "💰 Balanço Patrimonial",
                "📈 DRE",
                "⚠️ Análise de Risco"
            ])
            
            # ABA 1: Dados Cadastrais
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Informações Gerais")
                    st.write(f"**Razão Social:** {cadastro['nm_razao_social']}")
                    st.write(f"**Nome Fantasia:** {cadastro.get('nm_fantasia', 'N/A')}")
                    st.write(f"**CNPJ:** {cnpj_busca}")
                    st.write(f"**UF:** {cadastro['cd_uf']}")
                    st.write(f"**CNAE:** {cadastro.get('cd_cnae', 'N/A')}")
                    st.write(f"**Descrição CNAE:** {cadastro.get('de_cnae', 'N/A')}")
                
                with col2:
                    st.markdown("### Classificações")
                    st.write(f"**Setor:** {cadastro.get('cnae_divisao_descricao', 'N/A')}")
                    st.write(f"**Grande Porte:** {cadastro.get('empresa_grande_porte', 'N/A')}")
                    st.write(f"**Tipo ECD:** {cadastro.get('tipo_ecd', 'N/A')}")
                    st.write(f"**Natureza Jurídica:** {cadastro.get('nm_natureza_juridica_sefaz', 'N/A')}")
                    st.write(f"**Regime de Apuração:** {cadastro.get('nm_reg_apuracao', 'N/A')}")
                    st.write(f"**Simples Nacional:** {cadastro.get('sn_simples_nacional_rfb', 'N/A')}")
            
            # ABA 2: Indicadores Financeiros
            with tab2:
                if 'indicadores' in dados_empresa and not dados_empresa['indicadores'].empty:
                    st.markdown("### 📊 Indicadores Financeiros")

                    # ✅ Versão com mais colunas
                    df_indicadores = dados_empresa['indicadores'][[
                        'ativo_total', 'ativo_circulante', 'ativo_nao_circulante',
                        'passivo_total', 'passivo_circulante', 'passivo_nao_circulante',
                        'patrimonio_liquido', 'receita_liquida', 'lucro_bruto',
                        'resultado_liquido', 'custos_totais', 'despesas_totais',
                        'liquidez_corrente', 'liquidez_geral', 'endividamento_geral',
                        'composicao_endividamento', 'margem_liquida_perc', 'margem_bruta_perc',
                        'roa_retorno_ativo_perc', 'roe_retorno_patrimonio_perc'
                    ]].copy()
                    
                    df_indicadores = limpar_dataframe_para_exibicao(df_indicadores)
                    
                    st.dataframe(
                        df_indicadores.style.format({
                            'ativo_total': 'R$ {:,.2f}',
                            'ativo_circulante': 'R$ {:,.2f}',
                            'ativo_nao_circulante': 'R$ {:,.2f}',
                            'passivo_total': 'R$ {:,.2f}',
                            'passivo_circulante': 'R$ {:,.2f}',
                            'passivo_nao_circulante': 'R$ {:,.2f}',
                            'patrimonio_liquido': 'R$ {:,.2f}',
                            'receita_liquida': 'R$ {:,.2f}',
                            'lucro_bruto': 'R$ {:,.2f}',
                            'resultado_liquido': 'R$ {:,.2f}',
                            'custos_totais': 'R$ {:,.2f}',
                            'despesas_totais': 'R$ {:,.2f}',
                            'liquidez_corrente': '{:.4f}',
                            'liquidez_geral': '{:.4f}',
                            'endividamento_geral': '{:.4f}',
                            'composicao_endividamento': '{:.4f}',
                            'margem_liquida_perc': '{:.4f}%',
                            'margem_bruta_perc': '{:.4f}%',
                            'roa_retorno_ativo_perc': '{:.4f}%',
                            'roe_retorno_patrimonio_perc': '{:.4f}%'
                        }),
                        use_container_width=True
                    )
                else:
                    st.warning("Sem dados de indicadores financeiros.")
            
            # ABA 3: Balanço Patrimonial
            with tab3:
                if 'balanco' in dados_empresa and not dados_empresa['balanco'].empty:
                    anos_balanco = sorted([x // 100 for x in dados_empresa['balanco']['ano_referencia'].unique()], reverse=True)
                    ano_bp = st.selectbox("Selecione o Ano", anos_balanco, key='ano_bp')
                    criar_balanco_patrimonial(dados_empresa['balanco'], ano_bp)
                else:
                    st.warning("Sem dados de Balanço Patrimonial.")
            
            # ABA 4: DRE
            with tab4:
                if 'dre' in dados_empresa and not dados_empresa['dre'].empty:
                    anos_dre = sorted([x // 100 for x in dados_empresa['dre']['ano_referencia'].unique()], reverse=True)
                    ano_dre = st.selectbox("Selecione o Ano", anos_dre, key='ano_dre')
                    criar_dre(dados_empresa['dre'], ano_dre)
                else:
                    st.warning("Sem dados de DRE.")
            
            # ABA 5: Análise de Risco (COM COMPARAÇÃO SETORIAL)
            with tab5:
                if 'risco' in dados_empresa and not dados_empresa['risco'].empty:
                    st.markdown("### ⚠️ Análise de Risco Consolidada")
                    
                    risco_atual = dados_empresa['risco'].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        score = risco_atual['score_risco_total']
                        cor = 'metric-card-red' if score >= 7 else 'metric-card-yellow' if score >= 5 else 'metric-card-green'
                        st.markdown(f"""
                        <div class='{cor}'>
                            <h2>{score:.1f}</h2>
                            <p>Score de Risco Total</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='metric-card-blue'>
                            <h3>{risco_atual['classificacao_risco']}</h3>
                            <p>Classificação</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("Score Equação Contábil", f"{risco_atual['score_equacao_contabil']:.1f}")
                    
                    with col4:
                        st.metric("Indícios NEAF", int(risco_atual['qtd_indicios_neaf']))
                    
                    st.markdown("---")
                    
                    # ANÁLISE COMPARATIVA COM O SETOR
                    st.markdown("### 📊 Análise Comparativa com o Setor")
                    
                    setor_empresa = cadastro.get('cnae_divisao_descricao', None)
                    
                    if setor_empresa and 'indicadores' in dados_empresa and not dados_empresa['indicadores'].empty:
                        # Carregar dados do setor
                        with st.spinner("Carregando dados do setor para comparação..."):
                            df_setor = carregar_indicadores_agregados(engine, ano_selecionado)
                        
                        if df_setor is not None and not df_setor.empty:
                            setor_info = df_setor[df_setor['setor'] == setor_empresa]
                            
                            if not setor_info.empty:
                                setor_info = setor_info.iloc[0]
                                ind_empresa = dados_empresa['indicadores'].iloc[0]
                                
                                # Criar tabela comparativa
                                st.markdown(f"**Comparando com o setor:** {setor_empresa}")
                                st.markdown(f"**Quantidade de empresas no setor:** {int(setor_info['qtd_empresas']):,}")
                                
                                # Análise de cada indicador
                                indicadores_analise = {
                                    'Liquidez Corrente': {
                                        'valor_empresa': ind_empresa['liquidez_corrente'],
                                        'media_setor': setor_info['media_liquidez'],
                                        'ideal_min': 1.0,
                                        'ideal_max': 2.0
                                    },
                                    'Endividamento Geral': {
                                        'valor_empresa': ind_empresa['endividamento_geral'],
                                        'media_setor': setor_info['media_endividamento'],
                                        'ideal_min': 0.0,
                                        'ideal_max': 0.5
                                    },
                                    'Margem Líquida (%)': {
                                        'valor_empresa': ind_empresa['margem_liquida_perc'],
                                        'media_setor': setor_info['media_margem_liquida'],
                                        'ideal_min': 5.0,
                                        'ideal_max': 100.0
                                    },
                                    'ROA (%)': {
                                        'valor_empresa': ind_empresa['roa_retorno_ativo_perc'],
                                        'media_setor': setor_info['media_roa'],
                                        'ideal_min': 5.0,
                                        'ideal_max': 100.0
                                    },
                                    'ROE (%)': {
                                        'valor_empresa': ind_empresa['roe_retorno_patrimonio_perc'],
                                        'media_setor': setor_info['media_roe'],
                                        'ideal_min': 10.0,
                                        'ideal_max': 100.0
                                    }
                                }
                                
                                # ✅ NOVO: Limpar valores None do dicionário
                                for indicador in indicadores_analise:
                                    if indicadores_analise[indicador]['valor_empresa'] is None or \
                                       pd.isna(indicadores_analise[indicador]['valor_empresa']):
                                        indicadores_analise[indicador]['valor_empresa'] = 0
                                    
                                    if indicadores_analise[indicador]['media_setor'] is None or \
                                       pd.isna(indicadores_analise[indicador]['media_setor']):
                                        indicadores_analise[indicador]['media_setor'] = 0
                                
                                # Criar visualização para cada indicador
                                for indicador, valores in indicadores_analise.items():
                                    valor_emp = valores['valor_empresa']
                                    media_set = valores['media_setor']
                                    ideal_min = valores['ideal_min']
                                    ideal_max = valores['ideal_max']
                                    
                                    # Calcular desvio percentual
                                    if media_set != 0:
                                        desvio_perc = ((valor_emp - media_set) / abs(media_set)) * 100
                                    else:
                                        desvio_perc = 0
                                    
                                    # Determinar status
                                    if valor_emp == 0 and media_set == 0:
                                        status = "⚪ Sem dados"
                                        cor_status = "alert-medio"
                                    elif ideal_min <= valor_emp <= ideal_max:
                                        status = "✅ Normal"
                                        cor_status = "alert-positivo"
                                    elif abs(desvio_perc) > 50:
                                        status = "🚨 Anômalo (>50% de desvio)"
                                        cor_status = "alert-critico"
                                    elif abs(desvio_perc) > 25:
                                        status = "⚠️ Atenção (>25% de desvio)"
                                        cor_status = "alert-alto"
                                    else:
                                        status = "⚡ Levemente diferente"
                                        cor_status = "alert-medio"
                                    
                                    # Exibir análise
                                    st.markdown(f"""
                                    <div class='{cor_status}'>
                                        <h4>{indicador}</h4>
                                        <p><strong>Empresa:</strong> {valor_emp:.2f}</p>
                                        <p><strong>Média do Setor:</strong> {media_set:.2f}</p>
                                        <p><strong>Desvio:</strong> {desvio_perc:+.1f}%</p>
                                        <p><strong>Status:</strong> {status}</p>
                                        <p><small>Faixa ideal: {ideal_min:.1f} a {ideal_max:.1f}</small></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Gráfico de radar comparativo
                                st.markdown("---")
                                st.markdown("#### 📊 Comparação Visual com o Setor")
                                
                                fig = go.Figure()
                                
                                categorias = list(indicadores_analise.keys())
                                valores_empresa = [indicadores_analise[cat]['valor_empresa'] for cat in categorias]
                                valores_setor = [indicadores_analise[cat]['media_setor'] for cat in categorias]
                                
                                # Normalizar valores para o gráfico (0-100)
                                def normalizar(valor, minimo, maximo):
                                    if maximo == minimo:
                                        return 50
                                    return ((valor - minimo) / (maximo - minimo)) * 100
                                
                                valores_empresa_norm = []
                                valores_setor_norm = []
                                
                                for cat in categorias:
                                    val_emp = indicadores_analise[cat]['valor_empresa']
                                    val_set = indicadores_analise[cat]['media_setor']
                                    minimo = min(val_emp, val_set, 0)
                                    maximo = max(val_emp, val_set, indicadores_analise[cat]['ideal_max'])
                                    
                                    valores_empresa_norm.append(normalizar(val_emp, minimo, maximo))
                                    valores_setor_norm.append(normalizar(val_set, minimo, maximo))
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=valores_empresa_norm + [valores_empresa_norm[0]],
                                    theta=categorias + [categorias[0]],
                                    fill='toself',
                                    name='Empresa',
                                    line=dict(color='#e53e3e', width=3)
                                ))
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=valores_setor_norm + [valores_setor_norm[0]],
                                    theta=categorias + [categorias[0]],
                                    fill='toself',
                                    name='Média do Setor',
                                    line=dict(color='#3182ce', width=2),
                                    opacity=0.6
                                ))
                                
                                fig.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                    showlegend=True,
                                    title="Perfil da Empresa vs Setor (Normalizado 0-100)",
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Setor não encontrado na base de dados para comparação.")
                        else:
                            st.warning("Não foi possível carregar dados do setor para comparação.")
                    else:
                        st.info("Dados insuficientes para análise comparativa com o setor.")
                    
                    st.markdown("---")
                    
                    # Detalhamento dos scores
                    st.markdown("### 📊 Detalhamento dos Scores")
                    
                    scores = {
                        'Equação Contábil': risco_atual['score_equacao_contabil'],
                        'NEAF': risco_atual['score_neaf'],
                        'Risco Financeiro': risco_atual['score_risco_financeiro']
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(scores.values()),
                            y=list(scores.keys()),
                            orientation='h',
                            marker=dict(
                                color=list(scores.values()),
                                colorscale='RdYlGn_r',
                                cmin=0,
                                cmax=10
                            ),
                            text=[f'{v:.1f}' for v in scores.values()],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Componentes do Score de Risco",
                        xaxis_title="Pontuação",
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Alertas
                    if risco_atual['score_risco_total'] >= 7:
                        st.markdown("""
                        <div class='alert-critico'>
                            <h4>🚨 ALERTA CRÍTICO</h4>
                            <p>Esta empresa apresenta alto risco e deve ser priorizada para fiscalização.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risco_atual['score_risco_total'] >= 5:
                        st.markdown("""
                        <div class='alert-alto'>
                            <h4>⚠️ ALERTA ALTO</h4>
                            <p>Esta empresa apresenta riscos significativos e merece atenção.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='alert-positivo'>
                            <h4>✅ RISCO CONTROLADO</h4>
                            <p>Esta empresa apresenta baixo risco fiscal.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Histórico de risco
                    if len(dados_empresa['risco']) > 1:
                        st.markdown("---")
                        st.markdown("### 📈 Evolução do Risco")
                        
                        df_risco = dados_empresa['risco'].copy()
                        df_risco = limpar_dataframe_para_exibicao(df_risco)  # ✅ ADICIONAR
                        df_risco['ano'] = df_risco['ano_referencia'] // 100
                        df_risco = df_risco.sort_values('ano')
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df_risco['ano'],
                            y=df_risco['score_risco_total'],
                            mode='lines+markers',
                            name='Score Total',
                            line=dict(color='#e53e3e', width=3),
                            marker=dict(size=10)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df_risco['ano'],
                            y=df_risco['score_equacao_contabil'],
                            mode='lines+markers',
                            name='Equação Contábil',
                            line=dict(color='#3182ce', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df_risco['ano'],
                            y=df_risco['score_neaf'],
                            mode='lines+markers',
                            name='NEAF',
                            line=dict(color='#f59e0b', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Evolução dos Scores de Risco",
                            xaxis_title="Ano",
                            yaxis_title="Score",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Sem dados de análise de risco.")
        else:
            st.error("Empresa não encontrada no banco de dados.")
    else:
        st.info("👆 Digite um CNPJ válido (14 dígitos) e clique em Buscar para consultar os dados da empresa.")

# ---------------------------------------------------------------------------
# PÁGINA: FISCALIZAÇÃO INTELIGENTE (ML)
# ---------------------------------------------------------------------------

elif pagina == "🎯 Fiscalização Inteligente (ML)":
    st.markdown("<h1 class='main-header'>🎯 Fiscalização Inteligente com Machine Learning</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🤖 Sistema de Detecção de Anomalias Contábeis
    
    Este módulo utiliza algoritmos de Machine Learning para identificar empresas com comportamentos atípicos
    e maior probabilidade de irregularidades fiscais.
    
    **Técnicas utilizadas:**
    - 🔍 **Isolation Forest**: Detecção de anomalias multivariadas
    - 📊 **K-Means Clustering**: Agrupamento de empresas por perfil de risco
    - 📈 **Score Composto**: Combinação de análises estatísticas e ML
    """)
    
    st.markdown("---")
    
    # Carregar dados de alto risco
    with st.spinner("Carregando dados para análise de ML..."):
        df_alto_risco = carregar_empresas_alto_risco(engine, limite=1000)
    
    if df_alto_risco is not None and not df_alto_risco.empty:
        # Treinar modelo
        with st.spinner("Treinando modelo de Machine Learning..."):
            dados_ml, scaler = treinar_modelo_fiscalizacao(df_alto_risco)
        
        if dados_ml is not None:
            # Calcular score de ML
            df_ml_completo = calcular_score_ml(dados_ml, df_alto_risco)
            
            # Métricas do modelo
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                anomalias = (dados_ml['anomalia'] == -1).sum()
                st.markdown(f"""
                <div class='metric-card-red'>
                    <h3>{anomalias}</h3>
                    <p>Anomalias Detectadas</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                criticas = (df_ml_completo['prioridade_ml'] == 'Crítica').sum()
                st.markdown(f"""
                <div class='metric-card-red'>
                    <h3>{criticas}</h3>
                    <p>Prioridade Crítica</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                altas = (df_ml_completo['prioridade_ml'] == 'Alta').sum()
                st.markdown(f"""
                <div class='metric-card-yellow'>
                    <h3>{altas}</h3>
                    <p>Prioridade Alta</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                clusters = dados_ml['cluster'].nunique()
                st.markdown(f"""
                <div class='metric-card-blue'>
                    <h3>{clusters}</h3>
                    <p>Clusters Identificados</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualizações
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Distribuição de Prioridades ML")
                prioridade_counts = df_ml_completo['prioridade_ml'].value_counts()
                fig = px.pie(
                    values=prioridade_counts.values,
                    names=prioridade_counts.index,
                    color=prioridade_counts.index,
                    color_discrete_map={
                        'Crítica': '#d32f2f',
                        'Alta': '#f57c00',
                        'Média': '#fbc02d',
                        'Baixa': '#689f38'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Clusters de Risco")
                cluster_counts = dados_ml['cluster'].value_counts()
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Quantidade'},
                    color=cluster_counts.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            st.markdown("### 🔍 Análise Multidimensional")
            
            fig = px.scatter(
                df_ml_completo,
                x='ativo_milhoes',
                y='score_risco_total',
                color='prioridade_ml',
                size='score_ml_total',
                hover_data=['nm_razao_social', 'setor', 'cd_uf'],
                color_discrete_map={
                    'Crítica': '#d32f2f',
                    'Alta': '#f57c00',
                    'Média': '#fbc02d',
                    'Baixa': '#689f38'
                },
                labels={
                    'ativo_milhoes': 'Ativo (R$ Milhões)',
                    'score_risco_total': 'Score de Risco',
                    'prioridade_ml': 'Prioridade ML'
                }
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top prioridades
            st.markdown("---")
            st.markdown("### 🎯 Top 50 Empresas Prioritárias para Fiscalização (ML)")
            
            # ✅ Garantir tipo numérico
            df_ml_completo['score_ml_total'] = pd.to_numeric(df_ml_completo['score_ml_total'], errors='coerce').fillna(0)
            df_top_ml = df_ml_completo.nlargest(50, 'score_ml_total')
            
            df_exibir = df_top_ml[[
                'nm_razao_social', 'setor', 'cd_uf', 'score_ml_total',
                'prioridade_ml', 'score_risco_total', 'ativo_milhoes',
                'receita_milhoes', 'liquidez', 'endividamento'
            ]].copy()

            df_exibir = limpar_dataframe_para_exibicao(df_exibir)
            st.dataframe(
                df_exibir.style.format({
                    'score_ml_total': '{:.2f}',
                    'score_risco_total': '{:.2f}',
                    'ativo_milhoes': 'R$ {:,.2f}',
                    'receita_milhoes': 'R$ {:,.2f}',
                    'liquidez': '{:.2f}',
                    'endividamento': '{:.2%}'
                }).background_gradient(subset=['score_ml_total'], cmap='Reds', vmin=0, vmax=15)
                  .background_gradient(subset=['liquidez'], cmap='RdYlGn', vmin=0, vmax=2),
                use_container_width=True,
                height=600
            )
            
            # Exportar lista
            st.markdown("---")
            if st.button("📥 Exportar Lista Completa (CSV)"):
                csv = df_ml_completo.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"fiscalizacao_ml_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Não foi possível treinar o modelo de ML com os dados disponíveis.")
    else:
        st.error("Não foi possível carregar os dados para análise de ML.")

# ---------------------------------------------------------------------------
# PÁGINA: EMPRESAS ALTO RISCO
# ---------------------------------------------------------------------------

elif pagina == "⚠️ Empresas Alto Risco":
    st.markdown("<h1 class='main-header'>⚠️ Empresas de Alto Risco</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Lista de empresas com score de risco elevado
    
    Baseado em análises de:
    - ⚠️ Inconsistências na equação contábil
    - 📊 Indicadores financeiros atípicos
    - 🔍 Indícios de NEAF (Nota Fiscal de Entrada Ausente de Fornecedor)
    - 💰 Relevância financeira (ativo e receita)
    """)
    
    st.markdown("---")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider("Score Mínimo de Risco", 0, 10, 5)
    
    with col2:
        ufs = ['Todos', 'SC', 'PR', 'RS', 'SP', 'RJ', 'MG']
        uf_filtro = st.selectbox("Filtrar por UF", ufs)
    
    with col3:
        limite_registros = st.selectbox("Quantidade de registros", [50, 100, 200, 500], index=2)
    
    # Carregar dados
    with st.spinner("Carregando empresas de alto risco..."):
        df_alto_risco = carregar_empresas_alto_risco(engine, limite=limite_registros)
    
    if df_alto_risco is not None and not df_alto_risco.empty:
        # Aplicar filtros
        df_filtrado = df_alto_risco[df_alto_risco['score_risco_total'] >= min_score].copy()
        
        if uf_filtro != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['cd_uf'] == uf_filtro]
        
        # Estatísticas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Empresas Listadas", len(df_filtrado))
        
        with col2:
            media_score = df_filtrado['score_risco_total'].mean()
            st.metric("Score Médio", f"{media_score:.2f}")
        
        with col3:
            criticas = (df_filtrado['prioridade_fiscalizacao'] == 1).sum()
            st.metric("Prioridade Crítica", criticas)
        
        with col4:
            ativo_total = df_filtrado['ativo_milhoes'].sum()
            st.metric("Ativo Total", f"R$ {ativo_total:,.0f}M")
        
        st.markdown("---")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Distribuição por Classificação")
            class_counts = df_filtrado['classificacao_risco'].value_counts()
            fig = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                color=class_counts.index,
                color_discrete_map={
                    'Muito Alto': '#d32f2f',
                    'Alto': '#f57c00',
                    'Médio': '#fbc02d'
                },
                labels={'x': 'Classificação', 'y': 'Quantidade'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Distribuição por Prioridade")
            prior_counts = df_filtrado['prioridade_fiscalizacao'].value_counts().sort_index()
            fig = px.bar(
                x=prior_counts.index,
                y=prior_counts.values,
                color=prior_counts.values,
                color_continuous_scale='Reds',
                labels={'x': 'Prioridade', 'y': 'Quantidade'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabela principal
        st.markdown("---")
        st.markdown("### 📋 Listagem Detalhada")
        
        df_exibir = df_filtrado[[
            'nm_razao_social', 'setor', 'cd_uf', 'prioridade_fiscalizacao',
            'score_risco_total', 'classificacao_risco', 'qtd_indicios_neaf',
            'ativo_milhoes', 'receita_milhoes', 'liquidez', 'endividamento', 'margem_liquida'
        ]].copy()

        df_exibir = limpar_dataframe_para_exibicao(df_exibir)
        st.dataframe(
            df_exibir.style.format({
                'score_risco_total': '{:.2f}',
                'ativo_milhoes': 'R$ {:,.2f}',
                'receita_milhoes': 'R$ {:,.2f}',
                'liquidez': '{:.2f}',
                'endividamento': '{:.2%}',
                'margem_liquida': '{:.2f}%'
            }).background_gradient(subset=['score_risco_total'], cmap='Reds', vmin=min_score, vmax=10)
              .background_gradient(subset=['prioridade_fiscalizacao'], cmap='Reds', vmin=1, vmax=5),
            use_container_width=True,
            height=600
        )
        
        # Download
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📥 Exportar Lista (CSV)"):
                csv = df_filtrado.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"alto_risco_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 Gerar Relatório"):
                st.info("Funcionalidade em desenvolvimento")
    else:
        st.error("Não foi possível carregar os dados.")

# ---------------------------------------------------------------------------
# PÁGINA: INDICADORES FINANCEIROS
# ---------------------------------------------------------------------------

elif pagina == "📉 Indicadores Financeiros":
    st.markdown("<h1 class='main-header'>📉 Análise de Indicadores Financeiros</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Análise comparativa de indicadores financeiros por setor
    
    Indicadores disponíveis:
    - 💧 **Liquidez Corrente**: Capacidade de pagamento de curto prazo (ideal > 1.0)
    - 📊 **Endividamento Geral**: Nível de endividamento total (ideal < 0.5)
    - 💹 **Margem Líquida**: Rentabilidade sobre as vendas (ideal > 5%)
    - 📈 **ROA**: Retorno sobre o Ativo (ideal > 5%)
    - 🎯 **ROE**: Retorno sobre o Patrimônio Líquido (ideal > 10%)
    """)
    
    # Carregar dados
    with st.spinner("Carregando indicadores..."):
        df_setores = carregar_indicadores_agregados(engine, ano_selecionado)
    
    if df_setores is not None and not df_setores.empty:
        # Seletor de indicador
        indicador = st.selectbox(
            "Selecione o indicador para análise",
            [
                'Liquidez Corrente',
                'Endividamento Geral',
                'Margem Líquida',
                'ROA',
                'ROE'
            ]
        )
        
        # Mapear para coluna
        mapa_colunas = {
            'Liquidez Corrente': 'media_liquidez',
            'Endividamento Geral': 'media_endividamento',
            'Margem Líquida': 'media_margem_liquida',
            'ROA': 'media_roa',
            'ROE': 'media_roe'
        }
        
        coluna = mapa_colunas[indicador]
        
        # Estatísticas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            media = df_setores[coluna].mean()
            st.metric("Média Geral", f"{media:.2f}")
        
        with col2:
            mediana = df_setores[coluna].median()
            st.metric("Mediana", f"{mediana:.2f}")
        
        with col3:
            desvio = df_setores[coluna].std()
            st.metric("Desvio Padrão", f"{desvio:.2f}")
        
        with col4:
            maximo = df_setores[coluna].max()
            st.metric("Máximo", f"{maximo:.2f}")
        
        st.markdown("---")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### Top 15 Setores - {indicador}")
            # ✅ Garantir tipo numérico
            df_setores[coluna] = pd.to_numeric(df_setores[coluna], errors='coerce').fillna(0)
            df_top = df_setores.nlargest(15, coluna)
            
            fig = px.bar(
                df_top,
                x=coluna,
                y='setor',
                orientation='h',
                color=coluna,
                color_continuous_scale='RdYlGn' if indicador in ['Liquidez Corrente', 'Margem Líquida', 'ROA', 'ROE'] else 'RdYlGn_r',
                labels={coluna: indicador, 'setor': 'Setor'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"### Distribuição - {indicador}")
            fig = px.histogram(
                df_setores,
                x=coluna,
                nbins=30,
                labels={coluna: indicador},
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot
        st.markdown("---")
        st.markdown(f"### 📊 Análise de Dispersão - {indicador}")
        
        fig = px.box(
            df_setores,
            y=coluna,
            points='all',
            labels={coluna: indicador},
            color_discrete_sequence=['#06b6d4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # NOVA SEÇÃO: EMPRESAS SUSPEITAS
        st.markdown("---")
        st.markdown(f"### 🚨 Empresas Suspeitas - {indicador}")
        
        with st.spinner(f"Carregando empresas com {indicador} suspeito..."):
            df_suspeitas = carregar_empresas_suspeitas_indicador(engine, indicador, ano=ano_selecionado)
        
        if df_suspeitas is not None and not df_suspeitas.empty:
            st.markdown(f"**{len(df_suspeitas)} empresas** com valores críticos de {indicador}")
            
            # Explicar critérios
            criterios = {
                'Liquidez Corrente': 'Liquidez < 0.5 (risco de insolvência)',
                'Endividamento Geral': 'Endividamento > 0.8 (alto comprometimento)',
                'Margem Líquida': 'Margem < -10% (prejuízo significativo)',
                'ROA': 'ROA < -10% (rentabilidade muito negativa)',
                'ROE': 'ROE < -20% (retorno extremamente negativo)'
            }
            
            st.info(f"**Critério:** {criterios[indicador]}")
            
            # Gráfico de empresas suspeitas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Top 20 Valores Mais Críticos")
                df_top_suspeitas = df_suspeitas.head(20)
                
                fig = px.bar(
                    df_top_suspeitas,
                    x='valor_indicador',
                    y='nm_razao_social',
                    orientation='h',
                    color='score_risco_total',
                    color_continuous_scale='Reds',
                    labels={'valor_indicador': indicador, 'nm_razao_social': 'Empresa'},
                    hover_data=['setor', 'cd_uf', 'ativo_milhoes']
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribuição por UF")
                uf_counts = df_suspeitas['cd_uf'].value_counts()
                fig = px.pie(
                    values=uf_counts.values,
                    names=uf_counts.index,
                    title="Empresas Suspeitas por Estado"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Distribuição por Setor")
                setor_counts = df_suspeitas['setor'].value_counts().head(10)
                fig = px.bar(
                    x=setor_counts.values,
                    y=setor_counts.index,
                    orientation='h',
                    labels={'x': 'Quantidade', 'y': 'Setor'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabela detalhada
            st.markdown("#### 📋 Listagem Detalhada")
            
            df_exibir = df_suspeitas[[
                'nm_razao_social', 'cd_uf', 'setor', 'valor_indicador',
                'media_setor', 'desvio_setor', 'ativo_milhoes', 'receita_milhoes',
                'score_risco_total'
            ]].copy()
            
            df_exibir.columns = [
                'Razão Social', 'UF', 'Setor', f'{indicador}',
                'Média Setor', 'Desvio Setor', 'Ativo (R$M)', 'Receita (R$M)',
                'Score Risco'
            ]

            df_exibir = limpar_dataframe_para_exibicao(df_exibir)
            st.dataframe(
                df_exibir.style.format({
                    f'{indicador}': '{:.2f}',
                    'Média Setor': '{:.2f}',
                    'Desvio Setor': '{:.2f}',
                    'Ativo (R$M)': '{:.2f}',
                    'Receita (R$M)': '{:.2f}',
                    'Score Risco': '{:.1f}'
                }).background_gradient(subset=[f'{indicador}'], cmap='Reds')
                  .background_gradient(subset=['Score Risco'], cmap='Reds'),
                use_container_width=True,
                height=400
            )
            
            # Botão de exportação
            if st.button(f"📥 Exportar Empresas Suspeitas - {indicador}"):
                csv = df_suspeitas.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"empresas_suspeitas_{indicador.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.success(f"✅ Nenhuma empresa com valores críticos de {indicador} no ano {ano_selecionado}")
        
        # Tabela completa por setor
        st.markdown("---")
        st.markdown("### 📋 Tabela Completa por Setor")

        df_setores = limpar_dataframe_para_exibicao(df_setores)
        st.dataframe(
            df_setores.style.format({
                'qtd_empresas': '{:,.0f}',
                'media_ativo_milhoes': 'R$ {:,.2f}',
                'media_receita_milhoes': 'R$ {:,.2f}',
                'media_liquidez': '{:.2f}',
                'media_endividamento': '{:.2f}',
                'media_margem_liquida': '{:.2f}%',
                'media_roa': '{:.2f}%',
                'media_roe': '{:.2f}%'
            }).background_gradient(subset=[coluna], cmap='RdYlGn', vmin=df_setores[coluna].quantile(0.1), vmax=df_setores[coluna].quantile(0.9)),
            use_container_width=True,
            height=500
        )
    else:
        st.error("Não foi possível carregar os indicadores.")

# ---------------------------------------------------------------------------
# PÁGINA: PLANO DE CONTAS
# ---------------------------------------------------------------------------

elif pagina == "🗂️ Plano de Contas":
    st.markdown("<h1 class='main-header'>🗂️ Análise do Plano de Contas</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Sistema de contas contábeis mais utilizadas pelas empresas
    
    Esta análise mostra:
    - 📊 Contas mais comuns entre as empresas
    - 💰 Estatísticas de saldos (média, mínimo, máximo)
    - 🏢 Quantidade de empresas que utilizam cada conta
    - 📈 Distribuição por grupo de balanço
    """)
    
    # Carregar dados do plano de contas
    with st.spinner("Carregando plano de contas..."):
        df_plano = carregar_plano_contas_agregado(engine, ano_selecionado)
    
    if df_plano is not None and not df_plano.empty:
        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_contas = len(df_plano)
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{total_contas:,}</h3>
                <p>Contas Analisadas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            media_empresas = df_plano['qtd_empresas_usam'].mean()
            st.markdown(f"""
            <div class='metric-card-blue'>
                <h3>{media_empresas:.0f}</h3>
                <p>Média Empresas/Conta</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_saldo = df_plano['total_saldo_bilhoes'].sum()
            st.markdown(f"""
            <div class='metric-card-green'>
                <h3>R$ {total_saldo:.1f}B</h3>
                <p>Total em Saldos</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            conta_mais_usada = df_plano.iloc[0]['qtd_empresas_usam']
            st.markdown(f"""
            <div class='metric-card-yellow'>
                <h3>{conta_mais_usada:,}</h3>
                <p>Conta Mais Popular</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grupos_disponiveis = ['Todos'] + sorted(df_plano['descricao_grupo_balanco'].dropna().unique().tolist())
            grupo_filtro = st.selectbox("Filtrar por Grupo de Balanço", grupos_disponiveis)
        
        with col2:
            nivel_filtro = st.selectbox(
                "Filtrar por Nível",
                ['Todos'] + sorted(df_plano['nivel_conta'].dropna().unique().tolist())
            )
        
        with col3:
            min_empresas = st.number_input(
                "Mínimo de empresas",
                min_value=0,
                max_value=int(df_plano['qtd_empresas_usam'].max()),
                value=10
            )
        
        # Aplicar filtros
        df_filtrado = df_plano.copy()
        
        if grupo_filtro != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['descricao_grupo_balanco'] == grupo_filtro]
        
        if nivel_filtro != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['nivel_conta'] == nivel_filtro]
        
        df_filtrado = df_filtrado[df_filtrado['qtd_empresas_usam'] >= min_empresas]
        
        st.info(f"**{len(df_filtrado)} contas** encontradas com os filtros aplicados")
        
        # Gráficos
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Top 20 Contas Mais Utilizadas")
            # ✅ Garantir tipo numérico
            df_filtrado['qtd_empresas_usam'] = pd.to_numeric(df_filtrado['qtd_empresas_usam'], errors='coerce').fillna(0)
            df_top = df_filtrado.nlargest(20, 'qtd_empresas_usam')
            
            fig = px.bar(
                df_top,
                x='qtd_empresas_usam',
                y='nm_conta',
                orientation='h',
                color='total_saldo_bilhoes',
                color_continuous_scale='Blues',
                labels={'qtd_empresas_usam': 'Quantidade de Empresas', 'nm_conta': 'Conta'},
                hover_data=['cd_conta', 'descricao_grupo_balanco']
            )
            fig.update_layout(height=700)
            fig.update_yaxes(tickfont=dict(size=9))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Top 20 Maiores Saldos Totais")
            # ✅ Garantir tipo numérico antes de nlargest
            df_filtrado['total_saldo_bilhoes'] = pd.to_numeric(
                df_filtrado['total_saldo_bilhoes'], 
                errors='coerce'
            ).fillna(0)
            
            df_top_saldo = df_filtrado.nlargest(20, 'total_saldo_bilhoes')
            
            fig = px.bar(
                df_top_saldo,
                x='total_saldo_bilhoes',
                y='nm_conta',
                orientation='h',
                color='qtd_empresas_usam',
                color_continuous_scale='Greens',
                labels={'total_saldo_bilhoes': 'Saldo Total (R$ Bilhões)', 'nm_conta': 'Conta'},
                hover_data=['cd_conta', 'descricao_grupo_balanco']
            )
            fig.update_layout(height=700)
            fig.update_yaxes(tickfont=dict(size=9))
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribuições
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏢 Distribuição por Grupo de Balanço")
            grupo_counts = df_filtrado.groupby('descricao_grupo_balanco').agg({
                'cd_conta': 'count',
                'qtd_empresas_usam': 'sum'
            }).reset_index()
            grupo_counts.columns = ['Grupo', 'Qtd Contas', 'Total Uso']
            
            # ✅ ADICIONAR: Filtrar valores None/NaN/vazios
            grupo_counts = grupo_counts[grupo_counts['Grupo'].notna()]
            grupo_counts = grupo_counts[grupo_counts['Grupo'] != '']
            grupo_counts = grupo_counts[grupo_counts['Qtd Contas'] > 0]
            
            if len(grupo_counts) > 0:
                fig = px.pie(
                    grupo_counts,
                    values='Qtd Contas',
                    names='Grupo',
                    title='Quantidade de Contas por Grupo',
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem dados de grupos disponíveis para visualização")
        
        with col2:
            st.markdown("### 📊 Distribuição por Nível Hierárquico")
            if 'nivel_conta' in df_filtrado.columns:
                nivel_counts = df_filtrado['nivel_conta'].value_counts().sort_index()
                
                fig = px.bar(
                    x=nivel_counts.index,
                    y=nivel_counts.values,
                    labels={'x': 'Nível', 'y': 'Quantidade de Contas'},
                    color=nivel_counts.values,
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Informação de nível não disponível")
        
        # Análise de variabilidade
        st.markdown("---")
        st.markdown("### 📈 Análise de Variabilidade de Saldos")
        
        # Calcular coeficiente de variação
        df_filtrado['coef_variacao'] = (
            (df_filtrado['max_saldo_milhoes'] - df_filtrado['min_saldo_milhoes']) / 
            df_filtrado['media_saldo_milhoes'].abs()
        ) * 100
        
        # ✅ Garantir tipo numérico
        df_filtrado['coef_variacao'] = pd.to_numeric(df_filtrado['coef_variacao'], errors='coerce').fillna(0)
        df_variabilidade = df_filtrado.nlargest(20, 'coef_variacao')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_variabilidade['nm_conta'],
            y=df_variabilidade['media_saldo_milhoes'],
            name='Média',
            marker_color='lightblue',
            error_y=dict(
                type='data',
                symmetric=False,
                array=df_variabilidade['max_saldo_milhoes'] - df_variabilidade['media_saldo_milhoes'],
                arrayminus=df_variabilidade['media_saldo_milhoes'] - df_variabilidade['min_saldo_milhoes']
            )
        ))
        
        fig.update_layout(
            title='Top 20 Contas com Maior Variabilidade de Saldos',
            xaxis_title='Conta',
            yaxis_title='Saldo (R$ Milhões)',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela detalhada
        st.markdown("---")
        st.markdown("### 📋 Tabela Detalhada do Plano de Contas")
        
        # Preparar dados para exibição
        df_exibir = df_filtrado[[
            'cd_conta', 'nm_conta', 'descricao_grupo_balanco', 'nivel_conta',
            'qtd_empresas_usam', 'media_saldo_milhoes', 'min_saldo_milhoes',
            'max_saldo_milhoes', 'total_saldo_bilhoes'
        ]].copy()
        
        df_exibir.columns = [
            'Código', 'Nome da Conta', 'Grupo', 'Nível',
            'Qtd Empresas', 'Média Saldo (R$M)', 'Min Saldo (R$M)',
            'Max Saldo (R$M)', 'Total (R$B)'
        ]
        
        # Adicionar busca
        busca_conta = st.text_input("🔍 Buscar conta por nome ou código")
        
        if busca_conta:
            mascara = (
                df_exibir['Nome da Conta'].str.contains(busca_conta, case=False, na=False) |
                df_exibir['Código'].str.contains(busca_conta, case=False, na=False)
            )
            df_exibir = df_exibir[mascara]

        df_exibir = limpar_dataframe_para_exibicao(df_exibir)
        st.dataframe(
            df_exibir.style.format({
                'Qtd Empresas': '{:,.0f}',
                'Média Saldo (R$M)': '{:,.2f}',
                'Min Saldo (R$M)': '{:,.2f}',
                'Max Saldo (R$M)': '{:,.2f}',
                'Total (R$B)': '{:,.2f}'
            }).background_gradient(subset=['Qtd Empresas'], cmap='Blues')
              .background_gradient(subset=['Total (R$B)'], cmap='Greens'),
            use_container_width=True,
            height=500
        )
        
        # Exportar
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📥 Exportar Tabela Filtrada"):
                csv = df_exibir.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"plano_contas_{ano_selecionado}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Exportar Estatísticas"):
                # Criar resumo estatístico
                resumo = {
                    'Total de Contas': [len(df_filtrado)],
                    'Média Empresas/Conta': [df_filtrado['qtd_empresas_usam'].mean()],
                    'Total Saldo (Bilhões)': [df_filtrado['total_saldo_bilhoes'].sum()],
                    'Conta Mais Usada': [df_filtrado.iloc[0]['nm_conta']],
                    'Uso Máximo': [df_filtrado['qtd_empresas_usam'].max()]
                }
                df_resumo = pd.DataFrame(resumo)
                csv_resumo = df_resumo.to_csv(index=False)
                st.download_button(
                    label="Download Resumo CSV",
                    data=csv_resumo,
                    file_name=f"resumo_plano_contas_{ano_selecionado}.csv",
                    mime="text/csv"
                )
        
        # Insights
        st.markdown("---")
        st.markdown("### 💡 Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='alert-positivo'>
                <h4>📊 Contas Universais</h4>
                <p>Contas utilizadas por mais de 80% das empresas são consideradas essenciais e devem ser priorizadas em auditorias.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calcular e mostrar contas universais
            if len(df_filtrado) > 0:
                total_empresas_analise = df_filtrado['qtd_empresas_usam'].max()
                threshold_universal = total_empresas_analise * 0.8
                contas_universais = df_filtrado[df_filtrado['qtd_empresas_usam'] >= threshold_universal]
                
                if len(contas_universais) > 0:
                    st.write(f"**{len(contas_universais)} contas universais** encontradas:")
                    for _, conta in contas_universais.head(10).iterrows():
                        st.write(f"- {conta['cd_conta']}: {conta['nm_conta']} ({conta['qtd_empresas_usam']} empresas)")
                else:
                    st.write("Nenhuma conta universal identificada com os filtros atuais.")
        
        with col2:
            st.markdown("""
            <div class='alert-alto'>
                <h4>⚠️ Contas com Alta Variabilidade</h4>
                <p>Contas com grande diferença entre mínimo e máximo podem indicar inconsistências ou uso inadequado.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar contas com maior variabilidade
            if 'coef_variacao' in df_filtrado.columns:
                # ✅ Garantir tipo numérico
                df_filtrado['coef_variacao'] = pd.to_numeric(df_filtrado['coef_variacao'], errors='coerce').fillna(0)
                contas_variaveis = df_filtrado.nlargest(10, 'coef_variacao')
                st.write("**Top 10 contas mais variáveis:**")
                for _, conta in contas_variaveis.iterrows():
                    cv = conta['coef_variacao']
                    if pd.notna(cv) and not np.isinf(cv):
                        st.write(f"- {conta['cd_conta']}: {conta['nm_conta']} (CV: {cv:.1f}%)")
    else:
        st.error("Não foi possível carregar os dados do plano de contas.")

# ---------------------------------------------------------------------------
# PÁGINA: INDÍCIOS NEAF
# ---------------------------------------------------------------------------

elif pagina == "🔍 Indícios NEAF":
    st.markdown("<h1 class='main-header'>🔍 Análise de Indícios NEAF</h1>", unsafe_allow_html=True)

    st.markdown("""
    ### Sistema de Análise de Notas Fiscais de Entrada Ausente de Fornecedor

    O NEAF identifica operações onde:
    - 📄 A empresa declarou compras que o fornecedor não registrou vendas
    - ⚠️ Potencial simulação de operações ou notas frias
    - 🔍 Indícios de irregularidades fiscais
    """)

    st.markdown("---")

    # Carregar dados de score NEAF
    with st.spinner("Carregando dados de NEAF..."):
        df_score_neaf = carregar_score_neaf(engine, limite=500)

    if df_score_neaf is not None and not df_score_neaf.empty:
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_empresas = len(df_score_neaf)
            st.markdown(f"""
            <div class='metric-card-red'>
                <h3>{total_empresas:,}</h3>
                <p>Empresas com Indícios</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            total_indicios = df_score_neaf['qtd_total_indicios'].sum()
            st.markdown(f"""
            <div class='metric-card-yellow'>
                <h3>{total_indicios:,}</h3>
                <p>Total de Indícios</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            media_score = df_score_neaf['score_risco_neaf'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{media_score:.2f}</h3>
                <p>Score Médio de Risco</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            criticos = (df_score_neaf['classificacao_risco_neaf'] == 'CRÍTICO').sum()
            st.markdown(f"""
            <div class='metric-card-red'>
                <h3>{criticos}</h3>
                <p>Risco Crítico</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Gráficos
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Distribuição por Classificação de Risco")
            class_counts = df_score_neaf['classificacao_risco_neaf'].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                color=class_counts.index,
                color_discrete_map={
                    'CRÍTICO': '#d32f2f',
                    'ALTO': '#f57c00',
                    'MODERADO': '#fbc02d',
                    'BAIXO': '#689f38'
                },
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🏭 Top 10 Setores com Mais Indícios")
            setor_indicios = df_score_neaf.groupby('setor')['qtd_total_indicios'].sum().nlargest(10)
            fig = px.bar(
                x=setor_indicios.values,
                y=setor_indicios.index,
                orientation='h',
                color=setor_indicios.values,
                color_continuous_scale='Reds',
                labels={'x': 'Quantidade de Indícios', 'y': 'Setor'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        st.markdown("---")
        st.markdown("### 🎯 Análise Multidimensional de Risco NEAF")

        fig = px.scatter(
            df_score_neaf,
            x='qtd_total_indicios',
            y='qtd_tipos_indicios_distintos',
            size='score_risco_neaf',
            color='classificacao_risco_neaf',
            hover_data=['nm_razao_social', 'setor', 'cd_uf'],
            color_discrete_map={
                'CRÍTICO': '#d32f2f',
                'ALTO': '#f57c00',
                'MODERADO': '#fbc02d',
                'BAIXO': '#689f38'
            },
            labels={
                'qtd_total_indicios': 'Quantidade Total de Indícios',
                'qtd_tipos_indicios_distintos': 'Tipos Distintos de Indícios',
                'classificacao_risco_neaf': 'Classificação'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Tabela detalhada
        st.markdown("---")
        st.markdown("### 📋 Empresas com Indícios NEAF")

        # Filtros
        col1, col2, col3 = st.columns(3)

        with col1:
            classificacoes = ['Todas'] + list(df_score_neaf['classificacao_risco_neaf'].dropna().unique())
            filtro_class = st.selectbox("Classificação de Risco", classificacoes)

        with col2:
            ufs_neaf = ['Todas'] + sorted(df_score_neaf['cd_uf'].dropna().unique().tolist())
            filtro_uf = st.selectbox("UF", ufs_neaf, key='uf_neaf')

        with col3:
            min_indicios = st.number_input("Mínimo de Indícios", min_value=0, value=1)

        # Aplicar filtros
        df_filtrado_neaf = df_score_neaf.copy()

        if filtro_class != 'Todas':
            df_filtrado_neaf = df_filtrado_neaf[df_filtrado_neaf['classificacao_risco_neaf'] == filtro_class]

        if filtro_uf != 'Todas':
            df_filtrado_neaf = df_filtrado_neaf[df_filtrado_neaf['cd_uf'] == filtro_uf]

        df_filtrado_neaf = df_filtrado_neaf[df_filtrado_neaf['qtd_total_indicios'] >= min_indicios]

        st.info(f"**{len(df_filtrado_neaf)} empresas** encontradas com os filtros aplicados")

        df_exibir = df_filtrado_neaf[[
            'nm_razao_social', 'cnpj', 'setor', 'cd_uf',
            'qtd_total_indicios', 'qtd_tipos_indicios_distintos',
            'score_risco_neaf', 'classificacao_risco_neaf'
        ]].copy()

        df_exibir.columns = [
            'Razão Social', 'CNPJ', 'Setor', 'UF',
            'Total Indícios', 'Tipos Indícios',
            'Score Risco', 'Classificação'
        ]

        df_exibir = limpar_dataframe_para_exibicao(df_exibir)
        st.dataframe(
            df_exibir.style.format({
                'Score Risco': '{:.2f}'
            }).background_gradient(subset=['Score Risco'], cmap='Reds')
              .background_gradient(subset=['Total Indícios'], cmap='OrRd'),
            use_container_width=True,
            height=500
        )

        # Exportar
        if st.button("📥 Exportar Lista NEAF (CSV)"):
            csv = df_filtrado_neaf.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"neaf_indicios_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.warning("Não há dados de NEAF disponíveis ou a tabela ainda não foi populada.")

# ---------------------------------------------------------------------------
# PÁGINA: INCONSISTÊNCIAS CONTÁBEIS
# ---------------------------------------------------------------------------

elif pagina == "⚖️ Inconsistências Contábeis":
    st.markdown("<h1 class='main-header'>⚖️ Análise de Inconsistências Contábeis</h1>", unsafe_allow_html=True)

    st.markdown("""
    ### Detecção de Anomalias na Equação Patrimonial e Variações de Contas

    Este módulo analisa:
    - ⚖️ **Equação Contábil**: Ativo ≠ Passivo + Patrimônio Líquido
    - 📊 **Variações Anômalas**: Mudanças abruptas em contas específicas
    - 🔍 **Scores de Risco**: Classificação por gravidade da inconsistência
    """)

    st.markdown("---")

    # Tabs para diferentes análises
    tab1, tab2 = st.tabs(["⚖️ Equação Patrimonial", "📊 Variações de Contas"])

    with tab1:
        st.markdown("### Inconsistências na Equação Patrimonial")
        st.info("**Regra Básica:** Ativo Total = Passivo Total + Patrimônio Líquido")

        with st.spinner("Carregando inconsistências..."):
            df_equacao = carregar_inconsistencias_equacao(engine, ano=ano_selecionado, limite=500)

        if df_equacao is not None and not df_equacao.empty:
            # Métricas
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_incons = len(df_equacao)
                st.metric("Empresas com Inconsistência", total_incons)

            with col2:
                media_perc = df_equacao['percentual_diferenca'].mean()
                st.metric("Diferença Média (%)", f"{media_perc:.2f}%")

            with col3:
                criticos = (df_equacao['classificacao_inconsistencia'] == 'Crítica').sum()
                st.metric("Inconsistências Críticas", criticos)

            with col4:
                media_score = df_equacao['score_risco_equacao'].mean()
                st.metric("Score Médio", f"{media_score:.2f}")

            st.markdown("---")

            # Gráficos
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Distribuição por Classificação")
                class_counts = df_equacao['classificacao_inconsistencia'].value_counts()
                fig = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    color=class_counts.index,
                    color_discrete_map={
                        'Crítica': '#d32f2f',
                        'Alta': '#f57c00',
                        'Moderada': '#fbc02d',
                        'Diferença Mínima': '#689f38'
                    }
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Top 15 por Diferença Absoluta")
                df_equacao['diferenca_absoluta'] = pd.to_numeric(df_equacao['diferenca_absoluta'], errors='coerce').fillna(0)
                df_top = df_equacao.nlargest(15, 'diferenca_absoluta')

                fig = px.bar(
                    df_top,
                    x='diferenca_absoluta',
                    y='nm_razao_social',
                    orientation='h',
                    color='score_risco_equacao',
                    color_continuous_scale='Reds',
                    labels={'diferenca_absoluta': 'Diferença (R$)', 'nm_razao_social': 'Empresa'}
                )
                fig.update_layout(height=500)
                fig.update_yaxes(tickfont=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)

            # Tabela
            st.markdown("---")
            st.markdown("#### 📋 Detalhamento das Inconsistências")

            df_exibir = df_equacao[[
                'nm_razao_social', 'cnpj', 'setor', 'cd_uf',
                'ativo_total', 'passivo_pl_total', 'diferenca_absoluta',
                'percentual_diferenca', 'classificacao_inconsistencia', 'score_risco_equacao'
            ]].copy()

            df_exibir.columns = [
                'Razão Social', 'CNPJ', 'Setor', 'UF',
                'Ativo Total', 'Passivo + PL', 'Diferença',
                '% Diferença', 'Classificação', 'Score'
            ]

            df_exibir = limpar_dataframe_para_exibicao(df_exibir)
            st.dataframe(
                df_exibir.style.format({
                    'Ativo Total': 'R$ {:,.2f}',
                    'Passivo + PL': 'R$ {:,.2f}',
                    'Diferença': 'R$ {:,.2f}',
                    '% Diferença': '{:.2f}%',
                    'Score': '{:.2f}'
                }).background_gradient(subset=['Score'], cmap='Reds'),
                use_container_width=True,
                height=400
            )
        else:
            st.success("✅ Nenhuma inconsistência significativa na equação patrimonial encontrada!")

    with tab2:
        st.markdown("### Variações Anômalas de Contas")
        st.info("**Detecta:** Mudanças de mais de 100% ou reduções acima de 50% entre anos")

        with st.spinner("Carregando variações anômalas..."):
            df_variacoes = carregar_inconsistencias_variacoes(engine, ano=ano_selecionado, limite=500)

        if df_variacoes is not None and not df_variacoes.empty:
            # Métricas
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_var = len(df_variacoes)
                st.metric("Variações Detectadas", total_var)

            with col2:
                empresas_afetadas = df_variacoes['cnpj'].nunique()
                st.metric("Empresas Afetadas", empresas_afetadas)

            with col3:
                criticos = (df_variacoes['classificacao_variacao'] == 'Variação Extrema').sum()
                st.metric("Variações Críticas", criticos)

            with col4:
                media_var = df_variacoes['variacao_percentual'].abs().mean()
                st.metric("Variação Média (%)", f"{media_var:.1f}%")

            st.markdown("---")

            # Gráficos
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Distribuição por Classificação")
                class_counts = df_variacoes['classificacao_variacao'].value_counts()
                fig = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    color=class_counts.index,
                    color_discrete_map={
                        'Variação Extrema': '#d32f2f',
                        'Variação Muito Alta': '#f57c00',
                        'Variação Alta': '#fbc02d'
                    }
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Distribuição por Setor")
                setor_counts = df_variacoes['setor'].value_counts().head(10)
                fig = px.bar(
                    x=setor_counts.values,
                    y=setor_counts.index,
                    orientation='h',
                    color=setor_counts.values,
                    color_continuous_scale='OrRd',
                    labels={'x': 'Quantidade', 'y': 'Setor'}
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Tabela
            st.markdown("---")
            st.markdown("#### 📋 Detalhamento das Variações Anômalas")

            df_exibir = df_variacoes[[
                'nm_razao_social', 'cnpj', 'setor', 'cd_conta',
                'saldo_anterior', 'saldo_atual', 'variacao_absoluta',
                'variacao_percentual', 'classificacao_variacao', 'score_risco_variacao'
            ]].copy()

            df_exibir.columns = [
                'Razão Social', 'CNPJ', 'Setor', 'Conta',
                'Saldo Anterior', 'Saldo Atual', 'Variação Abs.',
                'Variação %', 'Classificação', 'Score'
            ]

            df_exibir = limpar_dataframe_para_exibicao(df_exibir)
            st.dataframe(
                df_exibir.style.format({
                    'Saldo Anterior': 'R$ {:,.2f}',
                    'Saldo Atual': 'R$ {:,.2f}',
                    'Variação Abs.': 'R$ {:,.2f}',
                    'Variação %': '{:.1f}%',
                    'Score': '{:.2f}'
                }).background_gradient(subset=['Score'], cmap='Reds'),
                use_container_width=True,
                height=400
            )
        else:
            st.success("✅ Nenhuma variação anômala significativa encontrada!")

# ---------------------------------------------------------------------------
# PÁGINA: BENCHMARK SETORIAL
# ---------------------------------------------------------------------------

elif pagina == "📈 Benchmark Setorial":
    st.markdown("<h1 class='main-header'>📈 Benchmark Setorial</h1>", unsafe_allow_html=True)

    st.markdown("""
    ### Análise Comparativa por Setor Econômico (CNAE)

    Compare indicadores financeiros entre setores:
    - 📊 **Médias setoriais** de ativo, receita e indicadores
    - 📈 **Percentis** para identificar outliers
    - 🏆 **Ranking** de desempenho por setor
    """)

    st.markdown("---")

    # Carregar benchmark
    with st.spinner("Carregando benchmark setorial..."):
        df_benchmark = carregar_benchmark_setorial(engine, ano=ano_selecionado)

    if df_benchmark is not None and not df_benchmark.empty:
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_setores = len(df_benchmark)
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{total_setores}</h3>
                <p>Setores Analisados</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            total_empresas = df_benchmark['qtd_empresas_setor'].sum()
            st.markdown(f"""
            <div class='metric-card-blue'>
                <h3>{total_empresas:,}</h3>
                <p>Total de Empresas</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            media_liquidez = df_benchmark['media_liquidez_corrente_setor'].mean()
            st.markdown(f"""
            <div class='metric-card-green'>
                <h3>{media_liquidez:.2f}</h3>
                <p>Liquidez Média Geral</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            media_margem = df_benchmark['media_margem_liquida_setor'].mean()
            st.markdown(f"""
            <div class='metric-card-yellow'>
                <h3>{media_margem:.2f}%</h3>
                <p>Margem Média Geral</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Seletor de indicador
        indicador_bench = st.selectbox(
            "Selecione o indicador para análise",
            [
                'Ativo Total Médio',
                'Receita Líquida Média',
                'Liquidez Corrente',
                'Endividamento',
                'Margem Líquida',
                'ROE'
            ]
        )

        # Mapear para coluna
        mapa_bench = {
            'Ativo Total Médio': 'media_ativo_total_setor',
            'Receita Líquida Média': 'media_receita_liquida_setor',
            'Liquidez Corrente': 'media_liquidez_corrente_setor',
            'Endividamento': 'media_endividamento_setor',
            'Margem Líquida': 'media_margem_liquida_setor',
            'ROE': 'media_roe_setor'
        }

        coluna_bench = mapa_bench[indicador_bench]

        # Gráficos
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 🏆 Top 15 Setores - {indicador_bench}")
            df_benchmark[coluna_bench] = pd.to_numeric(df_benchmark[coluna_bench], errors='coerce').fillna(0)
            df_top_bench = df_benchmark.nlargest(15, coluna_bench)

            fig = px.bar(
                df_top_bench,
                x=coluna_bench,
                y='cnae_divisao_descricao',
                orientation='h',
                color=coluna_bench,
                color_continuous_scale='Viridis',
                labels={coluna_bench: indicador_bench, 'cnae_divisao_descricao': 'Setor'}
            )
            fig.update_layout(height=600, showlegend=False)
            fig.update_yaxes(tickfont=dict(size=9))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🏭 Empresas por Setor (Top 15)")
            df_benchmark['qtd_empresas_setor'] = pd.to_numeric(df_benchmark['qtd_empresas_setor'], errors='coerce').fillna(0)
            df_top_emp = df_benchmark.nlargest(15, 'qtd_empresas_setor')

            fig = px.bar(
                df_top_emp,
                x='qtd_empresas_setor',
                y='cnae_divisao_descricao',
                orientation='h',
                color='qtd_empresas_setor',
                color_continuous_scale='Blues',
                labels={'qtd_empresas_setor': 'Quantidade de Empresas', 'cnae_divisao_descricao': 'Setor'}
            )
            fig.update_layout(height=600, showlegend=False)
            fig.update_yaxes(tickfont=dict(size=9))
            st.plotly_chart(fig, use_container_width=True)

        # Scatter comparativo
        st.markdown("---")
        st.markdown("### 📊 Análise Comparativa de Setores")

        fig = px.scatter(
            df_benchmark,
            x='media_liquidez_corrente_setor',
            y='media_margem_liquida_setor',
            size='qtd_empresas_setor',
            color='media_roe_setor',
            hover_name='cnae_divisao_descricao',
            color_continuous_scale='RdYlGn',
            labels={
                'media_liquidez_corrente_setor': 'Liquidez Corrente Média',
                'media_margem_liquida_setor': 'Margem Líquida Média (%)',
                'media_roe_setor': 'ROE Médio (%)'
            }
        )
        fig.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="Liquidez = 1")
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Margem = 0%")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Tabela completa
        st.markdown("---")
        st.markdown("### 📋 Tabela Completa de Benchmark Setorial")

        # Filtro de setor
        setores_disp = ['Todos'] + sorted(df_benchmark['cnae_divisao_descricao'].dropna().unique().tolist())
        setor_filtro = st.selectbox("Filtrar por Setor", setores_disp)

        df_filtrado_bench = df_benchmark.copy()
        if setor_filtro != 'Todos':
            df_filtrado_bench = df_filtrado_bench[df_filtrado_bench['cnae_divisao_descricao'] == setor_filtro]

        df_exibir = df_filtrado_bench[[
            'cnae_divisao_descricao', 'qtd_empresas_setor',
            'media_ativo_total_setor', 'media_receita_liquida_setor',
            'media_liquidez_corrente_setor', 'media_endividamento_setor',
            'media_margem_liquida_setor', 'media_roe_setor'
        ]].copy()

        df_exibir.columns = [
            'Setor', 'Qtd Empresas',
            'Ativo Médio', 'Receita Média',
            'Liquidez', 'Endividamento',
            'Margem Líquida %', 'ROE %'
        ]

        df_exibir = limpar_dataframe_para_exibicao(df_exibir)
        st.dataframe(
            df_exibir.style.format({
                'Qtd Empresas': '{:,.0f}',
                'Ativo Médio': 'R$ {:,.0f}',
                'Receita Média': 'R$ {:,.0f}',
                'Liquidez': '{:.2f}',
                'Endividamento': '{:.2f}',
                'Margem Líquida %': '{:.2f}',
                'ROE %': '{:.2f}'
            }).background_gradient(subset=['Liquidez'], cmap='RdYlGn', vmin=0, vmax=2)
              .background_gradient(subset=['Margem Líquida %'], cmap='RdYlGn', vmin=-10, vmax=20),
            use_container_width=True,
            height=500
        )

        # Exportar
        st.markdown("---")
        if st.button("📥 Exportar Benchmark (CSV)"):
            csv = df_benchmark.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"benchmark_setorial_{ano_selecionado}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.warning("Não há dados de benchmark setorial disponíveis ou a tabela ainda não foi populada.")

# =============================================================================
# 11. RODAPÉ
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p><strong>Sistema ECD - Escrituração Contábil Digital</strong></p>
    <p>Receita Estadual de Santa Catarina | Versão 2.1</p>
    <p><small>Atualizado com análises NEAF, Inconsistências e Benchmark Setorial</small></p>
</div>
""", unsafe_allow_html=True)