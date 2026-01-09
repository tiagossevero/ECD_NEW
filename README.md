# ECD Dashboard - Escrituração Contábil Digital

Sistema de análise e monitoramento de escriturações contábeis digitais para detecção de irregularidades fiscais e análise de risco empresarial.

**Versão:** 2.0
**Organização:** Receita Estadual de Santa Catarina (SEFAZ-SC)

---

## Sobre o Projeto

O ECD Dashboard é uma aplicação web desenvolvida em Streamlit para análise inteligente de dados contábeis submetidos através da Escrituração Contábil Digital. O sistema oferece:

- Análise financeira completa de empresas e setores econômicos
- Detecção de anomalias utilizando algoritmos de Machine Learning
- Identificação de empresas de alto risco fiscal
- Benchmarking setorial e comparativo
- Análise de inconsistências contábeis

---

## Funcionalidades

### 1. Visão Geral
- Métricas consolidadas de todas as empresas analisadas
- Distribuição por setor econômico (CNAE)
- Indicadores financeiros agregados

### 2. Análise por Setor
- Análise aprofundada por divisão CNAE
- Classificação de risco setorial
- Indicadores de saúde financeira por setor

### 3. Detalhamento de Empresa
- Perfil financeiro individual por CNPJ
- Balanço Patrimonial (Ativo/Passivo/Patrimônio Líquido)
- Demonstração do Resultado do Exercício (DRE)
- Indicadores financeiros calculados
- Comparativo multi-período

### 4. Fiscalização Inteligente (Machine Learning)
- **Isolation Forest** para detecção de outliers
- **K-Means Clustering** para agrupamento de empresas
- Score combinado de risco (ML + tradicional)
- Classificação de prioridade (Baixa/Média/Alta/Crítica)

### 5. Empresas de Alto Risco
- Listagem filtrada de empresas com alto score de risco
- Fatores de risco detalhados
- Análise em lote (500-1000 empresas)

### 6. Indicadores Financeiros
Dashboard com indicadores-chave:
- **Liquidez Corrente:** Ativo Circulante ÷ Passivo Circulante
- **Endividamento:** Passivo Total ÷ Ativo Total
- **Margem Líquida:** Lucro Líquido ÷ Receita Líquida
- **ROA:** Lucro Líquido ÷ Ativo Total
- **ROE:** Lucro Líquido ÷ Patrimônio Líquido

### 7. Plano de Contas
- Análise da estrutura de contas
- Hierarquia de 3 níveis
- Tendências de utilização

### 8. Indícios NEAF
- Indicadores de Auditoria Fiscal (NEAF)
- Classificação: CRÍTICO/ALTO/MÉDIO/BAIXO
- Distribuição de scores e tendências

### 9. Inconsistências Contábeis
- **Equação Patrimonial:** Verificação de Ativo = Passivo + PL
- **Variações de Contas:** Análise de anomalias em variações

### 10. Benchmark Setorial
- Comparativo de desempenho entre setores
- Médias e desvios padrão setoriais
- Exportação de dados (CSV)

---

## Tecnologias Utilizadas

| Categoria | Tecnologia |
|-----------|------------|
| **Framework Web** | Streamlit |
| **Linguagem** | Python 3 |
| **Processamento de Dados** | Pandas, NumPy |
| **Visualização** | Plotly (Express, Graph Objects) |
| **Machine Learning** | scikit-learn (Isolation Forest, K-Means, Random Forest) |
| **Banco de Dados** | Apache Impala via SQLAlchemy |
| **Autenticação** | LDAP com SSL/TLS |

---

## Requisitos

### Dependências Python

```txt
streamlit
pandas
numpy
plotly
sqlalchemy
scikit-learn
joblib
```

### Pré-requisitos de Infraestrutura

- Acesso à rede do servidor Impala (`bdaworkernode02.sef.sc.gov.br:21050`)
- Credenciais LDAP válidas para autenticação
- Python 3.8 ou superior

---

## Instalação

### 1. Clone o Repositório

```bash
git clone https://github.com/tiagossevero/ECD_NEW.git
cd ECD_NEW
```

### 2. Crie um Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as Dependências

```bash
pip install streamlit pandas numpy plotly sqlalchemy scikit-learn joblib impyla
```

### 4. Configure as Credenciais

Crie o arquivo de secrets do Streamlit:

```bash
mkdir -p ~/.streamlit
```

Crie o arquivo `~/.streamlit/secrets.toml`:

```toml
[impala_credentials]
user = "seu_usuario_ldap"
password = "sua_senha_ldap"
```

---

## Uso

### Executar a Aplicação

```bash
streamlit run "ECD (4).py"
```

A aplicação estará disponível em `http://localhost:8501`

### Acesso

A aplicação requer autenticação com a senha de acesso configurada no sistema.

---

## Estrutura do Projeto

```
ECD_NEW/
├── ECD (4).py          # Aplicação principal Streamlit (3.985 linhas)
├── ECD.json            # Backup de queries SQL do Hue
├── README.md           # Este arquivo
└── .git/               # Repositório Git
```

---

## Banco de Dados

### Conexão

- **Host:** bdaworkernode02.sef.sc.gov.br
- **Porta:** 21050
- **Database:** teste
- **Engine:** Apache Impala

### Tabelas Principais

| Tabela | Descrição |
|--------|-----------|
| `ecd_empresas_cadastro` | Cadastro de empresas com classificação CNAE |
| `ecd_indicadores_financeiros` | Indicadores financeiros calculados por setor |
| `ecd_balanco_patrimonial` | Dados do Balanço Patrimonial |
| `ecd_dre` | Demonstração do Resultado do Exercício |
| `ecd_saldos_contas_v2` | Saldos de contas contábeis |
| `ecd_plano_contas` | Plano de contas com hierarquia |
| `ecd_score_risco_consolidado` | Scores de risco consolidados |
| `ecd_neaf_indicios` | Indicadores NEAF para detecção de fraude |
| `ecd_neaf_score_risco` | Classificação de risco NEAF |
| `ecd_inconsistencias_equacao` | Inconsistências na equação patrimonial |
| `ecd_inconsistencias_variacoes` | Anomalias em variações de contas |
| `ecd_benchmark_setorial` | Benchmarks e comparativos setoriais |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    Fontes Externas                          │
│              (SEFAZ, Submissões ECD, CNAE)                  │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Apache Impala Data Warehouse                  │
│         (bdaworkernode02.sef.sc.gov.br:21050)               │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  usr_sat_ecd.*   │  │         teste database           │ │
│  │  (dados brutos)  │  │    (tabelas processadas ECD)     │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Python/SQLAlchemy                          │
│              (Conexão e ETL de dados)                       │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Aplicação Streamlit                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Pandas    │  │  Plotly     │  │   scikit-learn      │  │
│  │ (Processar) │  │ (Visualizar)│  │  (ML/Anomalias)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Dashboard Interativo (Browser)                 │
│     10 páginas de análise com filtros e visualizações       │
└─────────────────────────────────────────────────────────────┘
```

---

## Algoritmos de Machine Learning

### Isolation Forest
Utilizado para detectar **outliers financeiros** - empresas com comportamento atípico em relação ao seu setor.

### K-Means Clustering
Agrupa empresas com **perfis financeiros similares**, permitindo identificar grupos de risco e comparar empresas com pares.

### Score de Risco Combinado
Combina:
- Indicadores financeiros tradicionais
- Score de anomalia do Isolation Forest
- Indicadores NEAF
- Inconsistências contábeis

---

## Cache e Performance

A aplicação utiliza cache TTL do Streamlit para otimizar performance:

- **TTL padrão:** 3600 segundos (1 hora)
- **Cache separado** para dados de empresas e métricas agregadas
- **Resource caching** para conexão com banco de dados

---

## Glossário

| Termo | Significado |
|-------|-------------|
| **ECD** | Escrituração Contábil Digital |
| **CNAE** | Classificação Nacional de Atividades Econômicas |
| **NEAF** | Núcleo Especializado de Auditoria Fiscal |
| **SEFAZ** | Secretaria de Estado da Fazenda |
| **DRE** | Demonstração do Resultado do Exercício |
| **ROA** | Retorno sobre Ativos (Return on Assets) |
| **ROE** | Retorno sobre Patrimônio (Return on Equity) |

---

## Contribuição

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas alterações (`git commit -m 'Adicionar nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

---

## Histórico de Versões

| Versão | Data | Alterações |
|--------|------|------------|
| 2.0 | 2024 | Dashboard Streamlit com ML |
| 2.1 | 2024 | Correção de JOINs e formato ano_referencia |
| 2.2 | 2024 | Tooltips, UX e correção de erros de carregamento |

---

## Licença

Este projeto é de uso interno da Receita Estadual de Santa Catarina (SEFAZ-SC).

---

## Contato

**Equipe de Desenvolvimento**
Receita Estadual de Santa Catarina
SEFAZ-SC
