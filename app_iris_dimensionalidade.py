# -*- coding: utf-8 -*-
"""
=============================================================================
ESTUDO DE REDUÇÃO DE DIMENSIONALIDADE E SEPARABILIDADE - DATASET IRIS
=============================================================================
Sistema interativo em Streamlit para comparar:
  - Dados Originais (4D)
  - PCA (não supervisionado, 4D -> 2D)
  - LDA (supervisionado, 4D -> 2D)

Métricas avaliadas:
  - Separabilidade geométrica (Silhouette Score)
  - Performance de classificadores (KNN e Regressão Logística)

Autor: Engenheiro de ML / Dev Python Sênior (gerado via Claude)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =============================================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Redução de Dimensionalidade - Iris",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42  # Semente fixa para reprodutibilidade de todos os experimentos


# =============================================================================
# FUNÇÕES DE CARREGAMENTO E CACHE
# =============================================================================
@st.cache_data
def carregar_dados_iris():
    """
    Carrega o Dataset Iris nativo do scikit-learn e o organiza em um
    DataFrame do pandas, incluindo a coluna de classes (espécie) já
    traduzida para texto legível.
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["especie"] = df["target"].map(
        {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    )
    return df, iris.feature_names, iris.target_names


@st.cache_data
def padronizar_dados(X):
    """
    Padroniza (z-score) as features antes da redução de dimensionalidade.
    Essencial para PCA, pois a técnica é sensível à escala das variáveis.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


@st.cache_data
def aplicar_pca(X_scaled, n_componentes=2):
    """
    Aplica PCA (Análise de Componentes Principais) - método NÃO supervisionado.
    O PCA busca as direções de máxima variância dos dados, sem considerar
    os rótulos das classes.
    """
    pca = PCA(n_components=n_componentes, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    variancia_explicada = pca.explained_variance_ratio_
    return X_pca, variancia_explicada, pca


@st.cache_data
def aplicar_lda(X_scaled, y, n_componentes=2):
    """
    Aplica LDA (Linear Discriminant Analysis) - método SUPERVISIONADO.
    O LDA busca as direções que MAXIMIZAM a separação entre classes,
    pois utiliza os rótulos (y) durante o ajuste (fit).

    IMPORTANTE: por ser supervisionado, o LDA é mais suscetível a
    vazamento de dados (data leakage) se for ajustado usando o
    conjunto de teste. Nesta aplicação, o LDA usado para VISUALIZAÇÃO
    é ajustado no dataset completo (fins didáticos/exploratórios),
    mas na seção de CLASSIFICAÇÃO (item 4) o ajuste é feito
    estritamente sobre o conjunto de TREINO, conforme boas práticas.
    """
    lda = LDA(n_components=n_componentes)
    X_lda = lda.fit_transform(X_scaled, y)
    return X_lda, lda


def calcular_silhouette(X, y):
    """
    Calcula o Coeficiente de Silhueta para um conjunto de dados X
    dado os rótulos de classe y (rótulos reais utilizados como
    referência dos clusters "ideais").

    O Silhouette Score varia de -1 a +1:
      +1 -> clusters densos e bem separados (alta separabilidade)
       0 -> clusters sobrepostos/indistintos
      -1 -> amostras possivelmente atribuídas ao cluster errado
    """
    return silhouette_score(X, y)


# =============================================================================
# FUNÇÕES DE TREINAMENTO E AVALIAÇÃO DE CLASSIFICADORES
# =============================================================================
def treinar_avaliar_classificador(modelo, X_train, X_test, y_train, y_test):
    """
    Treina um classificador e retorna um dicionário com as métricas
    de performance (Acurácia, Precisão, Recall, F1-Score) calculadas
    no conjunto de TESTE, além da matriz de confusão.

    Usa average='macro' nas métricas multiclasse para dar peso igual
    às 3 classes do Iris, independentemente do tamanho de cada uma.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    matriz_confusao = confusion_matrix(y_test, y_pred)
    return metricas, matriz_confusao, y_pred


def preparar_dados_treino_teste(X, y, test_size=0.3):
    """
    Realiza a divisão treino/teste de forma estratificada (mantendo a
    proporção de classes em ambos os conjuntos). A estratificação é
    importante no Iris pois o dataset é perfeitamente balanceado
    (50 amostras por classe) e queremos preservar esse balanceamento.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def pipeline_classificacao_sem_vazamento(
    X_raw, y, modo, n_componentes, scaler_global, test_size=0.3
):
    """
    Pipeline CORRETO de classificação que evita vazamento de dados (data leakage).

    Regra de ouro: qualquer transformação que "aprende" parâmetros a partir
    dos dados (StandardScaler, PCA, LDA) deve ser ajustada (fit) APENAS
    com o conjunto de TREINO. O conjunto de teste é apenas transformado
    (transform) com os parâmetros já aprendidos.

    Parâmetros:
        X_raw: features originais (4D), ainda não padronizadas
        y: rótulos de classe
        modo: "original", "pca" ou "lda"
        n_componentes: número de componentes para PCA/LDA
        scaler_global: não utilizado diretamente (mantido por clareza de fluxo)

    Retorna:
        X_train_final, X_test_final, y_train, y_test
    """
    # 1) Divisão treino/teste ANTES de qualquer ajuste de transformação
    X_train_raw, X_test_raw, y_train, y_test = preparar_dados_treino_teste(
        X_raw, y, test_size=test_size
    )

    # 2) Padronização: fit SOMENTE no treino, transform no treino e no teste
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    if modo == "original":
        return X_train_scaled, X_test_scaled, y_train, y_test

    elif modo == "pca":
        # PCA: fit SOMENTE no treino (não supervisionado, mas ainda assim
        # deve respeitar a fronteira treino/teste)
        pca = PCA(n_components=n_componentes, random_state=RANDOM_STATE)
        X_train_reduzido = pca.fit_transform(X_train_scaled)
        X_test_reduzido = pca.transform(X_test_scaled)
        return X_train_reduzido, X_test_reduzido, y_train, y_test

    elif modo == "lda":
        # LDA: fit SOMENTE no treino, usando os rótulos de TREINO.
        # Isso é crítico: usar y_test aqui seria vazamento de dados grave,
        # pois o modelo "veria" informação da classe do conjunto de teste
        # antes mesmo de ser avaliado.
        lda = LDA(n_components=n_componentes)
        X_train_reduzido = lda.fit_transform(X_train_scaled, y_train)
        X_test_reduzido = lda.transform(X_test_scaled)
        return X_train_reduzido, X_test_reduzido, y_train, y_test

    else:
        raise ValueError(f"Modo inválido: {modo}")


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================
def plotar_scatter_2d(X_2d, especies, titulo, eixo_x, eixo_y):
    """Gera um scatter plot 2D interativo (Plotly) colorido por classe."""
    df_plot = pd.DataFrame(X_2d, columns=[eixo_x, eixo_y])
    df_plot["Espécie"] = especies.values

    fig = px.scatter(
        df_plot,
        x=eixo_x,
        y=eixo_y,
        color="Espécie",
        title=titulo,
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    fig.update_layout(
        height=450,
        legend_title_text="Espécie",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plotar_matriz_confusao(matriz, classes, titulo):
    """Gera um heatmap interativo (Plotly) para a matriz de confusão."""
    fig = px.imshow(
        matriz,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Previsto", y="Real", color="Qtd."),
        title=titulo,
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plotar_pairplot_plotly(df, feature_names):
    """
    Gera uma matriz de dispersão (equivalente ao pairplot do Seaborn)
    usando Plotly Express, permitindo interatividade (zoom, hover, etc.).
    """
    fig = px.scatter_matrix(
        df,
        dimensions=feature_names,
        color="especie",
        color_discrete_sequence=px.colors.qualitative.Set1,
        title="Matriz de Dispersão - Variáveis Originais (4D)",
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=True, marker=dict(size=4))
    fig.update_layout(height=750, legend_title_text="Espécie")
    return fig


# =============================================================================
# CARREGAMENTO INICIAL DOS DADOS
# =============================================================================
df, feature_names, target_names = carregar_dados_iris()
X_raw = df[feature_names].values
y = df["target"].values
especies = df["especie"]

X_scaled_full = padronizar_dados(X_raw)
X_pca_full, var_explicada, pca_model_full = aplicar_pca(X_scaled_full)
X_lda_full, lda_model_full = aplicar_lda(X_scaled_full, y)


# =============================================================================
# BARRA LATERAL (SIDEBAR) - CONTROLES INTERATIVOS
# =============================================================================
st.sidebar.title("⚙️ Painel de Controle")
st.sidebar.markdown("---")

st.sidebar.subheader("🔧 Hiperparâmetros")

k_vizinhos = st.sidebar.slider(
    "Número de vizinhos (K) - KNN",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Define quantos vizinhos mais próximos o algoritmo KNN consulta para classificar uma nova amostra.",
)

test_size_pct = st.sidebar.slider(
    "Tamanho do conjunto de teste (%)",
    min_value=10,
    max_value=50,
    value=30,
    step=5,
    help="Percentual dos dados reservado para teste (avaliação fora da amostra de treino).",
)
test_size = test_size_pct / 100.0

c_logreg = st.sidebar.slider(
    "Regularização (C) - Regressão Logística",
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.01,
    help="Inverso da força de regularização. Valores menores aumentam a regularização (modelo mais simples).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Sobre o Dataset")
st.sidebar.info(
    "**Iris Dataset**\n\n"
    "- 150 amostras\n"
    "- 4 features (cm): comprimento/largura de sépala e pétala\n"
    "- 3 classes: Setosa, Versicolor, Virginica\n"
    "- 50 amostras por classe (balanceado)"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Desenvolvido como estudo acadêmico sobre Redução de "
    "Dimensionalidade (PCA vs LDA) e Separabilidade de Classes."
)


# =============================================================================
# CABEÇALHO PRINCIPAL
# =============================================================================
st.title("🌸 Redução de Dimensionalidade e Separabilidade — Dataset Iris")
st.markdown(
    """
    Este aplicativo demonstra, de forma interativa, **como diferentes técnicas
    de redução de dimensionalidade** (PCA — não supervisionado — e LDA —
    supervisionado) afetam a **separabilidade geométrica das classes** e a
    **performance de classificadores** treinados sobre o Dataset Iris.
    """
)

st.markdown("---")


# =============================================================================
# ABAS PRINCIPAIS (TABS)
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 1. Exploração dos Dados",
        "🔬 2. PCA vs LDA",
        "📐 3. Separabilidade (Silhouette)",
        "🤖 4. Performance dos Classificadores",
    ]
)


# -----------------------------------------------------------------------
# TAB 1: CARREGAMENTO E ANÁLISE EXPLORATÓRIA
# -----------------------------------------------------------------------
with tab1:
    st.header("📊 Carregamento e Análise Exploratória")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de amostras", len(df))
    col2.metric("Número de features", len(feature_names))
    col3.metric("Número de classes", df["especie"].nunique())
    col4.metric("Amostras por classe", "50 (balanceado)")

    st.markdown("### 🗃️ Tabela Interativa - Dados Originais")
    st.dataframe(
        df.drop(columns=["target"]).rename(
            columns={c: c.replace(" (cm)", "").title() for c in feature_names}
        ),
        use_container_width=True,
        height=300,
    )

    st.markdown("### 🔍 Estatísticas Descritivas")
    st.dataframe(df[feature_names].describe().T, use_container_width=True)

    st.markdown("### 🌐 Matriz de Dispersão (Pairplot) - 4 Dimensões Originais")
    st.markdown(
        """
        O gráfico abaixo evidencia a **sobreposição parcial** entre as classes
        *Versicolor* e *Virginica* em quase todas as combinações de variáveis,
        enquanto *Setosa* já se mostra claramente separável mesmo no espaço
        original de 4 dimensões.
        """
    )
    fig_pairplot = plotar_pairplot_plotly(df, feature_names)
    st.plotly_chart(fig_pairplot, use_container_width=True)


# -----------------------------------------------------------------------
# TAB 2: REDUÇÃO DE DIMENSIONALIDADE (PCA vs LDA)
# -----------------------------------------------------------------------
with tab2:
    st.header("🔬 Redução de Dimensionalidade: PCA vs LDA")

    st.markdown(
        """
        - **PCA (Principal Component Analysis):** técnica **não supervisionada**
          que projeta os dados nas direções de **máxima variância**, ignorando
          os rótulos de classe.
        - **LDA (Linear Discriminant Analysis):** técnica **supervisionada**
          que projeta os dados nas direções que **maximizam a separação entre
          classes** (maximiza variância entre classes e minimiza variância
          dentro de cada classe), utilizando os rótulos durante o ajuste.
        """
    )

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(
            f"**Variância explicada pelo PCA:**\n\n"
            f"- PC1: {var_explicada[0]*100:.2f}%\n"
            f"- PC2: {var_explicada[1]*100:.2f}%\n"
            f"- **Total: {sum(var_explicada)*100:.2f}%**"
        )
    with col_info2:
        st.success(
            "**LDA:** com 3 classes, o LDA produz no máximo **2 componentes "
            "discriminantes** (n_classes - 1), que é exatamente o que "
            "usamos aqui para comparação direta com o PCA."
        )

    st.markdown("### 📍 Comparação Visual: PCA vs LDA")
    col_pca, col_lda = st.columns(2)

    with col_pca:
        fig_pca = plotar_scatter_2d(
            X_pca_full, especies, "PCA — Projeção 2D (Não Supervisionado)", "PC1", "PC2"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_lda:
        fig_lda = plotar_scatter_2d(
            X_lda_full, especies, "LDA — Projeção 2D (Supervisionado)", "LD1", "LD2"
        )
        st.plotly_chart(fig_lda, use_container_width=True)

    st.markdown(
        """
        > 💡 **Observação:** note como o LDA tende a produzir agrupamentos
        > mais compactos e com fronteiras mais nítidas entre as classes,
        > já que sua função objetivo é construída especificamente para
        > **maximizar a separação entre elas** — diferente do PCA, que
        > otimiza apenas a variância total, podendo ou não favorecer a
        > separação de classes.
        """
    )


# -----------------------------------------------------------------------
# TAB 3: MEDIDAS DE SEPARABILIDADE (SILHOUETTE SCORE)
# -----------------------------------------------------------------------
with tab3:
    st.header("📐 Medidas de Separabilidade entre Classes")

    # Cálculo do Silhouette Score nos três cenários
    sil_original = calcular_silhouette(X_scaled_full, y)
    sil_pca = calcular_silhouette(X_pca_full, y)
    sil_lda = calcular_silhouette(X_lda_full, y)

    df_silhouette = pd.DataFrame(
        {
            "Cenário": [
                "Dados Originais (4D)",
                "PCA (2D)",
                "LDA (2D)",
            ],
            "Dimensões": [4, 2, 2],
            "Silhouette Score": [sil_original, sil_pca, sil_lda],
        }
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette — Original (4D)", f"{sil_original:.4f}")
    col2.metric(
        "Silhouette — PCA (2D)",
        f"{sil_pca:.4f}",
        delta=f"{sil_pca - sil_original:+.4f}",
    )
    col3.metric(
        "Silhouette — LDA (2D)",
        f"{sil_lda:.4f}",
        delta=f"{sil_lda - sil_original:+.4f}",
    )

    st.markdown("### 📋 Tabela Comparativa")
    st.dataframe(
        df_silhouette.style.format({"Silhouette Score": "{:.4f}"}).background_gradient(
            subset=["Silhouette Score"], cmap="Greens"
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_bar_sil = px.bar(
        df_silhouette,
        x="Cenário",
        y="Silhouette Score",
        color="Cenário",
        text_auto=".4f",
        title="Comparação do Coeficiente de Silhueta entre Cenários",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar_sil.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_bar_sil, use_container_width=True)

    st.markdown("### 📖 Interpretação Matemática e Física")
    st.markdown(
        r"""
        O **Coeficiente de Silhueta** mede, para cada amostra $i$, o quão bem
        ela se encaixa no seu próprio cluster (classe) em comparação com o
        cluster vizinho mais próximo:

        $$
        s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
        $$

        Onde:
        - $a(i)$ = distância média entre $i$ e os demais pontos **da mesma classe**
          (coesão intra-cluster);
        - $b(i)$ = menor distância média entre $i$ e os pontos **da classe vizinha
          mais próxima** (separação inter-cluster).

        **Interpretação dos valores (variam de -1 a +1):**
        - **Próximo de +1:** a amostra está muito mais próxima do seu próprio
          grupo do que de qualquer outro — alta separabilidade.
        - **Próximo de 0:** a amostra está na fronteira entre dois clusters —
          ambiguidade/sobreposição.
        - **Negativo:** a amostra provavelmente está mais próxima de um
          cluster diferente do seu — sinal de má separação ou erro de
          rotulação.

        **Por que o LDA tende a apresentar Silhouette mais alto que o PCA?**
        Porque o LDA é **otimizado explicitamente** para maximizar a razão entre
        a dispersão *entre classes* (between-class scatter, $S_B$) e a dispersão
        *dentro das classes* (within-class scatter, $S_W$):

        $$
        J(w) = \frac{w^T S_B w}{w^T S_W w}
        $$

        Já o PCA maximiza apenas a variância total dos dados projetados, **sem
        nenhuma noção de classe** — por coincidência, essa direção de máxima
        variância pode (ou não) coincidir com a direção de melhor separação
        entre as espécies de Iris.
        """
    )


# -----------------------------------------------------------------------
# TAB 4: IMPACTO NA PERFORMANCE DE CLASSIFICADORES
# -----------------------------------------------------------------------
with tab4:
    st.header("🤖 Impacto na Performance de Classificadores")
    st.markdown(
        f"""
        Configuração atual (ajustável na barra lateral):
        **K = {k_vizinhos}** (KNN) · **C = {c_logreg}** (Regressão Logística) ·
        **Teste = {test_size_pct}%** dos dados.

        ⚠️ **Prevenção de Data Leakage:** para cada cenário (Original, PCA, LDA),
        a divisão treino/teste é feita **antes** de qualquer ajuste de
        `StandardScaler`, `PCA` ou `LDA`. Essas transformações são ajustadas
        (`fit`) **somente no conjunto de treino** e depois aplicadas
        (`transform`) ao conjunto de teste — isso é particularmente crítico
        para o LDA, que é supervisionado e usaria indevidamente os rótulos de
        teste se ajustado no dataset completo.
        """
    )

    cenarios = {
        "Original (4D)": "original",
        "PCA (2D)": "pca",
        "LDA (2D)": "lda",
    }

    classificadores = {
        "KNN": lambda: KNeighborsClassifier(n_neighbors=k_vizinhos),
        "Regressão Logística": lambda: LogisticRegression(
            C=c_logreg, max_iter=1000, random_state=RANDOM_STATE
        ),
    }

    resultados = []  # lista de dicts para montar o DataFrame comparativo final
    matrizes_confusao = {}  # armazena matrizes para plotagem posterior

    # Loop principal: para cada cenário de dados x cada classificador
    for nome_cenario, modo in cenarios.items():
        X_train, X_test, y_train, y_test = pipeline_classificacao_sem_vazamento(
            X_raw, y, modo=modo, n_componentes=2, scaler_global=None, test_size=test_size
        )

        for nome_clf, construtor_clf in classificadores.items():
            modelo = construtor_clf()
            metricas, matriz, y_pred = treinar_avaliar_classificador(
                modelo, X_train, X_test, y_train, y_test
            )

            resultados.append(
                {
                    "Cenário": nome_cenario,
                    "Classificador": nome_clf,
                    "Acurácia": metricas["Acurácia"],
                    "Precisão": metricas["Precisão"],
                    "Recall": metricas["Recall"],
                    "F1-Score": metricas["F1-Score"],
                }
            )
            matrizes_confusao[(nome_cenario, nome_clf)] = matriz

    df_resultados = pd.DataFrame(resultados)

    st.markdown("### 📋 Tabela Comparativa de Performance")
    st.dataframe(
        df_resultados.style.format(
            {
                "Acurácia": "{:.2%}",
                "Precisão": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
            }
        ).background_gradient(subset=["Acurácia", "F1-Score"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )

    # Gráfico de barras agrupado comparando F1-Score entre cenários e classificadores
    fig_comparativo = px.bar(
        df_resultados,
        x="Cenário",
        y="F1-Score",
        color="Classificador",
        barmode="group",
        text_auto=".2%",
        title="Comparação de F1-Score por Cenário e Classificador",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_comparativo.update_layout(height=420, yaxis_tickformat=".0%")
    st.plotly_chart(fig_comparativo, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧩 Matrizes de Confusão")

    classes_nomes = list(target_names)

    for nome_clf in classificadores.keys():
        st.markdown(f"#### {nome_clf}")
        cols = st.columns(3)
        for i, nome_cenario in enumerate(cenarios.keys()):
            matriz = matrizes_confusao[(nome_cenario, nome_clf)]
            fig_mc = plotar_matriz_confusao(
                matriz, classes_nomes, f"{nome_cenario}"
            )
            cols[i].plotly_chart(fig_mc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 Conclusões Esperadas")
    st.markdown(
        """
        - O cenário **LDA (2D)** tende a apresentar performance **igual ou
          superior** ao espaço **Original (4D)**, mesmo usando metade das
          dimensões — pois o LDA já incorpora informação de classe na própria
          redução.
        - O cenário **PCA (2D)**, por ser não supervisionado, pode apresentar
          uma **queda de performance** em relação ao original, especialmente
          se os componentes de maior variância não corresponderem às direções
          de melhor separação entre *Versicolor* e *Virginica* (as duas
          classes mais sobrepostas no Iris).
        - Esses resultados ilustram, na prática, a diferença fundamental
          entre **redução de dimensionalidade para representação** (PCA) e
          **redução de dimensionalidade para discriminação** (LDA).
        """
    )

st.markdown("---")
st.caption(
    "🌸 Sistema desenvolvido para fins acadêmicos — Estudo de Redução de "
    "Dimensionalidade e Separabilidade com o Dataset Iris (scikit-learn)."
)
