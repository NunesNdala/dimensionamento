# -*- coding: utf-8 -*-
"""
=============================================================================
ESTUDO DE REDUÇÃO DE DIMENSIONALIDADE E SEPARABILIDADE — MULTI-DATASET
=============================================================================
Sistema interativo em Streamlit para comparar:
  - Dados Originais (nD)
  - PCA (não supervisionado, nD -> 2D)
  - LDA (supervisionado, nD -> 2D)

Suporta múltiplos datasets simultâneos:
  - Iris (nativo scikit-learn)
  - Wine (nativo scikit-learn)
  - Breast Cancer (nativo scikit-learn)
  - Digits (nativo scikit-learn, primeiras 2 classes)
  - CSV personalizado (upload do usuário)

Métricas avaliadas:
  - Separabilidade geométrica (Silhouette Score)
  - Performance de classificadores (KNN e Regressão Logística)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    page_title="Redução de Dimensionalidade — Multi-Dataset",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42

# =============================================================================
# PALETA DE CORES POR DATASET (para distinguir visualmente nas comparações)
# =============================================================================
PALETA_DATASETS = [
    px.colors.qualitative.Set1,
    px.colors.qualitative.Set2,
    px.colors.qualitative.Pastel1,
    px.colors.qualitative.Dark2,
    px.colors.qualitative.Safe,
]


# =============================================================================
# FUNÇÕES DE CARREGAMENTO DE DATASETS NATIVOS
# =============================================================================
@st.cache_data
def carregar_iris():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["classe"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, list(iris.feature_names), list(iris.target_names), "Iris"


@st.cache_data
def carregar_wine():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    df["classe"] = df["target"].map(dict(enumerate(wine.target_names)))
    return df, list(wine.feature_names), list(wine.target_names), "Wine"


@st.cache_data
def carregar_breast_cancer():
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df["target"] = bc.target
    df["classe"] = df["target"].map(dict(enumerate(bc.target_names)))
    return df, list(bc.feature_names), list(bc.target_names), "Breast Cancer"


@st.cache_data
def carregar_digits():
    digits = load_digits()
    # Usar apenas dígitos 0-4 para melhor visualização
    mask = digits.target < 5
    df = pd.DataFrame(digits.data[mask], columns=[f"pixel_{i}" for i in range(64)])
    df["target"] = digits.target[mask]
    classes = [f"Dígito {i}" for i in range(5)]
    df["classe"] = df["target"].map(dict(enumerate(classes)))
    return df, [f"pixel_{i}" for i in range(64)], classes, "Digits (0-4)"


def carregar_csv_usuario(uploaded_file, coluna_alvo):
    """
    Carrega CSV com tratamento robusto:
    - Detecta separador automaticamente (virgula, ponto-e-virgula, tab, etc.)
    - Substitui marcadores de missing ('?', 'NA', etc.) por NaN
    - Remove colunas Unnamed e totalmente vazias
    - Converte colunas object para numerico quando possivel
    - Codifica colunas categoricas restantes com LabelEncoder
    - Remove linhas com NaN apos processamento
    """
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            na_values=["?", "NA", "N/A", "na", "n/a", ""],
        )

        # Remover colunas Unnamed e totalmente vazias
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.dropna(axis=1, how="all")

        if coluna_alvo not in df.columns:
            return None, None, None, None, f"Coluna '{coluna_alvo}' nao encontrada."

        df = df.dropna(subset=[coluna_alvo])
        feature_cols = [c for c in df.columns if c != coluna_alvo]

        # Tentar converter colunas object para numerico
        for col in feature_cols:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().mean() >= 0.3:
                    df[col] = converted

        numeric_cols = list(df[feature_cols].select_dtypes(include=[np.number]).columns)
        cat_cols = list(df[feature_cols].select_dtypes(include=["object", "str", "category"]).columns)

        for col in cat_cols:
            le_col = LabelEncoder()
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            le_col.fit(non_null.astype(str))
            df[col] = df[col].apply(
                lambda x, le=le_col: le.transform([str(x)])[0] if pd.notna(x) else np.nan
            )
            numeric_cols.append(col)

        if len(numeric_cols) == 0:
            return None, None, None, None, "Nenhuma coluna numerica ou convertivel encontrada alem da coluna alvo."

        feature_cols = numeric_cols

        le = LabelEncoder()
        df["target"] = le.fit_transform(df[coluna_alvo].astype(str))
        df["classe"] = df[coluna_alvo].astype(str)

        df = df.dropna(subset=feature_cols + ["target"])
        df = df.reset_index(drop=True)

        if len(df) < 10:
            return None, None, None, None, f"Poucos dados validos apos limpeza ({len(df)} linhas)."

        classes = list(le.classes_)
        nome = uploaded_file.name.replace(".csv", "")
        return df, feature_cols, classes, nome, None

    except Exception as e:
        return None, None, None, None, str(e)


# =============================================================================
# FUNÇÕES DE TRANSFORMAÇÃO (com cache por dataset)
# =============================================================================
def padronizar(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def aplicar_pca_2d(X_scaled):
    n = min(2, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    return X_pca, var, pca


def aplicar_lda_2d(X_scaled, y):
    n_classes = len(np.unique(y))
    n_comp = min(2, n_classes - 1, X_scaled.shape[1])
    lda = LDA(n_components=n_comp)
    X_lda = lda.fit_transform(X_scaled, y)
    return X_lda, lda


def calcular_silhouette(X, y):
    if len(np.unique(y)) < 2:
        return 0.0
    try:
        return silhouette_score(X, y)
    except Exception:
        return 0.0


# =============================================================================
# PIPELINE DE CLASSIFICAÇÃO SEM VAZAMENTO
# =============================================================================
def pipeline_classificacao(X_raw, y, modo, test_size=0.3):
    X_train_r, X_test_r, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_r)
    X_test_s = scaler.transform(X_test_r)

    if modo == "original":
        return X_train_s, X_test_s, y_train, y_test

    elif modo == "pca":
        n = min(2, X_train_s.shape[1])
        pca = PCA(n_components=n, random_state=RANDOM_STATE)
        return pca.fit_transform(X_train_s), pca.transform(X_test_s), y_train, y_test

    elif modo == "lda":
        n_classes = len(np.unique(y_train))
        n = min(2, n_classes - 1, X_train_s.shape[1])
        lda = LDA(n_components=n)
        return lda.fit_transform(X_train_s, y_train), lda.transform(X_test_s), y_train, y_test


def avaliar_classificador(modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    return {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }, confusion_matrix(y_test, y_pred)


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================
def scatter_2d(X_2d, classes_arr, titulo, eixo_x="PC1", eixo_y="PC2", paleta=None):
    if paleta is None:
        paleta = px.colors.qualitative.Set1
    cols = [eixo_x, eixo_y] if X_2d.shape[1] >= 2 else [eixo_x, eixo_x]
    df_p = pd.DataFrame(X_2d[:, :2], columns=[eixo_x, eixo_y] if X_2d.shape[1] >= 2 else [eixo_x, "Const"])
    df_p["Classe"] = classes_arr
    fig = px.scatter(
        df_p, x=df_p.columns[0], y=df_p.columns[1],
        color="Classe", title=titulo,
        color_discrete_sequence=paleta, opacity=0.8,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def bar_silhouette(df_sil, titulo="Silhouette Score por Cenário"):
    fig = px.bar(
        df_sil, x="Cenário", y="Silhouette Score", color="Dataset",
        barmode="group", text_auto=".4f",
        title=titulo,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(height=420, yaxis_range=[0, 1])
    return fig


def bar_f1_comparativo(df_res, titulo="F1-Score por Dataset e Cenário"):
    fig = px.bar(
        df_res, x="Dataset", y="F1-Score", color="Cenário",
        facet_col="Classificador", barmode="group",
        text_auto=".2%", title=titulo,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_layout(height=420, yaxis_tickformat=".0%")
    return fig


# =============================================================================
# PROCESSAMENTO DE UM DATASET (retorna todos os dados necessários)
# =============================================================================
def processar_dataset(df, feature_cols, classes, nome, k_vizinhos, c_logreg, test_size):
    X_raw = df[feature_cols].values
    y = df["target"].values
    classes_arr = df["classe"].values

    X_scaled, _ = padronizar(X_raw)
    X_pca, var_pca, _ = aplicar_pca_2d(X_scaled)
    X_lda, _ = aplicar_lda_2d(X_scaled, y)

    sil_orig = calcular_silhouette(X_scaled, y)
    sil_pca = calcular_silhouette(X_pca, y)
    sil_lda = calcular_silhouette(X_lda, y)

    resultados_clf = []
    matrizes = {}

    for modo_nome, modo_key in [("Original", "original"), ("PCA (2D)", "pca"), ("LDA (2D)", "lda")]:
        try:
            X_tr, X_te, y_tr, y_te = pipeline_classificacao(X_raw, y, modo_key, test_size)
        except Exception:
            continue

        for clf_nome, clf_fn in [
            ("KNN", lambda: KNeighborsClassifier(n_neighbors=k_vizinhos)),
            ("Regressão Logística", lambda: LogisticRegression(C=c_logreg, max_iter=1000, random_state=RANDOM_STATE)),
        ]:
            clf = clf_fn()
            try:
                metricas, matriz = avaliar_classificador(clf, X_tr, X_te, y_tr, y_te)
                resultados_clf.append({
                    "Dataset": nome, "Cenário": modo_nome, "Classificador": clf_nome,
                    **metricas,
                })
                matrizes[(modo_nome, clf_nome)] = matriz
            except Exception:
                continue

    return {
        "nome": nome,
        "df": df,
        "feature_cols": feature_cols,
        "classes": classes,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "X_lda": X_lda,
        "var_pca": var_pca,
        "classes_arr": classes_arr,
        "sil": {"Original (nD)": sil_orig, "PCA (2D)": sil_pca, "LDA (2D)": sil_lda},
        "resultados_clf": resultados_clf,
        "matrizes": matrizes,
        "y": y,
    }


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("⚙️ Painel de Controle")
st.sidebar.markdown("---")

# --- Seleção de Datasets Nativos ---
st.sidebar.subheader("📦 Datasets Nativos")
datasets_selecionados = st.sidebar.multiselect(
    "Selecione um ou mais datasets:",
    options=["Iris", "Wine", "Breast Cancer", "Digits (0-4)"],
    default=["Iris"],
)

# --- Upload de CSV(s) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📁 CSV Personalizado")
uploaded_files = st.sidebar.file_uploader(
    "Envie um ou mais arquivos CSV",
    type=["csv"],
    accept_multiple_files=True,
)

csv_configs = []
if uploaded_files:
    for uf in uploaded_files:
        # Ler colunas para seleção do alvo
        try:
            df_preview = pd.read_csv(uf, sep=None, engine="python", nrows=5,
                                      na_values=["?","NA","N/A","na","n/a",""])
            # Remover colunas Unnamed/vazias do preview tambem
            df_preview = df_preview.loc[:, ~df_preview.columns.str.startswith("Unnamed")]
            df_preview = df_preview.dropna(axis=1, how="all")
            uf.seek(0)
            col_alvo = st.sidebar.selectbox(
                f"Coluna alvo — {uf.name}",
                options=list(df_preview.columns),
                index=len(df_preview.columns) - 1,
                key=f"alvo_{uf.name}",
            )
            csv_configs.append((uf, col_alvo))
        except Exception as e:
            st.sidebar.error(f"Erro ao ler {uf.name}: {e}")

# --- Hiperparâmetros ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Hiperparâmetros")

k_vizinhos = st.sidebar.slider("K — KNN", 1, 20, 5)
test_size_pct = st.sidebar.slider("Tamanho do teste (%)", 10, 50, 30, 5)
test_size = test_size_pct / 100.0
c_logreg = st.sidebar.slider("C — Regressão Logística", 0.01, 10.0, 1.0, 0.01)

st.sidebar.markdown("---")
st.sidebar.caption("Estudo acadêmico — Redução de Dimensionalidade Multi-Dataset")


# =============================================================================
# CARREGAMENTO DOS DATASETS SELECIONADOS
# =============================================================================
loaders = {
    "Iris": carregar_iris,
    "Wine": carregar_wine,
    "Breast Cancer": carregar_breast_cancer,
    "Digits (0-4)": carregar_digits,
}

datasets_processados = []

for nome_ds in datasets_selecionados:
    df_ds, feats, classes, nome = loaders[nome_ds]()
    dados = processar_dataset(df_ds, feats, classes, nome, k_vizinhos, c_logreg, test_size)
    datasets_processados.append(dados)

for uf, col_alvo in csv_configs:
    uf.seek(0)
    df_csv, feats_csv, classes_csv, nome_csv, erro = carregar_csv_usuario(uf, col_alvo)
    if erro:
        st.sidebar.error(f"Erro em {uf.name}: {erro}")
    else:
        dados = processar_dataset(df_csv, feats_csv, classes_csv, nome_csv, k_vizinhos, c_logreg, test_size)
        datasets_processados.append(dados)

# =============================================================================
# CABEÇALHO
# =============================================================================
st.title("🔬 Redução de Dimensionalidade — Comparação Multi-Dataset")
st.markdown(
    "Compare **PCA** (não supervisionado) e **LDA** (supervisionado) "
    "em múltiplos datasets simultaneamente. Selecione os datasets na barra lateral."
)
st.markdown("---")

if not datasets_processados:
    st.warning("⚠️ Selecione ao menos um dataset na barra lateral para começar.")
    st.stop()

# =============================================================================
# ABAS PRINCIPAIS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Visão Geral",
    "🔬 2. PCA vs LDA (por dataset)",
    "📐 3. Silhouette Score",
    "🤖 4. Performance dos Classificadores",
    "⚖️ 5. Comparação Entre Datasets",
])


# -----------------------------------------------------------------------
# TAB 1: VISÃO GERAL DOS DATASETS
# -----------------------------------------------------------------------
with tab1:
    st.header("📊 Visão Geral dos Datasets Carregados")

    for dados in datasets_processados:
        nome = dados["nome"]
        df = dados["df"]
        feats = dados["feature_cols"]

        with st.expander(f"📁 {nome}", expanded=len(datasets_processados) == 1):
            cols = st.columns(4)
            cols[0].metric("Amostras", len(df))
            cols[1].metric("Features", len(feats))
            cols[2].metric("Classes", df["classe"].nunique())
            cols[3].metric("Amostras/Classe (média)", f"{len(df)/df['classe'].nunique():.0f}")

            col_tab, col_desc = st.columns([2, 1])
            with col_tab:
                st.markdown("**Distribuição de Classes**")
                dist = df["classe"].value_counts().reset_index()
                dist.columns = ["Classe", "Contagem"]
                fig_dist = px.bar(dist, x="Classe", y="Contagem", color="Classe",
                                  color_discrete_sequence=px.colors.qualitative.Set1)
                fig_dist.update_layout(height=300, showlegend=False,
                                       margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_desc:
                st.markdown("**Estatísticas Descritivas**")
                st.dataframe(df[feats].describe().T[["mean", "std", "min", "max"]]
                             .round(2), use_container_width=True)


# -----------------------------------------------------------------------
# TAB 2: PCA vs LDA POR DATASET
# -----------------------------------------------------------------------
with tab2:
    st.header("🔬 PCA vs LDA — Visualização por Dataset")

    for idx, dados in enumerate(datasets_processados):
        nome = dados["nome"]
        paleta = PALETA_DATASETS[idx % len(PALETA_DATASETS)]

        st.subheader(f"📌 {nome}")

        var = dados["var_pca"]
        var_total = sum(var[:2]) * 100

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            var_str = "\n".join([f"- PC{i+1}: {v*100:.2f}%" for i, v in enumerate(var[:2])])
            st.info(f"**Variância explicada (PCA):**\n\n{var_str}\n\n**Total: {var_total:.2f}%**")
        with col_info2:
            n_classes = len(dados["classes"])
            n_feats = len(dados["feature_cols"])
            n_comp_lda = min(2, n_classes - 1)
            st.success(
                f"**LDA:** {n_classes} classes → {n_comp_lda} componente(s) discriminante(s)\n\n"
                f"**Features originais:** {n_feats}D"
            )

        col_orig, col_pca, col_lda = st.columns(3)
        with col_orig:
            # PCA do espaço original reduzido para 2D apenas para vis
            fig_orig = scatter_2d(
                dados["X_pca"], dados["classes_arr"],  # reusa PCA para visualizar original também
                f"{nome} — Espaço Padronizado (via PCA)", "PC1", "PC2", paleta
            )
            st.plotly_chart(fig_orig, use_container_width=True)
            st.caption("(Projeção 2D via PCA para visualização do espaço original)")

        with col_pca:
            fig_pca = scatter_2d(
                dados["X_pca"], dados["classes_arr"],
                f"{nome} — PCA (Não Supervisionado)", "PC1", "PC2", paleta
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption(f"Variância retida: {var_total:.1f}%")

        with col_lda:
            X_lda = dados["X_lda"]
            if X_lda.shape[1] >= 2:
                X_lda_plot = X_lda
                eixo_x_lda, eixo_y_lda = "LD1", "LD2"
            else:
                # Apenas 1 componente (datasets com 2 classes): usa jitter no eixo Y
                X_lda_plot = np.column_stack([X_lda, np.zeros(len(X_lda))])
                eixo_x_lda, eixo_y_lda = "LD1", "Const"
            fig_lda = scatter_2d(
                X_lda_plot, dados["classes_arr"],
                f"{nome} — LDA (Supervisionado)", eixo_x_lda, eixo_y_lda, paleta
            )
            st.plotly_chart(fig_lda, use_container_width=True)

        st.markdown("---")


# -----------------------------------------------------------------------
# TAB 3: SILHOUETTE SCORE
# -----------------------------------------------------------------------
with tab3:
    st.header("📐 Comparação de Separabilidade — Silhouette Score")

    # Métricas por dataset
    for dados in datasets_processados:
        nome = dados["nome"]
        sil = dados["sil"]
        cols = st.columns(3)
        cols[0].metric(f"{nome} — Original (nD)", f"{sil['Original (nD)']:.4f}")
        cols[1].metric(f"{nome} — PCA (2D)", f"{sil['PCA (2D)']:.4f}",
                       delta=f"{sil['PCA (2D)'] - sil['Original (nD)']:+.4f}")
        cols[2].metric(f"{nome} — LDA (2D)", f"{sil['LDA (2D)']:.4f}",
                       delta=f"{sil['LDA (2D)'] - sil['Original (nD)']:+.4f}")

    st.markdown("---")

    # Tabela e gráfico comparativos
    rows_sil = []
    for dados in datasets_processados:
        for cenario, score in dados["sil"].items():
            rows_sil.append({"Dataset": dados["nome"], "Cenário": cenario, "Silhouette Score": score})

    df_sil = pd.DataFrame(rows_sil)

    st.markdown("### 📋 Tabela Comparativa")
    st.dataframe(
        df_sil.style.format({"Silhouette Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 📊 Gráfico Comparativo")
    fig_sil = bar_silhouette(df_sil)
    st.plotly_chart(fig_sil, use_container_width=True)

    # Gráfico radar (quando há múltiplos datasets)
    if len(datasets_processados) > 1:
        st.markdown("### 🕸️ Radar — Silhouette por Dataset e Cenário")
        fig_radar = go.Figure()
        cenarios = ["Original (nD)", "PCA (2D)", "LDA (2D)"]
        for dados in datasets_processados:
            valores = [dados["sil"][c] for c in cenarios]
            valores += [valores[0]]  # fechar o radar
            fig_radar.add_trace(go.Scatterpolar(
                r=valores,
                theta=cenarios + [cenarios[0]],
                fill="toself",
                name=dados["nome"],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown(
        r"""
        ### 📖 Referência Rápida
        O Silhouette Score $s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}$ varia de **-1** a **+1**:
        - **+1** → clusters densos e bem separados
        - **0** → sobreposição entre clusters
        - **-1** → amostras possivelmente no cluster errado
        """
    )


# -----------------------------------------------------------------------
# TAB 4: PERFORMANCE DOS CLASSIFICADORES
# -----------------------------------------------------------------------
with tab4:
    st.header("🤖 Performance dos Classificadores por Dataset")

    for dados in datasets_processados:
        nome = dados["nome"]
        df_res = pd.DataFrame(dados["resultados_clf"])
        if df_res.empty:
            st.warning(f"Sem resultados para {nome}.")
            continue

        with st.expander(f"📌 {nome}", expanded=len(datasets_processados) == 1):
            st.dataframe(
                df_res.drop(columns=["Dataset"]).style.format({
                    "Acurácia": "{:.2%}", "Precisão": "{:.2%}",
                    "Recall": "{:.2%}", "F1-Score": "{:.2%}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            fig_f1 = px.bar(
                df_res, x="Cenário", y="F1-Score", color="Classificador",
                barmode="group", text_auto=".2%",
                title=f"{nome} — F1-Score por Cenário",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig_f1.update_layout(height=380, yaxis_tickformat=".0%")
            st.plotly_chart(fig_f1, use_container_width=True)

            # Matrizes de confusão
            st.markdown("**Matrizes de Confusão — KNN**")
            classes_nomes = dados["classes"]
            cols_mc = st.columns(3)
            for i, cenario in enumerate(["Original", "PCA (2D)", "LDA (2D)"]):
                key = (cenario, "KNN")
                if key in dados["matrizes"]:
                    fig_mc = px.imshow(
                        dados["matrizes"][key],
                        x=classes_nomes, y=classes_nomes,
                        color_continuous_scale="Blues", text_auto=True,
                        labels=dict(x="Previsto", y="Real"),
                        title=cenario,
                    )
                    fig_mc.update_layout(height=320, margin=dict(l=5, r=5, t=40, b=5))
                    cols_mc[i].plotly_chart(fig_mc, use_container_width=True)


# -----------------------------------------------------------------------
# TAB 5: COMPARAÇÃO ENTRE DATASETS
# -----------------------------------------------------------------------
with tab5:
    st.header("⚖️ Comparação Consolidada Entre Datasets")

    if len(datasets_processados) < 2:
        st.info("💡 Selecione **2 ou mais datasets** na barra lateral para ativar esta aba.")
    else:
        # DataFrame consolidado de resultados
        todos_resultados = []
        for dados in datasets_processados:
            todos_resultados.extend(dados["resultados_clf"])
        df_todos = pd.DataFrame(todos_resultados)

        st.markdown("### 📋 Tabela Consolidada")
        st.dataframe(
            df_todos.style.format({
                "Acurácia": "{:.2%}", "Precisão": "{:.2%}",
                "Recall": "{:.2%}", "F1-Score": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 📊 F1-Score — Visão Consolidada")
        fig_cons = bar_f1_comparativo(df_todos)
        st.plotly_chart(fig_cons, use_container_width=True)

        # Heatmap: Dataset × Cenário (F1-Score médio dos classificadores)
        st.markdown("### 🌡️ Heatmap — F1-Score Médio (Dataset × Cenário)")
        df_heat = (
            df_todos.groupby(["Dataset", "Cenário"])["F1-Score"]
            .mean()
            .reset_index()
            .pivot(index="Dataset", columns="Cenário", values="F1-Score")
        )
        fig_heat = px.imshow(
            df_heat,
            text_auto=".2%",
            color_continuous_scale="Greens",
            title="F1-Score Médio (KNN + Log. Reg.) por Dataset e Cenário",
            labels=dict(color="F1-Score"),
        )
        fig_heat.update_layout(height=max(300, 80 * len(datasets_processados)))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Ranking final
        st.markdown("### 🏆 Ranking — Melhor Configuração por Dataset")
        ranking = (
            df_todos.sort_values("F1-Score", ascending=False)
            .groupby("Dataset")
            .first()
            .reset_index()[["Dataset", "Cenário", "Classificador", "F1-Score", "Acurácia"]]
        )
        ranking["F1-Score"] = ranking["F1-Score"].map("{:.2%}".format)
        ranking["Acurácia"] = ranking["Acurácia"].map("{:.2%}".format)
        ranking.index = range(1, len(ranking) + 1)
        st.dataframe(ranking, use_container_width=True)

        # Conclusão automática
        st.markdown("### 📌 Observações Automáticas")
        for dados in datasets_processados:
            nome = dados["nome"]
            sil = dados["sil"]
            melhor_cenario = max(sil, key=sil.get)
            lda_ganho = sil["LDA (2D)"] - sil["Original (nD)"]
            pca_ganho = sil["PCA (2D)"] - sil["Original (nD)"]

            if lda_ganho > 0.05:
                obs = f"✅ O **LDA** melhora significativamente a separabilidade (+{lda_ganho:.4f} vs. original)."
            elif lda_ganho > 0:
                obs = f"🟡 O **LDA** melhora levemente a separabilidade (+{lda_ganho:.4f} vs. original)."
            else:
                obs = f"🔴 O **LDA** não melhora a separabilidade ({lda_ganho:.4f} vs. original)."

            if pca_ganho >= 0:
                obs_pca = f"O **PCA** também mantém ou melhora a separabilidade ({pca_ganho:+.4f})."
            else:
                obs_pca = f"O **PCA** perde um pouco de separabilidade ({pca_ganho:.4f}) ao comprimir para 2D."

            st.markdown(f"**{nome}:** {obs} {obs_pca}")

st.markdown("---")
st.caption("🔬 Sistema de Redução de Dimensionalidade Multi-Dataset — PCA vs LDA")
