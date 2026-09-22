import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

st.set_page_config(page_title="Prédiction Performance Index", page_icon="🎓", layout="centered")

# ---- Chargement des artefacts ----
@st.cache_resource
def load_artifacts():
    model = jb.load("best_model.joblib")
    scaler = jb.load("scaler.joblib")   # Pipeline RobustScaler -> StandardScaler
    return model, scaler

model, scaler = load_artifacts()
feature_names = metadata["feature_names"]

st.title("🎓 Prédiction du Performance Index")
st.markdown(
    f"Modèle déployé : **{metadata['best_name']}** "
    f"(R² test = {metadata['r2_test']:.3f} | MAE test = {metadata['mae_test']:.2f})"
)

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction par fichier CSV"])

# ---------------- Onglet 1 : prédiction simple ----------------
with tab1:
    st.subheader("Renseigne les informations de l'élève")

    hours_studied = st.slider("Heures d'étude (par jour)", 0, 12, 5)
    previous_scores = st.slider("Score précédent", 0, 100, 70)
    extracurricular = st.selectbox("Activités extrascolaires", list(encoder.classes_))
    sleep_hours = st.slider("Heures de sommeil", 0, 12, 7)
    papers_practiced = st.slider("Nombre de sujets d'examen pratiqués", 0, 10, 3)

    if st.button("Prédire le Performance Index"):
        extracurricular_enc = encoder.transform([extracurricular])[0]

        input_df = pd.DataFrame(
            [[hours_studied, previous_scores, extracurricular_enc, sleep_hours, papers_practiced]],
            columns=feature_names,
        )
        input_scaled = scaler.transform(input_df.values)
        prediction = model.predict(input_scaled)[0]

        st.success(f"📊 Performance Index prédit : **{prediction:.2f}**")

# ---------------- Onglet 2 : prédiction par CSV ----------------
with tab2:
    st.subheader("Charge un fichier CSV")
    st.caption("Colonnes attendues : " + ", ".join(feature_names))
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        missing_cols = [c for c in feature_names if c not in data.columns]
        if missing_cols:
            st.error(f"Colonnes manquantes dans le fichier : {missing_cols}")
        else:
            data_proc = data.copy()
            data_proc["Extracurricular Activities"] = encoder.transform(
                data_proc["Extracurricular Activities"]
            )
            X = data_proc[feature_names].values
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)

            data["Performance Index (prédit)"] = preds
            st.dataframe(data)

            csv_out = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les résultats",
                csv_out,
                "predictions.csv",
                "text/csv",
            )
