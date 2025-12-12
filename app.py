import os

import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

import plotly.express as px

# ---------------------------------------------------
# Streamlit config
# ---------------------------------------------------
st.set_page_config(
    page_title="Tetuan – Zone 1 Power Consumption Forecast",
    layout="wide"
)

# ---------------------------------------------------
# PROJECT CONFIG
# ---------------------------------------------------
TARGET_COL = "Zone 1 Power Consumption"

FEATURE_COLS = [
    "Temperature", "Humidity", "Wind Speed",
    "general diffuse flows", "diffuse flows",
    TARGET_COL,
    "hour", "dayofweek", "is_weekend"
]

SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

# Pretrained models (all with window=24)
MODELS = {
    "GRU (window 24)": {
        "path": "gru_tetuan.h5",
        "window": 24
    },
    "LSTM (window 24)": {
        "path": "lstm_tetuan.h5",
        "window": 24
    },
    "CNN1D (window 24)": {
        "path": "cnn_tetuan.h5",
        "window": 24
    },
    # NEW: pretrained MLP model
    "MLP (window 24)": {
        "path": "mlp_tetuan.h5",
        "window": 24
    }
}

DEFAULT_MODEL_NAME = "GRU (window 24)"


# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------
def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window, :])
        ys.append(y[i + window, 0])
    return np.array(Xs), np.array(ys)


@st.cache_resource
def load_scalers():
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return scaler_X, scaler_y


@st.cache_resource
def load_model_by_name(model_name: str):
    config = MODELS[model_name]
    model_path = config["path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            f"Place it inside the TetuanApp folder."
        )

    model = load_model(model_path, compile=False)
    return model


def preprocess_tetuan(df: pd.DataFrame) -> pd.DataFrame:
    """Common preprocessing for Tetuan CSV."""
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    df = df.sort_values("DateTime")
    df.set_index("DateTime", inplace=True)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df_clean = df.dropna(subset=FEATURE_COLS)
    if df_clean.empty:
        raise ValueError("No data left after dropping NaNs in feature columns.")

    return df_clean


def evaluate_pretrained_model(df_clean, model_name, scaler_X, scaler_y):
    """
    Run selected pretrained model (GRU / LSTM / CNN1D / MLP)
    and return results_df, rmse, mae, mape.
    """
    config = MODELS[model_name]
    window = config["window"]
    model = load_model_by_name(model_name)

    # 1) Build scaled data
    X_raw = df_clean[FEATURE_COLS].values
    y_raw = df_clean[[TARGET_COL]].values

    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw)

    if len(X_scaled) <= window + 1:
        raise ValueError(
            f"Not enough data ({len(X_scaled)} samples) for window size {window}."
        )

    # 2) Create 3D sequences (samples, window, features)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window)

    # 3) Choose the right input shape depending on the model
    #    (3D for GRU/LSTM/CNN1D, 2D for MLP)
    in_shape = model.input_shape  # e.g. (None, 24, 9) or (None, 216)

    if len(in_shape) == 3:
        # Recurrent / CNN models -> expect 3D input
        X_input = X_seq
    elif len(in_shape) == 2:
        # Dense / MLP models -> expect 2D, flatten sequences
        X_input = X_seq.reshape(X_seq.shape[0], -1)
    else:
        raise ValueError(f"Unsupported model input shape: {in_shape}")

    # 4) Predict
    y_pred_scaled = model.predict(X_input, verbose=0)

    # inverse scale
    y_true = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # align timestamps
    timestamps = df_clean.index[window:window + len(y_true)]

    results_df = pd.DataFrame({
        "DateTime": timestamps,
        "Actual": y_true,
        "Predicted": y_pred
    })
    results_df["Error"] = results_df["Predicted"] - results_df["Actual"]
    results_df["Absolute Error"] = results_df["Error"].abs()
    results_df["Absolute % Error"] = (
        results_df["Absolute Error"]
        / results_df["Actual"].replace(0, np.nan)
        * 100
    )

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = results_df["Actual"] != 0
    mape = np.mean(
        np.abs(
            (results_df.loc[mask, "Actual"] - results_df.loc[mask, "Predicted"])
            / results_df.loc[mask, "Actual"]
        )
    ) * 100

    return results_df, rmse, mae, mape


# ---------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------
st.sidebar.title("⚙️ Configuration")

mode = st.sidebar.radio(
    "Mode",
    [
        "Pretrained models (comparison)",
        "GRU training playground"
    ]
)

if mode == "Pretrained models (comparison)":
    st.sidebar.subheader("🧠 Pretrained model")

    model_name = st.sidebar.selectbox(
        "Select model:",
        options=list(MODELS.keys()),
        index=list(MODELS.keys()).index(DEFAULT_MODEL_NAME)
    )

    max_points_plot = st.sidebar.slider(
        "Max points in plot",
        min_value=100,
        max_value=2000,
        value=300,
        step=50
    )
    max_rows_table = st.sidebar.slider(
        "Rows to show in table",
        min_value=20,
        max_value=1000,
        value=100,
        step=20
    )

    compare_all = st.sidebar.checkbox(
        "Show comparison dashboard (all models)",
        value=True
    )

else:
    # ---- Training playground (supports GRU / LSTM / CNN1D / MLP) ----
    st.sidebar.subheader("🧠 Training playground")

    model_choice = st.sidebar.selectbox(
        "Model to train:",
        ["GRU", "LSTM", "CNN1D", "MLP"]
    )

    window_size_train = st.sidebar.number_input(
        "Window size (steps)",
        min_value=12,
        max_value=48,
        value=24,
        step=1
    )

    units_main = st.sidebar.slider(
        "Main units / neurons",
        min_value=16,
        max_value=128,
        value=64,
        step=16
    )

    epochs = st.sidebar.slider(
        "Epochs",
        min_value=3,
        max_value=30,
        value=10,
        step=1
    )
    batch_size = st.sidebar.selectbox(
        "Batch size",
        options=[32, 64, 128],
        index=1  # 64
    )
    train_ratio = st.sidebar.slider(
        "Train ratio",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.05
    )
    last_n_rows = st.sidebar.slider(
        "Use last N rows for training",
        min_value=2000,
        max_value=10000,
        value=5000,
        step=1000
    )
    max_points_plot = st.sidebar.slider(
        "Max points in plot",
        min_value=100,
        max_value=1000,
        value=300,
        step=50
    )
    max_rows_table = st.sidebar.slider(
        "Rows to show in table",
        min_value=20,
        max_value=500,
        value=100,
        step=20
    )


# ---------------------------------------------------
# Main layout
# ---------------------------------------------------
st.title("⚡ Tetuan – Zone 1 Power Consumption Forecast (GRU / LSTM / CNN1D / MLP)")

uploaded_file = st.file_uploader("Upload Tetuan CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload the Tetuan CSV to begin.")
else:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📊 Data preview")
    st.dataframe(df_raw.head())

    if "DateTime" not in df_raw.columns:
        st.error("Your CSV is missing the 'DateTime' column.")
    else:
        # Common preprocessing
        try:
            df_clean = preprocess_tetuan(df_raw)
        except Exception as e:
            st.error(str(e))
            st.stop()

        # =====================================================
        # MODE 1 – PRETRAINED MODELS (COMPARISON)
        # =====================================================
        if mode == "Pretrained models (comparison)":
            st.markdown(
                "### 🧠 Pretrained models\n"
                "You can switch between GRU, LSTM, CNN1D and MLP, "
                "and optionally display a comparison dashboard."
            )

            try:
                scaler_X, scaler_y = load_scalers()
            except Exception as e:
                st.error(f"Error loading scalers: {e}")
                st.stop()

            # ---- Detailed view for selected model ----
            try:
                results_df_sel, rmse_sel, mae_sel, mape_sel = evaluate_pretrained_model(
                    df_clean, model_name, scaler_X, scaler_y
                )
            except Exception as e:
                st.error(f"Error evaluating {model_name}: {e}")
                st.stop()

            tab_overview, tab_table, tab_compare = st.tabs(
                ["📈 Selected model", "📋 Table (selected)", "📊 Comparison (all models)"]
            )

            with tab_overview:
                st.subheader(f"📈 Performance – {model_name}")

                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{rmse_sel:.2f}")
                c2.metric("MAE", f"{mae_sel:.2f}")
                c3.metric("MAPE (%)", f"{mape_sel:.2f}")

                st.subheader("📉 Actual vs Predicted (selected model)")
                plot_df = results_df_sel.tail(max_points_plot)
                fig = px.line(
                    plot_df,
                    x="DateTime",
                    y=["Actual", "Predicted"],
                    title=f"{model_name} – Actual vs Predicted",
                    labels={
                        "value": "Power Consumption",
                        "DateTime": "Time",
                        "variable": "Series"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab_table:
                st.subheader(f"📋 Actual vs Predicted – {model_name}")
                st.dataframe(results_df_sel.tail(max_rows_table))

                csv_data = results_df_sel.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download CSV (selected model)",
                    data=csv_data,
                    file_name=f"{model_name.replace(' ', '_')}_results.csv",
                    mime="text/csv"
                )

            with tab_compare:
                if compare_all:
                    st.subheader("📊 Comparison dashboard – all available models")

                    metrics_rows = []
                    combined_df = None

                    for m_name in MODELS.keys():
                        try:
                            res_df, rmse, mae, mape = evaluate_pretrained_model(
                                df_clean, m_name, scaler_X, scaler_y
                            )
                            metrics_rows.append({
                                "Model": m_name,
                                "RMSE": rmse,
                                "MAE": mae,
                                "MAPE (%)": mape
                            })

                            col_name = f"{m_name} Predicted"
                            tmp = res_df[["DateTime", "Predicted"]].rename(
                                columns={"Predicted": col_name}
                            )
                            if combined_df is None:
                                combined_df = res_df[["DateTime", "Actual"]].merge(
                                    tmp, on="DateTime", how="inner"
                                )
                            else:
                                combined_df = combined_df.merge(
                                    tmp, on="DateTime", how="inner"
                                )
                        except Exception as e:
                            st.warning(f"Skipping {m_name}: {e}")

                    if metrics_rows:
                        metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
                        st.write("### 🧮 Metrics comparison")
                        st.dataframe(metrics_df)

                        st.write("### 📈 Multi-model predictions (last points)")
                        if combined_df is not None and not combined_df.empty:
                            plot_cols = ["Actual"] + [
                                c for c in combined_df.columns if "Predicted" in c
                            ]
                            multi_plot_df = combined_df.tail(max_points_plot)
                            fig_cmp = px.line(
                                multi_plot_df,
                                x="DateTime",
                                y=plot_cols,
                                labels={
                                    "value": "Power Consumption",
                                    "DateTime": "Time",
                                    "variable": "Series"
                                },
                                title="Actual vs Predicted – All Models"
                            )
                            st.plotly_chart(fig_cmp, use_container_width=True)
                        else:
                            st.info("No combined data to plot.")
                    else:
                        st.info("No models could be evaluated for comparison.")
                else:
                    st.info(
                        "Activate 'Show comparison dashboard (all models)' "
                        "in the sidebar."
                    )

        # =====================================================
        # MODE 2 – TRAINING PLAYGROUND
        # =====================================================
        else:
            st.markdown(
                "### 🧪 GRU / LSTM / CNN1D / MLP – Training playground\n"
                "Train a model on the last N rows of the Tetuan dataset "
                "with the hyperparameters you choose in the sidebar."
            )

            if st.button("🚀 Train / Evaluate selected model"):
                with st.spinner(f"Training {model_choice} playground model..."):
                    df_train = df_clean.tail(last_n_rows)

                    X_raw = df_train[FEATURE_COLS].values
                    y_raw = df_train[[TARGET_COL]].values

                    scaler_X_pg = MinMaxScaler()
                    scaler_y_pg = MinMaxScaler()

                    X_scaled = scaler_X_pg.fit_transform(X_raw)
                    y_scaled = scaler_y_pg.fit_transform(y_raw)

                    X_seq, y_seq = create_sequences(
                        X_scaled, y_scaled, window_size_train
                    )

                    split_idx = int(len(X_seq) * train_ratio)
                    X_train_3d, X_test_3d = (
                        X_seq[:split_idx],
                        X_seq[split_idx:]
                    )
                    y_train, y_test = (
                        y_seq[:split_idx],
                        y_seq[split_idx:]
                    )

                    n_features = X_train_3d.shape[2]

                    # Flattened versions for MLP
                    X_train_flat = X_train_3d.reshape(X_train_3d.shape[0], -1)
                    X_test_flat = X_test_3d.reshape(X_test_3d.shape[0], -1)

                    # ----- Build chosen model -----
                    model = Sequential()

                    if model_choice == "GRU":
                        model.add(
                            GRU(
                                units_main,
                                input_shape=(window_size_train, n_features)
                            )
                        )
                        model.add(Dense(1))
                        X_tr, X_te = X_train_3d, X_test_3d

                    elif model_choice == "LSTM":
                        model.add(
                            LSTM(
                                units_main,
                                return_sequences=True,
                                input_shape=(window_size_train, n_features)
                            )
                        )
                        model.add(LSTM(units_main // 2))
                        model.add(Dense(1))
                        X_tr, X_te = X_train_3d, X_test_3d

                    elif model_choice == "CNN1D":
                        model.add(
                            Conv1D(
                                filters=64,
                                kernel_size=3,
                                activation="relu",
                                input_shape=(window_size_train, n_features)
                            )
                        )
                        model.add(MaxPooling1D(pool_size=2))
                        model.add(Flatten())
                        model.add(Dense(units_main, activation="relu"))
                        model.add(Dense(1))
                        X_tr, X_te = X_train_3d, X_test_3d

                    else:  # MLP
                        input_dim = X_train_flat.shape[1]
                        model.add(
                            Dense(
                                units_main,
                                activation="relu",
                                input_shape=(input_dim,)
                            )
                        )
                        model.add(Dense(units_main // 2, activation="relu"))
                        model.add(Dense(1))
                        X_tr, X_te = X_train_flat, X_test_flat

                    model.compile(optimizer="adam", loss="mse")

                    es = EarlyStopping(
                        monitor="val_loss",
                        patience=3,
                        restore_best_weights=True
                    )

                    model.fit(
                        X_tr,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[es],
                        verbose=0
                    )

                st.success(f"✅ {model_choice} playground training finished!")

                # Evaluation
                y_pred_scaled = model.predict(X_te, verbose=0)
                y_test_inv = scaler_y_pg.inverse_transform(
                    y_test.reshape(-1, 1)
                ).flatten()
                y_pred_inv = scaler_y_pg.inverse_transform(
                    y_pred_scaled
                ).flatten()

                all_timestamps = df_train.index[
                    window_size_train: window_size_train + len(X_seq)
                ]
                test_timestamps = all_timestamps[split_idx:]

                results_df = pd.DataFrame({
                    "DateTime": test_timestamps,
                    "Actual": y_test_inv,
                    "Predicted": y_pred_inv
                })
                results_df["Error"] = (
                    results_df["Predicted"] - results_df["Actual"]
                )
                results_df["Absolute Error"] = results_df["Error"].abs()
                results_df["Absolute % Error"] = (
                    results_df["Absolute Error"]
                    / results_df["Actual"].replace(0, np.nan)
                    * 100
                )

                rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
                mae = mean_absolute_error(y_test_inv, y_pred_inv)
                mask = results_df["Actual"] != 0
                mape = np.mean(
                    np.abs(
                        (
                            results_df.loc[mask, "Actual"]
                            - results_df.loc[mask, "Predicted"]
                        )
                        / results_df.loc[mask, "Actual"]
                    )
                ) * 100

                tab_overview, tab_table = st.tabs(
                    ["📈 Overview", "📋 Table & Download"]
                )

                with tab_overview:
                    st.subheader(
                        f"📊 Test Performance – {model_choice} Playground"
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE", f"{rmse:.2f}")
                    c2.metric("MAE", f"{mae:.2f}")
                    c3.metric("MAPE (%)", f"{mape:.2f}")

                    st.markdown(
                        f"- **Model:** {model_choice}\n"
                        f"- **Window size:** {window_size_train} steps\n"
                        f"- **Main units / neurons:** {units_main}\n"
                        f"- **Epochs:** {epochs}\n"
                        f"- **Batch size:** {batch_size}\n"
                        f"- **Train ratio:** {int(train_ratio * 100)}% train\n"
                        f"- **Rows used:** last {last_n_rows} rows of the dataset"
                    )

                    st.subheader(
                        "📈 Actual vs Predicted (test set, last points)"
                    )
                    plot_df = results_df.tail(max_points_plot)
                    fig = px.line(
                        plot_df,
                        x="DateTime",
                        y=["Actual", "Predicted"],
                        labels={
                            "value": "Power Consumption",
                            "DateTime": "Time",
                            "variable": "Series"
                        },
                        title=(
                            f"{model_choice} Playground – "
                            "Actual vs Predicted"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab_table:
                    st.subheader("📋 Actual vs Predicted – Test samples")
                    st.write(
                        f"Showing last **{min(max_rows_table, len(results_df))}** "
                        f"rows out of {len(results_df)} test samples."
                    )
                    st.dataframe(results_df.tail(max_rows_table))

                    csv_download = results_df.to_csv(
                        index=False
                    ).encode("utf-8")
                    st.download_button(
                        label="📥 Download full results (CSV)",
                        data=csv_download,
                        file_name=(
                            f"Tetuan_{model_choice}_playground_results.csv"
                        ),
                        mime="text/csv"
                    )
