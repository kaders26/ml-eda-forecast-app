import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Modeli yükle
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🛒 Talep Tahmini Uygulaması (Demand Forecasting)")

st.markdown("Bu uygulama, kullanıcıdan alınan girdilere göre satış tahmininde bulunur.")

# 🎛️ Sidebar: Kullanıcı girişi
st.sidebar.header("🔧 Tahmin Girdileri")

store = st.sidebar.selectbox("Mağaza No", options=list(range(1, 11)))
item = st.sidebar.selectbox("Ürün No", options=list(range(1, 51)))
year = st.sidebar.selectbox("Yıl", [2013, 2014, 2015, 2016])
month = st.sidebar.slider("Ay", 1, 12, 6)
dayofweek = st.sidebar.slider("Haftanın Günü (0=Pzt, 6=Paz)", 0, 6, 0)
is_weekend = 1 if dayofweek in [5, 6] else 0

lag_7 = st.sidebar.number_input("7 Günlük Gecikmeli Satış", value=50.0)
lag_30 = st.sidebar.number_input("30 Günlük Gecikmeli Satış", value=60.0)
rolling_std_7 = st.sidebar.number_input("7 Günlük Std Sapma", value=5.0)
sales_diff = st.sidebar.number_input("Satış Farkı", value=1.0)

# 📊 Tahmin işlemi
input_data = np.array([[store, item, year, month, dayofweek, is_weekend,
                        lag_7, lag_30, rolling_std_7, sales_diff]])


# Tahmini değer (örnek olarak modelin çıktısı burada)
prediction = model.predict(input_data)[0]

# 🔢 Tahmini tam sayıya yuvarla
rounded_prediction = int(round(prediction))

# 🎯 Tahmin Göstergesi
st.subheader("🎯 Tahmin Göstergesi")

# ⏱️ Hız göstergesi (gauge chart)
gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=rounded_prediction,
    number={'suffix': " adet", 'font': {'size': 24}},
    delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'steps': [
            {'range': [0, 30], 'color': "#F5F5F5"},
            {'range': [30, 70], 'color': "#B0E0E6"},
            {'range': [70, 100], 'color': "#90EE90"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 80
        }
    }
))

st.plotly_chart(gauge_fig, use_container_width=True)

# 📦 Sayı olarak da metrik göster
st.metric("📦 Tahmin Edilen Satış (Yuvarlanmış)", f"{rounded_prediction} adet")

# 📊 Satış Seviyesi Yorumlama
if rounded_prediction < 30:
    level = "📉 Düşük Talep"
    st.info(f"**Tahmin Seviyesi:** {level}")
elif rounded_prediction < 70:
    level = "📊 Orta Talep"
    st.warning(f"**Tahmin Seviyesi:** {level}")
else:
    level = "📈 Yüksek Talep"
    st.success(f"**Tahmin Seviyesi:** {level}")


# 🧠 Özellik Etki Analizi (Basit +1 değişim)
st.markdown("### 🔍 Özelliklerin Tahmine Etkisi (+1 değişimle)")

feature_names = ['store', 'item', 'year', 'month', 'dayofweek', 'is_weekend',
                 'lag_7', 'lag_30', 'rolling_std_7', 'sales_diff']

variations = []
pred_deltas = []

for i, name in enumerate(feature_names):
    sample = input_data.copy()
    sample[0, i] += 1  # Sadece bir değişkeni +1 arttır
    new_pred = model.predict(sample)[0]
    delta = new_pred - prediction
    variations.append(name)
    pred_deltas.append(delta)

# 🎨 Grafikle Göster
fig, ax = plt.subplots()
bars = ax.barh(variations, pred_deltas, color='skyblue')
ax.set_xlabel("Tahmindeki Değişim")
ax.set_title("Özelliklerin Etkisi (+1 değişim)")
st.pyplot(fig)


st.subheader("📈 Son 30 Günlük Satış Trendi (Simülasyon)")

days = list(range(-30, 1))
sales_simulated = [lag_30 + (i * sales_diff / 30) for i in range(-30, 1)]
sales_simulated[-1] = prediction

fig = go.Figure()
fig.add_trace(go.Scatter(x=days, y=sales_simulated, mode='lines+markers',
                         name='Simüle Satış', line=dict(color='orange', width=3)))
fig.add_hline(y=prediction, line_dash="dash", line_color="red", annotation_text="Tahmin", annotation_position="top right")

fig.update_layout(
    title="🕒 30 Günlük Satış Tahmini",
    xaxis_title="Gün (geçmişten bugüne)",
    yaxis_title="Satış Adedi",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)

st.subheader("🧬 SHAP Özellik Etkisi")

shap.initjs()

# Generate SHAP waterfall plot without passing ax argument
fig_shap = shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                                 base_values=explainer.expected_value,
                                                 data=input_data[0],
                                                 feature_names=feature_names))

# Create a Matplotlib figure to display the plot
fig, ax = plt.subplots()
shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                      base_values=explainer.expected_value,
                                      data=input_data[0],
                                      feature_names=feature_names))

st.pyplot(fig)

lower = prediction - 5
upper = prediction + 5

fig_ci = go.Figure()
fig_ci.add_trace(go.Scatter(
    x=["Tahmin"],
    y=[prediction],
    error_y=dict(type='data', array=[upper - prediction], arrayminus=[prediction - lower]),
    mode='markers',
    marker=dict(color='crimson', size=12)
))
fig_ci.update_layout(title="🎯 Tahmin Güven Aralığı",
                     yaxis_title="Satış Tahmini",
                     template="plotly_white")
st.plotly_chart(fig_ci)

# --- 📈 Çoklu Mağaza Simülasyonu ---
st.subheader("🏪 Çoklu Mağaza Simülasyonu")

num_stores = st.slider("Kaç Mağaza Gösterilsin?", 1, 10, 3)
days = list(range(-30, 1))

fig_multi = go.Figure()
for s in range(1, num_stores + 1):
    noise = np.random.normal(loc=0, scale=3, size=len(days))
    trend = [lag_30 + (i * sales_diff / 30) + s * 1.5 + noise[j] for j, i in enumerate(days)]

    fig_multi.add_trace(go.Scatter(
        x=days,
        y=trend,
        name=f"Mağaza {s}",
        mode="lines+markers",
        line=dict(width=2),
        marker=dict(size=6)
    ))

fig_multi.update_layout(
    title="📊 Mağazaların Simüle Satış Trendleri",
    xaxis_title="Gün (Geçmiş)",
    yaxis_title="Tahmini Satış",
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(fig_multi, use_container_width=True)

