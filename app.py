# app.py
import streamlit as st
import csv
import math
import io
from typing import List, Tuple

st.set_page_config(page_title="Prediksi Harga Rumah (Streamlit, tanpa library lain)", layout="centered")

st.title("Prediksi Harga Rumah (Linear Regression, tanpa library lain)")
st.markdown("""
Aplikasi demo regresi linear multivariat.  
Anda bisa pakai *dataset contoh* atau **unggah CSV** (kolom: `luas`, `kamar`, `harga`).
""")

# ---------- util CSV loader ----------
def load_csv(file_like) -> Tuple[List[List[float]], List[float]]:
    """
    Membaca CSV dengan header yang mengandung 'luas', 'kamar', 'harga' (case-insensitive).
    Mengembalikan X (list of feature lists) dan y (list).
    """
    text = file_like.read().decode("utf-8") if isinstance(file_like.read, type(lambda:0)) else file_like.read()
    # Reset if file_like supports seek
    try:
        file_like.seek(0)
    except Exception:
        pass
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []
    header = [h.strip().lower() for h in rows[0]]
    try:
        idx_luas = header.index("luas")
        idx_kamar = header.index("kamar")
        idx_harga = header.index("harga")
    except ValueError:
        st.error("CSV harus punya header kolom: luas, kamar, harga")
        return [], []
    X = []
    y = []
    for r in rows[1:]:
        if len(r) <= max(idx_luas, idx_kamar, idx_harga):
            continue
        try:
            luas = float(r[idx_luas])
            kamar = float(r[idx_kamar])
            harga = float(r[idx_harga])
            X.append([luas, kamar])
            y.append(harga)
        except:
            continue
    return X, y

# ---------- default dataset (small) ----------
default_X = [
    [50, 2],
    [60, 2],
    [80, 3],
    [100, 3],
    [120, 4],
    [150, 4],
    [180, 5],
    [200, 5],
]
# Harga (juta rupiah) - contoh sintetis
default_y = [150, 180, 240, 300, 360, 450, 540, 600]

# ---------- feature scaling ----------
def compute_means_stds(X: List[List[float]]):
    n = len(X)
    if n == 0:
        return [], []
    m = len(X[0])
    means = [0.0]*m
    stds = [0.0]*m
    for j in range(m):
        s = sum(X[i][j] for i in range(n))
        means[j] = s / n
    for j in range(m):
        s2 = sum((X[i][j] - means[j])**2 for i in range(n))
        stds[j] = math.sqrt(s2 / n) if s2 > 0 else 1.0
        if stds[j] == 0:
            stds[j] = 1.0
    return means, stds

def normalize_X(X: List[List[float]], means: List[float], stds: List[float]) -> List[List[float]]:
    return [[(X[i][j] - means[j]) / stds[j] for j in range(len(means))] for i in range(len(X))]

# ---------- model (gradient descent) ----------
def predict_single(weights: List[float], x: List[float]) -> float:
    # weights[0] = bias
    s = weights[0]
    for j in range(len(x)):
        s += weights[j+1] * x[j]
    return s

def compute_rmse(weights: List[float], X: List[List[float]], y: List[float]) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        p = predict_single(weights, X[i])
        s += (p - y[i])**2
    return math.sqrt(s / n)

def gradient_descent(X: List[List[float]], y: List[float], lr: float = 0.01, epochs: int = 500) -> Tuple[List[float], List[float]]:
    n = len(y)
    m = len(X[0])  # jumlah fitur
    weights = [0.0] * (m + 1)  # bias + m weights
    losses = []
    for ep in range(epochs):
        # compute gradients
        grads = [0.0] * (m + 1)
        loss = 0.0
        for i in range(n):
            xi = X[i]
            yi = y[i]
            pred = predict_single(weights, xi)
            err = pred - yi
            loss += err * err
            grads[0] += err  # gradient for bias
            for j in range(m):
                grads[j+1] += err * xi[j]
        # average
        loss = loss / n
        for k in range(len(grads)):
            grads[k] = grads[k] * (2.0 / n)
        # update weights
        for k in range(len(weights)):
            weights[k] -= lr * grads[k]
        losses.append(loss)
        # small stopping check (optional)
        if ep > 0 and abs(losses[-1] - losses[-2]) < 1e-9:
            break
    return weights, losses

# ---------- UI: load dataset ----------
st.sidebar.header("Data & Training")
use_default = st.sidebar.checkbox("Gunakan dataset contoh (default)", value=True)
uploaded = None
if not use_default:
    uploaded = st.sidebar.file_uploader("Unggah CSV (luas,kamar,harga)", type=["csv"])
if use_default or uploaded is None:
    X_raw = default_X
    y_raw = default_y
    st.sidebar.write("Menggunakan dataset contoh (8 baris).")
else:
    X_loaded, y_loaded = load_csv(uploaded)
    if not X_loaded:
        st.sidebar.error("Gagal membaca CSV — gunakan format header 'luas,kamar,harga' dan data numerik.")
        st.stop()
    X_raw = X_loaded
    y_raw = y_loaded
    st.sidebar.success(f"Dataset berhasil dimuat: {len(y_raw)} baris.")

# ---------- preprocess ----------
means, stds = compute_means_stds(X_raw)
X_norm = normalize_X(X_raw, means, stds)

# ---------- training controls ----------
st.sidebar.subheader("Pengaturan training")
lr = st.sidebar.number_input("Learning rate (lr)", min_value=1e-6, max_value=1.0, value=0.05, format="%.6f")
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=5000, value=1000, step=10)
train_button = st.sidebar.button("Latih model")

# ---------- show dataset ----------
st.subheader("Dataset (contoh / terunggah)")
cols = st.columns(len(X_raw[0]) + 1)
headers = ["harga"] + [f"fitur_{i+1}" for i in range(len(X_raw[0]))]
# simple table
for i, row in enumerate(X_raw):
    if i == 0:
        cols[0].write("Harga")
        for j in range(len(row)):
            cols[j+1].write("Fitur")
    # display as text
st.write("Preview 5 baris:")
for i in range(min(5, len(X_raw))):
    st.write(f"luas={X_raw[i][0]}  kamar={X_raw[i][1]}  → harga={y_raw[i]}")

# ---------- training ----------
if train_button:
    with st.spinner("Melatih model..."):
        weights, losses = gradient_descent(X_norm, y_raw, lr=lr, epochs=epochs)
        rmse = compute_rmse(weights, X_norm, y_raw)
    st.success("Training selesai.")
    st.write("Koefisien model (bias + w1..wn) — model memakai fitur ter-normalisasi:")
    st.write(weights)
    st.write(f"RMSE pada data pelatihan: {rmse:.3f}")
    # show loss chart
    st.subheader("Loss (MSE) selama training")
    # st.line_chart expects a sequence of numbers; we'll show MSE per epoch
    st.line_chart(losses)

    # allow prediction with sliders (input in original units)
    st.subheader("Prediksi (gunakan input berikut)")
    luas_input = st.number_input("Luas (m2)", min_value=0.0, value=100.0)
    kamar_input = st.number_input("Jumlah kamar", min_value=0.0, value=3.0)
    # normalize inputs
    x_in_norm = [(luas_input - means[0]) / stds[0], (kamar_input - means[1]) / stds[1]]
    pred = predict_single(weights, x_in_norm)
    st.write(f"Prediksi harga: **{pred:.2f}** (sama satuan dengan target di data — mis. juta Rupiah)")

    # option to save coefficients to file
    if st.button("Simpan koefisien ke file"):
        b = weights
        content = "bias," + ",".join(f"{v:.12f}" for v in b)
        st.download_button("Download CSV koefisien", data=content, file_name="koefisien_model.csv", mime="text/csv")
else:
    st.info("Tekan tombol 'Latih model' di sidebar untuk memulai training. Anda juga dapat mengunggah CSV di sidebar jika tidak ingin dataset contoh.")

# ---------- tambahan: contoh prediksi cepat tanpa training ----------
st.markdown("---")
st.subheader("Prediksi cepat tanpa melatih (model demo sederhana)")
st.write("Jika Anda ingin langsung mencoba prediksi tanpa training, saya sediakan model contoh sederhana:")
# example fixed model: harga = 2.5*luas + 50*kamar + 20 (misalnya)
luas_q = st.number_input("Luas (demo cepat)", min_value=0.0, value=90.0, key="demo_luas")
kamar_q = st.number_input("Kamar (demo cepat)", min_value=0.0, value=3.0, key="demo_kamar")
demo_pred = 2.5 * luas_q + 50.0 * kamar_q + 20.0
st.write(f"Prediksi demo: **{demo_pred:.2f}** (satuan sama seperti dataset contoh)")

st.markdown("""
**Catatan:**
- Aplikasi ini dibuat untuk demo pendidikan. Untuk produksi/distribusi gunakan library numerik (NumPy, scikit-learn) untuk stabilitas dan performa.
- Jika Anda ingin, saya bisa modifikasi kode agar mendukung fitur tambahan: validasi silang, regularisasi, lebih banyak fitur, atau ekspor model.
""")
