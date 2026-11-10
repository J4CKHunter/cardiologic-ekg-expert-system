# CardioLogic AI — Flask & SWI-Prolog Tabanlı EKG Uzman Sistemi

Bu proje, **Dijital Sinyal İşleme (DSP)** katmanı (`ekg_analyzer.py`) ile **SWI-Prolog Bilgi Tabanı** (`ekg_kurallari.pl`) çıkarım motorunu birleştiren tek sayfalık modern bir **Python / Flask** web uygulamasıdır.

## Klasör Yapısı

```text
ekg_flask_projesi/
├── app.py                  # Flask Web Sunucusu ve /api/analyze REST API uç noktası
├── ekg_analyzer.py         # 3 Katmanlı DSP + Öznitelik Çıkarımı + PySwip Prolog Motoru
├── ekg_kurallari.pl        # 7 Kategori / 19 Hastalık Prolog Bilgi Tabanı
├── requirements.txt        # Python bağımlılıkları
├── README.md               # Kurulum ve çalıştırma kılavuzu
└── templates/
    └── index.html          # Tek sayfalık interaktif web arayüzü (Tailwind + Chart.js)
```

---

## 1. Google Colab Üzerinden Doğrudan Çalıştırma

Google Colab, kendi içinde çalışan `localhost:5000` portunu dış dünyaya veya hücre içine güvenli bir proxy ile bağlayabildiği için **ngrok veya ek üyelik gerektirmeden** doğrudan Colab üzerinde çalıştırılabilir (`homework1 copy.ipynb` dosyasının son hücresinde hazır olarak bulunmaktadır):

```python
!sudo apt-get update && sudo apt-get install -y swi-prolog
!pip install flask pyswip opencv-python-headless scipy numpy pillow

import threading
from google.colab import output
from google.colab.output import eval_js
from app import app

# Flask sunucusunu arka planda başlat
threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False),
    daemon=True,
).start()

# 1) Tarayıcıda yeni sekmede açmak için tıklanabilir güvenli Colab linki:
print("Web Uygulaması Linki:", eval_js("google.colab.kernel.proxyPort(5000)"))

# 2) Arayüzü doğrudan Colab hücresinin içinde görüntülemek için:
output.serve_kernel_port_as_iframe(5000, height="850")
```

---

## 2. Kendi Bilgisayarında (`localhost:5000`) Çalıştırma Adımları

### Adım 1: SWI-Prolog Kurulumu (`pyswip` kütüphanesi için zorunludur)
* **Windows:** [swi-prolog.org](https://www.swi-prolog.org/Download.html) adresinden 64-bit Windows kurulum paketini indirip kurarken **"Add swipl to the system PATH"** kutucuğunu işaretleyin.
* **macOS (Homebrew):**
  ```bash
  brew install swi-prolog
  ```
* **Linux (Ubuntu / Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y swi-prolog
  ```

### Adım 2: Sanal Ortam (Virtual Environment) ve Python Paketlerinin Kurulumu
Proje klasörünün içinde terminal / komut istemini açarak aşağıdaki komutları çalıştırın:

```bash
# Sanal ortam oluşturma
python -m venv venv

# Sanal ortamı etkinleştirme:
# Windows (CMD / PowerShell):
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Gerekli kütüphaneleri yükleme
pip install -r requirements.txt
```

### Adım 3: Flask Sunucusunu Başlatma
```bash
python app.py
```
Terminalde `Running on http://127.0.0.1:5000` mesajını gördükten sonra tarayıcınızdan **`http://127.0.0.1:5000`** (veya `http://localhost:5000`) adresine giderek EKG görsellerini yükleyip analiz edebilirsiniz.
