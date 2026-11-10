import os
import json
import tempfile
from ekg_analyzer import (
    ECGImageProcessor,
    ECGFeatureExtractor,
    PrologDiagnosisEngine,
    ECGDatasetEvaluator,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROLOG_FILE = os.path.join(BASE_DIR, "ekg_kurallari.pl")
TEMPLATE_FILE = os.path.join(BASE_DIR, "templates", "index.html")

HASTALIK_BILGILERI = {
    "miyokard_iskemisi": {
        "kategori": "Kategori 1: Miyokard İskemisi ve Enfarktüs",
        "baslik": "Miyokard İskemisi (Myocardial Ischaemia)",
        "aciklama": "Kalp kasının oksijen ihtiyacının karşılanamaması sonucu gelişen ST çökmesi (downsloping), T dalgası düzleşmesi/negatifliği veya U dalgası negatifliği.",
        "aciliyet": "Yüksek"
    },
    "akut_miyokard_enfarktusu": {
        "kategori": "Kategori 1: Miyokard İskemisi ve Enfarktüs",
        "baslik": "Akut Miyokard Enfarktüsü (STEMI)",
        "aciklama": "Hiperakut T dalgaları, J noktasında konveks veya mezar taşı (tombstoning) ST yükselmesi ve/veya patolojik Q dalgaları ile karakterize akut koroner tıkanıklık.",
        "aciliyet": "Kritik / Acil"
    },
    "gecirilmis_miyokard_enfarktusu": {
        "kategori": "Kategori 1: Miyokard İskemisi ve Enfarktüs",
        "baslik": "Geçirilmiş Miyokard Enfarktüsü (Pathologic Q Waves)",
        "aciklama": "1 mm'den geniş veya 2 mm'den (ya da R boyunun 1/3'ünden) derin patolojik Q dalgaları ile karakterize nekroz bulgusu.",
        "aciliyet": "Orta / Klinik Takip"
    },
    "perikardit": {
        "kategori": "Kategori 2: Perikardiyal Hastalıklar",
        "baslik": "Akut Perikardit (Pericarditis)",
        "aciklama": "Yaygın konkav (eyer şeklinde) ST segment yükselmesi ve PR segment çökmesi ile karakterize perikard inflamasyonu.",
        "aciliyet": "Yüksek"
    },
    "perikardit_gec_evre": {
        "kategori": "Kategori 2: Perikardiyal Hastalıklar",
        "baslik": "Perikardit (Geç Evre - Evre II/III)",
        "aciklama": "PR çökmesi ile birlikte ST yükselmesinin düzelip T dalgalarının düzleştiği veya negatifleştiği perikardit evresi.",
        "aciliyet": "Orta"
    },
    "perikardiyal_efuzyon": {
        "kategori": "Kategori 2: Perikardiyal Hastalıklar",
        "baslik": "Perikardiyal Efüzyon / Tamponad Şüphesi",
        "aciklama": "Perikard boşluğunda sıvı birikimine bağlı düşük QRS voltajı (<4-5 mm) ve/veya atımdan atıma QRS genliğinin değiştiği elektriksel alternans.",
        "aciliyet": "Kritik / Acil"
    },
    "hiperkalemi": {
        "kategori": "Kategori 3: Elektrolit Bozuklukları",
        "baslik": "Hiperkalemi (Hyperkalaemia - K+ > 5.0 mmol/L)",
        "aciklama": "Sivri/çadır şeklinde (tented) T dalgaları, P dalgası basıklaşması, PR uzaması ve QRS genişlemesi ile karakterize potasyum yüksekliği.",
        "aciliyet": "Kritik / Acil"
    },
    "hipokalemi": {
        "kategori": "Kategori 3: Elektrolit Bozuklukları",
        "baslik": "Hipokalemi (Hypokalaemia - K+ < 3.0 mmol/L)",
        "aciklama": "ST segment çökmesi, T dalgası düzleşmesi/negatifliği, PR uzaması ve belirgin U dalgaları ile karakterize potasyum düşüklüğü.",
        "aciliyet": "Yüksek"
    },
    "hiperkalsemi": {
        "kategori": "Kategori 3: Elektrolit Bozuklukları",
        "baslik": "Hiperkalsemi (Hypercalcaemia - Ca2+ > 2.60 mmol/L)",
        "aciklama": "ST segmentinin kısalmasına bağlı kısalmış QT aralığı ve J noktasında Osborn (J) dalgaları ile karakterize kalsiyum yüksekliği.",
        "aciliyet": "Yüksek"
    },
    "hipokalsemi": {
        "kategori": "Kategori 3: Elektrolit Bozuklukları",
        "baslik": "Hipokalsemi (Hypocalcaemia - Ca2+ < 2.15 mmol/L)",
        "aciklama": "ST segmentinin uzamasına bağlı uzamış QT aralığı ile karakterize kalsiyum düşüklüğü.",
        "aciliyet": "Yüksek"
    },
    "brugada_tip1": {
        "kategori": "Kategori 4: Kalıtsal Kanalopatiler",
        "baslik": "Brugada Sendromu - Tip 1 (Coved ST Elevation)",
        "aciklama": "V1-V3 derivasyonlarında >2 mm kubbe (coved) şeklinde ST yükselmesi ve ardından negatif T dalgası (tanısal patern).",
        "aciliyet": "Kritik / Acil"
    },
    "brugada_tip2": {
        "kategori": "Kategori 4: Kalıtsal Kanalopatiler",
        "baslik": "Brugada Sendromu - Tip 2 (Saddleback ST >= 1mm)",
        "aciklama": "V1-V3 derivasyonlarında J noktasında >2 mm ve terminal ST kısmında >=1 mm eyer (saddle-shaped) ST yükselmesi.",
        "aciliyet": "Yüksek"
    },
    "brugada_tip3": {
        "kategori": "Kategori 4: Kalıtsal Kanalopatiler",
        "baslik": "Brugada Sendromu - Tip 3 (Saddleback ST < 1mm)",
        "aciklama": "V1-V3 derivasyonlarında J noktasında >2 mm ve terminal ST kısmında <1 mm eyer (saddle-shaped) ST yükselmesi.",
        "aciliyet": "Orta / Klinik Takip"
    },
    "wolff_parkinson_white": {
        "kategori": "Kategori 4: Kalıtsal Kanalopatiler",
        "baslik": "Wolff-Parkinson-White (WPW) Sendromu",
        "aciklama": "Kent demeti aksesuar yoluna bağlı kısa PR aralığı (<120 ms), QRS başlangıcında yavaş yükselen Delta dalgası ve geniş QRS (>120 ms).",
        "aciliyet": "Yüksek"
    },
    "uzun_qt_tip1": {
        "kategori": "Kategori 4: Kalıtsal Kanalopatiler",
        "baslik": "Kalıtsal Uzun QT Sendromu - Tip 1 (LQT1)",
        "aciklama": "Uzamış QT aralığı ve geniş tabanlı (broad) yüksek T dalgaları (egzersiz/stres/yüzme ile tetiklenir).",
        "aciliyet": "Yüksek"
    },
    "uzun_qt_tip2": {
        "kategori": "Kategori 4: Kalıtsal Kanalopatiler",
        "baslik": "Kalıtsal Uzun QT Sendromu - Tip 2 (LQT2)",
        "aciklama": "Uzamış QT aralığı ve çatallı/çentikli (bifid) T dalgaları (işitsel uyaranlarla tetiklenir).",
        "aciliyet": "Yüksek"
    },
    "uzun_qt_tip3": {
        "kategori": "Kategori 4: Kalıtsal Kanalopatiler",
        "baslik": "Kalıtsal Uzun QT Sendromu - Tip 3 (LQT3)",
        "aciklama": "Uzun izoelektrik ST segmenti sonrası geç başlayan (late-onset) T dalgaları ve uzamış QT aralığı (uyku/istirahatte tetiklenir).",
        "aciliyet": "Yüksek"
    },
    "arvd": {
        "kategori": "Kategori 5: Diğer Kalıtsal Hastalıklar",
        "baslik": "Aritmojenik Sağ Ventrikül Displazisi (ARVD)",
        "aciklama": "V1-V3 derivasyonlarında QRS sonunda Epsilon dalgası, lokalize QRS genişlemesi ve T dalgası negatifliği.",
        "aciliyet": "Kritik / Acil"
    },
    "cpvt": {
        "kategori": "Kategori 5: Diğer Kalıtsal Hastalıklar",
        "baslik": "Katekolaminerjik Polimorfik VT (CPVT)",
        "aciklama": "Adrenerjik deşarj sonrası gelişen, atımdan atıma QRS aksı/polaritesi değişen (alternan/bidireksiyonel) polimorfik ventriküler taşikardi.",
        "aciliyet": "Kritik / Acil"
    },
    "erken_repolarizasyon_sendromu": {
        "kategori": "Kategori 5: Diğer Kalıtsal Hastalıklar",
        "baslik": "Erken Repolarizasyon Sendromu (ERS / BER)",
        "aciklama": "QRS bitiminde belirgin J noktası çentiği (J-point notching) ve konkav ST segment yükselmesi.",
        "aciliyet": "Orta / Klinik Takip"
    },
    "sodyum_kanal_blokeri_toksisitesi": {
        "kategori": "Kategori 6: İlaç Toksisitesi",
        "baslik": "Sodyum Kanal Blokeri Toksisitesi",
        "aciklama": "Faz 0 sodyum girişinin bloke olmasına bağlı belirgin QRS genişlemesi (>100-120 ms) ve ventriküler aritmi riski.",
        "aciliyet": "Kritik / Acil"
    },
    "digoksin_etkisi_veya_toksisitesi": {
        "kategori": "Kategori 6: İlaç Toksisitesi",
        "baslik": "Digoksin Etkisi / Toksisitesi (Scooped ST Depression)",
        "aciklama": "Karakteristik kaşık şeklinde (scooped / Salvador Dali bıyığı / reverse-tick) ST çökmesi ve T dalgası değişiklikleri.",
        "aciliyet": "Yüksek"
    },
    "digoksin_toksisitesi_yavaslamis_iletim": {
        "kategori": "Kategori 6: İlaç Toksisitesi",
        "baslik": "Digoksin Toksisitesi (Yavaşlamış AV İletim)",
        "aciklama": "Artmış vagal (parasempatik) tonusa bağlı kaşık şeklinde ST çökmesi ve uzamış PR aralığı (>200 ms) / AV blok.",
        "aciliyet": "Yüksek"
    },
    "digoksin_toksisitesi_artmis_otomatiklik": {
        "kategori": "Kategori 6: İlaç Toksisitesi",
        "baslik": "Digoksin Toksisitesi (Artmış Otomatiklik & Kısa QT)",
        "aciklama": "Hücre içi kalsiyum artışına bağlı kaşık şeklinde (reverse-tick) ST çökmesi, kısalmış QT aralığı ve aritmi eğilimi.",
        "aciliyet": "Kritik / Acil"
    },
    "pulmoner_emboli": {
        "kategori": "Kategori 7: Diğer Klinik Durumlar",
        "baslik": "Pulmoner Emboli (S1Q3T3 / Sağ Yüklenme)",
        "aciklama": "Sinüs taşikardisi, S1Q3T3 paterni, sağ dal bloğu veya sağ aks sapması ile karakterize akut pulmoner tromboemboli.",
        "aciliyet": "Kritik / Acil"
    },
    "pulmoner_emboli_suphesi": {
        "kategori": "Kategori 7: Diğer Klinik Durumlar",
        "baslik": "Sinüs Taşikardisi / Pulmoner Emboli Şüphesi",
        "aciklama": "Açıklanamayan sinüs taşikardisi (>100 bpm); klinik bulgular eşliğinde pulmoner emboli açısından değerlendirilmelidir.",
        "aciliyet": "Orta"
    },
    "akut_inme": {
        "kategori": "Kategori 7: Diğer Klinik Durumlar",
        "baslik": "Akut İnme (Serebrovasküler Olay - Nörojenik EKG)",
        "aciklama": "Taşikardi, ST çökmesi, uzamış QTc aralığı, T dalgası düzleşmesi/negatifliği ve belirgin U dalgaları (Serebral T/U dalgaları).",
        "aciliyet": "Kritik / Acil"
    },
    "hipotermi": {
        "kategori": "Kategori 7: Diğer Klinik Durumlar",
        "baslik": "Hipotermi (Osborn / J Dalgası - <35°C)",
        "aciklama": "QRS-ST birleşiminde (J noktasında) karakteristik kubbe/tümsek şeklinde Osborn (J) dalgası ve/veya titreme artefaktı.",
        "aciliyet": "Kritik / Acil"
    },
    "derin_hipotermi": {
        "kategori": "Kategori 7: Diğer Klinik Durumlar",
        "baslik": "Derin Hipotermi (<32°C)",
        "aciklama": "Bradikardi ile birlikte PR, QRS ve QT aralıklarının tümünde belirgin uzama ve Osborn dalgaları.",
        "aciliyet": "Kritik / Acil"
    }
}


def analyze_ecg_file(file_path, filename="ekg.jpg", crop_box=None, lead_mode="auto"):
    """Görsel dosyasını DSP + Prolog boru hattından geçirip JSON-uyumlu sözlük döndürür."""
    processor = ECGImageProcessor(file_path, crop_box=crop_box, lead_mode=lead_mode)
    img = processor._load_image_bgr()
    panels = processor._extract_panels(img)
    comp_mm, c_mm, u_mm, l_mm, fs = panels[0]

    ham_veriler = processor.process_image()
    extractor = ECGFeatureExtractor()
    sembolik_bulgular = extractor.tum_bulgulari_cikar(ham_veriler)

    engine = PrologDiagnosisEngine(PROLOG_FILE)
    teshis_kodlari = engine.teshis_koy("hasta_web", sembolik_bulgular)
    mendeley_sinifi = ECGDatasetEvaluator.map_to_mendeley_class(sembolik_bulgular, teshis_kodlari)

    detayli_teshisler = []
    for kod in teshis_kodlari:
        bilgi = HASTALIK_BILGILERI.get(
            kod,
            {
                "kategori": "Klinik EKG Bulgusu",
                "baslik": kod.replace("_", " ").upper(),
                "aciklama": f"Prolog bilgi tabanı tarafından '{kod}' teşhisi doğrulandı.",
                "aciliyet": "Yüksek",
            },
        )
        detayli_teshisler.append({"kod": kod, **bilgi})

    step = max(1, len(comp_mm) // 350)
    signal_points = [round(float(x), 2) for x in comp_mm[::step]]
    time_points = [round(float(i * step) / fs, 3) for i in range(len(signal_points))]

    temiz_ham = {k: v for k, v in ham_veriler.items() if k not in ("alt_paneller", "kalite_raporu")}

    return {
        "status": "success",
        "filename": filename,
        "panel_count": len(panels),
        "fs": round(float(fs), 1),
        "mendeley_class": mendeley_sinifi,
        "quality_report": processor.last_quality_report,
        "raw_metrics": temiz_ham,
        "prolog_facts": sembolik_bulgular,
        "diagnoses": detayli_teshisler,
        "waveform": {"time_s": time_points, "amplitude_mm": signal_points},
    }


try:
    from flask import Flask, render_template, request, jsonify

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/analyze", methods=["POST"])
    def analyze_ecg():
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "Lütfen bir EKG görseli yükleyin."}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "Dosya seçilmedi."}), 400

        crop_box = None
        crop_str = request.form.get("crop_box", "").strip()
        if crop_str:
            try:
                parts = [float(x) for x in crop_str.split(",")]
                if len(parts) == 4:
                    crop_box = tuple(parts)
            except Exception:
                crop_box = None

        lead_mode = request.form.get("lead_mode", "auto").strip() or "auto"

        ext = os.path.splitext(uploaded_file.filename)[1].lower() or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            uploaded_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = analyze_ecg_file(tmp_path, uploaded_file.filename, crop_box=crop_box, lead_mode=lead_mode)
            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.route("/api/benchmark", methods=["GET", "POST"])
    def run_benchmark_api():
        try:
            evaluator = ECGDatasetEvaluator(PROLOG_FILE)
            docx_cand = os.path.join(BASE_DIR, "ecg-disease-patterns.docx")
            report = evaluator.run_comprehensive_benchmark(
                docx_path=docx_cand if os.path.exists(docx_cand) else None,
                user_uploads_dir=BASE_DIR,
            )
            return jsonify({"status": "success", "benchmark": report})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

except ImportError:
    app = None


if __name__ == "__main__":
    if app is not None:
        print("EKG Uzman Sistemi (Flask) başlatılıyor: http://127.0.0.1:5000")
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class FallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(TEMPLATE_FILE, "rb") as f:
                    self.wfile.write(f.read())

        print("Flask kurulu değil, standart HTTP sunucusu başlatılıyor: http://127.0.0.1:5000")
        HTTPServer(("0.0.0.0", 5000), FallbackHandler).serve_forever()
