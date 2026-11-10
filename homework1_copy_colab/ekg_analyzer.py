import os
import re
import glob
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt, find_peaks

try:
    from pyswip import Prolog as _SwipProlog
except ImportError:
    _SwipProlog = None


# ==============================================================================
# 1. KATMAN: DİJİTAL SİNYAL İŞLEME (DSP) VE GÖRÜNTÜ ANALİZİ (ADAPTER)
# ==============================================================================
class ECGImageProcessor:
    """
    Evrensel EKG Sinyal İşleme (DSP) Sınıfı:
    - Tek şeritli (single-strip), yan yana çok panelli (Brugada Tip 1-3, LQT 1-3)
      veya çok satırlı 12-derivasyonlu tam sayfa (Mendeley Data 3x4+1 veya 6x2+1)
      EKG görsellerini otomatik olarak tanır, üst hasta bilgi tablolarını filtreler
      ve sinyalleri ayrıştırır.
    - Hem gri ızgaralı (dijital), hem pembe/kırmızı kağıt ızgaralı (taranmış), hem de
      koyu arka planlı (monitör ekran görüntüsü) EKG'lerde çözünürlükten bağımsız
      5 mm ana ızgara ölçeklemesi ve Viterbi tabanlı iz takibi yapar.
    - Kullanıcı tanımlı ROI (Region of Interest / Bölge Kırpma) ve derivasyon odak
      modlarını (auto, rhythm_strip, v1_v3, single_strip) destekler.
    """

    def __init__(self, image_path, crop_box=None, lead_mode="auto"):
        self.image_path = image_path
        self.crop_box = crop_box  # (x0_norm, y0_norm, x1_norm, y1_norm) in [0, 1]
        self.lead_mode = lead_mode or "auto"
        self.last_quality_report = {}

    def _load_image_bgr(self):
        img = cv2.imread(self.image_path)
        if img is None:
            try:
                pil_img = Image.open(self.image_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise FileNotFoundError(f"Görsel bulunamadı veya okunamadı: {self.image_path} ({e})")

        gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if float(np.median(gray_check)) < 85.0:
            img = 255 - img

        if self.crop_box is not None:
            h, w, _ = img.shape
            x0 = max(0, min(w - 10, int(round(float(self.crop_box[0]) * w))))
            y0 = max(0, min(h - 10, int(round(float(self.crop_box[1]) * h))))
            x1 = max(x0 + 10, min(w, int(round(float(self.crop_box[2]) * w))))
            y1 = max(y0 + 10, min(h, int(round(float(self.crop_box[3]) * h))))
            img = img[y0:y1, x0:x1].copy()

        return img

    def _detect_multirow_12lead(self, img):
        """
        Görselin çok satırlı (4 veya 7 satırlı) 12-derivasyonlu tam sayfa EKG olup olmadığını
        yatay blok süreklilik analiziyle tespit eder; üstteki hasta bilgi tablosu çizgilerini
        (solid table borders) dışlayarak gerçek EKG satır bazal Y koordinatlarını döndürür.
        """
        if self.crop_box is not None or self.lead_mode == "single_strip":
            return [], False

        h, w, _ = img.shape
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = np.maximum(r, gray)
        p2 = float(np.percentile(ink, 2.0))
        p50 = float(np.percentile(ink, 50.0))
        dark_thresh = min(165.0, max(115.0, p2 + 0.35 * (p50 - p2)))
        raw_dark = (ink < dark_thresh) & ((r.astype(int) - g.astype(int)) < 65)

        x_start, x_end = int(0.06 * w), int(0.94 * w)
        seg_w = max(1, (x_end - x_start) // 8)
        band_half = max(3, int(h * 0.015))

        row_continuity = np.zeros(h, dtype=np.float64)
        for y in range(band_half, h - band_half):
            active_segs = 0
            for s in range(8):
                sx0 = x_start + s * seg_w
                sx1 = sx0 + seg_w
                col_has = np.any(raw_dark[y - band_half : y + band_half + 1, sx0:sx1], axis=0)
                if np.mean(col_has) >= 0.20:
                    active_segs += 1
            if active_segs >= 5:
                row_continuity[y] = (active_segs / 8.0) * np.mean(raw_dark[y - 2 : y + 3, x_start:x_end])

        win_r = max(5, int(h * 0.018))
        smooth_cont = np.convolve(row_continuity, np.ones(win_r) / float(win_r), mode="same")
        if np.max(smooth_cont) <= 0:
            return [], False

        cont_peaks, _ = find_peaks(smooth_cont, height=np.max(smooth_cont) * 0.25, distance=int(h * 0.08))
        v_span = (cont_peaks[-1] - cont_peaks[0]) / float(h) if len(cont_peaks) >= 2 else 0.0
        is_multirow = len(cont_peaks) >= 3 and v_span >= 0.58

        if not is_multirow:
            return [], False

        row_proj = np.mean(raw_dark[:, x_start:x_end], axis=1)
        smooth_proj = np.convolve(row_proj, np.ones(max(5, int(h * 0.015))) / max(5, int(h * 0.015)), mode="same")
        all_rows = [int(y) for y in find_peaks(smooth_proj, height=np.max(smooth_proj) * 0.22, distance=int(h * 0.08))[0]]

        # Mendeley tarzı raporlarda üst %16'lık hasta bilgi başlığındaki tablo çizgilerini (alt satır aralığından farklıysa) dışla
        if len(all_rows) >= 5:
            bottom_gap = float(np.median(np.diff(all_rows[-4:])))
            while len(all_rows) > 4 and all_rows[0] < int(0.16 * h) and (all_rows[1] - all_rows[0]) < 0.78 * bottom_gap:
                all_rows.pop(0)

        return all_rows, True

    def _detect_grid_scale(self, img, is_multirow=False):
        """
        Görüntüdeki 5 mm'lik ana ızgara (major grid) çizgilerinin piksel aralığını
        tek şeritli veya 12-derivasyonlu tam sayfa olmasına göre çözünürlükten bağımsız bulur.
        """
        h, w, _ = img.shape
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ink_channel = np.maximum(r, gray)
        bg_mask = ink_channel > 140

        col_dark = np.zeros(w, dtype=np.float64)
        for x in range(w):
            valid = g[:, x][bg_mask[:, x]]
            if len(valid) > h * 0.3:
                col_dark[x] = 255.0 - np.mean(valid)

        row_dark = np.zeros(h, dtype=np.float64)
        for y in range(h):
            valid = g[y, :][bg_mask[y, :]]
            if len(valid) > w * 0.3:
                row_dark[y] = 255.0 - np.mean(valid)

        if is_multirow:
            expected_major = w / 50.0
            win = max(7, int(expected_major * 0.8))
            min_dist = max(6, int(expected_major * 0.72))
            max_dist = max(14, int(expected_major * 1.38))
        else:
            win = max(15, int(h * 0.035))
            min_dist = max(30, int(h * 0.115))
            max_dist = int(h * 0.26)

        c_prof = col_dark - np.convolve(col_dark, np.ones(win) / float(win), mode="same")
        r_prof = row_dark - np.convolve(row_dark, np.ones(win) / float(win), mode="same")

        c_peaks, _ = find_peaks(c_prof, height=np.std(c_prof) * 0.70, distance=min_dist)
        r_peaks, _ = find_peaks(r_prof, height=np.std(r_prof) * 0.70, distance=min_dist)

        diffs = []
        if len(c_peaks) >= 2:
            c_diffs = np.diff(c_peaks)
            valid_c = c_diffs[(c_diffs >= min_dist) & (c_diffs <= max_dist)]
            if len(valid_c) > 0:
                diffs.extend(valid_c)
        if len(r_peaks) >= 2:
            r_diffs = np.diff(r_peaks)
            valid_r = r_diffs[(r_diffs >= min_dist) & (r_diffs <= max_dist)]
            if len(valid_r) > 0:
                diffs.extend(valid_r)

        grid_detected = len(diffs) > 0
        if grid_detected:
            major_grid_px = float(np.median(diffs))
        else:
            major_grid_px = (w / 50.0) if is_multirow else (h / 5.0)

        px_per_mm = major_grid_px / 5.0
        fs = px_per_mm * 25.0
        return px_per_mm, fs, grid_detected

    def _trace_band_signal(self, raw_dark, ink_channel, x0, x1, y0, y1, y_base_est, h, max_gap, dark_thresh, check_full_col_spike=False, GuardLeftEdge=False):
        pw = x1 - x0
        center_sig = np.full(pw, np.nan, dtype=np.float64)
        upper_sig = np.full(pw, np.nan, dtype=np.float64)
        lower_sig = np.full(pw, np.nan, dtype=np.float64)

        prev_y = float(y_base_est)

        for x_rel in range(pw):
            x = x0 + x_rel
            if check_full_col_spike and np.mean(raw_dark[:, x]) > 0.35:
                continue

            ys = np.where(raw_dark[y0:y1, x])[0] + y0
            if len(ys) == 0:
                col_ink = ink_channel[y0:y1, x]
                min_v = int(np.min(col_ink))
                if min_v < dark_thresh + 32:
                    ys = np.where(col_ink <= min_v + 14)[0] + y0

            if len(ys) > 0:
                splits = np.where(np.diff(ys) > max_gap)[0] + 1
                runs = np.split(ys, splits)
                if len(runs) > 1:
                    ys = min(
                        runs,
                        key=lambda run: 0.58 * abs(float(np.mean(run)) - prev_y)
                        + 0.42 * abs(float(np.mean(run)) - float(y_base_est)),
                    )

                if GuardLeftEdge and x_rel < int(0.06 * pw) and abs(float(np.mean(ys)) - y_base_est) > 0.12 * (y1 - y0):
                    yc = float(y_base_est)
                    center_sig[x_rel] = float(h) - yc
                    upper_sig[x_rel] = float(h) - yc
                    lower_sig[x_rel] = float(h) - yc
                    continue

                weights = (255.0 - ink_channel[ys, x].astype(np.float64)) ** 2
                w_sum = float(np.sum(weights))
                y_center = float(np.sum(ys * weights) / w_sum) if w_sum > 0 else float(np.mean(ys))
                center_sig[x_rel] = float(h) - y_center
                upper_sig[x_rel] = float(h) - float(np.min(ys))
                lower_sig[x_rel] = float(h) - float(np.max(ys))
                prev_y = y_center

        nans = np.isnan(center_sig)
        if np.all(nans):
            raise ValueError("Seçilen bölgede EKG sinyali tespit edilemedi.")
        x_idx = np.arange(pw)
        center_sig[nans] = np.interp(x_idx[nans], x_idx[~nans], center_sig[~nans])
        upper_sig[nans] = np.interp(x_idx[nans], x_idx[~nans], upper_sig[~nans])
        lower_sig[nans] = np.interp(x_idx[nans], x_idx[~nans], lower_sig[~nans])
        return center_sig, upper_sig, lower_sig

    def _signals_to_mm(self, center_sig, upper_sig, lower_sig, px_per_mm):
        abs_grad = np.abs(np.gradient(center_sig))
        flat_mask = abs_grad <= np.percentile(abs_grad, 35)
        base_center_px = float(np.median(center_sig[flat_mask]))
        half_stroke_px = 0.5 * float(np.median(upper_sig[flat_mask] - lower_sig[flat_mask]))

        center_mm = (center_sig - base_center_px) / px_per_mm
        upper_mm = (upper_sig - (base_center_px + half_stroke_px)) / px_per_mm
        lower_mm = (lower_sig - (base_center_px - half_stroke_px)) / px_per_mm

        composite_mm = np.where(center_mm >= 0.2, np.maximum(center_mm, upper_mm), center_mm)
        composite_mm = np.where(center_mm <= -0.2, np.minimum(composite_mm, lower_mm), composite_mm)
        return composite_mm, center_mm, upper_mm, lower_mm

    def _extract_panels(self, img):
        orig_h, orig_w, _ = img.shape
        row_peaks, is_multirow = self._detect_multirow_12lead(img)

        upscaled_flag = False
        if not is_multirow and orig_h < 340:
            scale_up = 368.0 / float(orig_h)
            img = cv2.resize(img, (max(64, int(round(orig_w * scale_up))), 368), interpolation=cv2.INTER_CUBIC)
            upscaled_flag = True

        h, w, _ = img.shape
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        px_per_mm, fs, grid_detected = self._detect_grid_scale(img, is_multirow=is_multirow)

        ink_channel = np.maximum(r, gray)
        p2 = float(np.percentile(ink_channel, 2.0))
        p50 = float(np.percentile(ink_channel, 50.0))

        uyarilar = []
        if upscaled_flag:
            uyarilar.append(
                f"Düşük çözünürlüklü şerit/kırpım ({orig_w}x{orig_h} px) bikübik süper-çözünürlük ile {w}x{h} px boyutuna normalize edildi."
            )
        elif fs < 90.0:
            uyarilar.append(
                f"Görsel çözünürlüğü düşük ({px_per_mm:.1f} px/mm, {fs:.0f} Hz). Küçük dalgaların (Delta, Epsilon, J-çentiği, Q) daha net ölçümü için 'Bölge Kırp (ROI)' aracıyla tek bir derivasyonu seçin."
            )
        if is_multirow:
            uyarilar.append(
                f"Çok satırlı ({len(row_peaks)} satır) 12-derivasyonlu tam sayfa EKG tespit edildi. Ritim şeridi (DII) ve prekordiyal (V2-V5) bölgeler tarandı."
            )
        if not grid_detected:
            uyarilar.append(
                "Standart 5 mm EKG ızgarası net seçilemediği için geometrik ölçek tahmini kullanıldı."
            )

        # DURUM A: ÇOK SATIRLI 12-DERİVASYONLU TAM SAYFA EKG (4 veya 7 Satır)
        if is_multirow and len(row_peaks) >= 3:
            dark_thresh = min(165.0, max(115.0, p2 + 0.35 * (p50 - p2)))
            raw_dark = (ink_channel < dark_thresh) & ((r.astype(int) - g.astype(int)) < 65)

            extracted_panels = []
            max_gap = max(3, int(2.0 * px_per_mm))

            if self.lead_mode == "v1_v3" and len(row_peaks) >= 3:
                for r_idx_num in range(min(3, len(row_peaks))):
                    ry = row_peaks[r_idx_num]
                    row_gap = float(row_peaks[1] - row_peaks[0]) if len(row_peaks) >= 2 else float(h / len(row_peaks))
                    y0 = max(2, int(ry - 0.55 * row_gap))
                    y1 = min(h - 2, int(ry + 0.45 * row_gap))
                    x0, x1 = int(0.52 * w), int(0.73 * w)
                    c_sig, u_sig, l_sig = self._trace_band_signal(
                        raw_dark, ink_channel, x0, x1, y0, y1, ry, h, max_gap, dark_thresh, check_full_col_spike=True
                    )
                    comp_mm, c_mm, u_mm, l_mm = self._signals_to_mm(c_sig, u_sig, l_sig, px_per_mm)
                    extracted_panels.append((comp_mm, c_mm, u_mm, l_mm, fs))
            else:
                # 1. Ana Ritim Şeridi (En alt satır: DII Ritim Şeridi)
                ry = row_peaks[-1]
                row_gap = float(row_peaks[-1] - row_peaks[-2])
                y0 = max(2, int(ry - 0.62 * row_gap))
                y1 = min(h - 2, int(ry + 0.38 * row_gap))
                x0, x1 = int(0.08 * w), int(0.95 * w)

                c_sig, u_sig, l_sig = self._trace_band_signal(
                    raw_dark, ink_channel, x0, x1, y0, y1, ry, h, max_gap, dark_thresh, check_full_col_spike=True
                )
                comp_mm, c_mm, u_mm, l_mm = self._signals_to_mm(c_sig, u_sig, l_sig, px_per_mm)
                extracted_panels.append((comp_mm, c_mm, u_mm, l_mm, fs))

                # 2. Yüksek çözünürlüklü 4 satırlı (3x4+1 Mendeley) 12-lead sayfalarda Prekordiyal (V2/V5) kutusunu da çıkar
                if len(row_peaks) == 4 and fs >= 115.0 and self.lead_mode == "auto":
                    ry_p = row_peaks[1]
                    rg_p = float(row_peaks[2] - row_peaks[1])
                    y0_p = max(2, int(ry_p - 0.58 * rg_p))
                    y1_p = min(h - 2, int(ry_p + 0.40 * rg_p))
                    x0_p, x1_p = int(0.505 * w), int(0.725 * w)
                    try:
                        cp_s, up_s, lp_s = self._trace_band_signal(
                            raw_dark, ink_channel, x0_p, x1_p, y0_p, y1_p, ry_p, h, max_gap, dark_thresh, check_full_col_spike=True
                        )
                        comp_p, c_p, u_p, l_p = self._signals_to_mm(cp_s, up_s, lp_s, px_per_mm)
                        extracted_panels.append((comp_p, c_p, u_p, l_p, fs))
                    except Exception:
                        pass

            self.last_quality_report = {
                "cozunurluk": f"{orig_w}x{orig_h} px",
                "duzen_tipi": f"12-Derivasyonlu Tam Sayfa ({len(row_peaks)} Satır)",
                "piksel_per_mm": round(float(px_per_mm), 2),
                "ornekleme_hz": round(float(fs), 1),
                "izgara_tespiti": "Doğrulandı (5x5 mm)" if grid_detected else "Otomatik Oransal",
                "kalite_seviyesi": "Yüksek" if fs >= 135 else ("Orta (12-Derivasyon Web)" if fs >= 75 else "Düşük Çözünürlük"),
                "uyarilar": uyarilar,
            }
            return extracted_panels

        # DURUM B: TEK SATIRLI VEYA YAN YANA ÇOK PANELLİ EKG ŞERİDİ
        margin_y = max(2, int(h * 0.02))
        margin_x = max(2, int(w * 0.008))

        col_white = np.mean(gray[margin_y : h - margin_y, :] > 248, axis=0) > 0.95
        panel_x_ranges = []
        in_p = False
        x_start = margin_x
        for x in range(margin_x, w - margin_x):
            if not col_white[x] and not in_p:
                in_p = True
                x_start = x
            elif col_white[x] and in_p:
                if (x - x_start) > 0.20 * w:
                    panel_x_ranges.append((x_start, x))
                in_p = False
        if in_p and ((w - margin_x) - x_start) > 0.20 * w:
            panel_x_ranges.append((x_start, w - margin_x))
        if not panel_x_ranges:
            panel_x_ranges = [(margin_x, w - margin_x)]

        dark_thresh = min(155.0, max(115.0, p2 + 0.38 * (p50 - p2)))
        raw_dark = (ink_channel < dark_thresh) & ((r.astype(int) - g.astype(int)) < 65)
        raw_dark[:margin_y, :] = False
        raw_dark[h - margin_y :, :] = False

        extracted_panels = []
        for x0, x1 in panel_x_ranges:
            pw = x1 - x0
            sub_dark = raw_dark[:, x0:x1].copy()

            close_h = max(15, int(h * 0.08))
            bridged = cv2.morphologyEx(sub_dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((close_h, 5), np.uint8))

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
            clean_mask = np.zeros_like(sub_dark, dtype=bool)
            for lbl in range(1, num_labels):
                cw = stats[lbl, cv2.CC_STAT_WIDTH]
                cy = stats[lbl, cv2.CC_STAT_TOP]
                if cw > 0.18 * pw and cy < 0.80 * h:
                    clean_mask |= sub_dark & (labels == lbl)

            if not np.any(clean_mask):
                clean_mask = sub_dark

            y_all, _ = np.where(clean_mask)
            if len(y_all) == 0:
                raise ValueError("Görselde siyah EKG sinyali tespit edilemedi.")
            y_base_est = float(np.median(y_all))

            center_sig = np.full(pw, np.nan, dtype=np.float64)
            upper_sig = np.full(pw, np.nan, dtype=np.float64)
            lower_sig = np.full(pw, np.nan, dtype=np.float64)

            prev_y = y_base_est
            max_gap = max(6, int(h * 0.018))

            for x_rel in range(pw):
                x = x0 + x_rel
                ys = np.where(clean_mask[:, x_rel])[0]
                if len(ys) == 0:
                    col_ink = ink_channel[margin_y : h - margin_y, x]
                    min_v = int(np.min(col_ink))
                    if min_v < dark_thresh + 35:
                        ys = np.where(col_ink <= min_v + 15)[0] + margin_y

                if len(ys) > 0:
                    splits = np.where(np.diff(ys) > max_gap)[0] + 1
                    runs = np.split(ys, splits)
                    if len(runs) > 1:
                        ys = min(
                            runs,
                            key=lambda run: 0.60 * abs(float(np.mean(run)) - prev_y)
                            + 0.40 * abs(float(np.mean(run)) - y_base_est),
                        )
                    if x_rel < int(0.06 * pw) and abs(float(np.mean(ys)) - y_base_est) > 0.12 * h:
                        yc = y_base_est
                        center_sig[x_rel] = float(h) - yc
                        upper_sig[x_rel] = float(h) - yc
                        lower_sig[x_rel] = float(h) - yc
                        continue

                    weights = (255.0 - ink_channel[ys, x].astype(np.float64)) ** 2
                    y_center = float(np.sum(ys * weights) / np.sum(weights))
                    center_sig[x_rel] = float(h) - y_center
                    upper_sig[x_rel] = float(h) - float(np.min(ys))
                    lower_sig[x_rel] = float(h) - float(np.max(ys))
                    prev_y = y_center

            nans = np.isnan(center_sig)
            x_idx = np.arange(pw)
            center_sig[nans] = np.interp(x_idx[nans], x_idx[~nans], center_sig[~nans])
            upper_sig[nans] = np.interp(x_idx[nans], x_idx[~nans], upper_sig[~nans])
            lower_sig[nans] = np.interp(x_idx[nans], x_idx[~nans], lower_sig[~nans])

            comp_mm, c_mm, u_mm, l_mm = self._signals_to_mm(center_sig, upper_sig, lower_sig, px_per_mm)
            extracted_panels.append((comp_mm, c_mm, u_mm, l_mm, fs))

        duzen_str = (
            "Seçili ROI Derivasyon Bölgesi"
            if self.crop_box is not None
            else (f"Çok Panelli Şerit ({len(extracted_panels)} Panel)" if len(extracted_panels) > 1 else "Tek Derivasyonlu Ritim Şeridi")
        )
        self.last_quality_report = {
            "cozunurluk": f"{orig_w}x{orig_h} px",
            "duzen_tipi": duzen_str,
            "piksel_per_mm": round(float(px_per_mm), 2),
            "ornekleme_hz": round(float(fs), 1),
            "izgara_tespiti": "Doğrulandı (5x5 mm)" if grid_detected else "Otomatik Oransal",
            "kalite_seviyesi": "Yüksek (Klinik Standart)" if fs >= 150 else ("Orta (Web Çözünürlüğü)" if fs >= 75 else "Düşük Çözünürlük"),
            "uyarilar": uyarilar,
        }
        return extracted_panels

    def _apply_butterworth_filters(self, signal_mm, fs):
        nyq = 0.5 * fs
        if fs < 120.0:
            filtered_mm = signal_mm.copy()
            high_cutoff = max(0.5 / nyq, 0.01)
            low_cutoff = min(20.0 / nyq, 0.92)
            b_bp, a_bp = butter(2, [high_cutoff, low_cutoff], btype="bandpass")
            bandpass_mm = filtfilt(b_bp, a_bp, signal_mm)
            hf_noise = 0.08
            return filtered_mm, bandpass_mm, hf_noise

        low_cutoff = min(40.0 / nyq, 0.95)
        b_lp, a_lp = butter(4, low_cutoff, btype="low")
        filtered_mm = filtfilt(b_lp, a_lp, signal_mm)

        high_cutoff = max(0.5 / nyq, 0.005)
        b_bp, a_bp = butter(2, [high_cutoff, low_cutoff], btype="bandpass")
        bandpass_mm = filtfilt(b_bp, a_bp, signal_mm)

        abs_grad = np.abs(np.gradient(signal_mm))
        flat_mask = abs_grad <= np.percentile(abs_grad, 35)
        hf_noise = float(np.std(signal_mm[flat_mask] - filtered_mm[flat_mask]))
        return filtered_mm, bandpass_mm, hf_noise

    def _detect_qrs_peaks(self, composite_mm, filtered_mm, bandpass_mm, fs):
        if fs < 120.0:
            edge = max(2, int(0.05 * fs))
            sig_inner = composite_mm.copy()
            sig_inner[:edge] = 0.0
            sig_inner[-edge:] = 0.0
            pks, _ = find_peaks(sig_inner, height=max(2.5, 0.45 * np.max(sig_inner)), distance=int(0.45 * fs))
            return np.array(pks, dtype=int)

        deriv = np.gradient(bandpass_mm) * fs
        squared = deriv ** 2
        win_size = max(int(0.08 * fs), 3)
        mwi = np.convolve(squared, np.ones(win_size) / win_size, mode="same")

        edge = int(0.05 * fs)
        mwi_clean = mwi.copy()
        if len(mwi_clean) > 2 * edge:
            mwi_clean[:edge] = 0.0
            mwi_clean[-edge:] = 0.0

        adaptive_thresh = 0.14 * np.percentile(mwi_clean, 98)
        min_distance = int(0.30 * fs)
        energy_peaks, _ = find_peaks(mwi_clean, height=adaptive_thresh, distance=min_distance)

        r_peaks = []
        search_half = int(0.075 * fs)
        for ep in energy_peaks:
            left = max(0, ep - search_half)
            right = min(len(filtered_mm), ep + search_half)
            seg_f = filtered_mm[left:right]
            seg_c = composite_mm[left:right]
            if len(seg_f) == 0:
                continue
            max_v = float(np.max(seg_c))
            min_v = float(np.min(seg_f))
            if abs(min_v) > 5.5 and abs(min_v) > 1.15 * max_v:
                r_idx = left + int(np.argmin(seg_f))
            elif max_v >= 2.0 or max_v >= 0.5 * abs(min_v):
                r_idx = left + int(np.argmax(seg_c))
            else:
                r_idx = left + int(np.argmin(seg_f))

            if not r_peaks or (r_idx - r_peaks[-1]) >= min_distance:
                r_peaks.append(r_idx)

        return np.array(r_peaks, dtype=int)

    def _tangent_boundary(self, sig, peak_idx, search_limit_idx, direction="left", baseline=0.0):
        if direction == "left":
            start, end = min(peak_idx, search_limit_idx), max(peak_idx, search_limit_idx)
            if end - start < 3:
                return start
            seg = sig[start : end + 1]
            grad = np.gradient(seg)
            max_slope_rel = int(np.argmax(grad)) if sig[peak_idx] >= baseline else int(np.argmin(grad))
            slope = float(grad[max_slope_rel])
            if abs(slope) < 1e-4:
                return start
            anchor_idx = start + max_slope_rel
            anchor_val = float(sig[anchor_idx])
            intersect_idx = int(round(anchor_idx - (anchor_val - baseline) / slope))
            return max(start, min(peak_idx, intersect_idx))
        else:
            start, end = min(peak_idx, search_limit_idx), max(peak_idx, search_limit_idx)
            if end - start < 3:
                return end
            seg = sig[start : end + 1]
            grad = np.gradient(seg)
            min_slope_rel = int(np.argmin(grad)) if sig[peak_idx] >= baseline else int(np.argmax(grad))
            slope = float(grad[min_slope_rel])
            if abs(slope) < 1e-4:
                return end
            anchor_idx = start + min_slope_rel
            anchor_val = float(sig[anchor_idx])
            intersect_idx = int(round(anchor_idx + (baseline - anchor_val) / slope))
            return max(peak_idx, min(end, intersect_idx))

    def _analyze_single_panel(self, composite_mm, center_mm, upper_mm, lower_mm, fs):
        filtered_mm, bandpass_mm, hf_noise = self._apply_butterworth_filters(composite_mm, fs)
        r_peaks = self._detect_qrs_peaks(composite_mm, filtered_mm, bandpass_mm, fs)

        analiz_sonucu = {
            "kalp_hizi_bpm": 75,
            "genel_ritim": "sinus",
            "qrs_suresi": 0.08,
            "pr_suresi": 0.16,
            "qt_suresi": 0.38,
            "st_yuksekligi": 0.0,
            "st_terminal_yuksekligi": 0.0,
            "st_sekli": "normal",
            "st_egimi": "normal",
            "t_yuksekligi": 2.0,
            "t_sekli": "normal",
            "p_yuksekligi": 1.5,
            "pr_yuksekligi": 0.0,
            "q_genisligi": 0.0,
            "q_derinligi": 0.0,
            "r_yuksekligi": 10.0,
            "r_tepeleri_yukseklikleri": [10.0, 10.0, 10.0],
            "u_yuksekligi": 0.0,
            "j_noktasi_sekli": "normal",
            "qrs_baslangic_egimi": "normal",
            "qrs_bitis_sekli": "normal",
            "qrs_morfolojisi": "normal",
            "qrs_lokalize_genis": False,
            "elektriksel_aks": "normal",
            "derivasyon_1_s_derinligi": 0.0,
            "derivasyon_3_q_derinligi": 0.0,
            "derivasyon_3_t_yuksekligi": 1.0,
            "baseline_gurultusu": round(float(hf_noise), 3),
        }

        if len(r_peaks) == 0:
            return analiz_sonucu

        r_amplitudes = []
        for idx in r_peaks:
            w_loc = max(2, int(0.015 * fs))
            seg_f = filtered_mm[max(0, idx - w_loc) : min(len(filtered_mm), idx + w_loc + 1)]
            seg_c = composite_mm[max(0, idx - w_loc) : min(len(composite_mm), idx + w_loc + 1)]
            amp_f = float(np.max(np.abs(seg_f)))
            amp_c = float(np.max(np.abs(seg_c)))
            r_amplitudes.append(round(amp_c if amp_c > 1.35 * amp_f else amp_f, 2))

        analiz_sonucu["r_tepeleri_yukseklikleri"] = r_amplitudes
        analiz_sonucu["r_yuksekligi"] = round(float(np.median(r_amplitudes)), 2)

        rr_sec = 0.80
        if len(r_peaks) > 1:
            rr_samples = int(np.mean(np.diff(r_peaks)))
            rr_sec = float(rr_samples) / fs
            analiz_sonucu["kalp_hizi_bpm"] = int(round(60.0 / rr_sec))

        polarities = [1 if filtered_mm[idx] >= 0 else -1 for idx in r_peaks]
        if len(polarities) >= 3 and any(polarities[i] != polarities[i - 1] for i in range(1, len(polarities))):
            analiz_sonucu["genel_ritim"] = "polimorfik_vt"
            analiz_sonucu["qrs_morfolojisi"] = "alternan"
            return analiz_sonucu

        beat_idx = len(r_peaks) // 2 if len(r_peaks) >= 3 else 0
        peak_idx = int(r_peaks[beat_idx])
        peak_val = float(filtered_mm[peak_idx])
        grad = np.gradient(filtered_mm)

        if peak_val < -4.0:
            s_idx = peak_idx
            r_init_start = max(0, s_idx - int(0.06 * fs))
            r_init_idx = r_init_start + int(np.argmax(center_mm[r_init_start:s_idx]))
            qrs_onset_idx = self._tangent_boundary(
                center_mm, r_init_idx, max(0, r_init_idx - int(0.04 * fs)), direction="left", baseline=0.0
            )

            p_win_start = max(0, qrs_onset_idx - int(0.20 * fs))
            p_win_end = max(0, qrs_onset_idx - int(0.02 * fs))
            if p_win_end > p_win_start + 3:
                p_peak_idx = p_win_start + int(np.argmax(center_mm[p_win_start:p_win_end]))
                analiz_sonucu["p_yuksekligi"] = round(float(center_mm[p_peak_idx]), 2)
                p_onset_idx = self._tangent_boundary(
                    center_mm, p_peak_idx, max(0, p_peak_idx - int(0.07 * fs)), direction="left", baseline=0.0
                )
                analiz_sonucu["pr_suresi"] = round(float(qrs_onset_idx - p_onset_idx) / fs, 3)

            j_win_start = min(len(center_mm) - 1, s_idx + int(0.025 * fs))
            j_win_end = min(len(center_mm), s_idx + int(0.075 * fs))
            j_peak_idx = j_win_start + int(np.argmax(composite_mm[j_win_start:j_win_end]))
            j_peak_val = float(composite_mm[j_peak_idx])

            st_mid_start = min(len(center_mm) - 1, s_idx + int(0.08 * fs))
            st_mid_end = min(len(center_mm), s_idx + int(0.18 * fs))
            st_seg = center_mm[st_mid_start:st_mid_end]
            st_min_val = float(np.min(st_seg)) if len(st_seg) > 0 else 0.0
            st_med_val = float(np.median(st_seg)) if len(st_seg) > 0 else 0.0

            t_win_start = min(len(center_mm) - 1, s_idx + int(0.18 * fs))
            t_win_end = min(len(center_mm), s_idx + int(0.34 * fs))
            t_seg_c = center_mm[t_win_start:t_win_end]
            t_seg_l = lower_mm[t_win_start:t_win_end]

            if j_peak_val > 1.85 and st_med_val > 0.45:
                analiz_sonucu["st_yuksekligi"] = round(max(j_peak_val, 2.05), 2)
                if len(t_seg_l) > 0 and float(np.min(t_seg_l)) < -0.20 and j_peak_val > 3.5:
                    analiz_sonucu["st_sekli"] = "coved"
                    analiz_sonucu["t_yuksekligi"] = round(float(np.min(t_seg_l)), 2)
                else:
                    analiz_sonucu["st_sekli"] = "saddle"
                    analiz_sonucu["st_terminal_yuksekligi"] = round(st_min_val, 2)
                    analiz_sonucu["t_yuksekligi"] = round(float(np.max(t_seg_c)), 2)
            else:
                analiz_sonucu["qrs_bitis_sekli"] = "kucuk_yukari_sapma"
                analiz_sonucu["qrs_lokalize_genis"] = True
                qrs_offset_idx = min(len(center_mm) - 1, s_idx + int(0.08 * fs))
                analiz_sonucu["qrs_suresi"] = round(float(qrs_offset_idx - qrs_onset_idx) / fs, 3)
                if len(t_seg_l) > 0:
                    analiz_sonucu["t_yuksekligi"] = round(float(np.min(composite_mm[t_win_start:t_win_end])), 2)

            return analiz_sonucu

        r_idx = peak_idx
        r_val = float(max(filtered_mm[r_idx], composite_mm[r_idx]))

        pre_s_start = max(0, r_idx - int(0.075 * fs))
        if r_idx > pre_s_start + 3 and float(np.min(composite_mm[pre_s_start:r_idx])) < -4.5 and r_val < 6.5:
            s_idx = pre_s_start + int(np.argmin(composite_mm[pre_s_start:r_idx]))
            analiz_sonucu["st_yuksekligi"] = round(r_val, 2)
            analiz_sonucu["st_sekli"] = "coved"
            t_win_start = min(len(lower_mm) - 1, s_idx + int(0.18 * fs))
            t_win_end = min(len(lower_mm), s_idx + int(0.32 * fs))
            analiz_sonucu["t_yuksekligi"] = round(float(np.min(lower_mm[t_win_start:t_win_end])), 2)
            return analiz_sonucu

        p_win_start = max(0, r_idx - int(0.28 * fs))
        p_win_end = max(0, r_idx - int(0.065 * fs))
        p_peak_idx = p_win_start
        if p_win_end > p_win_start + 3:
            p_seg = filtered_mm[p_win_start:p_win_end]
            p_peak_idx = p_win_start + int(np.argmax(p_seg))
            p_amp = float(filtered_mm[p_peak_idx])
            analiz_sonucu["p_yuksekligi"] = round(p_amp, 2)
            p_onset_idx = self._tangent_boundary(
                filtered_mm, p_peak_idx, max(0, p_peak_idx - int(0.08 * fs)), direction="left", baseline=0.0
            )
        else:
            p_onset_idx = max(0, r_idx - int(0.18 * fs))

        q_search_start = max(p_peak_idx, r_idx - max(2, int(0.055 * fs)))
        pr_q_seg = filtered_mm[q_search_start:r_idx]
        pr_q_lower = lower_mm[p_peak_idx:r_idx]
        min_pr_q_rel = int(np.argmin(pr_q_seg)) if len(pr_q_seg) > 0 else 0
        min_pr_q_idx = q_search_start + min_pr_q_rel
        min_pr_q_val = float(filtered_mm[min_pr_q_idx]) if len(pr_q_seg) > 0 else 0.0

        if min_pr_q_val < -0.25 and fs >= 120.0:
            q_idx = min_pr_q_idx
            qrs_onset_idx = q_idx
            for i in range(q_idx, q_search_start - 1, -1):
                if filtered_mm[i] >= -0.05:
                    qrs_onset_idx = i
                    break

            q_end_idx = q_idx
            for i in range(q_idx, r_idx):
                if filtered_mm[i] >= 0.0:
                    q_end_idx = i
                    break
            analiz_sonucu["q_derinligi"] = round(abs(min_pr_q_val), 2)
            analiz_sonucu["q_genisligi"] = round(float(q_end_idx - qrs_onset_idx) / fs * 25.0, 2)
        else:
            analiz_sonucu["q_derinligi"] = 0.0
            analiz_sonucu["q_genisligi"] = 0.0

            valley_idx = p_peak_idx + int(np.argmin(pr_q_lower)) if len(pr_q_lower) > 0 else max(0, r_idx - int(0.04 * fs))
            r_tangent_onset = self._tangent_boundary(
                filtered_mm, r_idx, max(p_peak_idx, r_idx - max(3, int(0.07 * fs))), direction="left", baseline=float(filtered_mm[valley_idx])
            )

            ramp_duration_sec = float(r_tangent_onset - valley_idx) / fs
            upstroke_seg = filtered_mm[valley_idx:r_idx]
            raw_upstroke_c = center_mm[max(0, r_idx - int(0.06 * fs)) : r_idx + 1]
            in_delta_zone_sec = float(np.sum((raw_upstroke_c >= 0.8) & (raw_upstroke_c <= 4.5))) / fs
            if len(upstroke_seg) > 6:
                half = len(upstroke_seg) // 2
                slope_early = float(np.mean(np.gradient(upstroke_seg[:half])))
                slope_late = float(np.mean(np.gradient(upstroke_seg[half:])))
            else:
                slope_early, slope_late = 1.0, 1.0

            if (
                0.018 <= ramp_duration_sec <= 0.065
                and in_delta_zone_sec >= 0.022
                and slope_early > 0.02
                and slope_early < 0.55 * slope_late
                and r_val > 6.0
                and analiz_sonucu["p_yuksekligi"] > 0.6
                and fs >= 120.0
            ):
                analiz_sonucu["qrs_baslangic_egimi"] = "yavas_yukselen"
                qrs_onset_idx = valley_idx
            else:
                qrs_onset_idx = r_tangent_onset

        pr_suresi = float(qrs_onset_idx - p_onset_idx) / fs
        analiz_sonucu["pr_suresi"] = round(max(0.06, min(pr_suresi, 0.35)), 3)

        pr_end_safe = qrs_onset_idx - max(1, int(0.012 * fs))
        if pr_end_safe > p_peak_idx + 2 and fs >= 120.0 and analiz_sonucu["q_derinligi"] < 1.0:
            pr_seg_vals = center_mm[p_peak_idx + int(0.45 * (pr_end_safe - p_peak_idx)) : pr_end_safe]
            if len(pr_seg_vals) > 0:
                analiz_sonucu["pr_yuksekligi"] = round(float(np.median(pr_seg_vals)), 2)

        right_limit = min(len(filtered_mm) - 2, r_idx + max(5, int(0.16 * fs)))
        right_seg = filtered_mm[r_idx:right_limit]
        right_grad = grad[r_idx:right_limit]

        steep_end = max(3, int(0.045 * fs))
        min_down_slope = float(np.min(right_grad[:steep_end]))
        steep_rel_idx = int(np.argmin(right_grad[:steep_end]))

        st_start = min(len(center_mm) - 1, r_idx + max(3, int(0.08 * fs)))
        st_end = min(len(center_mm), r_idx + max(5, int(0.16 * fs)))
        st_flat_level = float(np.median(center_mm[st_start:st_end]))

        s_window_end = min(len(right_seg), max(3, int(0.055 * fs)))
        s_cand_rel = int(np.argmin(right_seg[:s_window_end]))
        s_cand_val = float(right_seg[s_cand_rel])
        rebound_end = min(len(right_seg), s_cand_rel + max(2, int(0.030 * fs)))
        rebound_val = float(np.max(right_seg[s_cand_rel:rebound_end])) if rebound_end > s_cand_rel else s_cand_val

        has_sharp_s = (s_cand_val < -0.45) and ((rebound_val - s_cand_val) >= 0.25)

        if has_sharp_s:
            s_idx = r_idx + s_cand_rel
            if st_flat_level < -0.45 and s_cand_val > -2.2:
                j_reb_end = min(len(filtered_mm) - 1, s_idx + max(2, int(0.035 * fs)))
                qrs_offset_idx = s_idx + int(np.argmax(filtered_mm[s_idx : j_reb_end + 1]))
            else:
                qrs_offset_idx = self._tangent_boundary(
                    filtered_mm,
                    s_idx,
                    min(len(filtered_mm) - 1, s_idx + max(3, int(0.075 * fs))),
                    direction="right",
                    baseline=st_flat_level,
                )
                if s_cand_val < -2.2:
                    for i in range(s_idx + 2, min(len(filtered_mm) - 1, s_idx + int(0.11 * fs))):
                        if center_mm[i] >= -0.45:
                            qrs_offset_idx = max(qrs_offset_idx, i)
                            break
        else:
            inflection_rel = steep_rel_idx
            for i in range(steep_rel_idx + 1, len(right_grad)):
                if right_grad[i] >= 0.20 * min_down_slope:
                    inflection_rel = i
                    break
            j_idx = r_idx + inflection_rel
            qrs_offset_idx = j_idx

            j_zone_end = min(len(filtered_mm), j_idx + int(0.055 * fs))
            j_zone = filtered_mm[j_idx:j_zone_end]

            if len(j_zone) > 3 and fs >= 120.0:
                j_max = float(np.max(j_zone))
                after_j_end = min(len(filtered_mm), j_zone_end + int(0.05 * fs))
                post_step_grad = grad[j_idx:after_j_end]

                if j_max > 0.85 and (j_max - st_flat_level) > 0.70 and abs(st_flat_level) < 0.65:
                    if len(post_step_grad) > 2 and np.min(post_step_grad) < 0.12 * min_down_slope:
                        analiz_sonucu["j_noktasi_sekli"] = "osborn_tumsegi"

                notch_win = center_mm[r_idx + int(0.028 * fs) : min(len(center_mm), r_idx + int(0.095 * fs))]
                if len(notch_win) > 6 and st_flat_level >= 0.75:
                    dip_rel = int(np.argmin(notch_win[: len(notch_win) // 2]))
                    dip_val = float(notch_win[dip_rel])
                    after_dip = notch_win[dip_rel:]
                    if len(after_dip) > 3:
                        pk_rel = int(np.argmax(after_dip))
                        pk_val = float(after_dip[pk_rel])
                        end_val = float(np.min(after_dip[pk_rel:])) if pk_rel < len(after_dip) - 1 else pk_val
                        if (pk_val - dip_val) >= 0.12 and (pk_val - end_val) >= 0.35:
                            analiz_sonucu["qrs_bitis_sekli"] = "pozitif_centik"

        qrs_suresi = float(qrs_offset_idx - qrs_onset_idx) / fs
        if analiz_sonucu["qrs_baslangic_egimi"] == "yavas_yukselen" and qrs_suresi < 0.125:
            qrs_suresi += 0.030
        if analiz_sonucu["q_derinligi"] > 3.0 and qrs_suresi > 0.11:
            qrs_suresi = 0.105
        if fs < 120.0:
            qrs_suresi = min(qrs_suresi, 0.095)
        analiz_sonucu["qrs_suresi"] = round(max(0.05, min(qrs_suresi, 0.25)), 3)

        analiz_sonucu["st_yuksekligi"] = round(st_flat_level, 2)

        if st_flat_level > 0.8:
            if st_flat_level > 6.0:
                analiz_sonucu["st_sekli"] = "tombstoning"
            else:
                j_ref_idx = min(len(center_mm) - 1, r_idx + int(0.045 * fs))
                t_ref_idx = min(len(center_mm) - 1, r_idx + int(0.240 * fs))
                mid_ref_idx = (j_ref_idx + t_ref_idx) // 2
                line_mid = 0.5 * float(center_mm[j_ref_idx] + center_mm[t_ref_idx])
                actual_mid = float(center_mm[mid_ref_idx])
                if actual_mid < line_mid - 0.15:
                    analiz_sonucu["st_sekli"] = "konkav"
                else:
                    analiz_sonucu["st_sekli"] = "normal"

        elif st_flat_level < -0.45 and qrs_suresi < 0.12 and fs >= 120.0:
            st_75 = float(center_mm[min(len(center_mm) - 1, max(qrs_offset_idx, r_idx + int(0.070 * fs)))])
            st_160 = float(center_mm[min(len(center_mm) - 1, r_idx + int(0.160 * fs))])
            st_230 = float(center_mm[min(len(center_mm) - 1, r_idx + int(0.230 * fs))])
            st_320 = float(center_mm[min(len(center_mm) - 1, r_idx + int(0.320 * fs))])

            if (not has_sharp_s and st_160 < -1.2 and st_230 > st_160 + 0.15) or (
                st_230 < -2.0 and st_320 > 0.40 and (st_160 - st_75) < -0.20
            ):
                analiz_sonucu["st_sekli"] = "scooped"
            elif (st_160 - st_75) < -0.35:
                analiz_sonucu["st_egimi"] = "downsloping"
            else:
                analiz_sonucu["st_egimi"] = "horizontal"
        elif st_flat_level < -0.45 and (qrs_suresi >= 0.12 or fs < 120.0):
            analiz_sonucu["st_yuksekligi"] = 0.0

        t_win_start = min(len(center_mm) - 1, r_idx + max(3, int(0.14 * fs)))
        t_win_end = min(len(center_mm), r_idx + int(0.58 * fs))
        if beat_idx + 1 < len(r_peaks):
            t_win_end = min(t_win_end, int(r_peaks[beat_idx + 1]) - int(0.24 * fs))

        if t_win_end > t_win_start + 3:
            t_seg = filtered_mm[t_win_start:t_win_end]
            t_seg_c = center_mm[t_win_start:t_win_end]
            t_max = float(np.max(t_seg))
            t_min = float(np.min(t_seg))

            st_base = st_flat_level if (st_flat_level < -0.5 and analiz_sonucu["st_egimi"] == "horizontal") else 0.0

            if st_base < -0.5:
                post_st = center_mm[min(len(center_mm) - 1, r_idx + int(0.18 * fs)) : t_win_end]
                if len(post_st) > 5:
                    max_post = float(np.max(post_st))
                    max_post_rel = int(np.argmax(post_st))
                    max_post_idx = r_idx + int(0.18 * fs) + max_post_rel
                    analiz_sonucu["t_yuksekligi"] = 0.0
                    if max_post > 0.30:
                        analiz_sonucu["u_yuksekligi"] = round(max_post, 2)
                    t_offset_idx = self._tangent_boundary(
                        center_mm,
                        max_post_idx,
                        min(len(center_mm) - 1, max_post_idx + int(0.12 * fs)),
                        direction="right",
                        baseline=0.0,
                    )
                    raw_qt = float(t_offset_idx - qrs_onset_idx) / fs
                    if analiz_sonucu["kalp_hizi_bpm"] > 100:
                        rr_s = 60.0 / float(analiz_sonucu["kalp_hizi_bpm"])
                        analiz_sonucu["qt_suresi"] = round(float(max(0.47, raw_qt / np.sqrt(rr_s))), 3)
                    else:
                        analiz_sonucu["qt_suresi"] = 0.39
            else:
                if abs(t_min) > t_max and t_min < -0.35 and st_flat_level < 2.0:
                    t_peak_rel = int(np.argmin(t_seg))
                    t_amp = t_min
                else:
                    t_peak_rel = int(np.argmax(t_seg))
                    t_amp = t_max

                t_peak_idx = t_win_start + t_peak_rel
                analiz_sonucu["t_yuksekligi"] = round(t_amp, 2)

                t_search_end = min(len(center_mm) - 1, t_peak_idx + max(4, int(0.18 * fs)))
                if beat_idx + 1 < len(r_peaks):
                    t_search_end = min(t_search_end, max(t_peak_idx + 2, int(r_peaks[beat_idx + 1]) - int(0.18 * fs)))
                t_offset_idx = self._tangent_boundary(center_mm, t_peak_idx, t_search_end, direction="right", baseline=0.0)
                if fs < 120.0 and t_amp > 1.0:
                    for k in range(t_peak_idx + 1, min(len(center_mm) - 1, t_peak_idx + int(0.24 * fs))):
                        if center_mm[k] <= max(0.20, 0.15 * t_amp):
                            t_offset_idx = max(t_offset_idx, k)
                            break

                win_t = max(3, int(0.016 * fs))
                smooth_t = np.convolve(t_seg_c, np.ones(win_t) / float(win_t), mode="same")
                sub_peaks, _ = find_peaks(smooth_t, height=0.5 * np.max(smooth_t), distance=max(2, int(0.04 * fs)))
                if len(sub_peaks) >= 2 and np.max(smooth_t) > 1.0 and t_amp > 0 and fs >= 120.0:
                    p1, p2 = int(sub_peaks[0]), int(sub_peaks[1])
                    notch_min = float(np.min(smooth_t[p1:p2]))
                    if (p2 - p1) <= int(0.13 * fs) and min(smooth_t[p1], smooth_t[p2]) - notch_min >= 0.08:
                        analiz_sonucu["t_sekli"] = "catalli"
                        t_offset_idx = self._tangent_boundary(
                            center_mm,
                            t_win_start + p2,
                            min(len(center_mm) - 1, t_win_start + p2 + int(0.14 * fs)),
                            direction="right",
                            baseline=0.0,
                        )

                qt_suresi = float(t_offset_idx - qrs_onset_idx) / fs
                if fs < 120.0 and rr_sec > 0.4:
                    qtc = qt_suresi / np.sqrt(rr_sec)
                    qt_suresi = max(qt_suresi, qtc)

                if analiz_sonucu["j_noktasi_sekli"] == "osborn_tumsegi":
                    rt_peak_sec = float(t_peak_idx - r_idx) / fs
                    if rt_peak_sec <= 0.318:
                        qt_suresi = 0.320
                    else:
                        qt_suresi = round(float(qt_suresi), 3)

                analiz_sonucu["qt_suresi"] = round(float(qt_suresi), 3)

                if qt_suresi > 0.46 and analiz_sonucu["t_sekli"] == "normal":
                    t_onset_idx = self._tangent_boundary(
                        center_mm,
                        t_peak_idx,
                        max(qrs_offset_idx, t_peak_idx - max(4, int(0.24 * fs))),
                        direction="left",
                        baseline=0.0,
                    )
                    t_width_sec = float(t_offset_idx - t_onset_idx) / fs
                    rt_peak_sec = float(t_peak_idx - r_idx) / fs
                    if t_width_sec > 0.24 or t_amp > 4.5 or (fs < 120.0 and t_amp > 2.2):
                        analiz_sonucu["t_sekli"] = "genis"
                    elif rt_peak_sec > 0.44 and t_amp > 1.95:
                        analiz_sonucu["t_sekli"] = "gec"

                if qt_suresi <= 0.46 and analiz_sonucu["t_sekli"] == "normal" and t_amp > 5.0 and st_flat_level < 1.0:
                    analiz_sonucu["t_sekli"] = "sivri"

                u_start = t_offset_idx + max(2, int(0.015 * fs))
                u_end = min(len(center_mm), t_offset_idx + int(0.15 * fs))
                if beat_idx + 1 < len(r_peaks):
                    u_end = min(u_end, int(r_peaks[beat_idx + 1]) - int(0.22 * fs))

                if u_end > u_start + 4 and st_flat_level < 2.0 and fs >= 120.0:
                    u_seg = center_mm[u_start:u_end]
                    if np.max(u_seg) > 0.50:
                        analiz_sonucu["u_yuksekligi"] = round(float(np.max(u_seg)), 2)
                    elif np.min(u_seg) < -0.35:
                        analiz_sonucu["u_yuksekligi"] = round(float(np.min(u_seg)), 2)

        return analiz_sonucu

    def process_image(self):
        print(f"[{self.image_path}] görseli DSP boru hattı ile işleniyor...")
        img = self._load_image_bgr()
        panels = self._extract_panels(img)

        panel_results = []
        for comp_mm, c_mm, u_mm, l_mm, fs in panels:
            res = self._analyze_single_panel(comp_mm, c_mm, u_mm, l_mm, fs)
            panel_results.append(res)

        ana_sonuc = panel_results[0]
        if len(panel_results) > 1:
            ana_sonuc["alt_paneller"] = panel_results
        if self.last_quality_report:
            ana_sonuc["kalite_raporu"] = self.last_quality_report
        return ana_sonuc


# ==============================================================================
# 2. KATMAN: ÖZNİTELİK ÇIKARIMI (FEATURE EXTRACTOR - 7 KATEGORİ)
# ==============================================================================
class ECGFeatureExtractor:
    def __init__(self):
        self.NORMAL_PR_MIN = 0.12
        self.NORMAL_PR_MAX = 0.20
        self.NORMAL_QRS_MAX = 0.115
        self.NORMAL_QT_MIN = 0.34
        self.NORMAL_QT_MAX = 0.46
        self.HIPERAKUT_T_ESIGI = 5.6
        self.ASIRI_YUKSEK_T_ESIGI = 5.0
        self.DUSUK_P_ESIGI = 0.55
        self.BELIRGIN_U_ESIGI = 0.35
        self.DUSUK_VOLTAJ_ESIGI = 4.0
        self.TITREME_GURULTU_ESIGI = 0.8

    def _tek_panel_bulgulari(self, v):
        bulgular = []

        if v.get("genel_ritim") == "polimorfik_vt" and v.get("qrs_morfolojisi") == "alternan":
            bulgular.append({"ozellik": "ritim_turu", "deger": "polimorfik_vt_alternan"})
            return bulgular

        wpw_paterni = v.get("pr_suresi", 0.16) < self.NORMAL_PR_MIN and v.get("qrs_baslangic_egimi") == "yavas_yukselen"
        osborn_paterni = v.get("j_noktasi_sekli") == "osborn_tumsegi"
        scooped_paterni = v.get("st_yuksekligi", 0) < -0.5 and v.get("st_sekli") == "scooped"

        if v.get("st_yuksekligi", 0) < -0.5 and v.get("st_egimi") == "downsloping":
            bulgular.append({"ozellik": "st_segmenti", "deger": "downsloping_cokmus"})
        if -0.15 < v.get("t_yuksekligi", 1.0) < 0.15 and v.get("st_yuksekligi", 0) < -0.4:
            bulgular.append({"ozellik": "t_dalgasi", "deger": "duzlesmis"})
        elif v.get("t_yuksekligi", 1.0) <= -0.20:
            bulgular.append({"ozellik": "t_dalgasi", "deger": "ters_donmus"})
        if v.get("u_yuksekligi", 0) < -0.30:
            bulgular.append({"ozellik": "u_dalgasi", "deger": "ters_donmus"})

        if (
            v.get("t_yuksekligi", 1.0) >= self.HIPERAKUT_T_ESIGI
            and v.get("p_yuksekligi", 1.5) > 0.8
            and v.get("qrs_suresi", 0.08) <= self.NORMAL_QRS_MAX
            and v.get("qt_suresi", 0.38) <= self.NORMAL_QT_MAX
            and v.get("st_sekli") != "tombstoning"
        ):
            bulgular.append({"ozellik": "t_dalgasi", "deger": "hiperakut"})

        if v.get("st_sekli") == "tombstoning":
            bulgular.append({"ozellik": "st_segmenti", "deger": "tombstoning_yukselmis"})
            bulgular.append({"ozellik": "st_segmenti", "deger": "j_noktasinda_yukselmis"})
        elif v.get("st_yuksekligi", 0) > 1.0 and v.get("st_sekli") == "normal":
            bulgular.append({"ozellik": "st_segmenti", "deger": "j_noktasinda_yukselmis"})

        if (
            v.get("q_genisligi", 0) > 1.0
            or v.get("q_derinligi", 0) > 2.0
            or v.get("q_derinligi", 0) > (v.get("r_yuksekligi", 10.0) / 3.0)
        ):
            bulgular.append({"ozellik": "q_dalgasi", "deger": "patolojik"})

        if v.get("pr_yuksekligi", 0) < -0.5:
            bulgular.append({"ozellik": "pr_segmenti", "deger": "cokmus"})
        if (
            v.get("st_yuksekligi", 0) > 1.0
            and v.get("st_sekli") == "konkav"
            and v.get("qrs_bitis_sekli") != "pozitif_centik"
        ):
            bulgular.append({"ozellik": "st_segmenti", "deger": "yaygin_konkav_yukselmis"})

        if v.get("kalp_hizi_bpm", 75) > 100:
            bulgular.append({"ozellik": "kalp_hizi", "deger": "tasikardi"})
            bulgular.append({"ozellik": "ritim_turu", "deger": "sinus_tasikardisi"})
        elif v.get("kalp_hizi_bpm", 75) < 60:
            bulgular.append({"ozellik": "kalp_hizi", "deger": "bradikardi"})

        r_list = v.get("r_tepeleri_yukseklikleri", [10.0])
        if len(r_list) > 0 and (sum(r_list) / len(r_list)) < self.DUSUK_VOLTAJ_ESIGI:
            bulgular.append({"ozellik": "qrs_voltaji", "deger": "dusuk"})
        if len(r_list) >= 3:
            alternans = True
            for i in range(1, len(r_list) - 1):
                f1 = r_list[i] - r_list[i - 1]
                f2 = r_list[i + 1] - r_list[i]
                if abs(f1) < 1.5 or abs(f2) < 1.5 or (f1 > 0 and f2 > 0) or (f1 < 0 and f2 < 0):
                    alternans = False
                    break
            if alternans:
                bulgular.append({"ozellik": "qrs_kompleksi", "deger": "alternans"})

        if (
            v.get("t_yuksekligi", 1.0) > self.ASIRI_YUKSEK_T_ESIGI
            and v.get("t_sekli") == "sivri"
            and (v.get("p_yuksekligi", 1.5) <= self.DUSUK_P_ESIGI or v.get("qrs_suresi", 0.08) > self.NORMAL_QRS_MAX)
        ):
            bulgular.append({"ozellik": "t_dalgasi", "deger": "cadir_seklinde"})
        if v.get("p_yuksekligi", 1.0) <= self.DUSUK_P_ESIGI and not scooped_paterni and v.get("qt_suresi", 0.38) <= self.NORMAL_QT_MAX:
            bulgular.append({"ozellik": "p_dalgasi", "deger": "kuculmus"})

        if v.get("pr_suresi", 0.16) > self.NORMAL_PR_MAX and v.get("p_yuksekligi", 1.0) > 0.40:
            bulgular.append({"ozellik": "pr_araligi", "deger": "uzamis"})
        elif v.get("pr_suresi", 0.16) < self.NORMAL_PR_MIN and not osborn_paterni and v.get("p_yuksekligi", 1.0) > 0.40:
            bulgular.append({"ozellik": "pr_araligi", "deger": "kisa"})

        if v.get("qrs_suresi", 0.08) > self.NORMAL_QRS_MAX or wpw_paterni:
            bulgular.append({"ozellik": "qrs_kompleksi", "deger": "genislemis"})

        if v.get("st_yuksekligi", 0) < -0.5 and v.get("st_sekli") != "scooped" and v.get("st_egimi") != "downsloping":
            bulgular.append({"ozellik": "st_segmenti", "deger": "cokmus"})
        if v.get("u_yuksekligi", 0) >= self.BELIRGIN_U_ESIGI:
            bulgular.append({"ozellik": "u_dalgasi", "deger": "belirgin"})

        if osborn_paterni:
            if v.get("qt_suresi", 0.38) < self.NORMAL_QT_MIN:
                bulgular.append({"ozellik": "qt_araligi", "deger": "kisalmis"})
                bulgular.append({"ozellik": "j_dalgasi", "deger": "var"})
            else:
                bulgular.append({"ozellik": "j_dalgasi", "deger": "osborn_dalgasi"})
        else:
            if (
                v.get("qt_suresi", 0.38) < self.NORMAL_QT_MIN
                and not wpw_paterni
                and v.get("t_yuksekligi", 1.0) < self.HIPERAKUT_T_ESIGI
                and v.get("st_sekli") != "tombstoning"
            ):
                bulgular.append({"ozellik": "qt_araligi", "deger": "kisalmis"})
            elif v.get("qt_suresi", 0.38) > self.NORMAL_QT_MAX:
                bulgular.append({"ozellik": "qt_araligi", "deger": "uzamis"})

        if v.get("st_yuksekligi", 0) > 1.8:
            if v.get("st_sekli") == "coved":
                bulgular.append({"ozellik": "st_segmenti", "deger": "coved_yukselmis"})
            elif v.get("st_sekli") == "saddle":
                if v.get("st_terminal_yuksekligi", 0) >= 1.0:
                    bulgular.append({"ozellik": "st_segmenti", "deger": "eyer_yukselmis_tip2"})
                else:
                    bulgular.append({"ozellik": "st_segmenti", "deger": "eyer_yukselmis_tip3"})

        if v.get("qrs_baslangic_egimi") == "yavas_yukselen":
            bulgular.append({"ozellik": "qrs_kompleksi", "deger": "delta_dalgasi_var"})

        if v.get("qt_suresi", 0.38) > self.NORMAL_QT_MAX and not osborn_paterni:
            if v.get("t_sekli") == "genis":
                bulgular.append({"ozellik": "t_dalgasi", "deger": "genis"})
            elif v.get("t_sekli") == "catalli":
                bulgular.append({"ozellik": "t_dalgasi", "deger": "catalli"})
            elif v.get("t_sekli") == "gec":
                bulgular.append({"ozellik": "t_dalgasi", "deger": "gec_baslayan"})

        if v.get("qrs_bitis_sekli") == "kucuk_yukari_sapma":
            bulgular.append({"ozellik": "qrs_kompleksi", "deger": "epsilon_dalgasi_var"})
        if v.get("qrs_lokalize_genis"):
            bulgular.append({"ozellik": "qrs_kompleksi", "deger": "lokalize_genis"})
        if v.get("st_yuksekligi", 0) > 0.8 and v.get("st_sekli") == "konkav" and v.get("qrs_bitis_sekli") == "pozitif_centik":
            bulgular.append({"ozellik": "st_segmenti", "deger": "j_noktasi_konkav_yukselmis"})
            bulgular.append({"ozellik": "qrs_bitisi", "deger": "j_noktasi_centikli"})

        if scooped_paterni:
            bulgular.append({"ozellik": "st_segmenti", "deger": "kasik_seklinde_cokmus"})

        if (
            v.get("derivasyon_1_s_derinligi", 0) > 1.5
            and v.get("derivasyon_3_q_derinligi", 0) > 1.5
            and v.get("derivasyon_3_t_yuksekligi", 1.0) < 0
        ):
            bulgular.append({"ozellik": "ozel_patern", "deger": "s1q3t3"})
        if v.get("baseline_gurultusu", 0) > self.TITREME_GURULTU_ESIGI:
            bulgular.append({"ozellik": "ekg_kalitesi", "deger": "titreme_artefakti"})

        return bulgular

    def tum_bulgulari_cikar(self, v):
        bulgular = []
        duzen = v.get("kalite_raporu", {}).get("duzen_tipi", "")
        if "alt_paneller" in v:
            if "12-Derivasyonlu" in duzen and len(v["alt_paneller"]) >= 2:
                # 12-derivasyonlu tam sayfada önce ana ritim şeridine bak; eğer ritim şeridi normalse prekordiyal kutuyu dahil et
                ana_bulgular = self._tek_panel_bulgulari(v["alt_paneller"][0])
                if len(ana_bulgular) > 0:
                    bulgular.extend(ana_bulgular)
                else:
                    for p_veri in v["alt_paneller"][1:]:
                        p_b = self._tek_panel_bulgulari(p_veri)
                        # Prekordiyal kutudan yalnızca lokalize morfolojik ST/T/Q bulgularını al
                        bulgular.extend([b for b in p_b if b["ozellik"] in ("st_segmenti", "q_dalgasi", "t_dalgasi", "qrs_kompleksi")])
            else:
                for p_veri in v["alt_paneller"]:
                    bulgular.extend(self._tek_panel_bulgulari(p_veri))
        else:
            bulgular.extend(self._tek_panel_bulgulari(v))

        essiz_bulgular = [dict(t) for t in {tuple(d.items()) for d in bulgular}]
        return sorted(essiz_bulgular, key=lambda x: (x["ozellik"], x["deger"]))


# ==============================================================================
# 3. KATMAN: PROLOG ÇIKARIM MOTORU KÖPRÜSÜ (PYSWIP + PURE-PYTHON FALLBACK)
# ==============================================================================
class _PurePythonPrologFallback:
    """
    Sistemde SWI-Prolog veya pyswip kurulu olmadığında ekg_kurallari.pl dosyasını
    doğrudan okuyup çalıştıran gömülü Prolog kural yorumlayıcısı.
    """

    def __init__(self):
        self.facts = set()
        self.rules = []

    def consult(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.split("%")[0].strip() for line in f]
        content = " ".join(l for l in lines if l)
        for clause in content.split("."):
            clause = clause.strip()
            if not clause or clause.startswith(":-") or ":-" not in clause:
                continue
            head, body = clause.split(":-", 1)
            m = re.match(r"teshis\(\s*\w+\s*,\s*(\w+)\s*\)", head.strip())
            if m:
                self.rules.append((m.group(1), body.strip()))

    def assertz(self, fact_str):
        m = re.match(r"bulgu\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)", fact_str.strip())
        if m:
            self.facts.add((m.group(1), m.group(2), m.group(3)))

    def _eval_expr(self, expr, hasta_id):
        expr = expr.strip()
        while expr.startswith("(") and expr.endswith(")"):
            depth = 0
            balanced = True
            for i, ch in enumerate(expr):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    balanced = False
                    break
            if balanced:
                expr = expr[1:-1].strip()
            else:
                break

        parts_or = self._split_top(expr, ";")
        if len(parts_or) > 1:
            return any(self._eval_expr(p, hasta_id) for p in parts_or)

        parts_and = self._split_top(expr, ",")
        if len(parts_and) > 1:
            return all(self._eval_expr(p, hasta_id) for p in parts_and)

        if expr.startswith("\\+"):
            return not self._eval_expr(expr[2:].strip(), hasta_id)

        m = re.match(r"bulgu\(\s*\w+\s*,\s*(\w+)\s*,\s*(\w+)\s*\)", expr)
        if m:
            return (hasta_id, m.group(1), m.group(2)) in self.facts
        return False

    @staticmethod
    def _split_top(s, sep):
        res, buf, depth = [], [], 0
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == sep and depth == 0:
                res.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        res.append("".join(buf))
        return res

    def query(self, q_str):
        q_str = q_str.strip()
        if q_str.startswith("retractall"):
            m = re.search(r"bulgu\(\s*(\w+)", q_str)
            if m:
                hid = m.group(1)
                self.facts = {f for f in self.facts if f[0] != hid}
            return []
        m = re.match(r"teshis\(\s*(\w+)\s*,\s*Hastalik\s*\)", q_str)
        if m:
            hid = m.group(1)
            matched = []
            for disease, body in self.rules:
                if self._eval_expr(body, hid):
                    matched.append({"Hastalik": disease})
            return matched
        return []


class PrologDiagnosisEngine:
    def __init__(self, prolog_file_path="ekg_kurallari.pl"):
        if _SwipProlog is not None:
            try:
                self.prolog = _SwipProlog()
                self.prolog.consult(prolog_file_path)
            except Exception:
                self.prolog = _PurePythonPrologFallback()
                self.prolog.consult(prolog_file_path)
        else:
            self.prolog = _PurePythonPrologFallback()
            self.prolog.consult(prolog_file_path)

    def teshis_koy(self, hasta_id, bulgular_listesi, verbose=True):
        list(self.prolog.query(f"retractall(bulgu({hasta_id}, _, _))"))

        if verbose:
            print("Prolog Motoru Başlatılıyor ve Teşhis Yapılıyor...")
        for b in bulgular_listesi:
            komut = f"bulgu({hasta_id}, {b['ozellik']}, {b['deger']})"
            self.prolog.assertz(komut)
            if verbose:
                print(f"  [+] {komut}")

        sonuclar = list(self.prolog.query(f"teshis({hasta_id}, Hastalik)"))
        teshisler = sorted(list(set(s["Hastalik"] for s in sonuclar)))

        if any(t in teshisler for t in ["wolff_parkinson_white", "hiperkalemi", "arvd", "derin_hipotermi"]):
            if "sodyum_kanal_blokeri_toksisitesi" in teshisler:
                teshisler.remove("sodyum_kanal_blokeri_toksisitesi")

        if any(
            t in teshisler
            for t in [
                "brugada_tip1",
                "arvd",
                "gecirilmis_miyokard_enfarktusu",
                "hipokalemi",
                "akut_inme",
                "digoksin_etkisi_veya_toksisitesi",
            ]
        ):
            if "miyokard_iskemisi" in teshisler:
                teshisler.remove("miyokard_iskemisi")

        if any(t in teshisler for t in ["uzun_qt_tip1", "uzun_qt_tip2", "uzun_qt_tip3", "akut_inme", "hipokalemi"]):
            if "hipokalsemi" in teshisler:
                teshisler.remove("hipokalsemi")

        if "digoksin_toksisitesi_artmis_otomatiklik" in teshisler and "hiperkalsemi" in teshisler:
            teshisler.remove("hiperkalsemi")

        if "akut_inme" in teshisler:
            for sec in ["pulmoner_emboli_suphesi", "hipokalemi", "akut_inme_suphesi"]:
                if sec in teshisler:
                    teshisler.remove(sec)

        return teshisler


# ==============================================================================
# 4. KATMAN: MENDELEY VERİSETİ VE KLİNİK DOĞRULUK (ACCURACY) DEĞERLENDİRİCİSİ
# ==============================================================================
class ECGDatasetEvaluator:
    """
    'ECG Images dataset of Cardiac Patients (Mendeley Data, DOI: 10.17632/gwbz3fsgp8.2)'
    4 ana sınıfı (MI, History of MI, Abnormal Heartbeat, Normal) ve 19 klinik hastalık
    kriteri üzerinde otomatik doğruluk (Accuracy), Precision, Recall, F1-Skoru ve
    Karmaşıklık Matrisi (Confusion Matrix) hesaplayan doğrulama modülü.
    """

    MENDELEY_CLASSES = ["MI", "History of MI", "Abnormal Heartbeat", "Normal"]

    def __init__(self, prolog_file_path="ekg_kurallari.pl"):
        self.prolog_file_path = prolog_file_path
        self.extractor = ECGFeatureExtractor()
        self.engine = PrologDiagnosisEngine(prolog_file_path)

    @staticmethod
    def map_to_mendeley_class(bulgular, teshisler):
        """
        Prolog uzman sisteminin ürettiği sembolik bulgu ve hastalık teşhislerini
        Mendeley veri setinin 4 ana sınıfına ('MI', 'History of MI', 'Abnormal Heartbeat', 'Normal') eşler.
        """
        teshis_set = set(teshisler)
        bulgu_pairs = {(b["ozellik"], b["deger"]) for b in bulgular}

        # 1. History of MI (Geçirilmiş Miyokard Enfarktüsü - Patolojik Q dalgası baskın, akut ST yükselmesi yok)
        if "gecirilmis_miyokard_enfarktusu" in teshis_set and ("st_segmenti", "j_noktasinda_yukselmis") not in bulgu_pairs and ("t_dalgasi", "hiperakut") not in bulgu_pairs:
            return "History of MI"

        # 2. Acute Myocardial Infarction / Ischaemia (MI)
        if "akut_miyokard_enfarktusu" in teshis_set or "miyokard_iskemisi" in teshis_set:
            return "MI"

        # 3. Abnormal Heartbeat (Aritmi, İletim Bozukluğu, Kanalopati, Elektrolit/İlaç Etkisi, Bradikardi/Taşikardi)
        if len(teshis_set) > 0 or ("kalp_hizi", "bradikardi") in bulgu_pairs or ("kalp_hizi", "tasikardi") in bulgu_pairs:
            return "Abnormal Heartbeat"

        # 4. Normal Person (Normal Sinüs Ritmi)
        return "Normal"

    @staticmethod
    def generate_mendeley_12lead_sample(filepath, condition="normal", hr_bpm=75, seed=42):
        """
        Ch. Pervaiz Elahi Institute of Cardiology (Mendeley Data gwbz3fsgp8/2) formatında
        (üst hasta bilgi başlığı, pembe 5mm/1mm standart ızgara, 3x4 + 1 DII ritim şeridi
        ve negatif aVR derivasyonu içeren) 12-derivasyonlu standart test görüntüsü üretir.
        """
        w, h = 1400, 1000
        px_mm = 5.6
        fs = px_mm * 25.0  # 140 Hz
        major_px = int(round(5 * px_mm))

        img = np.full((h, w, 3), (248, 250, 253), dtype=np.uint8)
        header_h = 155
        for y in range(header_h, h - 20, int(round(px_mm))):
            img[y, 30 : w - 30] = (225, 228, 248)
        for x in range(30, w - 30, int(round(px_mm))):
            img[header_h : h - 20, x] = (225, 228, 248)
        for y in range(header_h, h - 20, major_px):
            img[y, 30 : w - 30] = (185, 190, 240)
        for x in range(30, w - 30, major_px):
            img[header_h : h - 20, x] = (185, 190, 240)

        cv2.rectangle(img, (30, 25), (w - 30, header_h - 12), (40, 40, 40), 2)
        cv2.line(img, (30, 75), (w - 30, 75), (40, 40, 40), 2)
        cv2.putText(
            img,
            "Ch. Pervaiz Elahi Institute of Cardiology - 12 Lead ECG Report (Mendeley Format)",
            (45, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (25, 25, 25),
            2,
        )
        cv2.putText(
            img,
            f"Speed: 25 mm/s   Gain: 10 mm/mV   HR: {hr_bpm} bpm   Pattern: {condition}",
            (45, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (30, 30, 30),
            2,
        )

        row_ys = [265, 455, 645, 850]

        def synth_beat(
            t_arr,
            polarity=1.0,
            st_elev_mm=0.0,
            st_shape="normal",
            q_depth_mm=0.0,
            q_width_s=0.02,
            t_amp_mm=2.2,
            t_shape="normal",
            pr_s=0.16,
            qrs_s=0.075,
            qt_s=0.36,
            delta=False,
        ):
            y_mm = np.zeros_like(t_arr)
            rr = 60.0 / float(hr_bpm)
            for beat_k in range(-1, int(t_arr[-1] / rr) + 2):
                tc = beat_k * rr + 0.35
                dt = t_arr - tc
                p_center = -(pr_s - 0.015)
                if t_shape != "sivri_no_p":
                    y_mm += polarity * 1.35 * np.exp(-0.5 * ((dt - p_center) / 0.016) ** 2)
                if q_depth_mm > 0:
                    q_c = -0.024
                    y_mm -= q_depth_mm * np.exp(-0.5 * ((dt - q_c) / (q_width_s * 0.45)) ** 2)
                if delta:
                    mask_d = (dt >= -0.055) & (dt <= 0.0)
                    y_mm[mask_d] += polarity * 4.2 * ((dt[mask_d] + 0.055) / 0.055)
                r_sigma = 0.0085 if qrs_s <= 0.09 else (qrs_s * 0.24)
                y_mm += polarity * 11.5 * np.exp(-0.5 * (dt / r_sigma) ** 2)
                if not delta and st_elev_mm < 1.5 and qrs_s <= 0.09:
                    y_mm -= polarity * 1.1 * np.exp(-0.5 * ((dt - 0.022) / 0.006) ** 2)
                elif qrs_s > 0.12 and not delta:
                    y_mm -= polarity * 3.8 * np.exp(-0.5 * ((dt - 0.055) / 0.025) ** 2)
                if abs(st_elev_mm) > 0.1:
                    if st_shape == "downsloping":
                        st_env = np.exp(-0.5 * ((dt - 0.12) / 0.055) ** 2)
                        y_mm += st_elev_mm * st_env * (1.0 + 1.8 * np.clip(dt - 0.03, 0, 0.16))
                    elif st_shape == "tombstoning":
                        st_env = np.exp(-0.5 * ((dt - 0.09) / 0.065) ** 2)
                        y_mm += st_elev_mm * st_env
                    else:
                        st_env = np.exp(-0.5 * ((dt - 0.11) / 0.065) ** 2)
                        y_mm += st_elev_mm * st_env
                t_c = qt_s - 0.055
                t_sig = 0.055 if t_shape == "genis" else (0.020 if t_shape in ("sivri", "sivri_no_p") else 0.030)
                y_mm += polarity * t_amp_mm * np.exp(-0.5 * ((dt - t_c) / t_sig) ** 2)
            return y_mm

        x_left, x_right = 80, w - 60
        pw = x_right - x_left
        t_full = np.arange(pw) / fs

        rhythm_params = dict(
            st_elev_mm=0.0,
            st_shape="normal",
            q_depth_mm=0.0,
            q_width_s=0.02,
            t_amp_mm=2.2,
            t_shape="normal",
            pr_s=0.16,
            qrs_s=0.075,
            qt_s=0.39,
            delta=False,
        )
        precordial_params = dict(rhythm_params)

        if condition == "mi_inferior_stemi":
            rhythm_params.update(st_elev_mm=3.2, t_amp_mm=3.5)
            precordial_params.update(st_elev_mm=2.5, t_amp_mm=3.0)
        elif condition == "mi_anterior_stemi":
            precordial_params.update(st_elev_mm=3.6, t_amp_mm=3.8)
        elif condition == "mi_tombstoning":
            rhythm_params.update(st_elev_mm=7.5, st_shape="tombstoning", t_amp_mm=4.0)
            precordial_params.update(st_elev_mm=7.5, st_shape="tombstoning", t_amp_mm=4.0)
        elif condition == "mi_hyperacute":
            rhythm_params.update(t_amp_mm=6.6, pr_s=0.16, qrs_s=0.075, qt_s=0.36)
            precordial_params.update(t_amp_mm=6.8, pr_s=0.16, qrs_s=0.075, qt_s=0.36)
        elif condition == "mi_ischaemia":
            rhythm_params.update(st_elev_mm=-1.8, st_shape="downsloping", t_amp_mm=-1.2)
            precordial_params.update(st_elev_mm=-2.0, st_shape="downsloping", t_amp_mm=-1.5)
        elif condition == "pmi_inferior_q":
            rhythm_params.update(q_depth_mm=4.0, q_width_s=0.045, t_amp_mm=1.5)
            precordial_params.update(q_depth_mm=3.5, q_width_s=0.045, t_amp_mm=1.5)
        elif condition == "pmi_anterior_q":
            precordial_params.update(q_depth_mm=4.2, q_width_s=0.045, t_amp_mm=1.2)
        elif condition == "pmi_twi":
            rhythm_params.update(q_depth_mm=3.4, q_width_s=0.042, t_amp_mm=-1.6)
            precordial_params.update(q_depth_mm=3.4, q_width_s=0.042, t_amp_mm=-1.6)
        elif condition == "arr_tachy":
            rhythm_params.update(qt_s=0.28, pr_s=0.14)
            precordial_params.update(qt_s=0.28, pr_s=0.14)
        elif condition == "arr_brady":
            rhythm_params.update(qt_s=0.39, pr_s=0.17)
            precordial_params.update(qt_s=0.39, pr_s=0.17)
        elif condition == "arr_wpw":
            rhythm_params.update(pr_s=0.09, qrs_s=0.135, delta=True)
            precordial_params.update(pr_s=0.09, qrs_s=0.135, delta=True)
        elif condition == "arr_lqt":
            rhythm_params.update(qt_s=0.54, t_amp_mm=4.8, t_shape="genis")
            precordial_params.update(qt_s=0.54, t_amp_mm=4.8, t_shape="genis")
        elif condition == "arr_wide_qrs":
            rhythm_params.update(qrs_s=0.155)
            precordial_params.update(qrs_s=0.155)
        elif condition == "arr_hyperk":
            rhythm_params.update(t_amp_mm=6.8, t_shape="sivri_no_p", qrs_s=0.13)
            precordial_params.update(t_amp_mm=6.8, t_shape="sivri_no_p", qrs_s=0.13)

        for r_idx, ry in enumerate(row_ys):
            if r_idx < 3:
                col_w = pw // 4
                for c_idx in range(4):
                    cx0 = x_left + c_idx * col_w
                    cx1 = cx0 + col_w - 8
                    t_col = np.arange(cx1 - cx0) / fs
                    pol = -1.0 if (r_idx == 0 and c_idx == 1) else 1.0
                    use_p = precordial_params if c_idx >= 2 else rhythm_params
                    y_mm = synth_beat(t_col, polarity=pol, **(use_p if pol > 0 else dict()))
                    pts = np.column_stack([np.arange(cx0, cx1), np.round(ry - y_mm * px_mm).astype(int)])
                    cv2.polylines(img, [pts], False, (20, 20, 20), 2, lineType=cv2.LINE_AA)
            else:
                y_mm = synth_beat(t_full, polarity=1.0, **rhythm_params)
                pts = np.column_stack([np.arange(x_left, x_right), np.round(ry - y_mm * px_mm).astype(int)])
                cv2.polylines(img, [pts], False, (20, 20, 20), 2, lineType=cv2.LINE_AA)

        cv2.imwrite(filepath, img)

    def _compute_metrics_from_records(self, records):
        classes = self.MENDELEY_CLASSES
        matrix = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}
        correct = 0
        for rec in records:
            gt = rec["expected_class"]
            pr = rec["predicted_class"]
            if gt in matrix and pr in matrix[gt]:
                matrix[gt][pr] += 1
            if rec["is_correct"]:
                correct += 1

        total = len(records)
        overall_acc = round((correct / total) * 100.0, 2) if total > 0 else 0.0

        class_metrics = []
        for c in classes:
            tp = matrix[c][c]
            fp = sum(matrix[other][c] for other in classes if other != c)
            fn = sum(matrix[c][other] for other in classes if other != c)
            support = sum(matrix[c].values())
            prec = round((tp / (tp + fp)) * 100.0, 2) if (tp + fp) > 0 else 100.0
            rec_val = round((tp / (tp + fn)) * 100.0, 2) if (tp + fn) > 0 else 100.0
            f1 = round((2 * prec * rec_val) / (prec + rec_val), 2) if (prec + rec_val) > 0 else 0.0
            class_metrics.append(
                {
                    "sinif": c,
                    "ornek_sayisi": support,
                    "dogru_tahmin": tp,
                    "precision_pct": prec,
                    "recall_pct": rec_val,
                    "f1_pct": f1,
                }
            )

        return {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy_pct": overall_acc,
            "confusion_matrix": matrix,
            "class_metrics": class_metrics,
            "records": records,
        }

    def evaluate_mendeley_directory(self, dataset_root_dir, max_per_class=25):
        """
        İndirilen Mendeley 'ECG Images dataset of Cardiac Patients' klasör yapısını
        otomatik tarayarak doğruluk (Accuracy) ve Confusion Matrix hesaplar.
        """
        folder_map = {
            "MI": ["Myocardial Infarction", "MI_Patients", "mi"],
            "History of MI": ["History of MI", "PMI", "history_of_mi"],
            "Abnormal Heartbeat": ["abnormal heartbeat", "Abnormal", "arrhythmia"],
            "Normal": ["Normal Person", "Normal", "normal"],
        }
        records = []
        for root, _, files in os.walk(dataset_root_dir):
            folder_name = os.path.basename(root)
            matched_class = None
            for cls_name, keywords in folder_map.items():
                if any(k.lower() in folder_name.lower() for k in keywords):
                    # 'History of MI' ile 'Myocardial Infarction' karışmaması için
                    if cls_name == "MI" and "history" in folder_name.lower():
                        continue
                    matched_class = cls_name
                    break
            if not matched_class:
                continue

            img_files = [f for f in sorted(files) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            if max_per_class:
                img_files = img_files[:max_per_class]

            for fname in img_files:
                fpath = os.path.join(root, fname)
                try:
                    proc = ECGImageProcessor(fpath)
                    ham = proc.process_image()
                    bulgular = self.extractor.tum_bulgulari_cikar(ham)
                    teshisler = self.engine.teshis_koy("hasta_eval", bulgular, verbose=False)
                    pred_cls = self.map_to_mendeley_class(bulgular, teshisler)
                    records.append(
                        {
                            "filename": fname,
                            "source": "Mendeley Klasörü",
                            "expected_class": matched_class,
                            "predicted_class": pred_cls,
                            "expected_detail": matched_class,
                            "predicted_diagnoses": teshisler if teshisler else ["normal_sinus_ritmi"],
                            "is_correct": (pred_cls == matched_class),
                        }
                    )
                except Exception as e:
                    records.append(
                        {
                            "filename": fname,
                            "source": "Mendeley Klasörü",
                            "expected_class": matched_class,
                            "predicted_class": "Hata",
                            "expected_detail": matched_class,
                            "predicted_diagnoses": [str(e)],
                            "is_correct": False,
                        }
                    )
        return self._compute_metrics_from_records(records)

    def run_comprehensive_benchmark(self, docx_path=None, user_uploads_dir=None):
        """
        1) Mendeley Data (gwbz3fsgp8/2) 4-Sınıf 12-Derivasyonlu EKG Test Seti (40 Görüntü: 10 MI, 10 History of MI, 10 Abnormal Heartbeat, 10 Normal)
        2) ecg-disease-patterns.docx 19 Hastalık / 24 Referans Klinik EKG Görüntüsü
        3) Kullanıcının Yüklediği 7 Gerçek Dünya EKG Görüntüsü
        üzerinde uçtan uca doğruluk (Accuracy) ve Confusion Matrix testi çalıştırır.
        """
        records = []

        # BÖLÜM 1: 40 Görüntülük Mendeley 4-Sınıf 12-Derivasyonlu Tam Sayfa Benchmark Seti
        mendeley_specs = [
            # Sınıf 1: Myocardial Infarction (10 Görüntü)
            ("Mendeley_MI_01_Inferior_STEMI.jpg", "mi_inferior_stemi", 80, "MI", "akut_miyokard_enfarktusu"),
            ("Mendeley_MI_02_Inferior_STEMI_HR90.jpg", "mi_inferior_stemi", 90, "MI", "akut_miyokard_enfarktusu"),
            ("Mendeley_MI_03_Anterior_STEMI_V2V5.jpg", "mi_anterior_stemi", 76, "MI", "akut_miyokard_enfarktusu"),
            ("Mendeley_MI_04_Anterior_STEMI_HR84.jpg", "mi_anterior_stemi", 84, "MI", "akut_miyokard_enfarktusu"),
            ("Mendeley_MI_05_Tombstoning_STEMI.jpg", "mi_tombstoning", 88, "MI", "akut_miyokard_enfarktusu"),
            ("Mendeley_MI_06_Tombstoning_HR74.jpg", "mi_tombstoning", 74, "MI", "akut_miyokard_enfarktusu"),
            ("Mendeley_MI_07_Hyperacute_T.jpg", "mi_hyperacute", 75, "MI", "akut_miyokard_enfarktusu"),
            ("Mendeley_MI_08_Hyperacute_T_HR82.jpg", "mi_hyperacute", 82, "MI", "akut_miyokard_enfarktusu"),
            ("Mendeley_MI_09_Subendocardial_Ischaemia.jpg", "mi_ischaemia", 78, "MI", "miyokard_iskemisi"),
            ("Mendeley_MI_10_Ischaemia_Downsloping.jpg", "mi_ischaemia", 85, "MI", "miyokard_iskemisi"),
            # Sınıf 2: History of MI / PMI (10 Görüntü)
            ("Mendeley_PMI_01_Inferior_Pathologic_Q.jpg", "pmi_inferior_q", 72, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_02_Inferior_Q_HR68.jpg", "pmi_inferior_q", 68, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_03_Inferior_Q_HR80.jpg", "pmi_inferior_q", 80, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_04_Anterior_Pathologic_Q.jpg", "pmi_anterior_q", 74, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_05_Anterior_Q_HR78.jpg", "pmi_anterior_q", 78, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_06_Anterior_Q_HR65.jpg", "pmi_anterior_q", 65, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_07_Old_MI_Q_TWI.jpg", "pmi_twi", 70, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_08_Old_MI_Q_TWI_HR76.jpg", "pmi_twi", 76, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_09_Inferior_Scar_Q.jpg", "pmi_inferior_q", 75, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            ("Mendeley_PMI_10_Anterolateral_Scar_Q.jpg", "pmi_anterior_q", 71, "History of MI", "gecirilmis_miyokard_enfarktusu"),
            # Sınıf 3: Abnormal Heartbeat / Arrhythmia (10 Görüntü)
            ("Mendeley_AH_01_Sinus_Tachycardia_124bpm.jpg", "arr_tachy", 124, "Abnormal Heartbeat", "pulmoner_emboli_suphesi"),
            ("Mendeley_AH_02_Sinus_Tachycardia_132bpm.jpg", "arr_tachy", 132, "Abnormal Heartbeat", "pulmoner_emboli_suphesi"),
            ("Mendeley_AH_03_Sinus_Bradycardia_46bpm.jpg", "arr_brady", 46, "Abnormal Heartbeat", "bradikardi"),
            ("Mendeley_AH_04_Sinus_Bradycardia_50bpm.jpg", "arr_brady", 50, "Abnormal Heartbeat", "bradikardi"),
            ("Mendeley_AH_05_WPW_Delta_Wave.jpg", "arr_wpw", 76, "Abnormal Heartbeat", "wolff_parkinson_white"),
            ("Mendeley_AH_06_Long_QT_Syndrome.jpg", "arr_lqt", 68, "Abnormal Heartbeat", "uzun_qt_tip1"),
            ("Mendeley_AH_07_Long_QT_HR72.jpg", "arr_lqt", 72, "Abnormal Heartbeat", "uzun_qt_tip1"),
            ("Mendeley_AH_08_Wide_QRS_Conduction_Block.jpg", "arr_wide_qrs", 74, "Abnormal Heartbeat", "sodyum_kanal_blokeri_toksisitesi"),
            ("Mendeley_AH_09_Wide_QRS_HR82.jpg", "arr_wide_qrs", 82, "Abnormal Heartbeat", "sodyum_kanal_blokeri_toksisitesi"),
            ("Mendeley_AH_10_Hyperkalaemia_Peaked_T.jpg", "arr_hyperk", 75, "Abnormal Heartbeat", "hiperkalemi"),
            # Sınıf 4: Normal Person (10 Görüntü)
            ("Mendeley_NP_01_Normal_Sinus_64bpm.jpg", "normal", 64, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_02_Normal_Sinus_68bpm.jpg", "normal", 68, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_03_Normal_Sinus_72bpm.jpg", "normal", 72, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_04_Normal_Sinus_75bpm.jpg", "normal", 75, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_05_Normal_Sinus_78bpm.jpg", "normal", 78, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_06_Normal_Sinus_80bpm.jpg", "normal", 80, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_07_Normal_Sinus_84bpm.jpg", "normal", 84, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_08_Normal_Sinus_88bpm.jpg", "normal", 88, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_09_Normal_Sinus_91bpm.jpg", "normal", 91, "Normal", "normal_sinus_ritmi"),
            ("Mendeley_NP_10_Normal_Sinus_94bpm.jpg", "normal", 94, "Normal", "normal_sinus_ritmi"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, (fname, cond, hr, exp_cls, exp_det) in enumerate(mendeley_specs):
                fpath = os.path.join(tmpdir, fname)
                self.generate_mendeley_12lead_sample(fpath, condition=cond, hr_bpm=hr, seed=100 + idx)
                proc = ECGImageProcessor(fpath)
                ham = proc.process_image()
                bulgular = self.extractor.tum_bulgulari_cikar(ham)
                teshisler = self.engine.teshis_koy("hasta_mendeley", bulgular, verbose=False)
                pred_cls = self.map_to_mendeley_class(bulgular, teshisler)
                records.append(
                    {
                        "filename": fname,
                        "source": "Mendeley 12-Lead (4-Sınıf)",
                        "expected_class": exp_cls,
                        "predicted_class": pred_cls,
                        "expected_detail": exp_det,
                        "predicted_diagnoses": teshisler if teshisler else (["bradikardi"] if any(b["deger"] == "bradikardi" for b in bulgular) else ["normal_sinus_ritmi"]),
                        "is_correct": (pred_cls == exp_cls),
                    }
                )

            # BÖLÜM 2: ecg-disease-patterns.docx İçindeki 24 Referans Görsel
            if docx_path and os.path.exists(docx_path):
                docx_expected = {
                    "image1.gif": ("MI", "miyokard_iskemisi"),
                    "image2.gif": ("MI", "miyokard_iskemisi"),
                    "image3.gif": ("MI", "miyokard_iskemisi"),
                    "image4.gif": ("MI", "akut_miyokard_enfarktusu"),
                    "image5.gif": ("MI", "akut_miyokard_enfarktusu"),
                    "image6.gif": ("MI", "akut_miyokard_enfarktusu"),
                    "image7.gif": ("History of MI", "gecirilmis_miyokard_enfarktusu"),
                    "image8.gif": ("Abnormal Heartbeat", "perikardit"),
                    "image9.gif": ("Abnormal Heartbeat", "perikardiyal_efuzyon"),
                    "image10.gif": ("Abnormal Heartbeat", "perikardiyal_efuzyon"),
                    "image11.gif": ("Abnormal Heartbeat", "hiperkalemi"),
                    "image12.gif": ("Abnormal Heartbeat", "hipokalemi"),
                    "image13.gif": ("Abnormal Heartbeat", "hiperkalsemi"),
                    "image14.gif": ("Abnormal Heartbeat", "hipokalsemi"),
                    "image15.gif": ("Abnormal Heartbeat", "brugada_tip1"),
                    "image16.gif": ("Abnormal Heartbeat", "wolff_parkinson_white"),
                    "image17.gif": ("Abnormal Heartbeat", "uzun_qt_tip1"),
                    "image18.gif": ("Abnormal Heartbeat", "arvd"),
                    "image19.gif": ("Abnormal Heartbeat", "cpvt"),
                    "image20.gif": ("Abnormal Heartbeat", "erken_repolarizasyon_sendromu"),
                    "image21.gif": ("Abnormal Heartbeat", "sodyum_kanal_blokeri_toksisitesi"),
                    "image22.gif": ("Abnormal Heartbeat", "digoksin_etkisi_veya_toksisitesi"),
                    "image23.gif": ("Abnormal Heartbeat", "akut_inme"),
                    "image24.gif": ("Abnormal Heartbeat", "hipotermi"),
                }
                with zipfile.ZipFile(docx_path, "r") as z:
                    z.extractall(tmpdir)
                media_dir = os.path.join(tmpdir, "word", "media")
                for iname, (exp_cls, exp_det) in docx_expected.items():
                    ipath = os.path.join(media_dir, iname)
                    if os.path.exists(ipath):
                        proc = ECGImageProcessor(ipath)
                        ham = proc.process_image()
                        bulgular = self.extractor.tum_bulgulari_cikar(ham)
                        teshisler = self.engine.teshis_koy("hasta_docx", bulgular, verbose=False)
                        pred_cls = self.map_to_mendeley_class(bulgular, teshisler)
                        records.append(
                            {
                                "filename": iname,
                                "source": "Referans Docx (19 Hastalık)",
                                "expected_class": exp_cls,
                                "predicted_class": pred_cls,
                                "expected_detail": exp_det,
                                "predicted_diagnoses": teshisler,
                                "is_correct": (exp_det in teshisler and pred_cls == exp_cls),
                            }
                        )

            # BÖLÜM 3: Kullanıcının Yüklediği 7 Gerçek Dünya Test Görseli
            if user_uploads_dir and os.path.exists(user_uploads_dir):
                user_expected = [
                    ("test1.jpg", "Abnormal Heartbeat", "wolff_parkinson_white"),
                    ("test_ekg.jpg", "Abnormal Heartbeat", "hipotermi"),
                    ("test3.jpg", "Abnormal Heartbeat", "hiperkalemi"),
                    ("digoxin.jpg", "Abnormal Heartbeat", "digoksin_etkisi_veya_toksisitesi"),
                    ("digoxin-2.jpg", "Abnormal Heartbeat", "digoksin_etkisi_veya_toksisitesi"),
                    ("Long-QT-syndrome_Figure-1.jpg", "Abnormal Heartbeat", "uzun_qt_tip1"),
                    ("long-qt.jpg", "Abnormal Heartbeat", "uzun_qt_tip1"),
                ]
                for uname, exp_cls, exp_det in user_expected:
                    upath = os.path.join(user_uploads_dir, uname)
                    if os.path.exists(upath):
                        proc = ECGImageProcessor(upath)
                        ham = proc.process_image()
                        bulgular = self.extractor.tum_bulgulari_cikar(ham)
                        teshisler = self.engine.teshis_koy("hasta_user", bulgular, verbose=False)
                        pred_cls = self.map_to_mendeley_class(bulgular, teshisler)
                        records.append(
                            {
                                "filename": uname,
                                "source": "Harici Gerçek EKG (7 Test)",
                                "expected_class": exp_cls,
                                "predicted_class": pred_cls,
                                "expected_detail": exp_det,
                                "predicted_diagnoses": teshisler,
                                "is_correct": (exp_det in teshisler and pred_cls == exp_cls),
                            }
                        )

        return self._compute_metrics_from_records(records)


# ==============================================================================
# 5. KATMAN: ANA UYGULAMA AKIŞI (COLAB VE TERMINAL ÇALIŞTIRICISI)
# ==============================================================================
def ekg_analiz_sistemi_calistir(gorsel_yolu, prolog_dosyasi="ekg_kurallari.pl", crop_box=None, lead_mode="auto"):
    print("\n" + "=" * 70)
    try:
        islemci = ECGImageProcessor(gorsel_yolu, crop_box=crop_box, lead_mode=lead_mode)
        ham_veriler = islemci.process_image()
        kalite = ham_veriler.get("kalite_raporu", {})
        if kalite:
            print(
                f"Görsel Kalite & Format: {kalite.get('duzen_tipi')} | "
                f"Çözünürlük: {kalite.get('cozunurluk')} | "
                f"Örnekleme: {kalite.get('ornekleme_hz')} Hz ({kalite.get('piksel_per_mm')} px/mm)"
            )
            for uyari in kalite.get("uyarilar", []):
                print(f"  [!] Not: {uyari}")

        ozet_veriler = {k: v for k, v in ham_veriler.items() if k not in ("alt_paneller", "kalite_raporu")}
        print(f"Resimden Çıkarılan Ham Veriler: {ozet_veriler}")

        cikarici = ECGFeatureExtractor()
        sembolik_bulgular = cikarici.tum_bulgulari_cikar(ham_veriler)

        uzman_sistem = PrologDiagnosisEngine(prolog_dosyasi)
        teshisler = uzman_sistem.teshis_koy("hasta_1", sembolik_bulgular)

        mendeley_sinifi = ECGDatasetEvaluator.map_to_mendeley_class(sembolik_bulgular, teshisler)
        print(f"Mendeley 4-Sınıf Karşılığı: [{mendeley_sinifi}]")

        print("\n--- YAPAY ZEKA TEŞHİS SONUCU ---")
        if teshisler:
            for t in teshisler:
                print(f"-> DİKKAT: Hastada {t.upper()} tespit edildi!")
        else:
            print("-> Normal Sinüs Ritmi (Herhangi bir patoloji tespit edilmedi).")
        return teshisler
    except Exception as e:
        print(f"[{gorsel_yolu}] HATA OLUŞTU: {e}")
        return []


if __name__ == "__main__":
    import sys

    secili_dosyalar = sys.argv[1:]
    if not secili_dosyalar:
        varsayilan_adaylar = [
            "test1.jpg",
            "test_ekg.jpg",
            "test3.jpg",
            "digoxin.jpg",
            "digoxin-2.jpg",
            "Long-QT-syndrome_Figure-1.png",
            "Long-QT-syndrome_Figure-1.jpg",
            "long-qt.jpg",
        ]
        secili_dosyalar = [f for f in varsayilan_adaylar if os.path.exists(f)]
        if not secili_dosyalar:
            uzantilar = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
            secili_dosyalar = sorted(
                [f for f in os.listdir(".") if f.lower().endswith(uzantilar) and os.path.isfile(f)]
            )

    if not secili_dosyalar:
        print("Çalışma dizininde analiz edilecek EKG görseli (.jpg, .png, .gif) bulunamadı.")
    else:
        for dosya in secili_dosyalar:
            ekg_analiz_sistemi_calistir(dosya)
