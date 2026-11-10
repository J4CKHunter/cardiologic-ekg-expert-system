% =========================================================
% EKG BİLGİ TABANI (KNOWLEDGE BASE)
% "ecg-disease-patterns.docx" Verilerine Göre Güncellenmiş
% Toplam 19 Hastalık / 7 Kategori
% =========================================================

% Python tarafından dinamik olarak fact ekleneceğini belirtir.
:- dynamic bulgu/3.

% =========================================================
% KATEGORİ 1: MİYOKARD İSKEMİSİ VE ENFARKTÜS
% (Myocardial Ischaemia & Acute Myocardial Infarction)
% =========================================================
% 1. Miyokard İskemisi (Downsloping ST depression, T flattening/inversion, U inversion)
teshis(Hasta, miyokard_iskemisi) :- bulgu(Hasta, st_segmenti, downsloping_cokmus).
teshis(Hasta, miyokard_iskemisi) :- bulgu(Hasta, t_dalgasi, duzlesmis).
teshis(Hasta, miyokard_iskemisi) :- bulgu(Hasta, t_dalgasi, ters_donmus).
teshis(Hasta, miyokard_iskemisi) :- bulgu(Hasta, u_dalgasi, ters_donmus).

% 2. Akut ve Geçirilmiş Miyokard Enfarktüsü (STEMI)
% (Hyperacute T waves, convex/tombstoning ST elevation at J point, new LBBB, pathologic Q waves + TWI)
teshis(Hasta, akut_miyokard_enfarktusu) :- bulgu(Hasta, t_dalgasi, hiperakut).
teshis(Hasta, akut_miyokard_enfarktusu) :- bulgu(Hasta, st_segmenti, j_noktasinda_yukselmis).
teshis(Hasta, akut_miyokard_enfarktusu) :- bulgu(Hasta, st_segmenti, tombstoning_yukselmis).
teshis(Hasta, akut_miyokard_enfarktusu) :- bulgu(Hasta, iletim, yeni_lbbb).
teshis(Hasta, akut_miyokard_enfarktusu) :- bulgu(Hasta, q_dalgasi, patolojik), bulgu(Hasta, t_dalgasi, ters_donmus).
teshis(Hasta, gecirilmis_miyokard_enfarktusu) :- bulgu(Hasta, q_dalgasi, patolojik).

% =========================================================
% KATEGORİ 2: PERİKARDİYAL HASTALIKLAR
% (Pericarditis & Pericardial Effusion)
% =========================================================
% 3. Perikardit (Widespread concave ST elevation, PR depression, T flattening/inversion stages)
teshis(Hasta, perikardit) :- bulgu(Hasta, st_segmenti, yaygin_konkav_yukselmis).
teshis(Hasta, perikardit) :- bulgu(Hasta, pr_segmenti, cokmus).
teshis(Hasta, perikardit_gec_evre) :- bulgu(Hasta, pr_segmenti, cokmus), bulgu(Hasta, t_dalgasi, duzlesmis).
teshis(Hasta, perikardit_gec_evre) :- bulgu(Hasta, pr_segmenti, cokmus), bulgu(Hasta, t_dalgasi, ters_donmus).

% 4. Perikardiyal Efüzyon (Sinus tachycardia, low QRS voltage, electrical alternans)
teshis(Hasta, perikardiyal_efuzyon) :- bulgu(Hasta, qrs_kompleksi, alternans).
teshis(Hasta, perikardiyal_efuzyon) :- bulgu(Hasta, qrs_voltaji, dusuk).
teshis(Hasta, perikardiyal_efuzyon) :- bulgu(Hasta, kalp_hizi, tasikardi), bulgu(Hasta, qrs_voltaji, dusuk).

% =========================================================
% KATEGORİ 3: ELEKTROLİT BOZUKLUKLARI
% (Hyperkalaemia, Hypokalaemia, Hypercalcaemia, Hypocalcaemia)
% =========================================================
% 5. Hiperkalemi (Tall tented T waves, reduced P wave, long PR, short QT, wide QRS, sine wave)
teshis(Hasta, hiperkalemi) :- bulgu(Hasta, t_dalgasi, cadir_seklinde).
teshis(Hasta, hiperkalemi) :- bulgu(Hasta, p_dalgasi, kuculmus), bulgu(Hasta, pr_araligi, uzamis).
teshis(Hasta, hiperkalemi) :- bulgu(Hasta, p_dalgasi, kuculmus), bulgu(Hasta, qrs_kompleksi, genislemis).
teshis(Hasta, hiperkalemi) :- bulgu(Hasta, qt_araligi, kisalmis), bulgu(Hasta, qrs_kompleksi, genislemis).
teshis(Hasta, hiperkalemi) :- bulgu(Hasta, dalga_sekli, sinuzoidal).

% 6. Hipokalemi (PR prolongation, ST depression, flat/inverted T waves, prominent U wave)
teshis(Hasta, hipokalemi) :- bulgu(Hasta, u_dalgasi, belirgin).
teshis(Hasta, hipokalemi) :- bulgu(Hasta, st_segmenti, cokmus), bulgu(Hasta, u_dalgasi, belirgin).
teshis(Hasta, hipokalemi) :- bulgu(Hasta, pr_araligi, uzamis), bulgu(Hasta, st_segmenti, cokmus), bulgu(Hasta, t_dalgasi, duzlesmis).
teshis(Hasta, hipokalemi) :- bulgu(Hasta, pr_araligi, uzamis), bulgu(Hasta, st_segmenti, cokmus), bulgu(Hasta, t_dalgasi, ters_donmus).

% 7. Hiperkalsemi (Shortened QT interval due to shortened ST segment, Osborn J waves, QRS/PR prolongation)
teshis(Hasta, hiperkalsemi) :- bulgu(Hasta, qt_araligi, kisalmis).
teshis(Hasta, hiperkalsemi) :- bulgu(Hasta, j_dalgasi, var).
teshis(Hasta, hiperkalsemi) :- bulgu(Hasta, qt_araligi, kisalmis), bulgu(Hasta, j_dalgasi, osborn_dalgasi).

% 8. Hipokalsemi (Prolonged QT interval due to prolonged ST segment)
teshis(Hasta, hipokalsemi) :- bulgu(Hasta, qt_araligi, uzamis).

% =========================================================
% KATEGORİ 4: KALITSAL KANALOPATİLER
% (Brugada Syndrome, WPW Syndrome, Congenital Long QT Syndrome)
% =========================================================
% 9. Brugada Sendromu (Type 1: coved STE >2mm + negative T; Type 2: saddle STE >2mm & terminal >=1mm; Type 3: saddle STE >2mm & terminal <1mm)
teshis(Hasta, brugada_tip1) :- bulgu(Hasta, st_segmenti, coved_yukselmis), bulgu(Hasta, t_dalgasi, ters_donmus).
teshis(Hasta, brugada_tip2) :- bulgu(Hasta, st_segmenti, eyer_yukselmis_tip2).
teshis(Hasta, brugada_tip3) :- bulgu(Hasta, st_segmenti, eyer_yukselmis_tip3).

% 10. Wolff-Parkinson-White (WPW) (Short PR <120ms, delta wave, wide QRS >120ms)
teshis(Hasta, wolff_parkinson_white) :- bulgu(Hasta, pr_araligi, kisa), bulgu(Hasta, qrs_kompleksi, delta_dalgasi_var), bulgu(Hasta, qrs_kompleksi, genislemis).

% 11. Kalıtsal Uzun QT Sendromu (LQT1: broad T; LQT2: bifid T; LQT3: late T)
teshis(Hasta, uzun_qt_tip1) :- bulgu(Hasta, qt_araligi, uzamis), bulgu(Hasta, t_dalgasi, genis).
teshis(Hasta, uzun_qt_tip2) :- bulgu(Hasta, qt_araligi, uzamis), bulgu(Hasta, t_dalgasi, catalli).
teshis(Hasta, uzun_qt_tip3) :- bulgu(Hasta, qt_araligi, uzamis), bulgu(Hasta, t_dalgasi, gec_baslayan).

% =========================================================
% KATEGORİ 5: DİĞER KALITSAL HASTALIKLAR
% (ARVD, CPVT, Early Repolarisation Syndrome)
% =========================================================
% 12. ARVD (Epsilon wave in V1-V3, TWI in V1-V3, localised QRS widening, VT with LBBB morphology)
teshis(Hasta, arvd) :- bulgu(Hasta, qrs_kompleksi, epsilon_dalgasi_var).
teshis(Hasta, arvd) :- bulgu(Hasta, qrs_kompleksi, lokalize_genis), bulgu(Hasta, t_dalgasi, ters_donmus).
teshis(Hasta, arvd) :- bulgu(Hasta, ritim_turu, vt_lbbb_morfolojisi).

% 13. CPVT (Polymorphic VT with alternating QRS morphology, ventricular fibrillation)
teshis(Hasta, cpvt) :- bulgu(Hasta, ritim_turu, polimorfik_vt_alternan).
teshis(Hasta, cpvt) :- bulgu(Hasta, ritim_turu, ventrikuler_fibrilasyon).

% 14. Erken Repolarizasyon Sendromu (Concave J-point ST elevation, J-point notching)
teshis(Hasta, erken_repolarizasyon_sendromu) :- bulgu(Hasta, st_segmenti, j_noktasi_konkav_yukselmis), bulgu(Hasta, qrs_bitisi, j_noktasi_centikli).
teshis(Hasta, erken_repolarizasyon_sendromu) :- bulgu(Hasta, qrs_bitisi, j_noktasi_centikli).

% =========================================================
% KATEGORİ 6: İLAÇ TOKSİSİTESİ
% (Sodium Channel Blockade & Digoxin Toxicity)
% =========================================================
% 15. Sodyum Kanal Blokeri Toksisitesi (QRS widening >100ms, ventricular arrhythmias)
teshis(Hasta, sodyum_kanal_blokeri_toksisitesi) :- bulgu(Hasta, qrs_kompleksi, genislemis).
teshis(Hasta, sodyum_kanal_blokeri_toksisitesi) :- bulgu(Hasta, ritim_turu, ventrikuler_tasikardi).
teshis(Hasta, sodyum_kanal_blokeri_toksisitesi) :- bulgu(Hasta, ritim_turu, ventrikuler_fibrilasyon).
teshis(Hasta, tca_toksisitesi) :- bulgu(Hasta, qrs_kompleksi, genislemis), bulgu(Hasta, kalp_hizi, tasikardi), bulgu(Hasta, elektriksel_aks, sag_aks_sapmasi).

% 16. Digoksin Toksisitesi (Scooped / down-sloping ST depression, T flattening/inversion, short QT, prolonged PR)
teshis(Hasta, digoksin_etkisi_veya_toksisitesi) :- bulgu(Hasta, st_segmenti, kasik_seklinde_cokmus).
teshis(Hasta, digoksin_toksisitesi_artmis_otomatiklik) :- bulgu(Hasta, st_segmenti, kasik_seklinde_cokmus), bulgu(Hasta, qt_araligi, kisalmis).
teshis(Hasta, digoksin_toksisitesi_yavaslamis_iletim) :- bulgu(Hasta, st_segmenti, kasik_seklinde_cokmus), bulgu(Hasta, pr_araligi, uzamis).

% =========================================================
% KATEGORİ 7: DİĞER (PULMONER EMBOLİ, AKUT İNME, HİPOTERMİ)
% =========================================================
% 17. Pulmoner Emboli (Sinus tachycardia, RBBB, RAD, P pulmonale, S1Q3T3)
teshis(Hasta, pulmoner_emboli_suphesi) :- bulgu(Hasta, ritim_turu, sinus_tasikardisi).
teshis(Hasta, pulmoner_emboli) :- bulgu(Hasta, ozel_patern, s1q3t3).
teshis(Hasta, pulmoner_emboli) :- bulgu(Hasta, iletim, sag_dal_blogu), bulgu(Hasta, elektriksel_aks, sag_aks_sapmasi).

% 18. Akut İnme (Stroke) (Tachycardia, ST depression, prolonged QT, T wave flattening/inversion, prominent U waves)
teshis(Hasta, akut_inme) :- bulgu(Hasta, kalp_hizi, tasikardi), bulgu(Hasta, qt_araligi, uzamis), bulgu(Hasta, st_segmenti, cokmus).
teshis(Hasta, akut_inme) :- bulgu(Hasta, kalp_hizi, tasikardi), bulgu(Hasta, st_segmenti, cokmus), bulgu(Hasta, t_dalgasi, duzlesmis).
teshis(Hasta, akut_inme) :- bulgu(Hasta, kalp_hizi, tasikardi), bulgu(Hasta, st_segmenti, cokmus), bulgu(Hasta, t_dalgasi, ters_donmus).
teshis(Hasta, akut_inme_suphesi) :- bulgu(Hasta, kalp_hizi, tasikardi), bulgu(Hasta, u_dalgasi, belirgin), bulgu(Hasta, qt_araligi, uzamis).

% 19. Hipotermi (Shivering movement artefact, Osborn J waves, PR/QRS/QT prolongation, bradyarrhythmias)
teshis(Hasta, hipotermi) :- bulgu(Hasta, j_dalgasi, osborn_dalgasi).
teshis(Hasta, hipotermi) :- bulgu(Hasta, ekg_kalitesi, titreme_artefakti).
teshis(Hasta, derin_hipotermi) :- bulgu(Hasta, pr_araligi, uzamis), bulgu(Hasta, qrs_kompleksi, genislemis), bulgu(Hasta, qt_araligi, uzamis), bulgu(Hasta, kalp_hizi, bradikardi).
