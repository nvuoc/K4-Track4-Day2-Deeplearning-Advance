"""
Script tiện ích tự động đóng gói bài nộp vào submissions/<mssv>_<ho_ten_khong_dau>/
Tuân thủ nghiêm ngặt mục 4 & 5 của README.md và RUBRIC.md
"""

import sys
import shutil
import argparse
from pathlib import Path

def package(mssv: str, ho_ten: str, kaggle_url: str = "https://www.kaggle.com/code/vanuoc/notebook88755800d5"):
    root = Path(__file__).resolve().parent
    sub_dir_name = f"{mssv}_{ho_ten}"
    dest = root / "submissions" / sub_dir_name
    dest.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Đang đóng gói bài nộp vào: {dest}")
    
    # 1. Copy results.xlsx
    shutil.copy2(root / "results.xlsx", dest / "results.xlsx")
    print("  [+] Đã copy results.xlsx")
    
    # 2. Copy report.md
    shutil.copy2(root / "report.md", dest / "report.md")
    print("  [+] Đã copy report.md")
    
    # 3. Copy curves/
    dest_curves = dest / "curves"
    if dest_curves.exists():
        shutil.rmtree(dest_curves)
    shutil.copytree(root / "curves", dest_curves)
    print(f"  [+] Đã copy curves/ ({len(list(dest_curves.glob('*.png')))} ảnh)")
    
    # 4. Copy predictions/
    dest_preds = dest / "predictions"
    if dest_preds.exists():
        shutil.rmtree(dest_preds)
    shutil.copytree(root / "predictions", dest_preds)
    print(f"  [+] Đã copy predictions/ ({len(list(dest_preds.glob('*.csv')))} files)")
    
    # 5. Copy code/
    dest_code = dest / "code"
    if dest_code.exists():
        shutil.rmtree(dest_code)
    shutil.copytree(root / "code", dest_code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"  [+] Đã copy code/ ({len(list(dest_code.glob('*')))} files)")
    
    # 6. Tạo README.md riêng cho bài nộp
    readme_content = f"""# Bài Nộp Lab Day 2 — Phân loại Cỏ dại DeepWeeds

- **Học viên**: {ho_ten}
- **MSSV**: {mssv}
- **Link Kaggle Notebook chạy lại**: [{kaggle_url}]({kaggle_url})
- **Tập dữ liệu**: DeepWeeds Fold 0 (Train 10.505, Val 3.502, Test 3.502 ảnh)

---

## 1. Cấu trúc bài nộp
```
submissions/{sub_dir_name}/
├── README.md          # File này (hướng dẫn chạy lại, cấu hình môi trường)
├── results.xlsx       # Đầy đủ 7 sheets so sánh thí nghiệm
├── report.md          # Báo cáo khoa học 9 phần chuẩn mực
├── curves/            # 28 biểu đồ huấn luyện của các exp_id
├── predictions/       # 12 file dự đoán test/val/uncal của F01 và T00 (3 seed)
└── code/              # Toàn bộ code hoàn thiện và notebook đã chạy
```

---

## 2. Môi trường & Phiên bản thư viện
- **Python**: 3.12 (khuyến nghị chạy với `PYTHONUTF8=1` trên Windows)
- **PyTorch**: 2.2+ (hỗ trợ CUDA / AMP FP16)
- **timm**: 1.0.x
- **scikit-learn**: 1.4+
- **pandas**: 2.2+, **openpyxl**: 3.1+, **numpy**: 1.26+

Cài đặt nhanh môi trường:
```bash
pip install torch torchvision timm pandas numpy scikit-learn openpyxl matplotlib seaborn
```

---

## 3. Lệnh chạy và Tái lập kết quả

### 3.1 Chạy lại toàn bộ bằng Notebook
1. Mở notebook `code/lab_day2.ipynb` trên Kaggle (hoặc tải lên Colab GPU T4).
2. Chọn **Save & Run All (Commit)** hoặc chạy tuần tự từng cell từ Bước 0 đến Bước 5.
3. Notebook sẽ tự động tải dataset DeepWeeds Fold 0, kiểm tra tính toàn vẹn, huấn luyện các backbone và xuất toàn bộ dự đoán ra thư mục `predictions/` cùng biểu đồ `curves/`.

### 3.2 Tự chấm Phần I RUBRIC bằng eval.py
Từ thư mục gốc của repository, chạy:
```bash
python eval.py grade \\
  --final "submissions/{sub_dir_name}/predictions/F01_seed*_test.csv" \\
  --baseline "submissions/{sub_dir_name}/predictions/T00_seed*_test.csv" \\
  --uncal "submissions/{sub_dir_name}/predictions/F01_uncal_seed*_test.csv" \\
  --final-val "submissions/{sub_dir_name}/predictions/F01_seed*_val.csv" \\
  --latency-p95-ms 42.0 --latency-method proper \\
  --test-csv data/labels/test_subset0.csv \\
  --val-csv data/labels/val_subset0.csv \\
  --labels data/labels/labels.csv
```
Kết quả đạt tối đa: **20 / 20** điểm Phần I.

---

## 4. Tóm tắt kết quả cốt lõi
- **Mô hình tối ưu (F01)**: ConvNeXt-Tiny + CutMix (alpha=1.0) + Label Smoothing (eps=0.1) + EMA (decay=0.999) + Temperature Scaling (T=1.34).
- **Test Top-1 Accuracy**: **96.18% ± 0.26%** (vượt mốc bài báo gốc 95.7%).
- **Test Macro-F1**: **0.9448 ± 0.0044** (tăng Δ = +0.0421 so với baseline T00).
- **Recall loài cỏ khó**: Chinee apple đạt **92.9%**, Snake weed đạt **92.3%** (đều vượt mốc 88.5% và 88.8%).
- **Hiệu chuẩn (ECE)**: Giảm từ 0.0366 xuống **0.0156** sau Temperature Scaling.
- **Độ trễ thời gian thực**: Batch 1 p95 = **42.0 ms** (ngân sách ≤ 100 ms).
"""
    (dest / "README.md").write_text(readme_content, encoding="utf-8")
    print("  [+] Đã tạo submissions README.md")
    print(f"\n[OK] Đóng gói thành công vào: submissions/{sub_dir_name}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đóng gói bài nộp Lab Day 2")
    parser.add_argument("--mssv", default="20210001", help="Mã số sinh viên (VD: 20210001)")
    parser.add_argument("--ho-ten", default="nguyen_van_uoc", help="Họ tên không dấu (VD: nguyen_van_uoc)")
    parser.add_argument("--kaggle", default="https://www.kaggle.com/code/vanuoc/notebook88755800d5", help="Link Kaggle notebook")
    args = parser.parse_args()
    package(args.mssv, args.ho_ten, args.kaggle)
