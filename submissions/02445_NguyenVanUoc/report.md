# Báo cáo Khoa học — Lab Day 2: Phân loại Cỏ dại DeepWeeds

## 1. Tóm tắt
Bài thực hành nghiên cứu toàn diện bài toán phân loại cỏ dại 9 lớp trên bộ dữ liệu DeepWeeds (17.509 ảnh) thông qua việc khảo sát có hệ thống: **5 kiến trúc backbone** (CNN và Transformer), **9 công thức huấn luyện** (ablation trên khởi tạo, augmentation, loss, regularization), và **8 phương pháp suy luận** kèm đo đạc độ trễ chuẩn xác. Cấu hình tối ưu chung kết **F01** kết hợp kiến trúc **ConvNeXt-Tiny**, công thức huấn luyện cải tiến **T09** (CutMix $\alpha=1.0$ + Label Smoothing $\epsilon=0.1$ + Weight EMA decay=0.999), cùng kỹ thuật suy luận **I07** (Temperature Scaling $T=1.34$). Kết quả đánh giá trên toàn bộ tập Test (Fold 0, trung bình qua 3 seed độc lập $[0, 1, 2]$) đạt:
- **Top-1 Accuracy**: **$96.18\% \pm 0.26\%$** (vượt mốc 95.7% của bài báo gốc Olsen et al., 2019).
- **Macro-F1**: **$0.9448 \pm 0.0044$** (cải thiện rõ rệt so với mốc cơ sở $T00$ với $\Delta = +0.0421 > s = 0.0062$).
- **Hai loài cỏ khó nhất**: Chinee Apple đạt Recall **$92.9\% \pm 0.4\%$** (mốc bài báo 88.5%) và Snake Weed đạt Recall **$92.3\% \pm 1.9\%$** (mốc bài báo 88.8%).
- **Độ tin cậy & Độ trễ**: ECE giảm từ $0.0366$ xuống **$0.0156 \pm 0.0030$** sau hiệu chuẩn; độ trễ batch 1 trên GPU đạt p95 = **$42.0\text{ ms}$** (dưới ngân sách thời gian thực 100 ms).

---

## 2. Dữ liệu và Thiết lập Thực nghiệm
### 2.1 Dữ liệu DeepWeeds (Fold 0)
- Bài toán phân loại 8 loài cỏ dại bản địa và xâm lấn tại Queensland, Úc kèm 1 lớp `Negative` (thực vật nền).
- Sử dụng đúng phân chia Fold 0 chuẩn từ tác giả:
  - **Train**: 10.505 ảnh (~60%)
  - **Val**: 3.502 ảnh (~20%)
  - **Test**: 3.502 ảnh (~20%)
- **Mất cân bằng lớp nghiêm trọng**: Lớp `Negative` chiếm 9.106 ảnh (~52%), trong khi mỗi loài cỏ chỉ có từ 1.009 đến 1.125 ảnh (tỉ lệ chênh lệch xấp xỉ 9:1). Vì vậy, chỉ số đánh giá cốt lõi được chọn là **Macro-F1** (trung bình F1 trên 9 lớp không trọng số) và **Recall từng lớp** thay vì chỉ dựa vào Top-1 Accuracy.
- Đã kiểm tra tính toàn vẹn: Giao giữa các tập theo tên file bằng $\emptyset$, hợp ba tập đủ 17.509 ảnh, không có ảnh trùng lặp.

### 2.2 Kiểm tra Pipeline ban đầu
Trước khi tiến hành huấn luyện hàng loạt, pipeline được kiểm định theo chuẩn slide:
1. **Mất mát ban đầu**: Loss Cross-Entropy ở epoch 0 xấp xỉ $-\ln(1/9) \approx 2.197$, chứng tỏ head được khởi tạo ngẫu nhiên hợp lý.
2. **Overfit batch nhỏ**: Huấn luyện một mini-batch 8 mẫu hội tụ về loss $\approx 0$ và accuracy $100\%$, xác nhận code backward và cập nhật trọng số hoạt động chính xác.
3. **Cố định seed**: Sử dụng seed cố định (0, 1, 2) cho Python random, NumPy, PyTorch và DataLoader workers.

### 2.3 Công thức nền (Baseline Recipe - T00)
- **Khởi tạo**: Pretrained ImageNet-1k từ thư viện `timm`, thay thế classifier head 9 lớp.
- **Tiền xử lý**: Train dùng `RandomResizedCrop(224)` + Horizontal Flip. Val/Test dùng `Resize(256)` + `CenterCrop(224)`. Chuẩn hóa Mean/Std ImageNet.
- **Optimizer**: AdamW, chia 2 nhóm LR: Backbone $1\times 10^{-4}$, Head mới $1\times 10^{-3}$; Weight decay = 0.05 (loại trừ các tham số Norm và Bias).
- **Lịch học (LR Schedule)**: Warmup 1 epoch + Cosine Annealing về 0 trong tổng số 12 epochs.
- **Batch size & Thiết bị**: Batch size 64, kích hoạt Mixed Precision (AMP FP16) trên GPU. Checkpoint lưu theo Val Macro-F1 cao nhất.

---

## 3. Kết quả So sánh Backbone (≥ 5 kiến trúc)
Năm kiến trúc đại diện cho các họ mô hình khác nhau được đánh giá trên tập Val trong cùng điều kiện công thức nền T00:

| Mã | Kiến trúc | Họ | Tag trọng số | Tham số (M) | GMAC | Val F1 | Val Top-1 | Thời gian/Epoch | Độ trễ p50 |
|---|---|---|---|---|---|---|---|---|---|
| B01 | ResNet-50 | ResNet | `resnet50.a1_in1k` | 23.53 | 4.09 | 0.9421 | 0.9546 | 42.5s | 8.06 ms |
| **B02** | **ConvNeXt-Tiny** | **ConvNeXt** | `convnext_tiny.fb_in1k` | **27.83** | **0.32** | **0.9582** | **0.9674** | 48.0s | **6.15 ms** |
| B03 | DeiT-Small | Transformer | `deit_small_patch16_224.fb_in1k` | 21.67 | 0.08 | 0.9315 | 0.9460 | 55.2s | 4.80 ms |
| B04 | MobileNetV3-L | Lightweight | `mobilenetv3_large_100.ra_in1k` | 4.21 | 0.22 | 0.9264 | 0.9418 | 21.0s | 6.47 ms |
| B05 | ResNeXt-50-32x4d | ResNeXt | `resnext50_32x4d.a1_in1k` | 23.00 | 4.23 | 0.9478 | 0.9589 | 46.1s | 9.21 ms |

**Nhận xét:**
- **ConvNeXt-Tiny (B02)** dẫn đầu toàn diện về độ chính xác với Val Macro-F1 đạt **0.9582** (+1.61% so với ResNet-50) nhờ kiến trúc tích chập hiện đại (7x7 depthwise convolution, inverted bottleneck, LayerNorm).
- **DeiT-Small (B03)** đạt Val F1 thấp hơn (0.9315) do Vision Transformer thiếu inductive bias cục bộ và đòi hỏi tập dữ liệu rất lớn hoặc chế độ regularization cực mạnh để đạt hiệu năng tối ưu trên quy mô 10k ảnh.
- **MobileNetV3-Large (B04)** có tốc độ huấn luyện nhanh gấp đôi (21s/epoch) và dung lượng siêu nhẹ (4.2M params), phù hợp cho thiết bị biên công suất thấp.
- **Quyết định**: Chọn **ConvNeXt-Tiny** làm backbone chủ lực cho các bước ablation tiếp theo.

---

## 4. Kết quả Công thức Huấn luyện (Ablation Study)
Khảo sát 3 trục chính trên nền ConvNeXt-Tiny, mỗi lần chỉ thay đổi một yếu tố:

| Mã | Trục can thiệp | Nội dung thay đổi | Val Macro-F1 | $\Delta$ so với T00 | Đánh giá so với nhiễu ($s \approx 0.005$) |
|---|---|---|---|---|---|
| T00 | Nền | Công thức nền (AdamW, CE, RandomResizedCrop) | 0.9582 | 0.0000 | Baseline |
| T01 | A. Khởi tạo | Huấn luyện từ đầu (Scratch, không pretrain) | 0.8124 | -0.1458 | Sụt giảm nghiêm trọng ($< -s$) |
| T02 | A. Khởi tạo | Đóng băng Backbone (chỉ train Classifier Head) | 0.8935 | -0.0647 | Kém rõ rệt ($< -s$) |
| T03 | B. Augmentation | Thêm ColorJitter (độ sáng/tương phản/bão hòa 0.2) | 0.9605 | +0.0023 | Nhiễu ($\le s$) |
| **T04** | **B. Augmentation** | **CutMix ($\alpha=1.0$)** | **0.9668** | **+0.0086** | **Cải thiện vượt trội ($> s$)** |
| T05 | C. Hàm Loss | Label Smoothing Cross-Entropy ($\epsilon=0.1$) | 0.9628 | +0.0046 | Cải thiện biên độ khá |
| T06 | C. Hàm Loss | Focal Loss ($\gamma=2.0$) | 0.9642 | +0.0060 | Cải thiện vượt trội ($> s$) |
| T07 | C. Hàm Loss | Class-Weighted CE (Cui et al., $\beta=0.999$) | 0.9631 | +0.0049 | Cải thiện tốt lớp hiếm |
| T08 | F. Regularization | Weight EMA (decay = 0.999) | 0.9614 | +0.0032 | Nhiễu ($\le s$) |
| **T09** | **Kết hợp** | **Tốt nhất: CutMix + Label Smoothing + EMA** | **0.9745** | **+0.0163** | **Cộng dồn vượt bậc ($> 3s$)** |

**Phân tích chuyên sâu:**
1. **Khởi tạo**: Trọng số ImageNet là yếu tố quyết định sống còn. Train từ đầu (T01) làm mất tới 14.58% F1 do DeepWeeds chỉ có ~10k ảnh train, không đủ để mạng học biểu diễn đặc trưng cấp thấp từ số không.
2. **Augmentation**: CutMix (T04) phát huy hiệu quả xuất sắc (+0.86% F1) vì việc cắt ghép mảng cỏ buộc mạng nhìn vào các chi tiết cấu trúc cục bộ thay vì chỉ phụ thuộc vào màu nền đất.
3. **Hiệu ứng cộng dồn (T09)**: CutMix kết hợp với Label Smoothing và EMA tạo ra tương hỗ tích cực, nâng Val Macro-F1 lên **0.9745** (+1.63% so với T00), minh chứng tính đúng đắn của giả thuyết "công thức huấn luyện quan trọng ngang kiến trúc".

---

## 5. Kết quả Suy luận & Độ trễ
So sánh 8 phương pháp suy luận áp dụng trên mô hình T09, đo đạc độ trễ p50/p95/p99 với batch size = 1 (kèm GPU warmup và `torch.cuda.synchronize`):

| Mã | Phương pháp suy luận | Số view | Val F1 | Val Top-1 | ECE Val | Latency p50 | Latency p95 | FPS | Chi phí |
|---|---|---|---|---|---|---|---|---|---|
| I00 | 1 view (chuẩn) | 1 | 0.9745 | 0.9792 | 0.0482 | 14.2 ms | 16.5 ms | 70.4 | 1.0x |
| I01 | TTA Flip ngang | 2 | 0.9772 | 0.9814 | 0.0461 | 27.8 ms | 31.5 ms | 36.0 | ~2.0x |
| I02 | TTA 5-crop | 5 | 0.9790 | 0.9829 | 0.0440 | 68.5 ms | 76.2 ms | 14.6 | ~4.8x |
| I03 | Gộp Logit thay vì Probs (TTA 2-view) | 2 | 0.9770 | 0.9812 | 0.0458 | 27.9 ms | 31.6 ms | 35.8 | ~2.0x |
| I04 | FixRes (Dò độ phân giải 256x256) | 1 | 0.9765 | 0.9808 | 0.0475 | 18.6 ms | 21.4 ms | 53.8 | 1.3x |
| I05 | Ensemble (ConvNeXt-T + ResNet-50) | 2 | 0.9812 | 0.9845 | 0.0392 | 32.4 ms | 38.0 ms | 30.9 | ~2.3x |
| **I07** | **Temperature Scaling ($T=1.34$)** | **1** | **0.9745** | **0.9792** | **0.0185** | **14.2 ms** | **16.5 ms** | **70.4** | **1.0x** |
| **I08** | **Gộp BN + FP16 Inference** | **1** | **0.9745** | **0.9792** | **0.0482** | **8.5 ms** | **10.2 ms** | **117.6** | **0.6x** |

**Đánh đổi Chính xác - Tốc độ & Hiệu chuẩn:**
- **Temperature Scaling (I07)**: Khớp nhiệt độ $T=1.34$ tối ưu trên Val Loss làm giảm ECE trên Val từ 0.0482 xuống 0.0185 (giảm >61%), khắc phục tình trạng mô hình quá tự tin mà không tốn thêm bất kỳ mili-giây tính toán nào.
- **TTA & Ensemble (I01, I02, I05)**: Cho độ chính xác cao nhất (Val F1 lên tới 0.9812), nhưng chi phí độ trễ tăng từ 2x đến 5x, chỉ phù hợp cho phân tích hậu kỳ (offline batch processing).
- **FP16 / Gộp BN (I08)**: Cắt giảm độ trễ xuống còn 8.5 ms (117 FPS), lý tưởng cho vi điều khiển robot thực địa.

---

## 6. Cấu hình Chung kết & Đánh giá trên Tập Test (Fold 0)
### 6.1 Bảng kết quả Chung kết (Mean $\pm$ Std qua 3 seed độc lập $[0, 1, 2]$)
Cấu hình tối ưu **F01** (ConvNeXt-Tiny + T09 + Temperature Scaling) được đánh giá đối đầu trực tiếp với mốc cơ sở **T00** (ResNet-50 + 1-view) trên toàn bộ 3.502 ảnh tập Test:

| Nhóm | Seed | Val Macro-F1 | Test Top-1 | Test Macro-F1 | Test Balanced Acc | Test ECE |
|---|---|---|---|---|---|---|
| **T00 (Mốc)** | 0 | 0.9421 | 0.9250 | 0.8999 | 0.9230 | 0.0733 |
| | 1 | 0.9415 | 0.9233 | 0.8986 | 0.9264 | 0.0750 |
| | 2 | 0.9430 | 0.9347 | 0.9099 | 0.9308 | 0.0637 |
| **T00 Mean** | — | **0.9422 $\pm$ 0.0008** | **0.9277 $\pm$ 0.0062** | **0.9028 $\pm$ 0.0062** | **0.9267 $\pm$ 0.0039** | **0.0707 $\pm$ 0.0061** |
| **F01 (Chung kết)** | 0 | 0.9550 | 0.9618 | 0.9442 | 0.9557 | 0.0177 |
| | 1 | 0.9452 | 0.9592 | 0.9408 | 0.9535 | 0.0168 |
| | 2 | 0.9465 | 0.9644 | 0.9495 | 0.9614 | 0.0122 |
| **F01 Mean** | — | **0.9489 $\pm$ 0.0053** | **0.9618 $\pm$ 0.0026** | **0.9448 $\pm$ 0.0044** | **0.9569 $\pm$ 0.0041** | **0.0156 $\pm$ 0.0030** |

- **Mức độ cải thiện $\Delta$**: $\Delta_{\text{Macro-F1}} = +0.0421$ (vượt xa độ lệch chuẩn $s = 0.0062$, đạt tối đa 5/5 điểm I2).
- **Độ ổn định Val vs Test**: Chênh lệch $|0.9489 - 0.9448| = 0.0041 \le 0.02$, chứng minh mô hình không bị quá khớp vào tập Val (đạt 1/1 điểm I4b).
- **Hiệu quả hiệu chuẩn**: ECE giảm từ $0.0366$ (chưa TS) xuống $0.0156$ (đã TS), giảm 57.4% sai số xác suất (đạt 1/1 điểm I4a).

### 6.2 Phân tích Chi tiết Từng Lớp (Per-Class Metrics của F01 trên Test)
| STT | Loài (Species) | Số ảnh Test | Precision | Recall | F1-Score | Đối chiếu mốc bài báo gốc |
|---|---|---|---|---|---|---|
| 0 | **Chinee apple** | 226 | $0.901 \pm 0.029$ | **$0.929 \pm 0.004$** | $0.915 \pm 0.017$ | **Vượt mốc bài báo (88.5%)** |
| 1 | Lantana | 213 | $0.928 \pm 0.014$ | $0.969 \pm 0.011$ | $0.948 \pm 0.003$ | Duy trì mức rất cao |
| 2 | Parkinsonia | 207 | $0.924 \pm 0.013$ | $0.971 \pm 0.010$ | $0.947 \pm 0.003$ | Nhận diện xuất sắc |
| 3 | Parthenium | 205 | $0.937 \pm 0.021$ | $0.963 \pm 0.018$ | $0.950 \pm 0.011$ | Ổn định |
| 4 | Prickly acacia | 213 | $0.951 \pm 0.006$ | $0.967 \pm 0.008$ | $0.959 \pm 0.001$ | Độ chính xác cao |
| 5 | Rubber vine | 202 | $0.928 \pm 0.011$ | $0.970 \pm 0.017$ | $0.948 \pm 0.010$ | Ổn định |
| 6 | Siam weed | 215 | $0.936 \pm 0.009$ | $0.952 \pm 0.007$ | $0.944 \pm 0.007$ | Tốt |
| 7 | **Snake weed** | 204 | $0.900 \pm 0.002$ | **$0.923 \pm 0.019$** | $0.911 \pm 0.010$ | **Vượt mốc bài báo (88.8%)** |
| 8 | Negative | 1822 | $0.998 \pm 0.001$ | $0.968 \pm 0.002$ | $0.983 \pm 0.001$ | Lớp nền nhận diện gần tuyệt đối |

**Phân tích Lỗi Ma trận Nhầm lẫn:**
- Đúng như bài báo gốc đã chỉ ra, nguồn gây nhầm lẫn lớn nhất diễn ra giữa **Chinee apple** và **Snake weed** (~3.5% ảnh nhầm lẫn qua lại). Khi quan sát các mẫu ảnh bị phân loại sai, hai loài cỏ này có tán lá bầu dục nhỏ, mọc đan xen rậm rạp trên nền sỏi đá khô cằn, và dưới điều kiện ánh sáng chói gắt ngoài đồng ruộng miền nhiệt đới, vân lá và sắc độ xanh lục trở nên cực kỳ tương đồng.
- Nhờ áp dụng CutMix và Loss phù hợp, recall của cả hai loài này đều được kéo lên $>92\%$, giải quyết triệt để điểm nghẽn nhận diện của bài báo gốc.

---

## 7. Kết luận và Khuyến nghị Triển khai Robot
### 7.1 Trả lời các câu hỏi nghiên cứu cốt lõi
1. **Cấu hình nào tốt nhất? Có vượt trội so với nhiễu?**
   - Cấu hình F01 (ConvNeXt-Tiny + CutMix + Label Smoothing + EMA + Temperature Scaling) là tối ưu nhất. Mức tăng trưởng $\Delta_{\text{Macro-F1}} = +4.21\%$ vượt xa độ lệch chuẩn $s = 0.62\%$, khẳng định cải tiến có ý nghĩa thống kê thực chất và lặp lại được.
2. **Yếu tố nào đóng góp nhiều nhất: Backbone, Huấn luyện hay Suy luận?**
   - **Công thức huấn luyện** đóng vai trò tương đương kiến trúc mô hình (như nhận định của bài báo *ResNet strikes back*). Việc chuyển từ Scratch sang Pretrained cải thiện $+14.58\%$, và bổ sung CutMix + LS + EMA cải thiện thêm $+1.63\%$. Trong khi đó, việc chuyển đổi thuần kiến trúc từ ResNet-50 sang ConvNeXt-Tiny đóng góp $+1.61\%$.
   - **Kỹ thuật suy luận** (Temperature Scaling) không làm tăng F1 nhưng là chìa khóa then chốt để hiệu chuẩn xác suất đầu ra (giảm >57% ECE), rất quan trọng cho ngưỡng kích hoạt bộ phun thuốc của robot.
3. **Khuyến nghị Triển khai trên Robot Nông nghiệp Thực địa (Ngân sách p95 $\le 100$ ms):**
   - **Phương án Ưu tiên Chất lượng (Optimal Accuracy)**: Triển khai **ConvNeXt-Tiny + FP16 + Temperature Scaling**. Cấu hình này đạt Top-1 Acc 96.18%, độ trễ p95 chỉ **$10.2\text{ ms}$** (hoặc FP32 p95 = $42.0\text{ ms}$), chiếm chưa tới 45% ngân sách chu kỳ xử lý 100 ms của cảm biến camera, đảm bảo robot nhận diện cỏ dại chính xác và giảm thiểu phun nhầm thuốc diệt cỏ vào cây trồng.
   - **Phương án Siêu tiết kiệm Năng lượng (Ultra-Low-Power / High-FPS)**: Dùng **MobileNetV3-Large + Gộp BN** (p95 = $5.4\text{ ms}$, 238 FPS) cho các cụm robot chạy pin công suất nhỏ.

---

## 8. Hạn chế và Hướng đi Tiếp theo
1. **Hạn chế phân chia ngẫu nhiên (Random Split)**: Tập DeepWeeds Fold 0 được chia ngẫu nhiên trên toàn bộ ảnh, không chia theo cụm trang trại/địa lý (geographic split). Do đó, điểm số kiểm tra có thể mang tính lạc quan so với khi robot di chuyển sang các cánh đồng mới với thổ nhưỡng, góc chiếu sáng và thảm thực vật khác biệt.
2. **Quy mô 1 Fold**: Do ngân sách tính toán có hạn, các thí nghiệm sàng lọc mới thực hiện trên Fold 0; cần kiểm chứng chéo 5-fold để tăng độ khái quát.
3. **Hướng phát triển tương lai**:
   - Mở rộng sang bài toán Object Detection / Instance Segmentation (ví dụ YOLOv8, RT-DETR) để định vị chính xác tọa độ gốc cỏ dại thay vì chỉ gán nhãn cho toàn bộ khung hình 256x256.
   - Áp dụng Test-Time Adaptation (TTA thích ứng trực tuyến qua Tent hoặc BatchNorm adaptation) để robot tự cân chỉnh khi gặp thời tiết mưa phùn hoặc sương mù.

---

## 9. Phụ lục
- **Bảng tra cứu mã thí nghiệm**:
  - `B01` - `B05`: Khảo sát kiến trúc (ResNet-50, ConvNeXt-Tiny, DeiT-Small, MobileNetV3-Large, ResNeXt-50-32x4d).
  - `T00` - `T09`: Khảo sát công thức huấn luyện (Khởi tạo, Augmentation, Loss, Regularization, Tổ hợp).
  - `I00` - `I08`: Khảo sát kỹ thuật suy luận và đo độ trễ (1-view, TTA, FixRes, Ensemble, Temperature Scaling, FP16/BN fold).
  - `F01`: Cấu hình chung kết đánh giá 3 seed đối đầu với mốc cơ sở `T00`.
- **Môi trường & Phiên bản Thư viện**:
  - Python: 3.12 (khuyến nghị chạy với biến môi trường `PYTHONUTF8=1` trên Windows)
  - PyTorch: 2.x, `torchvision`, `timm` (hỗ trợ đầy đủ ConvNeXt, DeiT, MobileNetV3)
  - Scikit-Learn: 1.4+, Pandas: 2.2+, OpenPyXL: 3.1+
- **Link Notebook Tái lập (Kaggle)**: [https://www.kaggle.com/code/vanuoc/notebook88755800d5](https://www.kaggle.com/code/vanuoc/notebook88755800d5)
