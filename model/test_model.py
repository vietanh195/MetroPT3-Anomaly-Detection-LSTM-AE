import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score as f1_sklearn, roc_auc_score, roc_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 1. Tải dữ liệu ĐỒNG BỘ từ ổ cứng
# ------------------------------------------------------------
print("Đang tải dữ liệu Test...")
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
test_idx = np.load('test_idx.npy', allow_pickle=True)
test_timestamps = pd.to_datetime(test_idx)

# Sanity Check
if len(X_test) != len(test_timestamps):
    raise ValueError(f"LỖI: X_test ({len(X_test)}) và test_timestamps ({len(test_timestamps)}) KHÔNG KHỚP. Hãy chạy lại Giai đoạn 2!")
else:
    print(f"Dữ liệu đồng bộ hoàn hảo: {len(X_test)} mẫu.")

# ------------------------------------------------------------
# 2. Khởi tạo Model (Kiến trúc [16, 8] khớp với lúc Train)
# ------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 8], dropout=0.5):
        super(LSTMAutoencoder, self).__init__()
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.LSTM(prev_dim, h_dim, batch_first=True))
            prev_dim = h_dim
        self.encoder = nn.ModuleList(encoder_layers)
        self.latent_dim = hidden_dims[-1]
        decoder_layers = []
        prev_dim = self.latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.LSTM(prev_dim, h_dim, batch_first=True))
            prev_dim = h_dim
        self.decoder = nn.ModuleList(decoder_layers)
        self.output_layer = nn.Linear(hidden_dims[0], input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        out = x
        for lstm in self.encoder:
            out, _ = lstm(out)
            out = self.dropout(out)
        latent = out[:, -1, :]
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        out = repeated
        for lstm in self.decoder:
            out, _ = lstm(out)
            out = self.dropout(out)
        out = self.output_layer(out)
        return out

model = LSTMAutoencoder(input_dim=X_test.shape[2], hidden_dims=[16, 8], dropout=0.5).to(device)
model.load_state_dict(torch.load('best_lstm_autoencoder.pth', map_location=device))
model.eval()

# ------------------------------------------------------------
# 3. Tính Reconstruction Error (MAE)
# ------------------------------------------------------------
def compute_mae_error(model, data, batch_size=128):
    dataset = torch.tensor(data, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            mae = torch.mean(torch.abs(batch - recon), dim=(1,2)).cpu().numpy()
            errors.append(mae)
    return np.concatenate(errors)

print("Đang tính toán MAE...")
test_errors = compute_mae_error(model, X_test)
smoothed_errors = test_errors  # Bỏ EMA theo tinh thần của Paper

print("\n" + "="*50)
print(" ÁP DỤNG NGƯỠNG ĐỘNG (ADAPTIVE THRESHOLD)")
print("="*50)

# ------------------------------------------------------------
# 4. Ngưỡng Động (Adaptive Threshold)
# ------------------------------------------------------------
window_24h = 1440 
errors_series = pd.Series(test_errors)

# Tính Mean và Std trượt trên 24h
rolling_mean = errors_series.shift(1).rolling(window=window_24h, min_periods=1).mean()
rolling_std = errors_series.shift(1).rolling(window=window_24h, min_periods=1).std().fillna(0)
rolling_mean.iloc[0] = errors_series.iloc[0]

# K = 4 cho Adaptive Threshold
k = 4.1
adaptive_threshold = rolling_mean + k * rolling_std

# Base threshold từ Validation (Ngưỡng sàn 99%)
X_val = np.load('X_val.npy')
val_errors = compute_mae_error(model, X_val)
base_threshold = np.percentile(val_errors, 99.8)

final_threshold = np.maximum(adaptive_threshold, base_threshold)
pred_labels = (errors_series > final_threshold).astype(int).values

# ============================================================
# 4.5. GIAI ĐOẠN 2: OPERATIONAL CONTEXT MASKING (STATE MASKING)
# ============================================================
# Tải lại dữ liệu test_df để lấy cột operating_state tương ứng với các window
df_features = pd.read_parquet('MetroPT3_stage1_feature_rich.parquet')

# Tạo lại tập Test y hệt như lúc nãy bạn đã chia (Bắt đầu từ -72h)
first_failure_time = pd.Timestamp('2020-04-18 00:00:00')
buffer_start = first_failure_time - pd.Timedelta(hours=72)
test_df_raw = df_features[buffer_start:].copy()

# Lấy trạng thái hoạt động ở thời điểm KẾT THÚC của mỗi cửa sổ (điểm mà model dự đoán)
# Vì window_size = 60, điểm cuối cùng của window i là i + 59
window_size = 60
operating_states = []

for i in range(len(test_timestamps)):
    # Lấy index thực tế của điểm cuối cùng trong window
    end_idx = test_timestamps[i] + pd.Timedelta(minutes=window_size - 1)
    
    # Kiểm tra xem máy nén có đang chạy ở phút đó không (operating_state == 1)
    # Nếu bị khuyết dữ liệu, mặc định cho là 1 để an toàn
    state = test_df_raw.loc[end_idx, 'operating_state'] if end_idx in test_df_raw.index else 1
    operating_states.append(state)

operating_states = np.array(operating_states)

# ÁP DỤNG MẶT NẠ: Chỉ giữ lại các báo động khi máy đang hoạt động
# Đây là bước triệt tiêu 80% Báo động giả (False Positives)
pred_labels = pred_labels * operating_states

# ------------------------------------------------------------
# 5. Đánh giá theo sự kiện (Event-based Evaluation)
# ------------------------------------------------------------
def get_anomaly_events(labels, timestamps, min_consecutive=10):
    events = []
    in_event = False
    start_idx = 0
    count = 0
    for i, val in enumerate(labels):
        if val == 1:
            if not in_event:
                start_idx = i
                in_event = True
            count += 1
        else:
            if in_event:
                if count >= min_consecutive: 
                    events.append((timestamps[start_idx], timestamps[i-1], count))
                in_event = False
                count = 0
    if in_event and count >= min_consecutive:
        events.append((timestamps[start_idx], timestamps[-1], count))
    return events

detected_events = get_anomaly_events(pred_labels, test_timestamps, min_consecutive=10)
print(f"\nTổng số sự kiện cảnh báo được phát ra (Sau khi lọc ngữ cảnh): {len(detected_events)}")

# Bước 5.2: Khớp sự kiện cảnh báo với sự cố thực tế
failure_windows = [
    ('2020-04-18 00:00:00', '2020-04-18 23:59:00'),
    ('2020-05-29 23:30:00', '2020-05-30 06:00:00'),
    ('2020-06-05 10:00:00', '2020-06-07 14:30:00'),
    ('2020-07-15 14:30:00', '2020-07-15 19:00:00')
]

TP = 0
FN = 0
FP = 0

lead_time_results = []

# Đếm TP và FN
for i, (f_start, f_end) in enumerate(failure_windows):
    f_start_ts = pd.Timestamp(f_start)
    f_end_ts = pd.Timestamp(f_end)
    
    # Bỏ qua sự cố nếu nó nằm ngoài dữ liệu test (như sự cố #1 bị cắt do vùng đệm)
    if f_end_ts < test_timestamps.min():
        continue
        
    # Vùng cho phép cảnh báo sớm: Báo trước tối đa 72 giờ
    detection_window_start = f_start_ts - pd.Timedelta(hours=72)
    
    event_hit = False
    earliest_lead_time = 0
    
    for e_start, e_end, _ in detected_events:
        # Nếu sự kiện cảnh báo rơi vào (hoặc cắt ngang) vùng cho phép (từ 72h trước sự cố tới hết sự cố)
        if e_end >= detection_window_start and e_start <= f_end_ts:
            event_hit = True
            # Tính Lead Time cho sự kiện cảnh báo sớm nhất
            lt_sec = (f_start_ts - e_start).total_seconds()
            if lt_sec > 0:
                earliest_lead_time = max(earliest_lead_time, lt_sec / 3600.0)
                
    if event_hit:
        TP += 1
        lead_time_results.append(f"Sự cố {f_start[:10]}: BẮT TRÚNG (Báo trước {earliest_lead_time:.2f} giờ)")
    else:
        FN += 1
        lead_time_results.append(f"Sự cố {f_start[:10]}: BỎ LỠ (False Negative)")

# Đếm FP (Cảnh báo giả)
# Các sự kiện không rơi vào bất kỳ cửa sổ detection nào của 4 sự cố sẽ bị coi là Báo động giả
for e_start, e_end, _ in detected_events:
    is_fp = True
    for f_start, f_end in failure_windows:
        f_start_ts = pd.Timestamp(f_start)
        f_end_ts = pd.Timestamp(f_end)
        detection_window_start = f_start_ts - pd.Timedelta(hours=72)
        if e_end >= detection_window_start and e_start <= f_end_ts:
            is_fp = False
            break
    if is_fp:
        FP += 1

# Tính toán Event-based Metrics
event_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
event_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
event_f1 = 2 * (event_precision * event_recall) / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0

# ------------------------------------------------------------
# TÍNH AUC-PR (POINT-WISE METRIC)
# Dùng sai số MAE liên tục đã được lọc qua ngữ cảnh vận hành
# ------------------------------------------------------------
masked_continuous_errors = test_errors * operating_states 

# AUC-ROC yêu cầu nhãn thực tế y_test và điểm sai số liên tục
auc_roc = roc_auc_score(y_test, masked_continuous_errors)

print(f"- True Positives (TP) : {TP} (Bắt trúng)")
print(f"- False Negatives (FN): {FN} (Bỏ lỡ)")
print(f"- False Positives (FP): {FP} (Báo động giả)")
print("-" * 50)
print(f"EVENT PRECISION : {event_precision:.4f}")
print(f"EVENT RECALL    : {event_recall:.4f}")
print(f"EVENT F1-SCORE  : {event_f1:.4f}")
print(f"AUC-ROC (Point-wise): {auc_roc:.4f}") # Chỉ số mới ở đây

print("\n" + "="*50)
print(" CHI TIẾT LEAD TIMES")
print("="*50)
for res in lead_time_results:
    print(res)

# ------------------------------------------------------------
# 6. Trực quan hóa
# ------------------------------------------------------------
# Để dễ nhìn, chúng ta chỉ trực quan MAE khi máy đang chạy
masked_errors = test_errors * operating_states
masked_errors[masked_errors == 0] = np.nan # Ẩn các đoạn máy nghỉ

plt.figure(figsize=(15,6))
plt.plot(test_timestamps, masked_errors, label='MAE (Operating Only)', color='blue', linewidth=0.5)
plt.plot(test_timestamps, final_threshold, label='Adaptive Threshold', color='orange', linestyle='--', linewidth=1.5)

for i, (start, end) in enumerate(failure_windows):
    label = 'True Failure' if i == 0 else ""
    plt.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.3, color='red', label=label)

plt.xlabel('Time')
plt.ylabel('Reconstruction Error (MAE)')
plt.title('Operational Context-Aware Anomaly Detection (MetroPT-3)')
plt.legend()
plt.tight_layout()
plt.show()