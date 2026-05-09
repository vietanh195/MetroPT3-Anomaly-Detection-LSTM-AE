import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Đọc dữ liệu & xử lý timestamp linh hoạt
# ------------------------------------------------------------
def optimize_memory(df):
    """Ép kiểu dữ liệu số để giảm RAM. Bỏ qua cột datetime."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"RAM ban đầu: {start_mem:.2f} MB")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'datetime64[ns]' or col_type == 'object':
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if col_type.kind in ['i', 'u']:
            if c_min >= 0 and c_max <= 1:
                df[col] = df[col].astype(np.uint8)
            elif c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= 0 and c_max <= np.iinfo(np.uint16).max:
                df[col] = df[col].astype(np.uint16)
            else:
                df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"RAM sau tối ưu: {end_mem:.2f} MB (tiết kiệm {100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df

data_path = '/kaggle/input/datasets/baerpear/metropt3/MetroPT3(AirCompressor).csv'
print("Đang đọc dữ liệu...")
# Read raw, then parse timestamps with mixed format handling
df = pd.read_csv(data_path, sep=',')
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
df.dropna(subset=['timestamp'], inplace=True)

# ------------------------------------------------------------
# 2. Sắp xếp, đặt index và xác định session_id
# ------------------------------------------------------------
df = df.sort_values('timestamp').reset_index(drop=True)
df.set_index('timestamp', inplace=True)

if df.index.duplicated().sum() > 0:
    df = df[~df.index.duplicated(keep='last')]

# Tối ưu bộ nhớ sau khi đã có index đúng
df = optimize_memory(df)

# Xác định session dựa trên gap > 30 phút
time_diff = df.index.to_series().diff().dt.total_seconds()
gap_threshold = 30 * 60
session_id = (time_diff > gap_threshold).cumsum()
df['session_id'] = session_id

# ------------------------------------------------------------
# 3. Resample THEO TỪNG SESSION
# ------------------------------------------------------------
print("Đang resample theo session...")

agg_dict = {
    'TP2': 'mean', 'TP3': 'mean', 'H1': 'mean', 'DV_pressure': 'mean',
    'Reservoirs': 'mean', 'Oil_temperature': 'mean', 'Motor_current': 'mean',
    'COMP': 'mean', 'DV_eletric': 'mean', 'Towers': 'mean', 'MPG': 'mean',
    'LPS': 'mean', 'Pressure_switch': 'mean', 'Oil_level': 'mean',
    'Caudal_impulses': 'last',
    'session_id': 'first'
}

def resample_session(session_df):
    if len(session_df) < 2:
        return pd.DataFrame()
    return session_df.resample('1min').agg(agg_dict)

df_1min = df.groupby('session_id', group_keys=False).apply(resample_session)

# ------------------------------------------------------------
# 4. Xử lý Air_flow_rate
# ------------------------------------------------------------
print("Đang tính toán lưu lượng khí...")
df_1min['Air_flow_rate'] = df_1min.groupby('session_id')['Caudal_impulses'].diff().fillna(0)
df_1min.loc[df_1min['Air_flow_rate'] < 0, 'Air_flow_rate'] = 0
df_1min['Air_flow_rate'] = df_1min['Air_flow_rate'].clip(lower=0)
df_1min.drop(columns=['Caudal_impulses'], inplace=True)

# ------------------------------------------------------------
# 5. Tạo đặc trưng operating_state
# ------------------------------------------------------------
df_1min['operating_state'] = (df_1min['Motor_current'] > 1.0).astype(np.uint8)

# ------------------------------------------------------------
# 6. BỔ SUNG ĐẶC TRƯNG ĐỘNG HỌC (Optimized)
# ------------------------------------------------------------
print("Đang tạo đặc trưng động học...")

# Gradient (tốc độ thay đổi)
for col in ['TP2', 'TP3', 'H1', 'Motor_current']:
    df_1min[f'{col}_grad'] = df_1min.groupby('session_id')[col].diff().fillna(0)

# Rolling stats (30 phút)
window = 30
df_1min['TP3_mean_30min'] = df_1min.groupby('session_id')['TP3'].transform(lambda x: x.rolling(window, min_periods=1).mean())
df_1min['TP3_std_30min']  = df_1min.groupby('session_id')['TP3'].transform(lambda x: x.rolling(window, min_periods=1).std())
df_1min['TP3_std_30min'] = df_1min['TP3_std_30min'].fillna(0)

# MPG on count
mpg_on = (df_1min['MPG'] > 0.5).astype(int)
mpg_rising = mpg_on.groupby(df_1min['session_id']).diff() == 1
df_1min['MPG_on_count_30min'] = mpg_rising.groupby(df_1min['session_id']).transform(lambda x: x.rolling(window, min_periods=1).sum())

# Motor current mean
df_1min['Motor_current_mean_30min'] = df_1min.groupby('session_id')['Motor_current'].transform(lambda x: x.rolling(window, min_periods=1).mean())

# Tốc độ tăng áp khi nạp (TP3_rise_rate)
df_1min['is_loading'] = (df_1min['Motor_current'] > 6.0).astype(int)
df_1min['load_group'] = (df_1min['is_loading'].diff() != 0).cumsum()

def compute_rise_rate(group):
    if group['is_loading'].iloc[0] == 1 and len(group) > 1:
        rate = (group['TP3'].iloc[-1] - group['TP3'].iloc[0]) / len(group)
    else:
        rate = 0.0
    return pd.Series(rate, index=group.index)

rise_rates = df_1min.groupby('load_group', group_keys=False).apply(compute_rise_rate)
df_1min['TP3_rise_rate'] = rise_rates

df_1min.drop(columns=['is_loading', 'load_group'], inplace=True)

# ------------------------------------------------------------
# 7. Lọc session quá ngắn
# ------------------------------------------------------------
min_session_length = 45
session_lengths = df_1min.groupby('session_id').size()
valid_sessions = session_lengths[session_lengths >= min_session_length].index
df_1min = df_1min[df_1min['session_id'].isin(valid_sessions)]
df_1min['session_id'] = df_1min['session_id'].astype('category').cat.codes

# ------------------------------------------------------------
# 8. Kiểm tra và lưu
# ------------------------------------------------------------
print("\n" + "="*50)
print("THÔNG TIN DATASET SAU GIAI ĐOẠN 1 (CÓ FEATURE ENGINEERING)")
print("="*50)
print(f"Số mẫu: {len(df_1min):,}")
print(f"Số session: {df_1min['session_id'].nunique()}")
print(f"Khoảng thời gian: {df_1min.index.min()} → {df_1min.index.max()}")
print(df_1min.info())
print(f"\nSố lượng NaN:\n{df_1min.isna().sum().sort_values(ascending=False)}")

# Lưu file
df_1min.to_parquet('MetroPT3_stage1_feature_rich.parquet', compression='snappy')
print("\nĐã lưu dữ liệu với đặc trưng mở rộng.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Tải dữ liệu từ Giai đoạn 1
df = pd.read_parquet('MetroPT3_stage1_feature_rich.parquet')

# 2. Chia Train/Test với vùng đệm 12h
first_failure_time = pd.Timestamp('2020-04-18 00:00:00')

# Dừng tập Train trước 96 giờ để đảm bảo không dính mầm mống lỗi
buffer_start = first_failure_time - pd.Timedelta(hours=96)
train_df = df[:buffer_start].copy()

# Bắt đầu tập Test ngay từ mốc 96 giờ trước lỗi
test_df = df[buffer_start:].copy()

print(f"Kích thước tập Train: {train_df.shape}")
print(f"Kích thước tập Test: {test_df.shape}")

# 3. Chuẩn hóa dữ liệu
features = [col for col in df.columns if col not in ['session_id']]  # giữ operating_state

scaler = MinMaxScaler()
train_df[features] = scaler.fit_transform(train_df[features])
test_df[features] = scaler.transform(test_df[features])

joblib.dump(scaler, 'minmax_scaler.pkl')
print("Đã lưu scaler.")

# 4. Tạo sliding window theo session
WINDOW_SIZE = 60

def create_windows(data, window_size):
    X, indices = [], []
    for sess in data['session_id'].unique():
        sess_data = data[data['session_id'] == sess]
        values = sess_data[features].values
        idx = sess_data.index
        if len(values) <= window_size:
            continue
        for i in range(len(values) - window_size):
            X.append(values[i:i+window_size])
            indices.append(idx[i])  # LƯU TIMESTAMP BẮT ĐẦU CỬA SỔ
    return np.array(X, dtype=np.float32), indices

print(f"Đang tạo cửa sổ trượt (Size={WINDOW_SIZE})...")
X_train, train_idx = create_windows(train_df, WINDOW_SIZE)
X_test, test_idx = create_windows(test_df, WINDOW_SIZE)

print(f"Đang tạo cửa sổ trượt (Size={WINDOW_SIZE})...")
X_train, train_idx = create_windows(train_df, WINDOW_SIZE)
X_test, test_idx = create_windows(test_df, WINDOW_SIZE)

# 5. Tách Validation (15% cuối của Train)
val_size = int(0.15 * len(X_train))
X_val = X_train[-val_size:]
X_train = X_train[:-val_size]

print(f"Kích thước X_train: {X_train.shape}")
print(f"Kích thước X_val: {X_val.shape}")
print(f"Kích thước X_test: {X_test.shape}")

# 6. Gán nhãn Ground Truth cho Test
failure_windows = [
    ('2020-04-18 00:00:00', '2020-04-18 23:59:00'),
    ('2020-05-29 23:30:00', '2020-05-30 06:00:00'),
    ('2020-06-05 10:00:00', '2020-06-07 14:30:00'),
    ('2020-07-15 14:30:00', '2020-07-15 19:00:00')
]

def label_windows(indices, window_size, failures):
    labels = np.zeros(len(indices))
    for start, end in failures:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        for i, t in enumerate(indices):
            window_end = t + pd.Timedelta(minutes=window_size-1)
            if not (window_end < start_ts or t > end_ts):
                labels[i] = 1
    return labels

y_test = label_windows(test_idx, WINDOW_SIZE, failure_windows)
print(f"Số cửa sổ lỗi trong Test: {y_test.sum()} / {len(y_test)}")

test_idx = np.array(test_idx, dtype='datetime64[ns]')
np.save('test_idx.npy', test_idx)

# 7. Lưu dữ liệu
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)