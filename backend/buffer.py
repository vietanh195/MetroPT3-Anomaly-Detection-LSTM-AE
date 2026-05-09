import pandas as pd
import numpy as np

class RealtimeBuffer:
    def __init__(self, window_size=60, rolling_window=30):
        self.window_size = window_size
        self.rolling_window = rolling_window
        self.raw_records = []
        self.history_1min = pd.DataFrame()
        
        # CHUẨN HÓA THỨ TỰ 25 CỘT THEO ĐÚNG PREPROCESSING.PY
        self.EXPECTED_COLUMNS = [
            'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature',
            'Motor_current', 'COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS',
            'Pressure_switch', 'Oil_level', 'Air_flow_rate', 'operating_state',
            'TP2_grad', 'TP3_grad', 'H1_grad', 'Motor_current_grad',
            'TP3_mean_30min', 'TP3_std_30min', 'MPG_on_count_30min',
            'Motor_current_mean_30min', 'TP3_rise_rate'
        ]
        
    def push_raw_record(self, raw_data_dict):
        self.raw_records.append(raw_data_dict)
        
        # 6 records = 1 phút
        if len(self.raw_records) >= 6:
            df_raw = pd.DataFrame(self.raw_records)
            resampled_row = df_raw.mean()
            
            if 'Caudal_impulses' in df_raw.columns:
                resampled_row['Caudal_impulses'] = df_raw['Caudal_impulses'].iloc[-1]
            
            self.raw_records = [] 
            
            # Thêm vào history_1min (dùng pd.concat để tối ưu)
            self.history_1min = pd.concat([self.history_1min, pd.DataFrame([resampled_row])], ignore_index=True)
            
            # Giữ lại số lượng dòng = window_size (60) + rolling_window (30) + buffer (10) = 100
            max_len = self.window_size + self.rolling_window + 10
            if len(self.history_1min) > max_len:
                self.history_1min = self.history_1min.iloc[-max_len:].reset_index(drop=True)
                
            return self._extract_features()
        return None

    def _extract_features(self):
        # Cần ít nhất 60 rows để model tạo window
        if len(self.history_1min) < self.window_size:
            return None
            
        df = self.history_1min.copy()
        
        # 1. Bổ sung tính năng cơ bản
        df['Air_flow_rate'] = df['Caudal_impulses'].diff().fillna(0).clip(lower=0)
        df['operating_state'] = (df['Motor_current'] > 1.0).astype(np.uint8)
        
        # 2. Gradient (Fix lỗi NaN triệt để)
        for col in ['TP2', 'TP3', 'H1', 'Motor_current']:
            df[f'{col}_grad'] = df[col].diff().fillna(0)
            
        # 3. Rolling stats (30 mins)
        w = self.rolling_window
        df['TP3_mean_30min'] = df['TP3'].rolling(w, min_periods=1).mean().fillna(0)
        df['TP3_std_30min'] = df['TP3'].rolling(w, min_periods=1).std().fillna(0)
        
        mpg_on = (df['MPG'] > 0.5).astype(int)
        mpg_rising = mpg_on.diff() == 1
        df['MPG_on_count_30min'] = mpg_rising.rolling(w, min_periods=1).sum().fillna(0)
        df['Motor_current_mean_30min'] = df['Motor_current'].rolling(w, min_periods=1).mean().fillna(0)
        
        # 4. TP3_rise_rate
        df['is_loading'] = (df['Motor_current'] > 6.0).astype(int)
        df['load_group'] = (df['is_loading'].diff() != 0).cumsum()
        
        df['TP3_rise_rate'] = 0.0
        for name, group in df.groupby('load_group'):
            if group['is_loading'].iloc[0] == 1 and len(group) > 1:
                rate = (group['TP3'].iloc[-1] - group['TP3'].iloc[0]) / len(group)
                df.loc[group.index, 'TP3_rise_rate'] = rate
                
        # 5. ÉP KIỂU VÀ CHỐT THỨ TỰ CỘT
        # Bỏ qua các cột thừa, chỉ lấy đúng 25 cột cần thiết
        try:
            df_final = df[self.EXPECTED_COLUMNS]
        except KeyError as e:
            print(f"Lỗi thiếu cột: {e}")
            return None
            
        # Lấy 60 row cuối cùng
        final_window = df_final.iloc[-self.window_size:]
        return final_window.values