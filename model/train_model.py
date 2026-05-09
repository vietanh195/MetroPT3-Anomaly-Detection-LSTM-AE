import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Thiết lập device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị sử dụng: {device}")

# ------------------------------------------------------------
# 2. Tải dữ liệu đã chuẩn bị
# ------------------------------------------------------------
X_train = np.load('X_train.npy')
X_val   = np.load('X_val.npy')
window_size = X_train.shape[1]
n_features = X_train.shape[2]

print(f"Train shape: {X_train.shape}")
print(f"Val shape  : {X_val.shape}")
print(f"Số đặc trưng: {n_features}")

# Chuyển thành TensorDataset
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.float32))

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------
# 3. Định nghĩa kiến trúc LSTM Autoencoder
# ------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 8], dropout=0.5):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        
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

model = LSTMAutoencoder(input_dim=n_features, hidden_dims=[16, 8], dropout=0.5).to(device)
print(model)

# ------------------------------------------------------------
# 4. Loss có trọng số (cập nhật)
# ------------------------------------------------------------
# Lấy tên đặc trưng từ scaler, nếu không có thì tạo tên giả
try:
    scaler = joblib.load('minmax_scaler.pkl')
    feature_names = list(scaler.feature_names_in_)
except:
    feature_names = [f'feat_{i}' for i in range(n_features)]

# Đảm bảo feature_names có đúng độ dài
if len(feature_names) != n_features:
    print(f"Cảnh báo: feature_names có {len(feature_names)} phần tử, n_features={n_features}. Sẽ tạo tên mới.")
    feature_names = [f'feat_{i}' for i in range(n_features)]

# Định nghĩa các đặc trưng quan trọng (gốc + động học)

important_keywords = [
    'operating_state', 'MPG', 'DV_eletric'
]

weights = torch.ones(n_features, dtype=torch.float32).to(device)

for i, name in enumerate(feature_names):

    if any(keyword in name for keyword in important_keywords):

        weights[i] = 1.5

print("Trọng số loss đã được gán (2.0 cho các đặc trưng quan trọng).")


# Trọng số cũ
# # Định nghĩa các đặc trưng quan trọng (gốc + động học)

# important_keywords = ['TP2', 'TP3', 'H1', 'Motor_current', 'Air_flow_rate',

#                       'TP3_grad', 'TP3_mean', 'TP3_std', 'TP3_rise', 'MPG_on_count']

# weights = torch.ones(n_features, dtype=torch.float32).to(device)

# for i, name in enumerate(feature_names):

#     if any(keyword in name for keyword in important_keywords):

#         weights[i] = 2.0

# print("Trọng số loss đã được gán (2.0 cho các đặc trưng quan trọng).")


criterion = nn.MSELoss(reduction='none')
def weighted_mse_loss(output, target):
    loss_per_element = criterion(output, target)
    weighted = loss_per_element * weights
    return weighted.mean()

# ------------------------------------------------------------
# 5. Huấn luyện với Early Stopping
# ------------------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

num_epochs = 120
patience = 10
best_val_loss = float('inf')
counter = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        inputs = batch[0].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = weighted_mse_loss(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * inputs.size(0)
    
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = weighted_mse_loss(outputs, inputs)
            total_val_loss += loss.item() * inputs.size(0)
    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_lstm_autoencoder.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping tại epoch {epoch+1}")
            break

model.load_state_dict(torch.load('best_lstm_autoencoder.pth'))
torch.save(model.state_dict(), 'final_lstm_autoencoder.pth')
print("Đã lưu model tốt nhất.")

plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_loss.png')
plt.show()