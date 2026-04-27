OPTIMIZED_SNN_CONFIG = {
    'input_dim': 256,          # Tăng từ 128 lên 256
    'hidden_dim': 128,         # Tăng từ 64 lên 128
    'num_classes': 8,
    'num_steps': 20,           # Tăng từ 10 lên 20
    'beta': 0.95,              # Hằng số thời gian của LIF
    'learning_rate': 0.0005,   # Giảm learning rate
    'batch_size': 32,
    'epochs': 200,             # Tăng số epoch
    'early_stopping_patience': 20,
    
    # Surrogate gradient parameters
    'surrogate_slope': 15.0,
    
    # Dropout để chống overfit
    'dropout_rate': 0.3
}
