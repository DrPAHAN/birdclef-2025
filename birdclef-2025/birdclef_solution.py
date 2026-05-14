import os, gc, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 1. УКАЖИТЕ ТОЧНЫЙ ПУТЬ К ПАПКЕ С РАСПАКОВАННЫМ ДАТАСЕТОМ
# Внутри этой папки ДОЛЖНЫ лежать train.csv, sample_submission.csv и папка train_audio/
DATA_DIR = '/Users/dr.pahan/Documents/PAC_python/birdclef-2025'

# Проверка структуры перед запуском
required = {
    'train.csv': 'CSV с метками',
    'train_audio': 'Папка с аудио',
    'sample_submission.csv': 'Пример сабмишна'
}
for name, desc in required.items():
    if not os.path.exists(os.path.join(DATA_DIR, name)):
        raise FileNotFoundError(f"❌ Не найдено: {name} ({desc})\nОжидаемый путь: {os.path.join(DATA_DIR, name)}")

CFG = {
    'seed': 42, 
    'n_folds': 5, 
    'epochs': 5,          # ⬇️ Уменьшено для быстрого локального теста
    'batch_size': 8,      # ⬇️ Уменьшено для Mac
    'lr': 1e-4, 
    'weight_decay': 1e-5, 
    'sample_rate': 32000,
    'duration': 5, 
    'n_mels': 128, 
    'fmin': 20, 
    'fmax': 16000,
    'hop_length': 320, 
    'n_fft': 1024, 
    'focal_gamma': 2.0, 
    'focal_alpha': 0.75,
    'device': 'cpu',      # ⬇️ Принудительно CPU (стабильнее на Mac, убирает MPS-ошибки)
    'paths': {
        'train_audio': os.path.join(DATA_DIR, 'train_audio'),
        'test_soundscapes': os.path.join(DATA_DIR, 'test_soundscapes'),
        'train_soundscapes': os.path.join(DATA_DIR, 'train_soundscapes'),
        'train_csv': os.path.join(DATA_DIR, 'train.csv'),
        'sample_sub': os.path.join(DATA_DIR, 'sample_submission.csv')
    }
}

print(f"✅ Пути проверены. Устройство: {CFG['device']}")

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(CFG['seed'])

# ================= 2. LABEL ENCODING =================
df_train = pd.read_csv(CFG['paths']['train_csv'])
# Собираем все уникальные виды из primary_label и secondary_labels
all_species = set(df_train['primary_label'].dropna().unique())
for sec in df_train['secondary_labels'].dropna():
    all_species.update(eval(sec) if isinstance(sec, str) else sec)
SPECIES_LIST = sorted(list(all_species))
SPECIES2IDX = {s: i for i, s in enumerate(SPECIES_LIST)}
N_CLASSES = len(SPECIES_LIST)  # 206

def encode_labels(row):
    target = torch.zeros(N_CLASSES)
    target[SPECIES2IDX[row['primary_label']]] = 1.0
    if pd.notna(row['secondary_labels']):
        for s in eval(row['secondary_labels']):
            if s in SPECIES2IDX:
                target[SPECIES2IDX[s]] = 1.0
    return target

df_train['target'] = df_train.apply(encode_labels, axis=1)

# ================= 3. DATASET =================
class BirdCLEFDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG['sample_rate'], n_mels=CFG['n_mels'],
            f_min=CFG['fmin'], f_max=CFG['fmax'],
            hop_length=CFG['hop_length'], n_fft=CFG['n_fft'],
            power=2.0, normalized=False
        )
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=48)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
        
    def __len__(self): return len(self.df)
    
    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != CFG['sample_rate']:
            wav = torchaudio.functional.resample(wav, sr, CFG['sample_rate'])
        target_len = CFG['sample_rate'] * CFG['duration']
        if wav.shape[1] > target_len:
            start = random.randint(0, wav.shape[1] - target_len) if self.mode=='train' else 0
            wav = wav[:, start:start+target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        return wav
    
    def augment_mel(self, mel):
        if self.mode != 'train': return mel
        if random.random() < 0.5: mel = self.time_mask(mel)
        if random.random() < 0.5: mel = self.freq_mask(mel)
        mel = mel * (0.8 + 0.4 * random.random())
        return mel
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(CFG['paths']['train_audio'], row['filename'])
        wav = self.load_audio(filepath)
        mel = self.mel_transform(wav).clamp(min=1e-10).log()
        mel = self.augment_mel(mel)
        return {'input': mel, 'target': row['target']}  # torchaudio уже возвращает [1, 128, T]

# ================= 4. MODEL & LOSS =================
class BioacousticModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # num_classes=0 убирает классификатор, но сохраняет global pooling
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, n_classes)
        
    def forward(self, x):
        # EfficientNet ожидает 3 канала. Если 1 -> дублируем
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
            
        # backbone уже возвращает [B, num_features] (обычно [B, 1280])
        feat = self.backbone(x)
        return self.head(feat)

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        return (self.alpha * (1 - pt)**self.gamma * bce).mean()

def macro_roc_auc(y_true, y_pred):
    valid = y_true.sum(axis=0) > 0
    if valid.sum() == 0: return 0.5
    return roc_auc_score(y_true[:, valid], y_pred[:, valid], average='macro')

# ================= 5. TRAINING =================
def train_fold(fold, train_idx, val_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    train_ds = BirdCLEFDataset(df_train.iloc[train_idx], 'train')
    val_ds = BirdCLEFDataset(df_train.iloc[val_idx], 'val')
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size']*2, shuffle=False, num_workers=2, pin_memory=True)
    
    model = BioacousticModel(N_CLASSES).to(CFG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    criterion = FocalBCELoss(alpha=CFG['focal_alpha'], gamma=CFG['focal_gamma'])
    
    best_auc, best_epoch = 0, 0
    for epoch in range(1, CFG['epochs']+1):
        # Train
        model.train(); epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'Fold {fold} | Train E{epoch}', leave=False)
        for batch in pbar:
            x, y = batch['input'].to(CFG['device']), batch['target'].to(CFG['device'])
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        scheduler.step()
        
        # Val
        model.eval(); all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['input'].to(CFG['device']), batch['target'].to(CFG['device'])
                logits = model(x)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(y.cpu().numpy())
        val_auc = macro_roc_auc(np.vstack(all_targets), np.vstack(all_preds))
        
        print(f'Fold {fold} | Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}')
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_fold{fold}_best.pt'))
    return best_auc

# ================= 6. INFERENCE =================
def infer_soundscapes(model_paths):
    model = BioacousticModel(N_CLASSES).to(CFG['device']).eval()
    # Load ensemble weights
    state_dicts = [torch.load(p, map_location=CFG['device'], weights_only=True) for p in model_paths]
    model.load_state_dict(state_dicts[0])
    
    results = []
    files = sorted(os.listdir(CFG['paths']['test_soundscapes']))
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=CFG['sample_rate'], n_mels=CFG['n_mels'],
        f_min=CFG['fmin'], f_max=CFG['fmax'], hop_length=CFG['hop_length'], n_fft=CFG['n_fft'], power=2.0
    )
    
    for fname in tqdm(files, desc='Inference'):
        wav, sr = torchaudio.load(os.path.join(CFG['paths']['test_soundscapes'], fname))
        if sr != CFG['sample_rate']: wav = torchaudio.functional.resample(wav, sr, CFG['sample_rate'])
        soundscape_id = fname.replace('.ogg', '')
        
        # 12 non-overlapping 5s windows
        window_len = CFG['sample_rate'] * 5
        file_preds = []
        for i in range(12):
            start = i * window_len
            segment = wav[:, start:start+window_len]
            mel = mel_transform(segment).clamp(min=1e-10).log().unsqueeze(0).to(CFG['device'])
            
            with torch.no_grad():
                preds = []
                # Multi-model averaging
                for sd in state_dicts:
                    model.load_state_dict(sd)
                    preds.append(torch.sigmoid(model(mel)).cpu().numpy())
                file_preds.append(np.mean(preds, axis=0))
        
        for sec_idx, pred in enumerate(file_preds):
            end_time = (sec_idx + 1) * 5
            row_id = f'soundscape_{soundscape_id}_{end_time}'
            results.append([row_id] + pred.flatten().tolist())
            
    return pd.DataFrame(results, columns=['row_id'] + SPECIES_LIST)

# ================= 7. MAIN EXECUTION =================
if __name__ == '__main__':
    # 1. Создаём локальную папку для чекпоинтов и результатов
    OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Cross-Validation
    kf = KFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
    best_aucs = []
    model_paths = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train)):
        fold_dir = os.path.join(OUTPUT_DIR, f'fold_{fold}')
        auc = train_fold(fold, tr_idx, val_idx, fold_dir)
        best_aucs.append(auc)
        model_paths.append(os.path.join(fold_dir, f'model_fold{fold}_best.pt'))
        
        # Безопасная очистка памяти (работает и на CPU, и на GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f'📊 CV Macro AUC: {np.mean(best_aucs):.4f} ± {np.std(best_aucs):.4f}')
    
    # 3. Инференс и формирование submission
    print('🚀 Starting inference...')
    sub_df = infer_soundscapes(model_paths)
    
    # Приводим колонки к формату sample_submission
    sample_sub = pd.read_csv(CFG['paths']['sample_sub'])
    species_cols = [c for c in sample_sub.columns if c != 'row_id']
    sub_df = sub_df[['row_id'] + species_cols]
    
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    sub_df.to_csv(submission_path, index=False)
    print(f'✅ submission.csv сохранён в: {submission_path}')