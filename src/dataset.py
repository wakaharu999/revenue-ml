import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class RevenueDataset(Dataset):
    """
    売上規模分類のためのカスタムDatasetクラス
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        # BERTの最大入力長は通常512トークン
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # TokenizerでテキストをBERTが読める数字（テンソル）に変換
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,    # [CLS], [SEP] などの特別トークンを追加
            max_length=self.max_length, # 最大長で切り捨て/パディング
            return_token_type_ids=False,
            padding='max_length',       # 短いテキストは0で埋める
            truncation=True,            # 512を超えたら切り捨てる
            return_attention_mask=True, # どこまでが実際のテキストかを示すマスク
            return_tensors='pt',        # PyTorchのテンソル形式で返す
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df: pd.DataFrame, tokenizer, max_length=512, batch_size=16, is_train=True):
    """
    DataFrameからDataLoaderを作成するヘルパー関数
    ※ dfには 'text' 列と 'revenue_class' 列が存在することを想定
    """
    ds = RevenueDataset(
        texts=df['text'].to_numpy(),
        labels=df['revenue_class'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,    # 学習時はシャッフルする
        num_workers=2,       # データロードの並列化（環境に合わせて調整）
        pin_memory=True      # GPUへの転送を高速化
    )