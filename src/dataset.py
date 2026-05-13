import torch
from torch.utils.data import Dataset
import pandas as pd

class CompanyPageDataset(Dataset):
    """
    企業 × ページカテゴリ ごとにテキストチャンクを束ねるカスタムデータセット
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.grouped_data = []

        # 企業名とページカテゴリでグループ化
        grouped = df.groupby(['company_name', 'page_category'])

        for (company, category), group in grouped:
            # そのページに属する全チャンクのリストを作成
            chunks = group['text'].tolist()[:15]  # 上限15チャンク（約6000文字）に制限
            label = group['label'].iloc[0]
            category_id = group['category_id'].iloc[0]

            self.grouped_data.append({
                'company_name': company,
                'category_id': category_id,
                'chunks': chunks,
                'label': label
            })

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        item = self.grouped_data[idx]

        # 複数チャンクを一括でトークナイズ
        encodings = self.tokenizer(
            item['chunks'],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors='pt'
        )

        return {
            'company_name': item['company_name'],
            'input_ids': encodings['input_ids'],           # (num_chunks, max_seq_len)
            'attention_mask': encodings['attention_mask'], # (num_chunks, max_seq_len)
            'category_id': torch.tensor(item['category_id'], dtype=torch.long),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


def custom_collate_fn(batch):
    """
    チャンク数 (num_chunks) が異なるデータを1つのミニバッチに揃えるための関数
    """
    labels = torch.stack([item['label'] for item in batch])
    category_ids = torch.stack([item['category_id'] for item in batch])
    company_names = [item['company_name'] for item in batch]

    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]

    # バッチ内で最大のチャンク数を取得（例：ある企業はチャンク3個、ある企業は10個なら、10に揃える）
    max_chunks = max(ids.size(0) for ids in input_ids_list)
    seq_len = input_ids_list[0].size(1)
    batch_size = len(batch)

    # ゼロ埋め用の空テンソルを作成 (batch_size, max_chunks, seq_len)
    batch_input_ids = torch.zeros((batch_size, max_chunks, seq_len), dtype=torch.long)
    batch_attention_mask = torch.zeros((batch_size, max_chunks, seq_len), dtype=torch.long)
    
    # 🌟 後でAttention Poolingをする際に、ゼロ埋めしたダミーチャンクを無視するためのマスク
    chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.float)

    # 実際のデータを流し込む
    for i in range(batch_size):
        num_chunks = input_ids_list[i].size(0)
        batch_input_ids[i, :num_chunks, :] = input_ids_list[i]
        batch_attention_mask[i, :num_chunks, :] = attention_mask_list[i]
        chunk_mask[i, :num_chunks] = 1.0  # データが存在するチャンクは1.0

    return {
        'company_names': company_names,
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'chunk_mask': chunk_mask,
        'category_ids': category_ids,
        'labels': labels
    }