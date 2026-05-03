import torch
import torch.nn as nn
from transformers import AutoModel

class RevenueClassifier(nn.Module):
    """
    日本語BERTを用いた売上規模分類モデル
    """
    def __init__(self, model_name="cl-tohoku/bert-base-japanese-v3", num_classes=5, dropout_rate=0.1):
        super().__init__()
        
        # 1. 事前学習済みの日本語BERT本体をロード
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 2. 過学習を防ぐためのドロップアウト層
        self.drop = nn.Dropout(p=dropout_rate)
        
        # 3. 最終分類層（全結合層）
        # BERTの出力次元（通常768）から、予測したいクラス数へ変換
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        順伝播処理
        """
        # BERTにテンソル化されたテキストを通す
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS]トークンの表現を抽出
        # ※ pooler_output が存在しないモデルへの安全対策として、
        # すべてのトークン出力（last_hidden_state）の先頭トークン（[:, 0, :]）を取得します
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # ドロップアウトを適用
        x = self.drop(cls_output)
        
        # 分類層を通して、各クラスの予測スコア（Logits）を出力
        logits = self.fc(x)
        
        return logits