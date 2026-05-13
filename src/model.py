import torch
import torch.nn as nn
from transformers import AutoModel

class HierarchicalAttentionBERT(nn.Module):
    def __init__(self, model_name="cl-tohoku/bert-base-japanese-v3", 
                 num_categories=8, num_classes=3, category_embed_size=64, projection_dim=256):
        super(HierarchicalAttentionBERT, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        
        # 1. Attention Pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 2. Category Embedding
        self.category_embedding = nn.Embedding(num_categories, category_embed_size)
        
        # 🌟 3. 独立した射影層 (MLP)
        # テキストとカテゴリ、それぞれのモダリティを共通の表現空間に飛ばす
        self.text_projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.category_projection = nn.Sequential(
            nn.Linear(category_embed_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 4. 最終分類器 (256 + 256 = 512次元を受け取る)
        self.classifier = nn.Linear(projection_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask, chunk_mask, category_ids):
        B, N, S = input_ids.size()
        
        # BERT処理
        flat_input_ids = input_ids.view(B * N, S)
        flat_attention_mask = attention_mask.view(B * N, S)
        bert_outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        cls_output = bert_outputs.pooler_output.view(B, N, -1)
        
        # Attention Pooling
        attn_weights = self.attention_pooling(cls_output).squeeze(-1)
        attn_weights = attn_weights.masked_fill(chunk_mask == 0, -1e9)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        pooled_text = torch.bmm(attn_probs.unsqueeze(1), cls_output).squeeze(1)
        
        # 🌟 射影層による次元変換と特徴抽出
        text_feat = self.text_projection(pooled_text)
        cat_feat = self.category_projection(self.category_embedding(category_ids))
        
        # 結合して分類
        fused_feat = torch.cat((text_feat, cat_feat), dim=1)
        return self.classifier(fused_feat)