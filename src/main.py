import pandas as pd

# データの読み込み（ファイルパスは環境に合わせて変更してください）
df = pd.read_parquet('data/train_features.parquet')

# 全体の行数と列数を確認
print(df.shape)

# 確認が終わったら exit() でPythonを終了
exit()