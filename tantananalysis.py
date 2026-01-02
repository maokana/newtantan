# =========================
# ニュータンタンメン口コミ分析（見栄え良い可視化版）
# =========================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)

# 1️⃣ ダミーデータ作成
data = {
    "store_name": [
        "川崎ニュータンタン本店", "元祖タンタンメン屋", "激辛タンタンメン館",
        "あっさりタンタンメン堂", "ボリュームタンタン亭", "ニュータンタン支店"
    ],
    "review_text": [
        "辛くてニンニクが効いている。麺も太くて食べ応えあり。",
        "スープが濃厚でコクがある。少し辛め。",
        "めちゃくちゃ辛い！汗だくになるがクセになる。",
        "あっさりして食べやすい。女性にもおすすめ。",
        "量が多くてお腹いっぱいになる。辛さは普通。",
        "辛さ控えめで優しい味。ニンニクは少なめ。"
    ]
}

df = pd.DataFrame(data)

# 2️⃣ TF-IDF ベクトル化
vectorizer = TfidfVectorizer(max_features=20)
X = vectorizer.fit_transform(df["review_text"])

# 3️⃣ KMeansクラスタリング
kmeans = KMeans(n_clusters=2, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# 4️⃣ PCAで2次元に圧縮して可視化
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())

df["pca_x"] = X_pca[:, 0]
df["pca_y"] = X_pca[:, 1]

plt.figure(figsize=(8,6))
palette = sns.color_palette("Set2", n_colors=2)
sns.scatterplot(data=df, x="pca_x", y="pca_y", hue="cluster",
                s=150, palette=palette, legend="full")

# 店舗名ラベルを追加
for i in range(df.shape[0]):
    plt.text(df["pca_x"][i]+0.01, df["pca_y"][i]+0.01,
             df["store_name"][i], fontsize=10)

plt.title("ニュータンタンメン店舗クラスタリング (PCAで2次元化)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title="クラスタ")
plt.tight_layout()
plt.show()

# 5️⃣ クラスタごとの特徴語（TF-IDF上位）
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(2):
    print(f"\nクラスタ {i} の特徴語:")
    for ind in order_centroids[i, :5]:  # 上位5語
        print(f" - {terms[ind]}")
