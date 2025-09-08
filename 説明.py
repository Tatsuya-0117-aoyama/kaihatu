いい質問！ここからは\*\*実運用での“調整の指針”\*\*をギュッとまとめます。まずはデフォルト（前回コード）で回して、下の順で調整していくのが効率的です。

---

# まずはこの設定でスタート（基準値）

* 事前学習（SimCLR）

  * `subjects_per_batch=16`, `samples_per_subject=8`（=128）
  * `temperature=0.2`
  * `pretrain_epochs=60`
  * `pretrain_temporal_pos=True`, `temporal_window=5`
  * `exclude_same_subject_negatives=True`
  * `use_subject_adversary=True`, `adversary_weight=0.2`
* 下流（CO回帰）

  * `freeze_backbone_epochs=10`
  * `epochs=200`, `batch_size=32`, `lr=1e-3`, `weight_decay=1e-4`
  * スケジューラは `cosine`

---

# うまくいかない時の“症状 → どう動かすか”

## A) メモリ落ち/OOMになる

* バッチを下げる：

  * 例1: `subjects_per_batch=12`, `samples_per_subject=6`（=72）
  * 例2: `8 × 8 = 64`
    **目安：合計 ≥ 64** を保てるとSimCLRが安定しやすいです。
* それでも厳しければ**AMP**（自動混合精度）を導入：

  * 事前学習＆下流学習のforward/backwardを `torch.cuda.amp.autocast()` と `GradScaler` で囲む。

## B) 事前学習で loss が下がるのに下流の相関が伸びない

* **温度 `temperature` を上げる**（0.3〜0.5）：ハードネガに寄り過ぎを抑えて、埋め込みの“均一性”を上げる
* **投影次元 `projection_dim` を上げる**（128 → 256）：表現のキャパを増やす
* **時間窓 `temporal_window` を縮める**（5 → 3）：ポジティブが似すぎ→過度な不変化の学習を避ける
* **凍結期間を短く**（`freeze_backbone_epochs=5`）：早めに全層微調整

## C) 被験者依存っぽい（テスト被験者で急に悪化）

* **逆学習を強める**：`adversary_weight=0.3`（最大0.5程度まで）
* **逆学習ウォームアップ**：最初の10epochは `adversary_weight=0` → 以降線形に0.2〜0.3へ
* **同一被験者を負例から除外** は維持（`exclude_same_subject_negatives=True`）
* それでもダメなら**時間窓を少し広げる**（5 → 8〜10）：時系列ゆらぎによりロバスト

## D) 事前学習の収束が鈍い / 不安定

* **温度を下げる**（0.2 → 0.1〜0.15）：ポジティブ重視
* **Augを弱める**：明るさ/コントラストの幅を少し縮める（コード内の±係数を半分に）
* **逆学習を弱める**：`adversary_weight=0.1` か**ウォームアップ**導入

## E) 下流学習で過学習（Val Loss↑ Corr↓）

* **凍結期間を延ばす**（`freeze_backbone_epochs=15`）→ ヘッド中心の学習を長めに
* **重み減衰↑**（`weight_decay=3e-4`）
* **データ拡張を下流にも（軽く）かける**：CO回帰用のDatasetに微弱ノイズ/輝度を追加（やり過ぎ注意）

---

# 調整しやすい「ツマミ」一覧（目安レンジ）

| 目的           | パラメータ                 | 下げると        | 上げると         | 推奨レンジ          |
| ------------ | --------------------- | ----------- | ------------ | -------------- |
| 埋め込みの分離性/均一性 | `temperature`         | ポジティブ重視（局所） | ネガティブ分散↑（均一） | 0.1〜0.5（基準0.2） |
| 表現キャパ        | `projection_dim`      | 軽量/単純       | 豊富/安定        | 64〜256（基準128）  |
| ID抑制         | `adversary_weight`    | 収束安定        | ID情報を強く排除    | 0.0〜0.5（基準0.2） |
| 時系列ロバスト      | `temporal_window`     | 近傍のみ        | 広い時間不変       | 3〜10（基準5）      |
| 対象の多様性（バッチ）  | `subjects_per_batch`  | ネガ不足        | ネガ充実（メモリ重）   | 8〜16           |
| ネガ/ポジの枚数     | `samples_per_subject` | 省メモリ        | 情報豊富（重）      | 6〜8            |

---

# 監視すべき指標（早めに異常を発見）

1. **事前学習**

   * Contrastive loss の推移（滑らかに下降）
   * （可能なら）**Alignment/Uniformity**

     * Alignment = 〓|z₁−z₂|² の平均（低いほど良い）
     * Uniformity = log(E exp(−2|zᵢ−zⱼ|²))（低いほど良い）
   * t-SNE/UMAPで**被験者クラスタリングが強すぎないか**（強い→adversary↑）

2. **下流（CO）**

   * **FoldごとのCorr/MAE**（被験者間のバラつき）
   * **被験者別MAE**（特定の属性で悪化→Aug/temporal\_window/逆学習を調整）

---

# すぐ効く小技（コード調整）

## 1) 逆学習ウォームアップ（簡易）

```python
# pretrain_encoder() の epoch ループ内で
warmup_epochs = 10
curr_adv_w = 0.0 if epoch < warmup_epochs else config.adversary_weight
...
adv_loss = 0.5*(ce(p1, sid.to(config.device)) + ce(p2, sid.to(config.device)))
loss = con_loss + curr_adv_w * adv_loss
```

## 2) 下流で“判別的LR”を使う（ヘッドは大きめ、Backboneは小さめ）

```python
# train_model() で optimizer 作成時に差し替え
backbone_params, head_params = [], []
for n,p in model.named_parameters():
    if any(k in n for k in ['fc','conv_final']):
        head_params.append(p)
    else:
        backbone_params.append(p)

optimizer = optim.Adam([
    {'params': head_params, 'lr': 1e-3, 'weight_decay': config.weight_decay},
    {'params': backbone_params, 'lr': 3e-4, 'weight_decay': config.weight_decay}
])
```

> これで `freeze_backbone_epochs` 経過後も、Backbone側は学習率低めで安定します。

## 3) 事前学習だけ AMP を入れる（メモリ節約 & 速度UP）

```python
scaler = torch.cuda.amp.GradScaler()
for v1, v2, sid in loader:
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        f1 = backbone.forward_features(v1)
        f2 = backbone.forward_features(v2)
        z1 = projector(f1); z2 = projector(f2)
        con_loss = info_nce_with_subject_mask(...)
        adv_loss = ...
        loss = con_loss + config.adversary_weight * adv_loss
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
    scaler.step(opt); scaler.update()
```

---

# 実験プロトコル（順番に）

1. **基準設定**（上の“まずは”）で1回フル実験（事前学習→下流）
2. **温度スイープ**：`temperature ∈ {0.2, 0.3, 0.5}`
3. **逆学習スイープ**＋**ウォームアップ有無**: `adversary_weight ∈ {0.0, 0.1, 0.2, 0.3}`
4. **バッチ設計**（OOMに合わせて）：`16×8 → 12×6 → 8×8`
5. **凍結期間**：`freeze_backbone_epochs ∈ {5, 10, 15}`

→ 各条件で**Fold平均Corr/MAE**と**被験者別MAE**を比較、最良を採用。

---

# 迷ったらこれ

* **被験者依存が強い** → `adversary_weight ↑`（0.3）＋`temperature ↑`（0.3〜0.5）
* **表現が弱く回帰が伸びない** → `projection_dim=256` ＋ `freeze_backbone_epochs=5`
* **学習が不安定** → `temperature ↓`（0.15〜0.2）＋ Aug弱め ＋ 逆学習ウォームアップ
* **OOM** → バッチ縮小 or AMP導入

---

必要なら、あなたのGPUメモリ（例: 8GB/12GB/24GB）に合わせた**最適バッチ設計**と、**AMP込みのパッチ**をこちらで作ります。
