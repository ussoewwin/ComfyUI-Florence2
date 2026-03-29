# Fix #3: generate() における attention_mask 未転送問題

## 問題の本質

`Florence2ForConditionalGeneration.generate()` が画像特徴量とテキスト入力を結合した後、
**`attention_mask` を内部の `language_model.generate()` に渡していなかった**。

これはコード上の潜在バグであり、transformers のバージョンとは無関係に存在していた。
しかし v4.x では内部的に `attention_mask=None` のケースを推定で補完する処理があったため
（特に `num_beams=1` の場合）、偶然動作していた。

v5.x では `GenerationMixin.generate()` の内部実装が変更され、
`attention_mask=None` の場合の補完ロジックが厳格化された結果、
**エラーは出ないが出力が空**（機能が死んでいる）という最も厄介な障害が発生した。

これが「開発者はエラーを直したと言っているが、機能は死んでいる」の**直接原因**である。

---

## 技術的な詳細

### Florence-2 の generate() フロー

```
Florence2ForConditionalGeneration.generate()
    │
    ├─ 1. input_ids → inputs_embeds に変換
    ├─ 2. pixel_values → image_features に変換 (_encode_image)
    ├─ 3. [image_features, text_embeds] を結合 → inputs_embeds
    │     同時に attention_mask も結合して生成
    │
    └─ 4. language_model.generate(inputs_embeds=..., attention_mask=???)
                                                     ^^^^^^^^^^^^^^^^
                                                     ここが問題
```

ステップ3で `_merge_input_ids_with_image_features()` が `inputs_embeds` と `attention_mask` の
両方を返す。しかし**ステップ4で `attention_mask` が `language_model.generate()` に渡されていなかった**。

### `_merge_input_ids_with_image_features()` の実装

```python
# modeling_florence2.py L2664-2686
def _merge_input_ids_with_image_features(self, image_features, inputs_embeds):
    batch_size, image_token_length = image_features.size()[:-1]
    device = image_features.device
    image_attention_mask = torch.ones(batch_size, image_token_length, device=device)

    task_prefix_embeds = inputs_embeds
    task_prefix_attention_mask = torch.ones(batch_size, task_prefix_embeds.size(1), device=device)

    # concat [image embeds, task prefix embeds]
    inputs_embeds = torch.cat([image_features, task_prefix_embeds], dim=1)
    attention_mask = torch.cat([image_attention_mask, task_prefix_attention_mask], dim=1)

    return inputs_embeds, attention_mask
```

この関数は正しく `attention_mask` を生成して返している。
問題は呼び出し側がこれを無視していたこと。

---

## 修正したファイル

`modeling_florence2.py`

---

## コードの詳細

### 修正前（オリジナル）

```python
def generate(self, input_ids, inputs_embeds=None, pixel_values=None, **kwargs):
    if inputs_embeds is None:
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_features = self._encode_image(pixel_values)
            inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)
            # ↑ attention_mask を受け取っているが…

    return self.language_model.generate(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        # ↓ attention_mask を渡していない！
        **kwargs
    )
```

`attention_mask` はローカル変数としてセットされるが、`language_model.generate()` には
`**kwargs` の中に入っていない（`kwargs` から pop されてもいないし、明示的にも渡されていない）。

### 修正後 (L2798-2823)

```python
def generate(self, input_ids, inputs_embeds=None, pixel_values=None, **kwargs):

    attention_mask = kwargs.pop('attention_mask', None)

    if inputs_embeds is None:
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_features = self._encode_image(pixel_values)
            inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)

    if attention_mask is None and inputs_embeds is not None:
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)

    return self.language_model.generate(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **kwargs
    )
```

---

## 修正の意味

### 1. `kwargs.pop('attention_mask', None)`

外部から `attention_mask` が渡された場合（ComfyUI の processor が生成する場合等）、
それを `kwargs` から取り出して使用する。
`pop` で取り出すのは、後で `**kwargs` として再度渡す際に重複しないようにするため。

### 2. `_merge_input_ids_with_image_features()` による上書き

画像入力がある場合、この関数が生成する `attention_mask` が
画像トークン + テキストトークンの正しい長さを持つ。
外部から渡された `attention_mask`（テキスト部分のみ）よりも適切であるため、上書きされる。

### 3. フォールバック生成

画像なし・外部マスクなしの場合のために、`inputs_embeds` の shape から
全トークンを attend する `attention_mask` を生成する。

### 4. `language_model.generate()` への明示的な転送

最も重要な修正。`attention_mask` を明示的に `language_model.generate()` に渡すことで、
transformers の `GenerationMixin` がエンコーダに正しい `attention_mask` を適用できる。

---

## なぜ attention_mask が重要か

### Encoder-Decoder モデルにおける attention_mask の役割

Florence-2 のテキスト生成部分は BART ベースの Encoder-Decoder モデルである。

```
inputs_embeds = [画像トークン(577個)] + [テキストトークン(可変)]
                 ↓
            Encoder (attention_mask で有効トークンを指定)
                 ↓
            Decoder (生成ループ)
```

`attention_mask` が `None` の場合、Encoder は入力のどのトークンに attention するべきか判断できない。

### beam search での深刻化

`num_beams > 1`（beam search）の場合、問題はさらに深刻になる。
beam search は入力をビーム数分だけ複製するが、`attention_mask=None` だとこの複製と
実際の入力の対応が崩れる。結果として：

- 出力が空文字列
- 出力が `</s>` のみ（即座に終了）
- 出力が意味不明な文字列

のいずれかになる。

---

## 関連修正: `num_beams` のデフォルト値変更

### 修正したファイル

`nodes.py`

### 修正箇所 (L379)

```python
"num_beams": ("INT", {"default": 1, "min": 1, "max": 64}),
```

オリジナルは `default: 3` だった。

### 変更の理由

1. **beam search の問題軽減**: `attention_mask` の修正により beam search は正常動作するが、
   デフォルトを greedy decoding (`num_beams=1`) にすることで、
   万が一の不具合リスクを低減する。

2. **推論速度**: beam search は beam 数分の並列処理が必要であり、
   VRAM 使用量と推論時間が増加する。多くのユースケースでは `num_beams=1` で十分。

3. **`do_sample=True` との整合性**: オリジナルコードでは `do_sample=True`（サンプリング）と
   `num_beams=3`（beam search）が同時にデフォルトになっていた。
   この組み合わせ自体は有効だが、意図しない挙動を引き起こしやすい。

---

## この問題が見つかりにくかった理由

1. **エラーが出ない**: `attention_mask=None` は有効な入力であり、transformers 内部で
   暗黙的に処理される。例外は発生せず、ログにも警告は出ない。

2. **v4.x では偶然動いていた**: v4.x の `GenerationMixin` は `attention_mask=None` の場合に
   `inputs_embeds.shape` から推定した全 `1` のマスクを自動生成する挙動があった。
   v5.x ではこの推定ロジックが変更された。

3. **開発者は「エラー修正」に集中**: transformers v5.x 対応として明示的なエラー
   （`ImportError`, `ValueError` 等）は修正されたが、
   「エラーは出ないが結果が空」という機能的バグは見落とされた。

4. **症状が環境依存**: `num_beams=1` + 特定のモデルサイズでは部分的に動作する場合がある。
   beam search 使用時に特に顕在化する。

---

## 影響範囲

| ファイル | 変更量 | 影響 |
|---------|--------|------|
| `modeling_florence2.py` | `generate()` メソッド 1箇所 | **全ての生成タスク** |
| `nodes.py` | `INPUT_TYPES` 1箇所 | デフォルトの推論設定 |

## テスト環境

- transformers 5.4.0
- num_beams=1 (greedy) および num_beams=3 (beam search) で動作確認
- caption, detailed_caption, OD, OCR 等の各タスクで確認

## 後方互換性

`attention_mask` を明示的に渡すことは、transformers v4.x でも正しい動作であり、副作用はない。
むしろ v4.x でも本来は明示的に渡すべきだった（暗黙的な推定に依存すべきでなかった）。
