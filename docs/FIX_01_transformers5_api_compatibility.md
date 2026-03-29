# Fix #1: transformers 5.x API 互換性 & Flash Attention 2 対応

## 問題の本質

`transformers` ライブラリが v4.x → v5.x にメジャーアップデートされた際、内部APIに複数の破壊的変更が入った。
Florence-2 のカスタムモデルコード（`modeling_florence2.py`）は transformers v4.x の内部実装に依存していたため、
v5.x 環境では**モデルの初期化・ロード段階**でクラッシュまたは不正な状態になる。

この問題は大きく4つのサブ問題に分解される：

| # | 問題 | 発生箇所 | 症状 |
|---|------|---------|------|
| A | `_tie_or_clone_weights()` 削除 | `_tie_weights()` | `AttributeError` / 重み共有の破壊 |
| B | `post_init()` の挙動変更 | `__init__()` | 初期化時のハング・クラッシュ |
| C | `is_flash_attn_greater_or_equal_2_10` 関数削除 | import 文 | `ImportError` |
| D | `_supports_flash_attn` 属性の新規要求 | モデル初期化バリデーション | `ValueError: does not support Flash Attention 2` |

---

## A. `_tie_or_clone_weights()` の削除

### 背景

BART ベースのモデルでは、encoder・decoder・lm_head が同一の embedding weight を共有する「weight tying」が行われる。
transformers v4.x では `PreTrainedModel._tie_or_clone_weights()` というユーティリティメソッドでこれを実現していたが、
**v5.x ではこのメソッドが削除された**。

### 修正したファイル

`modeling_florence2.py`

### 修正箇所

**`Florence2LanguageModel._tie_weights()`** (L1964-1971):

```python
def _tie_weights(self):
    if self.config.tie_word_embeddings:
        if hasattr(self, '_tie_or_clone_weights'):
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
        else:
            self.encoder.embed_tokens.weight = self.shared.weight
            self.decoder.embed_tokens.weight = self.shared.weight
```

**`Florence2LanguageForConditionalGeneration._tie_weights()`** (L2091-2100):

```python
def _tie_weights(self):
    if self.config.tie_word_embeddings:
        if hasattr(self, '_tie_or_clone_weights'):
            self._tie_or_clone_weights(self.model.encoder.embed_tokens, self.model.shared)
            self._tie_or_clone_weights(self.model.decoder.embed_tokens, self.model.shared)
            self._tie_or_clone_weights(self.lm_head, self.model.shared)
        else:
            self.model.encoder.embed_tokens.weight = self.model.shared.weight
            self.model.decoder.embed_tokens.weight = self.model.shared.weight
            self.lm_head.weight = self.model.shared.weight
```

### 修正の意味

`hasattr` でメソッドの存在を確認し、v4.x では従来通り `_tie_or_clone_weights()` を使い、
v5.x では直接 `.weight` 属性を代入する。これにより **両バージョンとの後方互換性**を維持する。

重み共有が壊れると、encoder/decoder/lm_head がそれぞれ独立したパラメータとなり、
モデルの推論結果が根本的に狂う（出力がゴミになる、あるいは空になる）。

---

## B. `post_init()` の挙動変更

### 背景

`PreTrainedModel.post_init()` は初期化後のウェイト初期化や weight tying を行うメソッド。
transformers v5.x ではこの内部実装が変更され、`_tie_weights()` の呼び出しタイミングやバリデーションが厳格化された。

Florence-2 は `accelerate` の `init_empty_weights` コンテキスト内でモデルを作成し、
後から手動で weight を設定するというロードフローを取る。
v5.x の `post_init()` はこのパターンと相性が悪く、空のテンソルに対して不正なバリデーションが走る。

### 修正したファイル

`modeling_florence2.py`

### 修正箇所

**`Florence2LanguageModel.__init__()`** (L1961-1962):

```python
# Initialize weights and apply final processing
if not version.parse(transformers.__version__) >= version.parse('5.0.0'):
    self.post_init()
```

**`Florence2LanguageForConditionalGeneration.__init__()`** (L2087-2089):

```python
# Initialize weights and apply final processing
if not version.parse(transformers.__version__) >= version.parse('5.0.0'):
    self.post_init()
```

また、`_tied_weights_keys` クラス属性も同様に条件付きに（L1948-1949, L2077-2078）:

```python
class Florence2LanguageModel(Florence2LanguagePreTrainedModel):
    if not version.parse(transformers.__version__) >= version.parse('5.0.0'):
        _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
```

### 修正の意味

v5.x では `post_init()` を呼ばず、weight tying は `load_model()` 関数内で
`model.language_model.tie_weights()` を手動で呼ぶ形に変更した。
`_tied_weights_keys` は v5.x の `from_pretrained()` が使わないロードパスでは不要であり、
逆に存在すると不整合を起こす可能性があるため、条件付きで除外した。

---

## C. `is_flash_attn_greater_or_equal_2_10` の削除

### 背景

transformers v4.x では Flash Attention 2.10 以上かどうかを判定する専用関数
`is_flash_attn_greater_or_equal_2_10` が `transformers.utils` に存在した。
v5.x ではこれが汎用の `is_flash_attn_greater_or_equal(version_string)` に統合され、
旧関数は削除された。

### エラーメッセージ

```
ImportError: cannot import name 'is_flash_attn_greater_or_equal_2_10' from 'transformers.utils'
```

### 修正したファイル

`modeling_florence2.py`

### 修正箇所 (L44-53)

```python
try:
    from transformers.utils import is_flash_attn_greater_or_equal_2_10
except ImportError:
    try:
        from transformers.utils import is_flash_attn_greater_or_equal
        def is_flash_attn_greater_or_equal_2_10():
            return is_flash_attn_greater_or_equal("2.10")
    except ImportError:
        def is_flash_attn_greater_or_equal_2_10():
            return True
```

### 修正の意味

3段階のフォールバック：
1. v4.x: 元の関数をそのまま使用
2. v5.x: 新しい汎用関数で `"2.10"` を指定して同等の判定
3. どちらも存在しない場合: `True` を返す（Flash Attention がインストール済みと仮定）

---

## D. `_supports_flash_attn` クラス属性の新規要求

### 背景

transformers v5.x では、モデル初期化時に attention 実装のバリデーションが厳格化された。
`attn_implementation="flash_attention_2"` を指定した場合、モデルクラスに
`_supports_flash_attn = True` （v5.x で新設）が設定されていないと `ValueError` が発生する。

v4.x では `_supports_flash_attn_2 = True` のみで十分だったが、
v5.x ではそれに加えて `_supports_flash_attn = True` が必須になった。

### エラーメッセージ

```
ValueError: Florence2ForConditionalGeneration does not support Flash Attention 2 yet.
```

### 修正したファイル

`modeling_florence2.py`

### 修正箇所

**`Florence2LanguagePreTrainedModel`** (L1435-1437):

```python
_supports_flash_attn_2 = True   # v4.x 互換
_supports_flash_attn = True     # v5.x 新規要求
_supports_sdpa = True
```

**`Florence2PreTrainedModel`** (L2366-2368):

```python
_supports_flash_attn_2 = True   # v4.x 互換
_supports_flash_attn = True     # v5.x 新規要求
_supports_sdpa = True
```

### 修正の意味

Florence-2 は内部で Flash Attention 2 / SDPA の両方をサポートしている。
これらのクラス属性を **両方の PreTrainedModel 基底クラス**に追加することで、
transformers v5.x のバリデーションを通過できるようになる。

---

## 追加修正: `configuration_florence2.py` - `forced_bos_token_id` 設定順序

### 背景

`Florence2LanguageConfig.__init__()` で `forced_bos_token_id` を設定していたが、
`super().__init__(**kwargs)` がこの値を上書きしていた。
transformers v5.x では `PretrainedConfig.__init__()` の内部処理が変わり、
この上書きが顕在化した。

config.json で `forced_bos_token_id: 0` と指定されていても、
`super().__init__()` の処理後に `None` や別の値になる場合がある。

### 修正したファイル

`configuration_florence2.py`

### 修正箇所 (L253-266)

```python
forced_bos = kwargs.pop('forced_bos_token_id', bos_token_id)

super().__init__(
    num_labels=num_labels,
    pad_token_id=pad_token_id,
    bos_token_id=bos_token_id,
    eos_token_id=eos_token_id,
    is_encoder_decoder=is_encoder_decoder,
    decoder_start_token_id=decoder_start_token_id,
    forced_eos_token_id=forced_eos_token_id,
    **kwargs,
)

self.forced_bos_token_id = forced_bos
```

### 修正の意味

`forced_bos_token_id` を `kwargs` から先に取り出し（`pop`）、
`super().__init__()` の**後に**明示的に設定する。
これにより、config.json の値が確実に反映される。

`forced_bos_token_id` はデコーダの最初のトークンを強制するパラメータであり、
この値が不正だと生成結果が完全に壊れる。

---

## 追加修正: `nodes.py` - `generation_config` の整合性確保

### 背景

transformers v5.x では `GenerationMixin.generate()` が `generation_config` を参照する際のデフォルト値処理が変更された。
`init_empty_weights` で作成したモデルには `generation_config` が正しく設定されない場合がある。

### 修正したファイル

`nodes.py`

### 修正箇所 - `load_model()` 関数 (L78-91)

```python
if len(tokenizer) != model.language_model.model.shared.num_embeddings:
    model.resize_token_embeddings(len(tokenizer))
    model.language_model.tie_weights()

# Ensure generation_config has correct values from the language model config
lang_config = config.text_config
if hasattr(model.language_model, 'generation_config'):
    gen_cfg = model.language_model.generation_config
    gen_cfg.decoder_start_token_id = getattr(lang_config, 'decoder_start_token_id', 2)
    gen_cfg.eos_token_id = getattr(lang_config, 'eos_token_id', 2)
    gen_cfg.pad_token_id = getattr(lang_config, 'pad_token_id', 1)
    gen_cfg.forced_bos_token_id = getattr(lang_config, 'forced_bos_token_id', 0)
    gen_cfg.forced_eos_token_id = getattr(lang_config, 'forced_eos_token_id', 2)
```

### 修正の意味

2つの処理を追加：

1. **トークナイザーのサイズチェック**: 特殊トークン（`<loc_0>`〜`<loc_999>` 等）追加後の
   トークナイザーの語彙数がモデルの embedding サイズと一致しない場合、
   `resize_token_embeddings()` で拡張し、`tie_weights()` を再実行する。

2. **`generation_config` の明示設定**: `decoder_start_token_id`, `eos_token_id`, `pad_token_id`,
   `forced_bos_token_id`, `forced_eos_token_id` を config から手動で設定する。
   これが不正だとデコードが開始できない・停止しない等の問題が発生する。

---

## 影響範囲

| ファイル | 変更量 | 影響 |
|---------|--------|------|
| `modeling_florence2.py` | 6箇所 | モデル初期化・weight tying・FA2 サポート |
| `configuration_florence2.py` | 1箇所 | config パラメータの正確性 |
| `nodes.py` | 1箇所 | ロード後の整合性確保 |

## テスト環境

- transformers 5.4.0
- PyTorch 2.x (CUDA)
- Florence-2-large / Florence-2-large-ft で動作確認

## 後方互換性

すべての修正は `hasattr` チェックまたは `version.parse()` による分岐で実装しており、
**transformers v4.x 環境でも動作する**。
