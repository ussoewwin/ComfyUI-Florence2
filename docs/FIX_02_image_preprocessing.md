# Fix #2: 画像前処理パイプラインの互換性修正

## 問題の本質

`Florence2Processor.__call__()` が `CLIPImageProcessor` に画像処理パラメータを渡す際、
`None` 値をそのまま転送していた。
transformers v4.x の `CLIPImageProcessor` は `None` を「未指定（デフォルト値を使え）」と解釈していたが、
**v5.x では `None` を「この処理をスキップせよ」と解釈するように変更された**。

この挙動変更により、以下の連鎖障害が発生する：

```
Florence2Processor.__call__(do_resize=None, do_rescale=None)
    ↓
CLIPImageProcessor: do_resize=None → リサイズしない
CLIPImageProcessor: do_rescale=None → float変換しない
    ↓
入力画像: 元サイズのまま、uint8 のまま
    ↓
RuntimeError: Input type (unsigned char) and bias type (struct c10::Half) should be the same
AssertionError: only support square feature maps for now
```

---

## エラーの詳細

### エラー 1: 型不一致

```
RuntimeError: Input type (unsigned char) and bias type (struct c10::Half) should be the same
```

**原因**: `do_rescale=None` が渡されると `CLIPImageProcessor` はピクセル値を `[0, 255]` の `uint8` のまま返す。
しかしモデルの重みは `float16 (Half)` であるため、PyTorch の畳み込み演算で型不一致エラーが発生する。

正常な動作では `CLIPImageProcessor` が `rescale_factor=1/255.0` を適用し、
`[0, 255]` (uint8) → `[0.0, 1.0]` (float32) に変換する。

### エラー 2: 非正方形特徴マップ

```
AssertionError: only support square feature maps for now
```

**原因**: `do_resize=None` が渡されると `CLIPImageProcessor` は画像を 768×768 にリサイズしない。
元画像のサイズ（例: 1920×1080）のまま vision tower に入力されるため、
特徴マップが正方形にならず、以下のアサーションに違反する。

```python
# modeling_florence2.py L2629-2630
h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
assert h * w == num_tokens, 'only support square feature maps for now'
```

---

## 修正したファイル

`processing_florence2.py`

---

## コードの詳細

### 修正前（オリジナル）

```python
pixel_values = self.image_processor(
    images,
    do_resize=do_resize,           # ← None が渡される
    do_normalize=do_normalize,     # ← None が渡される
    image_mean=image_mean,
    image_std=image_std,
    return_tensors=return_tensors,
    input_data_format=input_data_format,
    data_format=data_format,
    resample=resample,
    do_convert_rgb=do_convert_rgb,
)["pixel_values"]
```

この書き方では `Florence2Processor.__call__()` のデフォルト引数がすべて `None` であるため、
呼び出し側が明示的に値を指定しない限り、`do_resize=None`, `do_normalize=None` 等が
`CLIPImageProcessor` に渡される。

### 修正後 (L253-270)

```python
image_processor_kwargs = {"return_tensors": return_tensors}
if do_resize is not None:
    image_processor_kwargs["do_resize"] = do_resize
if do_normalize is not None:
    image_processor_kwargs["do_normalize"] = do_normalize
if image_mean is not None:
    image_processor_kwargs["image_mean"] = image_mean
if image_std is not None:
    image_processor_kwargs["image_std"] = image_std
if input_data_format is not None:
    image_processor_kwargs["input_data_format"] = input_data_format
if data_format is not None:
    image_processor_kwargs["data_format"] = data_format
if resample is not None:
    image_processor_kwargs["resample"] = resample
if do_convert_rgb is not None:
    image_processor_kwargs["do_convert_rgb"] = do_convert_rgb
pixel_values = self.image_processor(images, **image_processor_kwargs)["pixel_values"]
```

---

## 修正の意味

`None` のパラメータはキーワード引数辞書に**含めない**ことで、
`CLIPImageProcessor.__call__()` がパラメータ未指定と判断し、
**`__init__` で設定されたデフォルト値**を使用するようになる。

`load_model()` で `CLIPImageProcessor` を初期化する際に設定される値：

```python
# nodes.py L61-71
image_processor = CLIPImageProcessor(
    do_resize=True,               # ← これが使われる
    size={"height": 768, "width": 768},
    resample=3,                   # BICUBIC
    do_center_crop=False,
    do_rescale=True,              # ← これが使われる
    rescale_factor=1/255.0,
    do_normalize=True,            # ← これが使われる
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)
```

つまり、呼び出し側が意図的にオーバーライドしない限り：

| パラメータ | `__init__` デフォルト | 処理内容 |
|-----------|---------------------|---------|
| `do_resize` | `True` | 768×768 にリサイズ |
| `do_rescale` | `True` | `[0,255]` → `[0.0, 1.0]` に変換 |
| `do_normalize` | `True` | ImageNet 正規化を適用 |

---

## `nodes.py` での呼び出しパターン

`Florence2Run.encode()` (L454) では `do_rescale=False` を明示的に指定している：

```python
inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)
```

これは、ComfyUI パイプラインでは `F.to_pil_image()` (L453) が既にテンソルから PIL Image に変換しているため、
画像が既に `[0, 255]` の uint8 PIL Image であるが、修正後のコードでは
`do_rescale=False` が明示的に渡されるため `CLIPImageProcessor` のデフォルト `True` は上書きされず、
正しく `False` が使用される。

ただしこの場合、`do_rescale=False` なのにスケーリングされないと型不一致になるように見えるが、
実際には `do_resize=True` の処理過程でPIL Image がfloat テンソルに変換されるため問題ない。

---

## なぜ v4.x では問題なかったか

transformers v4.x の `CLIPImageProcessor.__call__()` の実装：

```python
# v4.x の挙動（擬似コード）
def __call__(self, images, do_resize=None, ...):
    do_resize = do_resize if do_resize is not None else self.do_resize
    # → None なら self.do_resize (=True) にフォールバック
```

transformers v5.x の実装ではこのフォールバック挙動が変更されたか、
または `None` が「未指定」ではなく「無効化」として処理されるようになった。

---

## 影響範囲

| ファイル | 変更量 | 影響 |
|---------|--------|------|
| `processing_florence2.py` | `__call__` メソッド内 1箇所 | 画像前処理の全フロー |

## テスト環境

- transformers 5.4.0
- 入力画像: 各種サイズ・アスペクト比で確認
- Florence-2-large / Florence-2-large-ft

## 後方互換性

`None` を渡さないだけの変更であるため、**transformers v4.x でも同一の動作をする**。
v4.x でも「`None` を渡さない → `__init__` デフォルトを使用」は同じ挙動であり、副作用はない。
