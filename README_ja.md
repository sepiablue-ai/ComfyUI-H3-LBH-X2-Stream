# ComfyUI H3 LBH X2 Stream

[English](README.md) | 日本語

MiniMax H3を384×672で3step生成し、LBHで映像latentを544×960へ拡大、最後の1stepを高解像度のH3で仕上げるComfyUIノードです。X2 VAEと非同期NVENC保存で1088×1920を出力します。音声は途中latentを引き継ぎ、合計4step生成します。

022を土台にした、画質・音質の確認を経て選ばれた3+1構成をまとめています。全4stepを低解像度で完了してから拡大する4+0は、画質・音質NGのため含めていません。

## 生成の流れ

| 工程 | 映像 | 音声 |
|---|---|---|
| Step 1〜3 | H3、384×672 | 通常の生成を継続 |
| LBH | clean video latentを544×960へ拡大、元の残りsigmasへ再ノイズ化 | 空間拡大せず、低解像度samplerの途中audioを引き継ぐ |
| Step 4 | H3、544×960、高解像度の画像条件 | 最後のstepでも更新 |
| 保存 | INT8 X2 VAE、1088×1920、H.264 NVENC | AAC、32kHz、ステレオ |

`simple`の元4stepを`SplitSigmas(step=3)`で分割します。LBHは切替地点で1回だけ使用します。音声固定や2step化はしていません。CONSTの既存再ノイズ化を保持し、RALUのσ補正は使用しません。

## 含まれる高速化

| 技術 | 設定 |
|---|---|
| F12 | FC1のみ、層19・22〜29・47〜49をネイティブConvRot W4A4。全4stepに適用 |
| QKV再利用 | ネイティブSLAの同一attention呼び出し内だけ。最大336MiB、呼び出し終了で解放 |
| SLA | keep 5%、min_tokens 4096、extra_tokens 256、音声・条件行はexact |
| FFN | 8192行以下は1分割、それを超えると8分割。seq_threshold 4096 |
| 画像条件 | 高解像度画像でTEを1回だけ実行。低・高のVAE keyframeを先に作成 |
| 出力 | [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream)のINT8 X2 VAEと非同期NVENC保存 |

FC2・QKV・OutProjの重みは元のINT8を維持します。VSA Gateは使用しません。実験用の環境変数、forward回数制限、監査ログ、旧検証フォルダは不要です。

## 必要な環境

WindowsとNVIDIA CUDA向けです。開発検証はRTX 4070 12GB、ComfyUI 0.38.0、Python 3.13.13、PyTorch 2.14.0+cu130、comfy-kitchen 0.2.36で行いました。約48GiB RAMのPCでの結果で、より小さいRAMの下限は未確認です。他GPU・OS・モデルで同じ速度や品質を保証するものではありません。

ComfyUIにネイティブMiniMax H3、BlockSparseAttention、ConvRot W4A4、Dynamic VRAM、Sol-Attnが必要です。X2 Streamの`fp16_accumulation=true`には対応するPyTorchが必要です。ComfyUI本体やPyTorchはsetupで更新しません。

## Windowsセットアップ

このrepoを`ComfyUI/custom_nodes/ComfyUI-H3-LBH-X2-Stream/`へ配置するか、任意のフォルダからsetup.batを実行してください。ComfyUIを停止してから準備します。

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe"
```

専用Pythonを省略すると、そのComfyUIのPortable Python、`.venv`、`venv`を探します。候補が1つでなければ明示指定が必要です。グローバルPythonは拒否します。setupはコードとサンプル画像を配置し、モデルと必要APIを確認します。既存の異なるノードやサンプル画像を上書きしません。ComfyUIを起動しません。

X2 StreamやPython依存関係が未導入なら、専用Pythonを指定して次を実行します。

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --install-dependencies
```

未導入のX2 Streamはcommit `eb148a3d0f91a273668bbe027bfcbafb85f2c5e8`から取得します。既存のX2 Streamは更新しません。pipはその専用Pythonに対して`--no-deps`で実行し、PyTorchを自動で置換しません。モデルはダウンロードしません。

検査のみ、またはF12を先に準備する場合:

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --check --gpu-check
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --prepare-f12
```

`--check`は配置・pip・モデル変換を行いません。`--gpu-check`は小さいネイティブW4A4のCPU offload/reload試験です。動画は生成しません。

F12は初回のみ`models/h3_lbh_x2/f12/`へ12個のshardを保存します。合計約0.93GBです。F12 Loaderの`prepare_if_missing=true`でも準備できます。既存の50層FC1キャッシュを持っている場合は、Loaderの`cache_directory`、またはsetupの`--cache`で指定できます。そのうち12層だけを使い、Gateを読み込まず、元ファイルを変更しません。不完全・不一致のキャッシュは上書きしません。

## モデル

重み・変換済みshard・動画は同梱しません。次のモデルを自分で配置してください。共有フォルダはComfyUI標準の`extra_model_paths.yaml`で指定できます。LBHの共有パスには`latent_upscale_models`も必要です。[設定例](extra_model_paths.yaml.example)を参照してください。setupだけ追加設定を読む場合は`--extra-model-paths PATH`を使います。

| ファイル | ComfyUI内の場所 | 配布元 |
|---|---|---|
| minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors | models/diffusion_models | [MATLOWAI](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/tree/main/diffusion_models) |
| qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | models/text_encoders | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| minimax_h3_video_vae_int8_convrot.safetensors | models/vae | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| minimax_h3_audio_vae_fp32.safetensors | models/vae | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| MiniMax-H3-X2-Detail-v1.safetensors | models/vae | [X2 Detail VAE](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE) |
| minimax_h3_latent_upscaler_3d_fp16.safetensors | models/latent_upscale_models | [LBH](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) |

LBHの現行配布名`minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`も選択できます。実測で使用した重みのSHA-256は[benchmark.json](docs/benchmark.json)に記録しています。現行名のファイルと実測重みが同一かは未確認です。ダウンロード後、実際のファイル名をノードで選び直してください。

## ComfyUIで使う

1. 初回だけ[00_prepare_x2.json](examples/00_prepare_x2.json)を読み込み、元のX2 VAEからINT8版を準備します。X2 Streamで既に準備済みなら省略できます。
2. [10秒](examples/i2va_lbh_3p1_10s.json)、[14秒](examples/i2va_lbh_3p1_14s.json)、または[5秒](examples/i2va_lbh_3p1_05s.json)をComfyUIへドラッグします。API版は[examples/api](examples/api)です。
3. LoadImageで画像を選び、H3 LBH Shared Image Conditioningのプロンプトを変更します。初期サンプル画像はsetupがinputへコピーします。
4. 6つのモデル選択、F12のキャッシュ場所、NVENCのGPU indexを確認してQueueします。

画像は1枚入力のI2VAです。FL2VAモデルを使用していますが、終端画像を指定するワークフローではありません。PILの中央クロップとLanczosリサイズで、検証と同じ低・高解像度の条件画像を作ります。サンプルは元WebPをPILでデコードしたPNGで同梱し、ComfyUIのWebPデコーダーによる数画素の差を避けています。6通りの条件画像が検証時のPNGと完全一致することを確認済みです。音声の継続には、低samplerのoutput0 → Separate AV → Concat AVという接続が必要です。LBHには低samplerのoutput1、つまりclean predicted latentを渡します。

全例はseed43、res_multistep/simple4、video/audio sigma shift12/3、24fps、CQ18、FP16 accumulation有効です。フレーム数は5秒124、10秒243、14秒345で、実尺は約5.167・10.125・14.375秒です。保存先はComfyUIのoutput/H3_LBH_X2_Streamです。

## 実測と検証範囲

| 設定 | 022 | LBH3+1 | 短縮 |
|---|---:|---:|---:|
| 10秒 | 106.879秒 | 85.393秒 | 21.486秒 |
| 14秒 | 154.834秒 | 116.594秒 | 38.240秒 |

これは正式版を作る前の検証用ワークフローの実測です。初回モデルロード、条件付け、sampler、LBH、音声、X2と保存を含み、サーバー起動・アップロード・検査・一度だけの重み準備を除きます。各1回で、実行日の揺れを含みます。当時の5秒3+1は未生成でした。配布版の5秒は、その後の新規導入検証で生成を確認しています。

配布版では実験用監査を削除し、画像の準備とF12の準備機能を通常利用向けにまとめています。配布版でも新規ComfyUI 0.39.0にsetupし、5・10・14秒exampleの生成を確認しました。[検証記録](docs/validation.json)と[実測記録](docs/benchmark.json)で、元の実測と配布版の検査を区別しています。

6ノードの登録、4組のGUI/APIワークフローの入力検査・接続往復、setup.batの環境検査、12層のF12新規変換と元キャッシュとの完全一致、50層モデルの実読み込みを確認しました。再ノイズ化はFP32/BF16の4条件で元の計算と完全一致し、QKVとFFNの計算本体も維持しています。初期の部品検査では動画を再生成せず、その後、下記の新規導入検証で配布ワークフローを実行しました。

保守用の7件のテストは、repo内で次のように実行できます。標準ライブラリのみを使い、QKVの336MiB上限・入力変更時の再計算・解放と、setupの再実行・上書き拒否を確認します。

```bat
"C:\ComfyUI\.venv\Scripts\python.exe" -B -m unittest discover -s tests -v
```

既存masterのComfyUI 0.38.0にもsetup.batで導入し、5秒exampleを57.506秒で生成、全映像・音声デコード検査を通過しました。依存更新・022変更なし。5・10・14秒はGUIのワークフロー一覧へ登録済みです。[master検証記録](docs/master_validation.json)。

## 新規導入での生成確認

ComfyUI 0.39.0を新規cloneし、専用venvでsetup.batを実行しました。X2準備用と5・10・14秒exampleをそのまま実行し、すべて成功しました。動画3本は1088×1920・24fps、AAC 32kHzステレオで全デコード検査を通過しています。

| example | 実行時間 | モデル再利用 |
|---|---:|---|
| 5秒 | 58.802秒 | 新規プロセスの初回動画 |
| 10秒 | 86.289秒 | 新規プロセスの初回動画 |
| 14秒 | 112.983秒 | 10秒の後、同一プロセスで別画像・別プロンプト |

初回だけのF12準備とX2変換、サーバー起動、出力検査は動画生成時間に含めません。既存PyTorch等は標準venvから参照し、0.39.0が新たに要求する5パッケージだけを検証用venvへ導入しました。標準venvは更新していません。これは依存まで完全に空の環境から導入した試験ではありません。GUIのデータとAPI版の一致を確認してAPI経由で実行しており、ブラウザーでのドラッグ操作・Queueボタンは未検査です。今回出力の主観的な画質・音質は未評価です。[詳細記録](docs/fresh_install_validation.json)。

## ライセンス

本パッケージのコードはGPL-3.0です。同梱LBHコードのMITライセンス表示はvendor/LBH_LICENSE.txtに保持しています。モデル重みと派生重みには各配布元の条件が適用されます。詳細と出典は[NOTICE.md](NOTICE.md)を参照してください。
