# heytux model benchmark - run1

One record per model; see the per-slot JSON under this dir for full detail.


## chat

| model | load s | no-warmup s | pp64 t/s | tg64 t/s | TTFT s | decode t/s | RSS MB | PSS MB | quant |
|-|-|-|-|-|-|-|-|-|-|
| Gemma-3n-E2B-it-Q4_K_M | 19.7 | 8.1 | 17.7 | 6.8 | 1.92 | 6.8 | 4309 | 4308 | Q4_K_M |
| Gemma-4-E2B-it-Q4_K_M | 35.3 | 15.4 | 15.8 | 6.5 | 1.39 | 6.4 | 4335 | 4333 | Q4_K_M |
| Granite-4.0-1B-Q4_K_M | 11.0 | 7.3 | 27.1 | 10.4 | 0.57 | 10.3 | 2279 | 2277 | Q4_K_M |
| Granite-4.0-H-1B-Q4_K_M | 10.0 | 5.8 | 25.4 | 8.8 | 0.78 | 8.9 | 2714 | 2712 | Q4_K_M |
| LFM2.5-1.2B-Instruct-Q4_K_M | 8.1 | 4.7 | 39.2 | 14.9 | 0.96 | 14.8 | 1564 | 1562 | Q4_K_M |
| Llama-3.2-1B-Instruct-Q4_K_M | 11.1 | 7.8 | 41.4 | 13.3 | 0.40 | 12.8 | 1796 | 1794 | Q4_K_M |
| Qwen3-0.6B-Q4_K_M | 6.6 | 4.7 | 81.3 | 25.2 | 0.26 | 23.9 | 1157 | 1155 | Q4_K_M |
| Qwen3-1.7B-Q4_K_M | 13.5 | 8.5 | 28.9 | 9.7 | 0.71 | 9.6 | 2552 | 2550 | Q4_K_M |
| Qwen3.5-0.8B-Q4_0 | 8.5 | 6.1 | 74.4 | 15.9 | 0.36 | 15.6 | 1779 | 1777 | Q4_0 |
| Qwen3.5-0.8B-Q4_K_M | 9.7 | 7.7 | 60.0 | 15.2 | 0.43 | 14.9 | 1806 | 1804 | Q4_K_M |
| Qwen3.5-2B-Q4_0 | 13.8 | 9.1 | 31.5 | 7.7 | 0.80 | 7.7 | 3111 | 3109 | Q4_0 |
| Qwen3.5-2B-Q4_K_M | 16.5 | 11.3 | 24.0 | 7.1 | 1.01 | 7.1 | 3328 | 3326 | Q4_K_M |
| Qwen3.5-4B-Q4_K_M | 29.9 | 31.0 | 9.1 | 3.2 | 2.64 | 3.2 | 4687 | 4685 | Q4_K_M |

load s = time-to-health (warmup on); decode t/s = server timings predicted_per_second.

## router

| model | tool acc | none prec | none recall | latency ms | RSS MB |
|-|-|-|-|-|-|
| FunctionGemma-270M-it-Q8_0 | 55% | 55% | 100% | 1817 | 602 |
| LFM2.5-230M-Q8_0 | 93% | 94% | 100% | 3145 | 580 |
| LFM2.5-350M-Q8_0 | 91% | 100% | 87% | 2920 | 847 |

Needle3 baseline: acc 59%, none P/R 100%/29%, 246 ms

### router: failed / skipped

| model | status | detail |
|-|-|-|
| Needle3 | failed | RuntimeError("llama-server exited rc=1, log tail:\narams_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)\n0. |
| xLAM-2-1B-fc-r-Q4_K_M | failed | <HTTPError 500: 'Internal Server Error'> |

## embedding

| model | embed ms | router acc | none prec | none recall | thr | RSS MB |
|-|-|-|-|-|-|-|
| EmbeddingGemma-300M-Q8_0 | 41 | 62% | 100% | 32% | 0.50 | 711 |
| Qwen3-Embedding-0.6B-Q8_0 | 78 | 45% | - | 0% | 0.50 | 1535 |
| all-MiniLM-L6-v2-Q8_0 | 37 | 95% | 100% | 90% | 0.50 | 89 |
| bge-m3-Q4_K_M | 112 | 45% | - | 0% | 0.50 | 788 |

### embedding: failed / skipped

| model | status | detail |
|-|-|-|
| embeddinggemma-300m-ONNX-q4 | skipped | embedding runtime onnx needs onnxruntime, no adapter (id=embeddinggemma-300m-ONNX-q4) |
| multilingual-e5-small-Q8_0 | failed | RuntimeError("llama-server exited rc=1, log tail:\ning n_batch = n_ubatch = 512 to avoid assertion failure\n0.00.039.156 |

## asr

| model | RTF | WER clean | WER quiet(-20dB) | WER noise(10.0 dB SNR) | WER cs | RSS MB |
|-|-|-|-|-|-|-|
| moonshine-base-en-quantized | 0.092 | 6% | 6% | 26% | - | 313 |
| moonshine-tiny-en-quantized | 0.061 | 7% | 8% | 21% | - | 197 |
| parakeet-tdt-0.6b-v3-int8 | 0.190 | 39% | 38% | 38% | 48% | 1021 |
| parakeet-tdt-110m-en-int8 | 0.066 | 39% | 37% | 38% | - | 278 |
| whisper-base.en | 0.490 | 7% | 6% | 6% | - | 262 |
| whisper-small-int8 | 1.583 | 8% | 7% | 8% | 108% | 534 |
| whisper-tiny-int8 | 0.267 | 13% | 13% | 9% | 118% | 198 |

**moonshine-base-en-quantized** ref/hyp samples:
  - [clean] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [clean] ref='turn off wifi' hyp='Turn off wi-fi' wer=0.667
  - [clean] ref='enable bluetooth' hyp='Enable Bluetooth.' wer=0.0
  - [quiet] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [quiet] ref='turn off wifi' hyp='Turn off wi-fi' wer=0.667
  - [quiet] ref='enable bluetooth' hyp='Enable Bluetooth' wer=0.0
  - [noise] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [noise] ref='turn off wifi' hyp='Turn off wi-fi.' wer=0.667
  - [noise] ref='enable bluetooth' hyp='Enable bluetooth' wer=0.0

**moonshine-tiny-en-quantized** ref/hyp samples:
  - [clean] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [clean] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [clean] ref='enable bluetooth' hyp='Enable bluetooth.' wer=0.0
  - [quiet] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [quiet] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [quiet] ref='enable bluetooth' hyp='Enable bluetooth.' wer=0.0
  - [noise] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [noise] ref='turn off wifi' hyp='Turn off wifi.' wer=0.0
  - [noise] ref='enable bluetooth' hyp='Enable Bluetooth.' wer=0.0

**parakeet-tdt-0.6b-v3-int8** ref/hyp samples:
  - [clean] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [clean] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [clean] ref='enable bluetooth' hyp='Enable Bluetooth.' wer=0.0
  - [quiet] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [quiet] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [quiet] ref='enable bluetooth' hyp='Enable Bluetooth.' wer=0.0
  - [noise] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [noise] ref='turn off wifi' hyp='Turn off Wi Fi.' wer=0.667
  - [noise] ref='enable bluetooth' hyp='Enable Bluetooth.' wer=0.0

**parakeet-tdt-110m-en-int8** ref/hyp samples:
  - [clean] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [clean] ref='turn off wifi' hyp='Turn off Wi Fi.' wer=0.667
  - [clean] ref='enable bluetooth' hyp='Enable Bluetooth' wer=0.0
  - [quiet] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [quiet] ref='turn off wifi' hyp='Turn off Wi Fi' wer=0.667
  - [quiet] ref='enable bluetooth' hyp='enable Bluetooth.' wer=0.0
  - [noise] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [noise] ref='turn off wifi' hyp='Turn off Wi Fi.' wer=0.667
  - [noise] ref='enable bluetooth' hyp='Enable Bluetooth' wer=0.0

**whisper-base.en** ref/hyp samples:
  - [clean] ref='turn on the flashlight' hyp='turn on the flashlight.' wer=0.0
  - [clean] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [clean] ref='enable bluetooth' hyp='Enable Bluetooth.' wer=0.0
  - [quiet] ref='turn on the flashlight' hyp='turn on the flashlight.' wer=0.0
  - [quiet] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [quiet] ref='enable bluetooth' hyp='enable Bluetooth.' wer=0.0
  - [noise] ref='turn on the flashlight' hyp='turn on the flashlight.' wer=0.0
  - [noise] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [noise] ref='enable bluetooth' hyp='Enable Bluetooth.' wer=0.0

**whisper-small-int8** ref/hyp samples:
  - [clean] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [clean] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [clean] ref='enable bluetooth' hyp='enable Bluetooth' wer=0.0
  - [quiet] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [quiet] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [quiet] ref='enable bluetooth' hyp='enable bluechip' wer=0.5
  - [noise] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [noise] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [noise] ref='enable bluetooth' hyp='Enable Bluetooth.' wer=0.0

**whisper-tiny-int8** ref/hyp samples:
  - [clean] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [clean] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [clean] ref='enable bluetooth' hyp='Enable Bluetooth' wer=0.0
  - [quiet] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [quiet] ref='turn off wifi' hyp='Turn off Wi-Fi' wer=0.667
  - [quiet] ref='enable bluetooth' hyp='Enable Bluetooth' wer=0.0
  - [noise] ref='turn on the flashlight' hyp='Turn on the flashlight.' wer=0.0
  - [noise] ref='turn off wifi' hyp='Turn off Wi-Fi.' wer=0.667
  - [noise] ref='enable bluetooth' hyp='enable Bluetooth' wer=0.0

**moonshine-base-en-quantized** real captures (no ground truth):
  - 20261004-104617-775.wav: 'Turn on the flashlight'
  - 20261004-104626-343.wav: 'Turn off the flashlight'
  - 20261004-104636-295.wav: 'Can you tell me what is the capital city of Bernal?'
  - 20261004-105446-093.wav: ''
  - 20261004-105454-238.wav: 'Turn off the flashlight'
  - 20261004-110654-831.wav: 'Linux ah for good'
  - 20261004-110705-791.wav: ''
  - 20261004-114822-756.wav: 'What time it is?'

**moonshine-tiny-en-quantized** real captures (no ground truth):
  - 20261004-104617-775.wav: "They're not on the flashlight."
  - 20261004-104626-343.wav: 'Them an object flashlight'
  - 20261004-104636-295.wav: 'Can you tell me what is the capital city of Brna?'
  - 20261004-105446-093.wav: 'Return on the flashlight.'
  - 20261004-105454-238.wav: 'Turn off the flashlight.'
  - 20261004-110654-831.wav: "It's a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little bit of a little"
  - 20261004-110705-791.wav: 'So, give up. Yeah. Directly passively. Come say goodbye, thank you.'
  - 20261004-114822-756.wav: "What time it is? I think it's gas and acid is a bit. I think it's gas and acid is a bit."

**parakeet-tdt-0.6b-v3-int8** real captures (no ground truth):
  - 20261004-104617-775.wav: 'Turn on the flashlight.'
  - 20261004-104626-343.wav: ''
  - 20261004-104636-295.wav: 'Can you tell me what is the capital city of Brno?'
  - 20261004-105446-093.wav: 'And turn on the flashlight.'
  - 20261004-105454-238.wav: 'Turn on the flashlight.'
  - 20261004-110654-831.wav: 'Linux.'
  - 20261004-110705-791.wav: 'Ya quitas a te conseguir en keyboard.'
  - 20261004-114822-756.wav: "What time it is. I think it's some minutes."

**parakeet-tdt-110m-en-int8** real captures (no ground truth):
  - 20261004-104617-775.wav: 'Turn on the flashlight.'
  - 20261004-104626-343.wav: 'Then obviously flashlight.'
  - 20261004-104636-295.wav: 'Can you tell me what is the capital city of Bernot?'
  - 20261004-105446-093.wav: 'Turn on the flashlight.'
  - 20261004-105454-238.wav: 'Turn on the flashlight.'
  - 20261004-110654-831.wav: 'Linuxita.'
  - 20261004-110705-791.wav: "That's the consecutive."
  - 20261004-114822-756.wav: 'What time it is'

**whisper-base.en** real captures (no ground truth):
  - 20261004-104617-775.wav: 'Turn on the flashlight.'
  - 20261004-104626-343.wav: 'Then over the flashlight'
  - 20261004-104636-295.wav: 'Can you tell me what is the capital city of Bernal?'
  - 20261004-105446-093.wav: 'and turn on the flashlight.'
  - 20261004-105454-238.wav: 'Turn off the flashlight'
  - 20261004-110654-831.wav: 'Linux.'
  - 20261004-110705-791.wav: 'Yeah, the circuit has to be executed at any time.'
  - 20261004-114822-756.wav: 'What time it is.'

**whisper-small-int8** real captures (no ground truth):
  - 20261004-104617-775.wav: 'turn on the flashlight'
  - 20261004-104626-343.wav: 'Then on the flash drive.'
  - 20261004-104636-295.wav: 'Can you tell me what is the capital city of Brno?'
  - 20261004-105446-093.wav: 'and turn on the flashlight'
  - 20261004-105454-238.wav: 'Turn off the flashlight'
  - 20261004-110654-831.wav: 'Linux.'
  - 20261004-110705-791.wav: "Yeah, I'll get you some of the keys you got in keyboard."
  - 20261004-114822-756.wav: 'What time it is?'

**whisper-tiny-int8** real captures (no ground truth):
  - 20261004-104617-775.wav: 'turn around the flash light'
  - 20261004-104626-343.wav: 'then an awful fresh bread'
  - 20261004-104636-295.wav: 'Can you tell me what is the capital city of Bernou?'
  - 20261004-105446-093.wav: "I'm turning on the flashlight."
  - 20261004-105454-238.wav: "I'm gonna turn it on on the flashlight."
  - 20261004-110654-831.wav: '"Lenux" "A" "Sachita" "A" "Lenux" "A" "Sachita" "A" "Lenux'
  - 20261004-110705-791.wav: '- Yeah, there are a few tasks to consider, thank you.'
  - 20261004-114822-756.wav: 'What time it is?'

### asr: failed / skipped

| model | status | detail |
|-|-|-|
| streaming-zipformer-en-20M | failed | RuntimeError('Got invalid dimensions for input: x for the following indices\n index: 1 Got: 140 Expected: 39\n Please fi |
| streaming-zipformer-kroko | failed | worker exit rc=255 |

## tts

| model | RTF | TTFA s | RSS MB | wavs |
|-|-|-|-|-|
| kokoro-int8-en-v0_19 | 2.628 | 5.829 | 343 | /home/miro/heytux/bench/results/run1/tts/kokoro-int8-en-v0_19 |
| kokoro-int8-multi-lang-v1_1 | 2.430 | 4.690 | 420 | /home/miro/heytux/bench/results/run1/tts/kokoro-int8-multi-lang-v1_1 |
| vits-piper-cs_CZ-jirka-medium | 0.170 | 0.514 | 246 | /home/miro/heytux/bench/results/run1/tts/vits-piper-cs_CZ-jirka-medium |
| vits-piper-en_US-amy-medium | 0.169 | 0.445 | 229 | /home/miro/heytux/bench/results/run1/tts/vits-piper-en_US-amy-medium |
| vits-piper-en_US-lessac-low-int8 | 0.273 | 0.542 | 149 | /home/miro/heytux/bench/results/run1/tts/vits-piper-en_US-lessac-low-int8 |
| vits-piper-en_US-lessac-medium-int8 | 0.407 | 0.755 | 155 | /home/miro/heytux/bench/results/run1/tts/vits-piper-en_US-lessac-medium-int8 |

TTFA is full offline synth time (no streaming).

### tts: failed / skipped

| model | status | detail |
|-|-|-|
| kitten-mini-en-v0_8 | failed | worker exit rc=255 |
| kitten-nano-en-v0_8-int8 | failed | worker exit rc=255 |
| matcha-icefall-en_US-ljspeech | failed | worker exit rc=255 |
| supertonic-tts-int8 | failed | worker exit rc=255 |

## vad

| model | xRT | silero IoU | is silero | RSS MB |
|-|-|-|-|-|
| silero-vad-onnx-community | 111 | 1.000 | yes | 76 |

### vad: failed / skipped

| model | status | detail |
|-|-|-|
| silero_vad | failed | FileNotFoundError("/home/miro/sherpa/silero_vad.onnx: none of ('*int8.onnx', '*.onnx')") |
| ten-vad | failed | worker exit rc=255 |

## vision

| model | load s | TTFT s | tok/s | RSS MB |
|-|-|-|-|-|
| LFM2.5-VL-450M-Q4_K_M | 4.1 | 1.76 | 40.6 | 684 |
| SmolVLM2-500M-Q8_0 | 6.5 | 12.77 | 27.1 | 1050 |
  - LFM2.5-VL-450M-Q4_K_M: 'A square divided into four equal quadrants with a gradient of colors.'
  - SmolVLM2-500M-Q8_0: 'A colorful grid of squares arranged in a 4x4 grid, with each square having a gradient of colors from light to dark.'

### vision: failed / skipped

| model | status | detail |
|-|-|-|
| Gemma-4-E2B-mmproj | skipped | vision: Gemma-4-E2B-mmproj is an mmproj projector; benchmark with its base chat gguf |
| Qwen3.5-0.8B-mmproj | skipped | vision: Qwen3.5-0.8B-mmproj is an mmproj projector; benchmark with its base chat gguf |
| Qwen3.5-2B-mmproj | skipped | vision: Qwen3.5-2B-mmproj is an mmproj projector; benchmark with its base chat gguf |

## wake

| model | status |
|-|-|

### wake: failed / skipped

| model | status | detail |
|-|-|-|
| openWakeWord-base | skipped | wake slot skipped for now (brief) |

## zram / swap tuning

| config | model | load s | tg tok/s | TTFT s | MemAvail MB | swap MB | pswpout | zram ratio | OOM |
|-|-|-|-|-|-|-|-|-|-|
| zram-zstd-2.7g-s100-pc3 | Qwen3.5-4B-Q4_K_M | 41.6 | 3.35 | 2.37 | 811 | 828 | 642607 | 8.9 | no |
| zram-zstd-2.7g-s100-pc3 | Gemma-4-E2B-it-Q4_K_M | 24.4 | 6.48 | 1.14 | 2664 | 862 | 13349 | 13.0 | no |
| zram-zstd-2.7g-s60-pc3 | Qwen3.5-4B-Q4_K_M | 38.8 | 3.38 | 2.32 | 1028 | 1045 | 694191 | 13.8 | no |
| zram-zstd-2.7g-s60-pc3 | Gemma-4-E2B-it-Q4_K_M | 23.5 | 6.62 | 1.12 | 2856 | 1075 | 23836 | 20.2 | no |
| zram-zstd-4g-s100-pc3 | Qwen3.5-4B-Q4_K_M | 42.7 | 3.36 | 2.40 | 960 | 976 | 734989 | 8.7 | no |
| zram-zstd-4g-s100-pc3 | Gemma-4-E2B-it-Q4_K_M | 24.0 | 6.50 | 1.14 | 2700 | 894 | 988 | 12.4 | no |
| zram-zstd-4g-s180-pc0 | Qwen3.5-4B-Q4_K_M | 45.0 | 3.32 | 2.35 | 1022 | 1034 | 854034 | 8.5 | no |
| zram-zstd-4g-s180-pc0 | Gemma-4-E2B-it-Q4_K_M | 23.6 | 6.60 | 1.14 | 2775 | 967 | 1297 | 12.0 | no |
| zram-zstd-4g-s60-pc0 | Qwen3.5-4B-Q4_K_M | 45.7 | 3.32 | 2.40 | 1005 | 1019 | 864749 | 8.1 | no |
| zram-zstd-4g-s60-pc0 | Gemma-4-E2B-it-Q4_K_M | 23.1 | 6.69 | 1.14 | 2750 | 947 | 512 | 11.3 | no |

Skipped configs: zram-lz4-4g-s100-pc3 (RuntimeError("algo lz4 not supported (have ['zstd'])"))
