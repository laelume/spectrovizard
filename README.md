<p align="center">
  <img src="https://raw.githubusercontent.com/laelume/spectrovizard/main/img/spectrovizard_icon.jpg" alt="spectrovizard" width="128">
</p>

# SpectroVizard

Lightweight spectrogram visualizer for Codium (and VSCode). 
Displays audio spectrogram on click in native file editor. 

## Supported formats

WAV, MP3, FLAC, OGG, M4A, AAC, PCM

## Usage

Open a supported audio file from the Explorer. The spectrogram opens as the editor for that file.  
Scroll horizontally to navigate long files. Hover over the spectrogram to read time and frequency at the cursor.

## Parameters

Adjust in the left panel. Changes apply on mouse release.

| Parameter | Effect |
|-----------|--------|
| NFFT | Frequency resolution. Higher = more frequency detail, less time detail. |
| Hop length | Time resolution. Lower = more time detail. |
| Window | Spectral window function. Hann is a safe default. |
| Scale | dB or linear magnitude. |
| dB range | Colour map dynamic range. Reduce to increase contrast. |
| Gain | Brightness offset in dB. |
| Colormap | viridis / magma / inferno / plasma / greys / hot / jet / turbo |
| Freq min | Set minimum frequency value. Blank = full Nyquist. |
| Freq max | Set maximum frequency value. Blank = full Nyquist. |
| Channel | Left, right, or mixed mono. |

## PCM files

Raw `.pcm` has no header. Provide format via sidecar or filename:

**Sidecar** — place `filename.pcm.json` alongside the file:
```json
{ "sr": 44100, "channels": 1, "dtype": "f32le" }
```
**Filename** — encode tokens in the name: signal_sr44100_ch1_f32le.pcm
If neither is present a dialog will prompt on open.

## License

Copyright 2026 laelume
MIT License