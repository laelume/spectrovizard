
# SpectroVizard: a lightweight spectrogram visualizer for codium/vscode

<p align="center">
  <img src="https://raw.githubusercontent.com/laelume/spectrovizard/main/img/spectrovizard_icon.jpg" alt="spectrovizard" width="200">
</p>

## Basic Usage

Navigate through audio files in Explorer, and spectrograms open on-click. Wizardry!  
You can scroll, zoom, and adjust FFT parameters to quickly inspect audio files.  

## Supported formats

WAV, MP3, FLAC, OGG, M4A, AAC, PCM

## FFT Parameters

| Parameter     | Effect |
|-----------    |--------|
| NFFT          | Time-Frequency resolution (higher -> more frequency detail, lower -> more time detail) |
| Hop length    | Time window. shorter -> more time detail |
| Window        | Spectral window function (Hann is a safe default) |
| Scale         | dB or linear magnitude |
| dB range      | Colour map dynamic range (Reduce to increase contrast) |
| Gain          | Brightness offset in dB |
| Colormap      | viridis / magma / inferno / plasma / greys / hot / jet / turbo |
| Freq min      | Set minimum frequency value (Blank = full Nyquist) |
| Freq max      | Set maximum frequency value (Blank = full Nyquist) |
| Channel       | Left, right, or mixed mono |

## Note about PCM files

Raw `.pcm` has no header. Provide format via sidecar or filename:

**Sidecar** — place `filename.pcm.json` alongside the file:
```json
{ "sr": 44100, "channels": 1, "dtype": "f32le" }
```
**Filename** expects tokens to be encoded in file name, i.e. signal_sr44100_ch1_f32le.pcm
If none are present, a dialog will prompt on open as a flag. 

## License

Copyright 2026 laelume  
MIT License  