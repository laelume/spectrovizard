// Copyright 2026 laelume. 
//
// Custom editor provider: registers a WebviewPanel for each audio file opened
// via the customEditors contribution in package.json.
// Audio bytes are transferred once to the webview; all DSP runs client-side
// in media/spectro.js using the Web Audio API OfflineAudioContext.

import * as vscode from 'vscode';
import * as path   from 'path';
import * as fs     from 'fs';
import { COLORMAP_NAMES, getLUT, ColormapName } from './colormaps';

// === === === === === === === ===
// S P E C T R O   P R O V I D E R
// ==== ==== ==== ==== ==== ====

export class SpectroEditorProvider implements vscode.CustomReadonlyEditorProvider {
    /**
     * Register this provider with the extension context and return a disposable.
     */
    static register(context: vscode.ExtensionContext): vscode.Disposable {
        return vscode.window.registerCustomEditorProvider(
            'spectrovizard.spectroView',
            new SpectroEditorProvider(context),
            {
                webviewOptions: { retainContextWhenHidden: true },
                supportsMultipleEditorsPerDocument: false
            }
        );
    }

    constructor(private readonly ctx: vscode.ExtensionContext) {}

    /**
     * Called by VSCode when a file matching the selector is opened.
     * Builds the webview HTML, serialises LUTs, transfers audio bytes.
     */
    async openCustomDocument(
        uri: vscode.Uri,
        _openContext: vscode.CustomDocumentOpenContext,
        _token: vscode.CancellationToken
    ): Promise<vscode.CustomDocument> {
        // CustomReadonlyEditorProvider requires a document object;
        // no state needed beyond the URI for a read-only viewer.
        return { uri, dispose: () => {} };
    }

    /**
     * Called by VSCode to populate the webview panel for an opened document.
     */
    async resolveCustomEditor(
        document: vscode.CustomDocument,
        panel: vscode.WebviewPanel,
        _token: vscode.CancellationToken
    ): Promise<void> {
        const uri  = document.uri;
        const ext  = path.extname(uri.fsPath).toLowerCase().replace('.', '');
        const name = path.basename(uri.fsPath);

        panel.webview.options = { enableScripts: true };

        // --- LUT serialisation ---
        // Each LUT is transferred as a base64 string to avoid JSON number array bloat.
        const lutPayload: Record<string, string> = {};
        for (const cname of COLORMAP_NAMES) {
            const buf = getLUT(cname as ColormapName);
            lutPayload[cname] = Buffer.from(buf).toString('base64');
        }

        // --- Audio bytes ---
        // Read the full file into a Buffer; transfer as base64.
        // For streaming, the webview handles chunking internally via
        // Web Audio API decodeAudioData on overlapping byte windows.
        const audioBytes  = fs.readFileSync(uri.fsPath);
        const audioB64    = audioBytes.toString('base64');

        // --- PCM metadata resolution ---
        // For bare .pcm files: attempt sidecar → filename → prompt dialog.
        let pcmMeta: PcmMeta | null = null;
        if (ext === 'pcm') {
            pcmMeta = await resolvePcmMeta(uri);
            if (!pcmMeta) {
                // User cancelled dialog; show error and abort render.
                vscode.window.showErrorMessage(
                    `SpectroVizard: PCM format unknown for ${name}. Open cancelled.`
                );
                return;
            }
        }

        panel.webview.html = buildHtml(
            panel.webview,
            this.ctx,
            name,
            ext,
            audioB64,
            lutPayload,
            pcmMeta
        );
    }
}

// === === === === === === === ===
// P C M   M E T A D A T A
// ==== ==== ==== ==== ==== ====

interface PcmMeta {
    sampleRate : number;
    channels   : number;
    dtype      : 'f32le' | 'f32be' | 'i16le' | 'i16be' | 'i32le' | 'i32be' | 'u8';
}

/**
 * Resolve PCM format metadata via sidecar JSON, then filename convention,
 * then a prompt dialog. Returns null if the user cancels.
 *
 * Sidecar convention:   <file>.pcm.json  →  { "sr": 44100, "channels": 1, "dtype": "f32le" }
 * Filename convention:  name_sr44100_ch1_f32le.pcm
 */
async function resolvePcmMeta(uri: vscode.Uri): Promise<PcmMeta | null> {
    const fpath = uri.fsPath;

    // 1. Sidecar
    const sidecarPath = fpath + '.json';
    if (fs.existsSync(sidecarPath)) {
        try {
            const raw  = JSON.parse(fs.readFileSync(sidecarPath, 'utf8'));
            const meta = parseSidecarPcm(raw);
            if (meta) return meta;
        } catch { /* fall through */ }
    }

    // 2. Filename convention
    const fromName = parsePcmFilename(path.basename(fpath));
    if (fromName) return fromName;

    // 3. Dialog
    return promptPcmMeta();
}

/**
 * Parse sidecar JSON object into PcmMeta; returns null on validation failure.
 */
function parseSidecarPcm(raw: Record<string, unknown>): PcmMeta | null {
    const DTYPES = ['f32le','f32be','i16le','i16be','i32le','i32be','u8'];
    const sr   = Number(raw['sr']       ?? raw['sample_rate'] ?? 0);
    const ch   = Number(raw['channels'] ?? raw['ch']          ?? 1);
    const dt   = String(raw['dtype']    ?? '');
    if (!sr || !DTYPES.includes(dt)) return null;
    return { sampleRate: sr, channels: ch, dtype: dt as PcmMeta['dtype'] };
}

/**
 * Parse PCM metadata from filename tokens _srNNNN_chN_dtype.
 * Example: signal_sr22050_ch1_f32le.pcm
 */
function parsePcmFilename(basename: string): PcmMeta | null {
    const srMatch  = basename.match(/_sr(\d+)/i);
    const chMatch  = basename.match(/_ch(\d+)/i);
    const dtMatch  = basename.match(/_(f32le|f32be|i16le|i16be|i32le|i32be|u8)/i);
    if (!srMatch) return null;
    return {
        sampleRate : parseInt(srMatch[1], 10),
        channels   : chMatch  ? parseInt(chMatch[1],  10) : 1,
        dtype      : dtMatch  ? dtMatch[1].toLowerCase() as PcmMeta['dtype'] : 'f32le'
    };
}

/**
 * Show input-box prompts to collect PCM format from the user.
 * Returns null if the user cancels any step.
 */
async function promptPcmMeta(): Promise<PcmMeta | null> {
    const srStr = await vscode.window.showInputBox({
        prompt      : 'PCM sample rate (Hz)',
        value       : '44100',
        validateInput: v => /^\d+$/.test(v) ? null : 'Integer required'
    });
    if (!srStr) return null;

    const chStr = await vscode.window.showInputBox({
        prompt      : 'PCM channel count',
        value       : '1',
        validateInput: v => /^\d+$/.test(v) ? null : 'Integer required'
    });
    if (!chStr) return null;

    const dtypes = ['f32le','f32be','i16le','i16be','i32le','i32be','u8'];
    const dtype  = await vscode.window.showQuickPick(dtypes, {
        placeHolder: 'PCM sample dtype'
    });
    if (!dtype) return null;

    return {
        sampleRate : parseInt(srStr, 10),
        channels   : parseInt(chStr, 10),
        dtype      : dtype as PcmMeta['dtype']
    };
}

// === === === === === === === ===
// H T M L   B U I L D E R
// ==== ==== ==== ==== ==== ====

/**
 * Construct the full webview HTML string.
 * Inlines all LUTs and audio data as JSON/base64; loads spectro.js from media/.
 */
function buildHtml(
    webview  : vscode.Webview,
    ctx      : vscode.ExtensionContext,
    filename : string,
    ext      : string,
    audioB64 : string,
    luts     : Record<string, string>,
    pcmMeta  : PcmMeta | null
): string {
    // Resolve the media/spectro.js URI via the webview resource system.
    const scriptUri = webview.asWebviewUri(
        vscode.Uri.joinPath(ctx.extensionUri, 'media', 'spectro.js')
    );

    const pcmJson = pcmMeta ? JSON.stringify(pcmMeta) : 'null';
    const lutJson = JSON.stringify(luts);

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none';
               script-src 'unsafe-inline' ${webview.cspSource};
               style-src  'unsafe-inline';">
<title>SpectroVizard — ${filename}</title>
<style>
  /* === === === === === === === === */
  /* L A Y O U T                    */
  /* ==== ==== ==== ==== ==== ====  */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --panel-w    : 220px;
    --bg         : #1e1e1e;
    --fg         : #cccccc;
    --accent     : #569cd6;
    --border     : #3c3c3c;
    --ctrl-bg    : #2d2d2d;
    --ctrl-hover : #3a3a3a;
    --font       : 11px/1.5 'Consolas', 'Courier New', monospace;
  }

  body {
    display         : flex;
    flex-direction  : row;
    width           : 100vw;
    height          : 100vh;
    overflow        : hidden;
    background      : var(--bg);
    color           : var(--fg);
    font            : var(--font);
  }

  /* --- Left control panel --- */
  #panel {
    width           : var(--panel-w);
    min-width       : var(--panel-w);
    height          : 100%;
    display         : flex;
    flex-direction  : column;
    border-right    : 1px solid var(--border);
    padding         : 10px 8px;
    overflow-y      : auto;
    gap             : 6px;
  }

  #panel h2 {
    font-size    : 11px;
    font-weight  : 700;
    letter-spacing: 0.08em;
    color        : var(--accent);
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  .section-label {
    font-size    : 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color        : #888;
    margin-top   : 8px;
    margin-bottom: 2px;
  }

  .ctrl-row {
    display         : flex;
    flex-direction  : column;
    gap             : 2px;
  }

  .ctrl-row label {
    font-size : 10px;
    color     : var(--fg);
  }

  .ctrl-row input[type=range] {
    width      : 100%;
    accent-color: var(--accent);
  }

  .ctrl-row select,
  .ctrl-row input[type=number] {
    width        : 100%;
    background   : var(--ctrl-bg);
    color        : var(--fg);
    border       : 1px solid var(--border);
    border-radius: 2px;
    padding      : 2px 4px;
    font         : var(--font);
  }

  .ctrl-row select:hover,
  .ctrl-row input[type=number]:hover {
    background: var(--ctrl-hover);
  }

  .val-display {
    font-size : 9px;
    color     : #888;
    text-align: right;
  }

  #info-box {
    font-size   : 9px;
    color       : #888;
    margin-top  : auto;
    padding-top : 8px;
    border-top  : 1px solid var(--border);
    word-break  : break-all;
  }

  /* --- Right spectrogram viewport --- */
  #viewport {
    flex        : 1;
    display     : flex;
    flex-direction: column;
    overflow    : hidden;
    position    : relative;
  }

  #freq-axis {
    position   : absolute;
    right      : 0;
    top        : 0;
    width      : 44px;
    height     : calc(100% - 24px);
    pointer-events: none;
  }

  #spectro-scroll {
    flex        : 1;
    overflow-x  : auto;
    overflow-y  : hidden;
    position    : relative;
  }

  #spectro-canvas {
    display     : block;
    height      : calc(100vh - 24px);
    image-rendering: pixelated;
  }

  #time-axis {
    height      : 24px;
    overflow    : hidden;
    position    : relative;
  }

  #time-canvas {
    display     : block;
    height      : 24px;
  }

  #loading-overlay {
    position    : absolute;
    inset       : 0;
    background  : rgba(30,30,30,0.85);
    display     : flex;
    align-items : center;
    justify-content: center;
    font-size   : 12px;
    color       : var(--accent);
    z-index     : 10;
  }

  #progress-bar-wrap {
    width       : 60%;
    height      : 4px;
    background  : var(--border);
    border-radius: 2px;
    margin-top  : 8px;
  }

  #progress-bar {
    height      : 4px;
    width       : 0%;
    background  : var(--accent);
    border-radius: 2px;
    transition  : width 0.1s;
  }

  .overlay-inner { text-align: center; }

  #error-msg {
    display     : none;
    position    : absolute;
    inset       : 0;
    background  : rgba(30,30,30,0.9);
    color       : #f48771;
    align-items : center;
    justify-content: center;
    font-size   : 12px;
    z-index     : 11;
    padding     : 20px;
    text-align  : center;
  }
</style>
</head>
<body>

<!-- === === === === === === === === -->
<!-- C O N T R O L   P A N E L      -->
<!-- ==== ==== ==== ==== ==== ====   -->
<div id="panel">
  <h2>SpectroVizard</h2>
  <div id="filename-label" style="font-size:9px;color:#888;word-break:break-all"></div>

  <div class="section-label">T R A N S F O R M</div>

  <div class="ctrl-row">
    <label>N F F T</label>
    <select id="ctrl-nfft">
      <option value="128">128</option>
      <option value="256">256</option>
      <option value="512" selected>512</option>
      <option value="1024">1024</option>
      <option value="2048">2048</option>
      <option value="4096">4096</option>
    </select>
  </div>

  <div class="ctrl-row">
    <label>H O P   L E N G T H</label>
    <select id="ctrl-hop">
      <option value="32">32</option>
      <option value="64">64</option>
      <option value="128" selected>128</option>
      <option value="256">256</option>
      <option value="512">512</option>
    </select>
  </div>

  <div class="ctrl-row">
    <label>W I N D O W</label>
    <select id="ctrl-window">
      <option value="hann" selected>Hann</option>
      <option value="hamming">Hamming</option>
      <option value="blackman">Blackman</option>
      <option value="bartlett">Bartlett</option>
      <option value="rect">Rectangular</option>
    </select>
  </div>

  <div class="section-label">S C A L I N G</div>

  <div class="ctrl-row">
    <label>S C A L E</label>
    <select id="ctrl-scale">
      <option value="db" selected>dB (power)</option>
      <option value="linear">Linear (magnitude)</option>
    </select>
  </div>

  <div class="ctrl-row">
    <label>D B   R A N G E &nbsp;<span class="val-display" id="dbrange-val">80</span></label>
    <input type="range" id="ctrl-dbrange" min="20" max="160" step="5" value="80">
  </div>

  <div class="ctrl-row">
    <label>G A I N &nbsp;<span class="val-display" id="gain-val">0</span> dB</label>
    <input type="range" id="ctrl-gain" min="-40" max="40" step="1" value="0">
  </div>

  <div class="section-label">D I S P L A Y</div>

  <div class="ctrl-row">
    <label>C O L O R M A P</label>
    <select id="ctrl-cmap">
      <option value="viridis" selected>viridis</option>
      <option value="magma">magma</option>
      <option value="inferno">inferno</option>
      <option value="plasma">plasma</option>
      <option value="greys">greys</option>
      <option value="hot">hot</option>
      <option value="jet">jet</option>
      <option value="turbo">turbo</option>
    </select>
  </div>

  <div class="ctrl-row">
    <label>F R E Q   M A X   (H z)</label>
    <input type="number" id="ctrl-fmax" min="1" step="100" value="">
    <span class="val-display" id="fmax-hint">default: Nyquist</span>
  </div>

  <div class="ctrl-row">
    <label>C H A N N E L</label>
    <select id="ctrl-channel">
      <option value="0">Ch 1 (L)</option>
      <option value="1">Ch 2 (R)</option>
      <option value="mix">Mix</option>
    </select>
  </div>

  <div id="info-box">
    <div id="info-sr">—</div>
    <div id="info-dur">—</div>
    <div id="info-ch">—</div>
    <div id="info-cursor">—</div>
  </div>
</div>

<!-- === === === === === === === === -->
<!-- S P E C T R O   V I E W P O R T -->
<!-- ==== ==== ==== ==== ==== ====   -->
<div id="viewport">
  <div id="loading-overlay">
    <div class="overlay-inner">
      <div>Decoding audio…</div>
      <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
    </div>
  </div>
  <div id="error-msg"></div>

  <div id="spectro-scroll">
    <canvas id="spectro-canvas"></canvas>
  </div>
  <div id="time-axis">
    <canvas id="time-canvas"></canvas>
  </div>
  <canvas id="freq-axis"></canvas>
</div>

<!-- Inline payload — LUTs and audio transferred once on load -->
<script>
  window.SPECTRO_PAYLOAD = {
    filename : ${JSON.stringify(filename)},
    ext      : ${JSON.stringify(ext)},
    audioB64 : ${JSON.stringify(audioB64)},
    luts     : ${lutJson},
    pcmMeta  : ${pcmJson}
  };
</script>
<script src="${scriptUri}"></script>
</body>
</html>`;
}