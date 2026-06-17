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

        panel.webview.options = {
            enableScripts      : true,
            localResourceRoots : [vscode.Uri.joinPath(this.ctx.extensionUri, 'media')]
        };
        
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

        // --- Persistent params ---
        // Read previously saved params from globalState and inject into payload.
        // The webview reads window.SPECTRO_SAVED_PARAMS on init and restores controls.
        const savedParams = this.ctx.globalState.get<string>('spectrovizard.params', '{}');

        // Listen for param-save messages posted from the webview
        panel.webview.onDidReceiveMessage(msg => {
            if (msg.type === 'saveParams') {
                this.ctx.globalState.update('spectrovizard.params', JSON.stringify(msg.params));
            }
        }, undefined, this.ctx.subscriptions);

        panel.webview.html = buildHtml(
            panel.webview,
            this.ctx,
            name,
            ext,
            audioB64,
            lutPayload,
            pcmMeta,
            savedParams
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
    pcmMeta  : PcmMeta | null,
    savedParams  : string
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
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --panel-w    : 230px;
    --bg         : #1e1e1e;
    --fg         : #cccccc;
    --accent     : #569cd6;
    --accent-dim : #2d4f6e;
    --border     : #3c3c3c;
    --ctrl-bg    : #2d2d2d;
    --ctrl-hover : #3a3a3a;
    --btn-active : #569cd6;
    --btn-active-fg: #ffffff;
    --font       : 11px/1.5 'Consolas', 'Courier New', monospace;
    --font-sm    : 9px/1.4  'Consolas', 'Courier New', monospace;
  }

  body {
    display        : flex;
    flex-direction : row;
    width          : 100vw;
    height         : 100vh;
    overflow       : hidden;
    background     : var(--bg);
    color          : var(--fg);
    font           : var(--font);
  }

  /* --- Panel --- */
  #panel {
    width          : var(--panel-w);
    min-width      : var(--panel-w);
    height         : 100%;
    display        : flex;
    flex-direction : column;
    border-right   : 1px solid var(--border);
    padding        : 10px 8px;
    overflow-y     : auto;
    gap            : 4px;
  }

  #panel h2 {
    font-size      : 11px;
    font-weight    : 700;
    letter-spacing : 0.08em;
    color          : var(--accent);
    text-transform : uppercase;
    margin-bottom  : 2px;
  }

  .section-label {
    font            : var(--font-sm);
    text-transform  : uppercase;
    letter-spacing  : 0.1em;
    color           : #888;
    margin-top      : 8px;
    margin-bottom   : 2px;
  }

  .ctrl-row {
    display        : flex;
    flex-direction : column;
    gap            : 2px;
  }

  .ctrl-row label {
    font           : var(--font-sm);
    color          : #aaa;
  }

  /* --- Inline button group --- */
  .btn-group {
    display        : flex;
    flex-wrap      : nowrap;
    gap            : 2px;
  }

  .btn-group button {
    flex           : 1 1 0;
    min-width      : 0;
    padding        : 2px 3px;
    background     : var(--ctrl-bg);
    color          : var(--fg);
    border         : 1px solid var(--border);
    border-radius  : 2px;
    font           : var(--font-sm);
    cursor         : pointer;
    white-space    : nowrap;
    overflow       : hidden;
    text-overflow  : ellipsis;
    text-align     : center;
  }

  .btn-group button:hover {
    background     : var(--ctrl-hover);
  }

  .btn-group button.active {
    background     : var(--btn-active);
    color          : var(--btn-active-fg);
    border-color   : var(--btn-active);
  }

  /* Overflow dropdown at end of btn-group */
  .btn-group select.overflow-select {
    width          : 20px;
    min-width      : 20px;
    padding        : 2px 0;
    background     : var(--ctrl-bg);
    color          : var(--fg);
    border         : 1px solid var(--border);
    border-radius  : 2px;
    font           : var(--font-sm);
    cursor         : pointer;
    text-align     : center;
  }

  /* --- Sliders --- */
  .ctrl-row input[type=range] {
    width          : 100%;
    accent-color   : var(--accent);
  }

  /* --- Number inputs --- */
  .ctrl-row input[type=number] {
    width          : 100%;
    background     : var(--ctrl-bg);
    color          : var(--fg);
    border         : 1px solid var(--border);
    border-radius  : 2px;
    padding        : 2px 4px;
    font           : var(--font);
  }

  .val-display {
    font           : var(--font-sm);
    color          : #888;
    text-align     : right;
  }

  /* --- Zoom controls --- */
  #zoom-row {
    display        : flex;
    gap            : 4px;
    margin-top     : 4px;
    align-items    : center;
  }

  #zoom-row label {
    font           : var(--font-sm);
    color          : #aaa;
    white-space    : nowrap;
  }

  #zoom-row input[type=range] {
    flex           : 1;
    accent-color   : var(--accent);
  }

  #zoom-row .val-display {
    min-width      : 32px;
  }

  /* --- Info box --- */
  #info-box {
    font           : var(--font-sm);
    color          : #888;
    margin-top     : auto;
    padding-top    : 8px;
    border-top     : 1px solid var(--border);
    word-break     : break-all;
  }

  /* --- Viewport --- */
  #viewport {
    flex           : 1;
    display        : flex;
    flex-direction : column;
    overflow       : hidden;
    position       : relative;
  }

  #freq-axis {
    position       : absolute;
    right          : 0;
    top            : 0;
    width          : 44px;
    height         : calc(100% - 24px);
    pointer-events : none;
  }

  #spectro-scroll {
    flex           : 1;
    overflow-x     : auto;
    overflow-y     : hidden;
    position       : relative;
    cursor         : crosshair;
  }

  #spectro-canvas {
    display        : block;
    image-rendering: pixelated;
  }

  #time-axis { height: 24px; overflow: hidden; }
  #time-canvas { display: block; height: 24px; }

  #loading-overlay {
    position       : absolute;
    inset          : 0;
    background     : rgba(30,30,30,0.85);
    display        : flex;
    align-items    : center;
    justify-content: center;
    font-size      : 12px;
    color          : var(--accent);
    z-index        : 10;
  }

  #progress-bar-wrap {
    width          : 60%;
    height         : 4px;
    background     : var(--border);
    border-radius  : 2px;
    margin-top     : 8px;
  }

  #progress-bar {
    height         : 4px;
    width          : 0%;
    background     : var(--accent);
    border-radius  : 2px;
    transition     : width 0.1s;
  }

  .overlay-inner { text-align: center; }

  #error-msg {
    display        : none;
    position       : absolute;
    inset          : 0;
    background     : rgba(30,30,30,0.9);
    color          : #f48771;
    align-items    : center;
    justify-content: center;
    font-size      : 12px;
    z-index        : 11;
    padding        : 20px;
    text-align     : center;
  }
</style>
</head>
<body>

<div id="panel">
  <h2>SpectroVizard</h2>
  <div id="filename-label" style="font:var(--font-sm);color:#888;word-break:break-all"></div>

  <div class="section-label">T R A N S F O R M</div>

  <div class="ctrl-row">
    <label>N F F T</label>
    <div class="btn-group" id="bg-nfft">
      <button data-val="128">128</button>
      <button data-val="256">256</button>
      <button data-val="512" class="active">512</button>
      <button data-val="1024">1024</button>
      <button data-val="2048">2048</button>
      <button data-val="4096">4096</button>
    </div>
  </div>

  <div class="ctrl-row">
    <label>H O P</label>
    <div class="btn-group" id="bg-hop">
      <button data-val="32">32</button>
      <button data-val="64">64</button>
      <button data-val="128" class="active">128</button>
      <button data-val="256">256</button>
      <button data-val="512">512</button>
    </div>
  </div>

  <div class="ctrl-row">
    <label>W I N D O W</label>
    <div class="btn-group" id="bg-window">
      <button data-val="hann" class="active">hann</button>
      <button data-val="hamming">hamm</button>
      <button data-val="blackman">blk</button>
      <button data-val="bartlett">bart</button>
      <button data-val="rect">rect</button>
    </div>
  </div>

  <div class="section-label">S C A L I N G</div>

  <div class="ctrl-row">
    <label>Y   A X I S</label>
    <div class="btn-group" id="bg-scale">
      <button data-val="linear" class="active">lin</button>
      <button data-val="log">log</button>
      <button data-val="mel">mel</button>
    </div>
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
    <div class="btn-group" id="bg-cmap">
      <button data-val="viridis" class="active">vir</button>
      <button data-val="magma">mag</button>
      <button data-val="inferno">inf</button>
      <button data-val="plasma">pla</button>
      <button data-val="greys">gry</button>
      <button data-val="hot">hot</button>
      <button data-val="jet">jet</button>
      <button data-val="turbo">tur</button>
    </div>
  </div>

<div class="ctrl-row">
    <label>F R E Q   M A X   (H z)</label>
    <div style="display:flex;gap:2px;align-items:center">
      <input type="text" inputmode="numeric" id="ctrl-fmax" placeholder="Nyquist"
                  style="flex:1;background:var(--ctrl-bg);color:var(--fg);border:1px solid var(--border);border-radius:2px;padding:2px 4px;font:var(--font);-moz-appearance:textfield;">
            <span id="fmax-hint" style="display:none"></span>

      <button id="btn-fmax-reset" title="Reset to Nyquist"
              style="background:var(--ctrl-bg);border:1px solid var(--border);border-radius:2px;color:var(--fg);cursor:pointer;padding:2px 4px;font-size:11px;line-height:1">↺</button>
    </div>
  </div>

  <div class="ctrl-row">
    <label>F R E Q   M I N   (H z)</label>
    <div style="display:flex;gap:2px;align-items:center">
      <input type="text" inputmode="numeric" id="ctrl-fmin" placeholder="0"
             style="flex:1;background:var(--ctrl-bg);color:var(--fg);border:1px solid var(--border);border-radius:2px;padding:2px 4px;font:var(--font);-moz-appearance:textfield;">
      <button id="btn-fmin-reset" title="Reset to 0 Hz"
              style="background:var(--ctrl-bg);border:1px solid var(--border);border-radius:2px;color:var(--fg);cursor:pointer;padding:2px 4px;font-size:11px;line-height:1">↺</button>
    </div>
  </div>

  <div class="section-label">Z O O M</div>

  <div class="ctrl-row">
    <label>T I M E &nbsp;<span class="val-display" id="zoom-time-val">1.0x</span></label>
    <div id="zoom-row">
      <input type="range" id="ctrl-zoom-time" min="-4" max="4" step="0.1" value="0">
    </div>
  </div>

  <div class="ctrl-row">
    <label>F R E Q &nbsp;<span class="val-display" id="zoom-freq-val">1.0x</span></label>
    <div id="zoom-row-freq">
      <input type="range" id="ctrl-zoom-freq" min="-4" max="4" step="0.1" value="0">
    </div>
  </div>

  <div id="info-box">
    <div id="info-sr">—</div>
    <div id="info-dur">—</div>
    <div id="info-ch">—</div>
    <div id="info-cursor">—</div>
  </div>
</div>

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

<script>
  window.onerror = function(msg, src, line, col, err) {
    document.getElementById('error-msg').style.display = 'flex';
    document.getElementById('error-msg').textContent   = msg + ' ' + src + ':' + line;
  };

  window.SPECTRO_PAYLOAD = {
    filename : ${JSON.stringify(filename)},
    ext      : ${JSON.stringify(ext)},
    audioB64 : ${JSON.stringify(audioB64)},
    luts     : ${lutJson},
    pcmMeta  : ${pcmJson},
    savedParams : ${JSON.stringify(savedParams)}
  };
</script>
<script src="${scriptUri}"></script>
</body>
</html>`;
}