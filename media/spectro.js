// Copyright 2026 laelume. All rights reserved.
// BSD 3-Clause License.
//
// All DSP runs client-side. No Python subprocess. No Node.js audio libraries.
//
// Pipeline:
//   1. Base64 audio bytes → ArrayBuffer
//   2. Web Audio API decodeAudioData → AudioBuffer (WAV/MP3/FLAC/OGG/M4A)
//   3. PCM path: manual ArrayBuffer → Float32Array interleave decode
//   4. Per-chunk STFT via Cooley-Tukey radix-2 FFT (pure JS)
//   5. Power → dB / linear scaling → LUT colormap → canvas ImageData
//   6. Chunk-based streaming: render visible window only; recompute on scroll
//   7. Independent time (px/frame) and freq (fmin/fmax) zoom axes
//   8. Parameters persisted via postMessage → VSCode globalState
//
// References:
//   [1] Cooley & Tukey (1965). Mathematics of Computation 19(90):297-301.
//       DOI: https://doi.org/10.1090/S0025-5718-1965-0178586-1
//   [2] Harris (1978). Proc. IEEE 66(1):51-83.
//       DOI: https://doi.org/10.1109/PROC.1978.10837

'use strict';

// acquireVsCodeApi() must be called exactly once per webview lifetime.
// Hoisted to module scope to prevent re-invocation on every saveParams() call.
const _vscode = acquireVsCodeApi();

// === === === === === === === ===
// C O N S T A N T S
// ==== ==== ==== ==== ==== ====

// Frames per streaming chunk. Larger = fewer draw calls; smaller = faster scroll response.
const CHUNK_FRAMES = 512;

// Frequency axis canvas width (px).
const FREQ_AXIS_W  = 44;

// Time axis canvas height (px).
const TIME_AXIS_H  = 24;

// File size warning threshold (uncompressed byte estimate).
const WARN_SIZE_MB = 50;

// Zoom slider maps [-4, 4] → scale factor via 2^x.
// At 0: 1x. At 4: 16x. At -4: ~0.06x.
const ZOOM_BASE    = 2;

// === === === === === === === ===
// S T A T E
// ==== ==== ==== ==== ==== ====

/**
 * Global viewer state. All rendering reads exclusively from this object.
 * Mutated by control handlers, param restore, and render pipeline.
 */
const STATE = {
    audioBuffer  : null,   // AudioBuffer from decodeAudioData
    samples      : null,   // Float32Array — active channel samples
    sampleRate   : 0,

    // STFT params
    nfft         : 512,
    hop          : 128,
    windowType   : 'hann',
    scale        : 'db',
    dbRange      : 80,
    gainDb       : 0,
    cmap         : 'viridis',
    channel      : 0,
    fmax         : null,   // null = Nyquist
    fmin         : 0,

    // Zoom: stored as slider values [-4,4]; applied as 2^val scale factors
    zoomTime     : 0,      // >0 = expand (more px per frame); <0 = compress
    zoomFreq     : 0,      // >0 = narrow visible freq band (zoom in)

    // Decoded LUT tables: name → Uint8Array(768)
    luts         : {},

    // Computed STFT cache
    totalFrames  : 0,
    magCache     : null,   // Float32Array[totalFrames * (nfft/2+1)]
    tileCache    : new Map(),

    dirty        : true,
    rendering    : false,
    lastScrollX  : 0,
    rafHandle    : null,

    globalPeak : 1.0,
    peakReady : false,
};

// === === === === === === === ===
// P A R A M   P E R S I S T E N C E
// ==== ==== ==== ==== ==== ====

/**
 * Keys saved and restored across file navigations via VSCode globalState.
 */
const PERSIST_KEYS = [
    'nfft', 'hop', 'windowType', 'scale', 'dbRange', 'gainDb',
    'cmap', 'channel', 'fmax', 'fmin', 'zoomTime', 'zoomFreq'
];

// === === === === === === === ===
// E N T R Y   P O I N T
// ==== ==== ==== ==== ==== ====

(function init() {
    /**
     * Main entry: decode payload, wire controls, restore params, start render.
     * Controls are wired before params are restored so that syncDomToState()
     * correctly sets .active states without being overwritten by wiring defaults.
     */
    const payload = window.SPECTRO_PAYLOAD;
    if (!payload) { showError('SPECTRO_PAYLOAD missing.'); return; }

    // Decode LUTs from base64
    for (const [name, b64] of Object.entries(payload.luts)) {
        STATE.luts[name] = base64ToUint8(b64);
    }

    for (const [name, lut] of Object.entries(STATE.luts)) {
        console.log(name,
            'idx0',   lut[0], lut[1], lut[2],
            'idx255', lut[765], lut[766], lut[767]);
    }

    document.getElementById('filename-label').textContent = payload.filename;

    // Wire controls first, then restore — order is critical
    wireControls();
    wireScroll();
    wireCursorInfo();

    // Restore saved params after wiring so DOM reflects saved state correctly
    restoreParams(payload.savedParams);

    decodeAudio(payload)
        .then(buf => {
            STATE.audioBuffer = buf;
            STATE.sampleRate  = buf.sampleRate;
            updateInfoBox(buf);
            updateChannelSelector(buf.numberOfChannels);
            applyChannelSelection(buf);
            computeGlobalPeak();
            hideLoading();
            scheduleRender(true);
        })

        .catch(err => {
            console.error('decode pipeline error:', err);
            showError(err.stack || err.message);
        });

})();



/**
 * Restore STATE and DOM controls from a saved params value.
 * Handles both single and double JSON-encoded strings defensively,
 * since globalState round-trips through JSON.stringify in the host.
 *
 * @param {string|object} savedJson
 */
function restoreParams(savedJson) {
    let saved = {};
    try {
        let val = savedJson;
        // Unwrap double-encoded string if necessary
        if (typeof val === 'string') val = JSON.parse(val);
        if (typeof val === 'string') val = JSON.parse(val);
        if (val && typeof val === 'object') saved = val;
    } catch { return; }

    for (const key of PERSIST_KEYS) {
        if (saved[key] !== undefined) STATE[key] = saved[key];
    }

    syncDomToState();
}

/**
 * Push current STATE param values into DOM controls.
 * Button groups: set .active on matching data-val button.
 * Sliders / inputs: set .value directly.
 */
function syncDomToState() {
    setBtnGroupActive('bg-nfft',   String(STATE.nfft));
    setBtnGroupActive('bg-hop',    String(STATE.hop));
    setBtnGroupActive('bg-window', STATE.windowType);
    setBtnGroupActive('bg-scale',  STATE.scale);
    setBtnGroupActive('bg-cmap',   STATE.cmap);

    setVal('ctrl-dbrange',   STATE.dbRange);
    setVal('ctrl-gain',      STATE.gainDb);
    setVal('ctrl-fmax',      STATE.fmax ?? '');
    setVal('ctrl-fmin',      STATE.fmin ?? 0);
    setVal('ctrl-zoom-time', STATE.zoomTime);
    setVal('ctrl-zoom-freq', STATE.zoomFreq);

    setText('dbrange-val',   STATE.dbRange);
    setText('gain-val',      STATE.gainDb);
    setText('zoom-time-val', zoomLabel(STATE.zoomTime));
    setText('zoom-freq-val', zoomLabel(STATE.zoomFreq));
}

/**
 * Collect current STATE param values and post to the extension host for persistence.
 * Uses the module-level _vscode handle; never calls acquireVsCodeApi() more than once.
 */
function saveParams() {
    const params = {};
    for (const key of PERSIST_KEYS) params[key] = STATE[key];
    _vscode.postMessage({ type: 'saveParams', params });
}

// === === === === === === === ===
// A U D I O   D E C O D E
// ==== ==== ==== ==== ==== ====

/**
 * Decode base64 audio payload into a Web Audio API AudioBuffer.
 * Container formats (WAV/MP3/FLAC/OGG/M4A) decoded via decodeAudioData.
 * Bare PCM decoded manually via decodePcm().
 *
 * @param {object} payload - window.SPECTRO_PAYLOAD
 * @returns {Promise<AudioBuffer>}
 */
async function decodeAudio(payload) {
    const arrayBuf = base64ToArrayBuffer(payload.audioB64);
    setProgress(20);

    if (payload.ext === 'pcm') {
        return decodePcm(arrayBuf, payload.pcmMeta);
    }

    const offCtx = new OfflineAudioContext(1, 1, 44100);
    setProgress(40);

    return new Promise((resolve, reject) => {
        offCtx.decodeAudioData(
            arrayBuf,
            buf => { setProgress(90); resolve(buf); },
            err  => reject(new Error(String(err)))
        );
    });
}

/**
 * Manually decode a raw PCM ArrayBuffer into an AudioBuffer.
 * Supported dtypes: f32le, f32be, i16le, i16be, i32le, i32be, u8.
 *
 * @param {ArrayBuffer} arrayBuf
 * @param {{ sampleRate: number, channels: number, dtype: string }} meta
 * @returns {AudioBuffer}
 */
function decodePcm(arrayBuf, meta) {
    const { sampleRate, channels, dtype } = meta;
    const view        = new DataView(arrayBuf);
    const bytesPerSmp = dtypeBytes(dtype);
    const totalSmp    = Math.floor(arrayBuf.byteLength / bytesPerSmp);
    const frameCount  = Math.floor(totalSmp / channels);

    const offCtx   = new OfflineAudioContext(channels, frameCount, sampleRate);
    const audioBuf = offCtx.createBuffer(channels, frameCount, sampleRate);

    for (let ch = 0; ch < channels; ch++) {
        const chData = audioBuf.getChannelData(ch);
        for (let i = 0; i < frameCount; i++) {
            const byteOff = (i * channels + ch) * bytesPerSmp;
            chData[i]     = readPcmSample(view, byteOff, dtype);
        }
    }

    setProgress(80);
    return audioBuf;
}

/**
 * Return bytes-per-sample for a PCM dtype string.
 */
function dtypeBytes(dtype) {
    if (dtype === 'u8')                               return 1;
    if (dtype === 'i16le' || dtype === 'i16be')       return 2;
    if (dtype === 'i32le' || dtype === 'i32be')       return 4;
    if (dtype === 'f32le' || dtype === 'f32be')       return 4;
    throw new Error(`Unknown dtype: ${dtype}`);
}

/**
 * Read one normalised float sample [-1, 1] from a DataView at byteOffset.
 */
function readPcmSample(view, offset, dtype) {
    switch (dtype) {
        case 'f32le': return view.getFloat32(offset, true);
        case 'f32be': return view.getFloat32(offset, false);
        case 'i16le': return view.getInt16(offset, true)   / 32768.0;
        case 'i16be': return view.getInt16(offset, false)  / 32768.0;
        case 'i32le': return view.getInt32(offset, true)   / 2147483648.0;
        case 'i32be': return view.getInt32(offset, false)  / 2147483648.0;
        case 'u8':    return (view.getUint8(offset) - 128) / 128.0;
        default:      return 0;
    }
}

// === === === === === === === ===
// C H A N N E L   S E L E C T I O N
// ==== ==== ==== ==== ==== ====

/**
 * Extract the active channel (or downmixed mono) from AudioBuffer into STATE.samples.
 *
 * @param {AudioBuffer} buf
 */
function applyChannelSelection(buf) {
    const nCh  = buf.numberOfChannels;
    const nSmp = buf.length;
    const ch   = STATE.channel;

    if (ch === 'mix' && nCh > 1) {
        const out = new Float32Array(nSmp);
        for (let c = 0; c < nCh; c++) {
            const src = buf.getChannelData(c);
            for (let i = 0; i < nSmp; i++) out[i] += src[i];
        }
        const inv = 1.0 / nCh;
        for (let i = 0; i < nSmp; i++) out[i] *= inv;
        STATE.samples = out;
    } else {
        const idx     = (typeof ch === 'number') ? Math.min(ch, nCh - 1) : 0;
        STATE.samples = buf.getChannelData(idx).slice();
    }
}

/**
 * Compute global peak from raw sample amplitude, not FFT output.
 * Independent of nfft, hop, and window — computed once after channel selection.
 */
function computeGlobalPeak() {
    const samples = STATE.samples;
    let peak = 1e-12;
    for (let i = 0; i < samples.length; i++) {
        const a = Math.abs(samples[i]);
        if (a > peak) peak = a;
    }
    // Store as power equivalent for consistency with magCache units:
    // magCache holds re²+im² (power); peak amplitude² gives the power reference.
    STATE.globalPeak = peak * peak * (STATE.nfft / 2) * (STATE.nfft / 2);
    STATE.peakReady  = true;
}

// === === === === === === === ===
// S T F T   C O R E
// ==== ==== ==== ==== ==== ====

/**
 * Cooley-Tukey radix-2 DIT in-place FFT.
 * Input: interleaved Float32Array [re0,im0,re1,im1,...], length = 2*N, N = power of 2.
 *
 * Reference: [1] DOI:10.1090/S0025-5718-1965-0178586-1
 *
 * @param {Float32Array} data
 */
function fftInPlace(data) {
    const n = data.length >> 1;

    // Bit-reversal permutation
    let j = 0;
    for (let i = 1; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            let t = data[2*i];   data[2*i]   = data[2*j];   data[2*j]   = t;
            t     = data[2*i+1]; data[2*i+1] = data[2*j+1]; data[2*j+1] = t;
        }
    }

    // Butterfly stages
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let cRe = 1.0, cIm = 0.0;
            for (let k = 0; k < (len >> 1); k++) {
                const uRe = data[2*(i+k)],     uIm = data[2*(i+k)+1];
                const h   = i + k + (len >> 1);
                const vRe = data[2*h]   * cRe - data[2*h+1] * cIm;
                const vIm = data[2*h]   * cIm + data[2*h+1] * cRe;
                data[2*(i+k)]   = uRe + vRe;  data[2*(i+k)+1] = uIm + vIm;
                data[2*h]       = uRe - vRe;  data[2*h+1]     = uIm - vIm;
                const nRe = cRe * wRe - cIm * wIm;
                cIm       = cRe * wIm + cIm * wRe;
                cRe       = nRe;
            }
        }
    }
}

/**
 * Build a spectral window function of length n.
 * Reference: [2] DOI:10.1109/PROC.1978.10837
 *
 * @param  {string}     type - 'hann'|'hamming'|'blackman'|'bartlett'|'rect'
 * @param  {number}     n
 * @returns {Float32Array}
 */
function makeWindow(type, n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const t = i / (n - 1);
        switch (type) {
            case 'hann':
                w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * t)); break;
            case 'hamming':
                w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * t); break;
            case 'blackman':
                w[i] = 0.42 - 0.5  * Math.cos(2 * Math.PI * t)
                            + 0.08 * Math.cos(4 * Math.PI * t); break;
            case 'bartlett':
                w[i] = 1 - Math.abs((2*i - (n-1)) / (n-1)); break;
            default: // rect
                w[i] = 1.0; break;
        }
    }
    return w;
}

/**
 * Compute STFT power spectrum for frames [startFrame, endFrame).
 * Writes into STATE.magCache at correct frame offsets.
 *
 * @param {number} startFrame
 * @param {number} endFrame
 */
function computeChunk(startFrame, endFrame) {
    const { samples, nfft, hop, windowType } = STATE;
    const win   = makeWindow(windowType, nfft);
    const nBins = (nfft >> 1) + 1;
    const buf   = new Float32Array(nfft * 2);

    for (let f = startFrame; f < endFrame; f++) {
        const offset  = f * hop;
        buf.fill(0);
        const copyLen = Math.min(nfft, samples.length - offset);
        if (copyLen <= 0) break;

        for (let k = 0; k < copyLen; k++) {
            buf[2*k]   = samples[offset + k] * win[k];
            buf[2*k+1] = 0;
        }

        fftInPlace(buf);

        // Store raw power (re²+im²). No scaling applied here.
        // Intensity mapping occurs in renderTile(); y-axis spacing in binForRow().
        const cacheOff = f * nBins;
        for (let b = 0; b < nBins; b++) {
            const re = buf[2*b], im = buf[2*b+1];
            STATE.magCache[cacheOff + b] = re*re + im*im;
        }
    }
}

// === === === === === === === ===
// Z O O M
// ==== ==== ==== ==== ==== ====

/**
 * Convert a zoom slider value [-4, 4] to a linear scale factor via 2^val.
 *
 * @param  {number} sliderVal
 * @returns {number}
 */
function zoomToScale(sliderVal) {
    return Math.pow(ZOOM_BASE, sliderVal);
}

/**
 * Format zoom scale factor for display label.
 *
 * @param  {number} sliderVal
 * @returns {string}
 */
function zoomLabel(sliderVal) {
    return zoomToScale(sliderVal).toFixed(2) + 'x';
}

/**
 * Compute pixel width per STFT frame given the current time zoom level.
 * At zoomTime=0: 1px/frame. At zoomTime=4: 16px/frame.
 * Minimum 1px/frame to prevent sub-pixel tile artefacts.
 *
 * @returns {number}
 */
function pxPerFrame() {
    return Math.max(1, zoomToScale(STATE.zoomTime));
}

/**
 * Compute effective [fminHz, fmaxHz] after applying manual bounds and freq zoom.
 * Freq zoom narrows the visible band symmetrically around the centre frequency.
 * Manual fmin/fmax are applied first; zoom then contracts the band inward.
 *
 * @returns {{ fminHz: number, fmaxHz: number }}
 */
function effectiveFreqBand() {
    const nyquist = STATE.sampleRate / 2;
    const rawMax  = (STATE.fmax && STATE.fmax < nyquist) ? STATE.fmax : nyquist;
    const rawMin  = (STATE.fmin && STATE.fmin > 0)       ? STATE.fmin : 0;
    const centre  = (rawMax + rawMin) / 2;
    const halfBand = (rawMax - rawMin) / 2;
    const scale   = zoomToScale(STATE.zoomFreq);
    // Zoom > 1 narrows the band (zoom in); zoom < 1 widens (clamped to raw bounds)
    const newHalf = halfBand / scale;
    return {
        fminHz : Math.max(rawMin, centre - newHalf),
        fmaxHz : Math.min(rawMax, centre + newHalf)
    };
}

/**
 * Map a canvas pixel row to a frequency bin index given the current y-axis mode.
 * row=0 → fmaxHz (top); row=viewH-1 → fminHz (bottom).
 *
 * Modes:
 *   linear — uniform Hz spacing (equal bin steps)
 *   log    — logarithmic Hz spacing (equal ratio steps, Audacity default)
 *   mel    — mel-scale spacing (perceptually uniform)
 *
 * Mel conversion: m = 2595 * log10(1 + f/700)
 * Reference: O'Shaughnessy (1987) "Speech Communication", p.150.
 *
 * @param {number} row    - pixel row (0 = top)
 * @param {number} viewH  - canvas height in pixels
 * @param {number} nBins  - number of FFT bins (nfft/2 + 1)
 * @param {number} fminHz
 * @param {number} fmaxHz
 * @returns {number} bin index clamped to [0, nBins-1]
 */
function binForRow(row, viewH, nBins, fminHz, fmaxHz) {
    const nyquist = STATE.sampleRate / 2;
    // Fraction from top: 0 at row=0, 1 at row=viewH-1
    const frac = (row + 0.5) / viewH;

    let freqHz;
    if (STATE.scale === 'linear') {
        // Uniform Hz interpolation
        freqHz = fmaxHz - frac * (fmaxHz - fminHz);

    } else if (STATE.scale === 'log') {
        // Logarithmic interpolation — avoid log(0) by clamping fminHz
        const lo = Math.max(fminHz, 1);
        const hi = fmaxHz;
        freqHz = hi * Math.pow(lo / hi, frac);

    } else {
        // Mel scale
        const melMin = 2595 * Math.log10(1 + fminHz / 700);
        const melMax = 2595 * Math.log10(1 + fmaxHz / 700);
        const mel    = melMax - frac * (melMax - melMin);
        freqHz       = 700 * (Math.pow(10, mel / 2595) - 1);
    }

    // Convert Hz to bin index
    const bin = Math.round(freqHz / nyquist * (nBins - 1));
    return Math.max(0, Math.min(nBins - 1, bin));
}


// === === === === === === === ===
// R E N D E R   P I P E L I N E
// ==== ==== ==== ==== ==== ====

/**
 * Invalidate caches and schedule a render frame.
 *
 * @param {boolean} recompute - if true, discard magCache and recompute FFT
 */
function scheduleRender(recompute) {
    if (!STATE.samples) return;

    if (recompute) {
        const nBins       = (STATE.nfft >> 1) + 1;
        STATE.totalFrames = Math.ceil((STATE.samples.length - STATE.nfft) / STATE.hop) + 1;
        STATE.magCache    = new Float32Array(STATE.totalFrames * nBins);
        STATE.tileCache   = new Map();
        STATE.dirty       = true;
        STATE.peakReady   = false;
        STATE.globalPeak  = 1.0;
    } else {
        STATE.tileCache = new Map();
        STATE.dirty     = true;
    }

    if (STATE.rafHandle) cancelAnimationFrame(STATE.rafHandle);
    STATE.rafHandle = requestAnimationFrame(renderFrame);
}

/**
 * Single animation frame: compute and paint the visible canvas region.
 */
function renderFrame() {
    STATE.rafHandle = null;
    if (!STATE.samples || STATE.rendering) return;
    STATE.rendering = true;

    const scroll      = document.getElementById('spectro-scroll');
    const canvas      = document.getElementById('spectro-canvas');
    const viewW       = scroll.clientWidth - FREQ_AXIS_W;
    const viewH       = scroll.clientHeight;
    const ppf         = pxPerFrame();

    const nBins       = (STATE.nfft >> 1) + 1;
    const totalFrames = STATE.totalFrames;
    const canvasW     = Math.round(totalFrames * ppf);

    canvas.width  = canvasW;
    canvas.height = viewH;

    const scrollX    = scroll.scrollLeft;
    const firstFrame = Math.floor(scrollX / ppf);
    const lastFrame  = Math.min(totalFrames, Math.ceil((scrollX + viewW) / ppf));

    const chunkStart = Math.floor(firstFrame / CHUNK_FRAMES) * CHUNK_FRAMES;
    const chunkEnd   = Math.min(
        totalFrames,
        Math.ceil(lastFrame / CHUNK_FRAMES) * CHUNK_FRAMES
    );

    for (let c = chunkStart; c < chunkEnd; c += CHUNK_FRAMES) {
        if (!STATE.tileCache.has(c) || STATE.dirty) {
            const frameEnd = Math.min(c + CHUNK_FRAMES, totalFrames);
            computeChunk(c, frameEnd);
            const tile = renderTile(c, frameEnd, nBins, viewH, ppf);
            STATE.tileCache.set(c, { img: tile, x: Math.round(c * ppf) });
        }
    }

    STATE.dirty = false;

    const ctx2d = canvas.getContext('2d');
    ctx2d.clearRect(0, 0, canvasW, viewH);
    for (const { img, x } of STATE.tileCache.values()) {
        ctx2d.putImageData(img, x, 0);
    }

    drawTimeAxis(totalFrames, canvasW, ppf);
    drawFreqAxis(viewH);

    STATE.rendering = false;
}

/**
 * Render one chunk of frames into an ImageData.
 * Applies dB/linear scaling, gain, freq zoom, and LUT colormap.
 * LUT index 0 = lowest intensity (dark), 255 = highest (bright).
 *
 * @param {number} startFrame
 * @param {number} endFrame
 * @param {number} nBins
 * @param {number} viewH   - canvas height in pixels
 * @param {number} ppf     - pixels per frame (time zoom)
 * @returns {ImageData}
 */
function renderTile(startFrame, endFrame, nBins, viewH, ppf) {
    /**
     * Render one chunk of frames into an ImageData.
     * dB:     val=1 at 0 dBFS, val=0 at -dbRange dBFS.
     * Linear: magnitude normalised to [0,1] with fixed 1.0 reference.
     *         Gain control handles brightness instead of per-tile normalisation.
     * LUT index 0 = dark, 255 = bright (correct for all included colormaps).
     */
    const lut      = STATE.luts[STATE.cmap];
    const gainLin  = Math.pow(10, STATE.gainDb / 20);
    const nFrames  = endFrame - startFrame;
    const tileW    = Math.round(nFrames * ppf);
    const imgData  = new ImageData(tileW, viewH);
    const pix      = imgData.data;

    const { fminHz, fmaxHz } = effectiveFreqBand();
    const nyquist  = STATE.sampleRate / 2;
    const fmaxBin  = Math.min(nBins - 1, Math.round(fmaxHz / nyquist * (nBins - 1)));
    const fminBin  = Math.max(0,         Math.round(fminHz / nyquist * (nBins - 1)));
    const usedBins = Math.max(1, fmaxBin - fminBin + 1);

    for (let px = 0; px < tileW; px++) {
        const f        = Math.min(nFrames - 1, Math.floor(px / ppf));
        const cacheOff = (startFrame + f) * nBins;

        for (let px = 0; px < tileW; px++) {
            const f        = Math.min(nFrames - 1, Math.floor(px / ppf));
            const cacheOff = (startFrame + f) * nBins;

            for (let row = 0; row < viewH; row++) {
                // row=0 → top → fmaxHz; row=viewH-1 → bottom → fminHz
                const bin = fminBin + Math.round((1 - (row + 0.5) / viewH) * (usedBins - 1));
                const pow = STATE.magCache[cacheOff + bin] * gainLin * gainLin;

                let val;
                if (STATE.scale === 'db') {
                    const db = 20 * Math.log10(Math.sqrt(pow) + 1e-12);
                    val = Math.max(0, Math.min(1, (db + STATE.dbRange) / STATE.dbRange));
                } else {
                    val = Math.min(1, Math.sqrt(pow));
                }

                const lutIdx = Math.min(255, Math.floor(val * 255)) * 3;
                const pixOff = (row * tileW + px) * 4;
                pix[pixOff]     = lut[lutIdx];
                pix[pixOff + 1] = lut[lutIdx + 1];
                pix[pixOff + 2] = lut[lutIdx + 2];
                pix[pixOff + 3] = 255;
            }
        }

    return imgData;
    }
}

// === === === === === === === ===
// A X E S
// ==== ==== ==== ==== ==== ====

/**
 * Draw time axis ticks and labels below the spectrogram.
 *
 * @param {number} totalFrames
 * @param {number} canvasW
 * @param {number} ppf         - pixels per frame
 */
function drawTimeAxis(totalFrames, canvasW, ppf) {
    const axCanvas  = document.getElementById('time-canvas');
    axCanvas.width  = canvasW;
    axCanvas.height = TIME_AXIS_H;
    const ctx = axCanvas.getContext('2d');
    ctx.clearRect(0, 0, canvasW, TIME_AXIS_H);
    ctx.fillStyle   = '#888';
    ctx.strokeStyle = '#555';
    ctx.font        = '9px Consolas, monospace';
    ctx.textAlign   = 'center';

    const totalSec = (totalFrames * STATE.hop) / STATE.sampleRate;
    const nTicks   = Math.max(2, Math.floor(canvasW / 100));
    for (let t = 0; t <= nTicks; t++) {
        const frac  = t / nTicks;
        const x     = Math.round(frac * canvasW);
        const sec   = frac * totalSec;
        const label = sec < 60
            ? sec.toFixed(2) + 's'
            : Math.floor(sec / 60) + 'm' + (sec % 60).toFixed(1) + 's';
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 4); ctx.stroke();
        ctx.fillText(label, x, 14);
    }
}

/**
 * Draw frequency axis ticks and labels to the right of the spectrogram.
 * Reflects the effective freq band after fmin/fmax and freq zoom.
 *
 * @param {number} viewH
 */
function drawFreqAxis(viewH) {
    const axCanvas  = document.getElementById('freq-axis');
    axCanvas.width  = FREQ_AXIS_W;
    axCanvas.height = viewH;
    const ctx = axCanvas.getContext('2d');
    ctx.clearRect(0, 0, FREQ_AXIS_W, viewH);
    ctx.fillStyle   = '#888';
    ctx.strokeStyle = '#555';
    ctx.font        = '9px Consolas, monospace';
    ctx.textAlign   = 'right';

    const { fminHz, fmaxHz } = effectiveFreqBand();
    const nTicks = Math.max(2, Math.floor(viewH / 40));

    for (let t = 0; t <= nTicks; t++) {
        const frac   = t / nTicks;
        const y      = Math.round(frac * viewH);

        // Derive displayed frequency from the same row→bin→Hz mapping used in renderTile()
        const row    = Math.round(frac * viewH);
        const bin    = binForRow(row, viewH, (STATE.nfft >> 1) + 1, fminHz, fmaxHz);
        const freqHz = bin / ((STATE.nfft >> 1)) * (STATE.sampleRate / 2);


        const label  = freqHz >= 1000
            ? (freqHz / 1000).toFixed(1) + 'k'
            : Math.round(freqHz) + '';
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(4, y); ctx.stroke();
        ctx.fillText(label, FREQ_AXIS_W - 6, y + 3);
    }
}

// === === === === === === === ===
// C O N T R O L   W I R I N G
// ==== ==== ==== ==== ==== ====

/**
 * Attach all control event handlers.
 * Button groups use click delegation; sliders use mouseup + change.
 * saveParams() is called after every state mutation.
 */
function wireControls() {
    // --- Button groups ---
    wireBtnGroup('bg-nfft', val => {
        STATE.nfft = parseInt(val, 10);
        scheduleRender(true);
        saveParams();
    });

    wireBtnGroup('bg-hop', val => {
        STATE.hop = parseInt(val, 10);
        scheduleRender(true);
        saveParams();
    });

    wireBtnGroup('bg-window', val => {
        STATE.windowType = val;
        scheduleRender(true);
        saveParams();
    });

    wireBtnGroup('bg-scale', val => {
        STATE.scale = val;
        scheduleRender(false);
        saveParams();
    });

    wireBtnGroup('bg-cmap', val => {
        STATE.cmap = val;
        scheduleRender(false);
        saveParams();
    });

    // --- Sliders ---
    onMouseup('ctrl-dbrange', () => {
        STATE.dbRange = parseInt(getVal('ctrl-dbrange'), 10);
        setText('dbrange-val', STATE.dbRange);
        scheduleRender(false);
        saveParams();
    });

    // Live label update during drag; render only on mouseup
    document.getElementById('ctrl-dbrange').addEventListener('input', () => {
        setText('dbrange-val', document.getElementById('ctrl-dbrange').value);
    });

    onMouseup('ctrl-gain', () => {
        STATE.gainDb = parseInt(getVal('ctrl-gain'), 10);
        setText('gain-val', STATE.gainDb);
        scheduleRender(false);
        saveParams();
    });

    document.getElementById('ctrl-gain').addEventListener('input', () => {
        setText('gain-val', document.getElementById('ctrl-gain').value);
    });

    // --- Freq bounds ---
    onBlur('ctrl-fmax', () => {
        const v    = getVal('ctrl-fmax');
        STATE.fmax = v ? parseFloat(v) : null;
        scheduleRender(false);
        saveParams();
    });

    onBlur('ctrl-fmin', () => {
        const v    = getVal('ctrl-fmin');
        STATE.fmin = v ? parseFloat(v) : 0;
        scheduleRender(false);
        saveParams();
    });


    // --- Freq reset buttons ---
    document.getElementById('btn-fmax-reset')?.addEventListener('click', () => {
        setVal('ctrl-fmax', '');
        STATE.fmax = null;
        scheduleRender(false);
        saveParams();
    });

    document.getElementById('btn-fmin-reset')?.addEventListener('click', () => {
        setVal('ctrl-fmin', '');
        STATE.fmin = 0;
        scheduleRender(false);
        saveParams();
    });

    // --- Zoom sliders ---
    onMouseup('ctrl-zoom-time', () => {
        STATE.zoomTime = parseFloat(getVal('ctrl-zoom-time'));
        setText('zoom-time-val', zoomLabel(STATE.zoomTime));
        scheduleRender(false);
        saveParams();
    });

    document.getElementById('ctrl-zoom-time').addEventListener('input', () => {
        setText('zoom-time-val', zoomLabel(parseFloat(
            document.getElementById('ctrl-zoom-time').value
        )));
    });

    onMouseup('ctrl-zoom-freq', () => {
        STATE.zoomFreq = parseFloat(getVal('ctrl-zoom-freq'));
        setText('zoom-freq-val', zoomLabel(STATE.zoomFreq));
        scheduleRender(false);
        saveParams();
    });

    document.getElementById('ctrl-zoom-freq').addEventListener('input', () => {
        setText('zoom-freq-val', zoomLabel(parseFloat(
            document.getElementById('ctrl-zoom-freq').value
        )));
    });
}

/**
 * Attach click delegation to a button group container.
 * On click: sets .active on the clicked button and calls cb with data-val.
 *
 * @param {string}   groupId - element id of the .btn-group container
 * @param {function} cb      - called with the data-val string of the clicked button
 */
function wireBtnGroup(groupId, cb) {
    const group = document.getElementById(groupId);
    if (!group) return;
    group.addEventListener('click', evt => {
        const btn = evt.target.closest('button');
        if (!btn) return;
        group.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        cb(btn.dataset.val);
    });
}

/**
 * Set .active on the button whose data-val matches val; clear all others.
 *
 * @param {string} groupId
 * @param {string} val
 */
function setBtnGroupActive(groupId, val) {
    const group = document.getElementById(groupId);
    if (!group) return;
    group.querySelectorAll('button').forEach(b => {
        b.classList.toggle('active', b.dataset.val === val);
    });
}

/**
 * Attach mouseup + change handlers to an element by id.
 *
 * @param {string}   id
 * @param {function} fn
 */
function onMouseup(id, fn) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('mouseup', fn);
    el.addEventListener('change',  fn);
}

/**
 * Attach blur + change handlers to an element by id.
 *
 * @param {string}   id
 * @param {function} fn
 */
function onBlur(id, fn) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('blur',   fn);
    el.addEventListener('change', fn);
}

/**
 * Re-render on horizontal scroll when crossing chunk boundaries.
 */
function wireScroll() {
    const scroll = document.getElementById('spectro-scroll');
    scroll.addEventListener('scroll', () => {
        const x         = scroll.scrollLeft;
        const ppf       = pxPerFrame();
        const prevChunk = Math.floor(STATE.lastScrollX / (CHUNK_FRAMES * ppf));
        const currChunk = Math.floor(x               / (CHUNK_FRAMES * ppf));
        if (prevChunk !== currChunk) {
            STATE.lastScrollX = x;
            if (STATE.rafHandle) cancelAnimationFrame(STATE.rafHandle);
            STATE.rafHandle   = requestAnimationFrame(renderFrame);
        }
    });
}

/**
 * Show time/frequency readout on mousemove over the spectrogram canvas.
 */
function wireCursorInfo() {
    const canvas = document.getElementById('spectro-canvas');
    canvas.addEventListener('mousemove', evt => {
        const rect    = canvas.getBoundingClientRect();
        const x       = evt.clientX - rect.left;
        const y       = evt.clientY - rect.top;
        const ppf     = pxPerFrame();
        const frame   = Math.floor(x / ppf);
        const timeSec = (frame * STATE.hop) / STATE.sampleRate;

        const { fminHz, fmaxHz } = effectiveFreqBand();
        const freqHz = fminHz + (fmaxHz - fminHz) * (1 - y / canvas.height);

        document.getElementById('info-cursor').textContent =
            `t=${timeSec.toFixed(3)}s  f=${Math.round(freqHz)}Hz`;
    });

    canvas.addEventListener('mouseleave', () => {
        document.getElementById('info-cursor').textContent = '—';
    });
}

// === === === === === === === ===
// U I   H E L P E R S
// ==== ==== ==== ==== ==== ====

/**
 * Populate the audio info box with decoded buffer metadata.
 */
function updateInfoBox(buf) {
    document.getElementById('info-sr').textContent  = `SR: ${buf.sampleRate} Hz`;
    document.getElementById('info-dur').textContent = `dur: ${buf.duration.toFixed(3)} s`;
    document.getElementById('info-ch').textContent  = `ch: ${buf.numberOfChannels}`;

    if (window.SPECTRO_PAYLOAD.audioB64.length * 0.75 > WARN_SIZE_MB * 1024 * 1024) {
        document.getElementById('info-dur').textContent += '  ⚠ large';
    }
}

/**
 * Rebuild the channel selector as a btn-group after audio decode.
 * Only shows valid options for the current file's channel count.
 * Wires click handler and reflects saved STATE.channel.
 *
 * @param {number} nCh - number of channels in the decoded AudioBuffer
 */
function updateChannelSelector(nCh) {
    let group = document.getElementById('bg-channel');
    if (!group) {
        const panel   = document.getElementById('panel');
        const wrapper = document.createElement('div');
        wrapper.className = 'ctrl-row';
        wrapper.innerHTML =
            '<label>C H A N N E L</label>' +
            '<div class="btn-group" id="bg-channel"></div>';
        panel.insertBefore(wrapper, document.getElementById('info-box'));
        group = document.getElementById('bg-channel');
    }

    group.innerHTML = '';
    const options = [{ val: '0', label: 'L' }];
    if (nCh > 1) {
        options.push({ val: '1',   label: 'R'   });
        options.push({ val: 'mix', label: 'mix' });
    }

    const savedCh = String(STATE.channel);
    for (const opt of options) {
        const btn = document.createElement('button');
        btn.dataset.val = opt.val;
        btn.textContent = opt.label;
        if (opt.val === savedCh) btn.classList.add('active');
        group.appendChild(btn);
    }

    wireBtnGroup('bg-channel', val => {
        STATE.channel = val === 'mix' ? 'mix' : parseInt(val, 10);
        if (STATE.audioBuffer) {
            applyChannelSelection(STATE.audioBuffer);
            scheduleRender(true);
            saveParams();
        }
    });
}

/**
 * Hide the loading overlay.
 */
function hideLoading() {
    const ov = document.getElementById('loading-overlay');
    if (ov) ov.style.display = 'none';
}

/**
 * Set the progress bar fill percentage.
 */
function setProgress(pct) {
    const bar = document.getElementById('progress-bar');
    if (bar) bar.style.width = pct + '%';
}

/**
 * Display an error message in the error overlay.
 */
function showError(msg) {
    hideLoading();
    const el = document.getElementById('error-msg');
    if (!el) { console.error('showError:', msg); return; }
    el.textContent   = msg;
    el.style.display = 'flex';
}


// === === === === === === === ===
// D O M   U T I L I T I E S
// ==== ==== ==== ==== ==== ====

/** Get value of an input element by id. */
function getVal(id) {
    return document.getElementById(id)?.value ?? '';
}

/** Set value of an input element by id. */
function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
}

/** Set textContent of an element by id. */
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

// === === === === === === === ===
// B Y T E   U T I L I T I E S
// ==== ==== ==== ==== ==== ====

/**
 * Decode a base64 string to a Uint8Array.
 */
function base64ToUint8(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
}

/**
 * Decode a base64 string to an ArrayBuffer.
 */
function base64ToArrayBuffer(b64) {
    return base64ToUint8(b64).buffer;
}