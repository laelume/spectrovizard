// Copyright 2026 laelume. All rights reserved.
// BSD 3-Clause License.
//
// All DSP runs client-side. No Python subprocess. No Node.js audio libraries.
//
// Pipeline overview:
//   1. Base64 audio bytes → ArrayBuffer
//   2. Web Audio API decodeAudioData → AudioBuffer (handles WAV/MP3/FLAC/OGG/M4A)
//   3. PCM path: manual ArrayBuffer → Float32Array conversion
//   4. Per-chunk STFT via Cooley-Tukey radix-2 FFT (pure JS)
//   5. Power spectrum → dB scaling → LUT colormap → canvas pixel rows
//   6. Chunk-based streaming: render visible window, recompute on scroll
//
// References:
//   [1] Cooley & Tukey (1965) "An algorithm for the machine calculation of
//       complex Fourier series." Mathematics of Computation 19(90):297–301.
//       DOI: https://doi.org/10.1090/S0025-5718-1965-0178586-1
//   [2] Harris (1978) "On the use of windows for harmonic analysis with the
//       discrete Fourier transform." Proc. IEEE 66(1):51–83.
//       DOI: https://doi.org/10.1109/PROC.1978.10837
//   [3] Smith (2011) "Spectral Audio Signal Processing."
//       https://ccrma.stanford.edu/~jos/sasp/

/* global window, document, AudioContext, OfflineAudioContext */

'use strict';

// === === === === === === === ===
// C O N S T A N T S
// ==== ==== ==== ==== ==== ====

// Number of STFT frames to compute per rendering chunk.
// Larger values reduce draw calls; smaller values improve scroll responsiveness.
const CHUNK_FRAMES   = 512;

// Minimum canvas pixel column width per STFT frame.
const PX_PER_FRAME   = 1;

// Frequency axis canvas width in pixels.
const FREQ_AXIS_W    = 44;

// Time axis canvas height in pixels.
const TIME_AXIS_H    = 24;

// Maximum base64 inline audio size (bytes). Files larger than this are
// flagged in the info box but still processed; no hard limit is enforced
// because the Web Audio API handles large buffers natively.
const WARN_SIZE_MB   = 50;

// === === === === === === === ===
// S T A T E
// ==== ==== ==== ==== ==== ====

/**
 * Global viewer state. Mutated by control handlers and render calls.
 * All rendering reads from this object exclusively.
 */
const STATE = {
    // AudioBuffer decoded from the file
    audioBuffer  : null,

    // Flat Float32Array of the active channel samples
    samples      : null,

    // Sample rate in Hz (from AudioBuffer or pcmMeta)
    sampleRate   : 0,

    // Current STFT parameters (mirrored from DOM controls)
    nfft         : 512,
    hop          : 128,
    windowType   : 'hann',
    scale        : 'db',
    dbRange      : 80,
    gainDb       : 0,
    cmap         : 'viridis',
    channel      : 0,    // 0=L, 1=R, 'mix'=average
    fmax         : null, // null = Nyquist

    // Decoded LUT tables: name → Uint8Array(768)
    luts         : {},

    // Total number of STFT frames for the current audio/params
    totalFrames  : 0,

    // Per-frame magnitude cache: Float32Array[totalFrames * (nfft/2+1)]
    // Populated lazily by computeChunk().
    magCache     : null,

    // Rendered tile cache: Map<chunkIndex, ImageData>
    tileCache    : new Map(),

    // Whether a full recompute is pending (params changed)
    dirty        : true,

    // Render lock: prevent concurrent renders
    rendering    : false,

    // Scroll position tracking for tile eviction
    lastScrollX  : 0,

    // Animation frame handle
    rafHandle    : null
};

// === === === === === === === ===
// E N T R Y   P O I N T
// ==== ==== ==== ==== ==== ====

(function init() {
    /**
     * Main entry: decode payload, wire controls, start render loop.
     */
    const payload = window.SPECTRO_PAYLOAD;
    if (!payload) { showError('SPECTRO_PAYLOAD missing.'); return; }

    // Decode LUTs from base64
    for (const [name, b64] of Object.entries(payload.luts)) {
        STATE.luts[name] = base64ToUint8(b64);
    }

    document.getElementById('filename-label').textContent = payload.filename;

    // Decode audio bytes → AudioBuffer, then kick off initial render
    decodeAudio(payload)
        .then(buf => {
            STATE.audioBuffer = buf;
            STATE.sampleRate  = buf.sampleRate;
            updateInfoBox(buf);
            updateChannelSelector(buf.numberOfChannels);
            applyChannelSelection(buf);
            hideLoading();
            scheduleRender(true);
        })
        .catch(err => {
            showError(`Audio decode failed: ${err.message}`);
        });

    wireControls();
    wireScroll();
    wireCursorInfo();
})();

// === === === === === === === ===
// A U D I O   D E C O D E
// ==== ==== ==== ==== ==== ====

/**
 * Decode base64 audio payload into a Web Audio API AudioBuffer.
 * Handles WAV/MP3/FLAC/OGG/M4A via decodeAudioData, and bare PCM manually.
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

    // Use OfflineAudioContext as a container for decodeAudioData.
    // Sample rate is provisional (44100); the decoded buffer carries the real SR.
    const offCtx = new OfflineAudioContext(1, 1, 44100);
    setProgress(40);

    return new Promise((resolve, reject) => {
        offCtx.decodeAudioData(
            arrayBuf,
            buf => { setProgress(90); resolve(buf); },
            err => reject(new Error(String(err)))
        );
    });
}

/**
 * Manually decode a raw PCM ArrayBuffer into an AudioBuffer.
 * Supports dtypes: f32le, f32be, i16le, i16be, i32le, i32be, u8.
 *
 * @param {ArrayBuffer} arrayBuf  - raw bytes
 * @param {object}      meta      - { sampleRate, channels, dtype }
 * @returns {AudioBuffer}
 */
function decodePcm(arrayBuf, meta) {
    const { sampleRate, channels, dtype } = meta;
    const view        = new DataView(arrayBuf);
    const bytesPerSmp = dtypeBytes(dtype);
    const totalSmp    = Math.floor(arrayBuf.byteLength / bytesPerSmp);
    const framesTotal = Math.floor(totalSmp / channels);

    const offCtx  = new OfflineAudioContext(channels, framesTotal, sampleRate);
    const audioBuf = offCtx.createBuffer(channels, framesTotal, sampleRate);

    for (let ch = 0; ch < channels; ch++) {
        const chData = audioBuf.getChannelData(ch);
        for (let i = 0; i < framesTotal; i++) {
            const byteOff = (i * channels + ch) * bytesPerSmp;
            chData[i] = readPcmSample(view, byteOff, dtype);
        }
    }

    setProgress(80);
    return audioBuf;
}

/**
 * Return bytes per sample for a PCM dtype string.
 */
function dtypeBytes(dtype) {
    if (dtype === 'u8')               return 1;
    if (dtype === 'i16le' || dtype === 'i16be') return 2;
    if (dtype === 'i32le' || dtype === 'i32be') return 4;
    if (dtype === 'f32le' || dtype === 'f32be') return 4;
    throw new Error(`Unknown dtype: ${dtype}`);
}

/**
 * Read a single normalised float sample [-1, 1] from a DataView at byteOffset.
 */
function readPcmSample(view, offset, dtype) {
    switch (dtype) {
        case 'f32le': return view.getFloat32(offset, true);
        case 'f32be': return view.getFloat32(offset, false);
        case 'i16le': return view.getInt16(offset, true)  / 32768.0;
        case 'i16be': return view.getInt16(offset, false) / 32768.0;
        case 'i32le': return view.getInt32(offset, true)  / 2147483648.0;
        case 'i32be': return view.getInt32(offset, false) / 2147483648.0;
        case 'u8':    return (view.getUint8(offset) - 128) / 128.0;
        default:      return 0;
    }
}

// === === === === === === === ===
// C H A N N E L   S E L E C T I O N
// ==== ==== ==== ==== ==== ====

/**
 * Extract the active channel (or downmixed mono) from AudioBuffer
 * into STATE.samples as a flat Float32Array.
 *
 * @param {AudioBuffer} buf
 */
function applyChannelSelection(buf) {
    const nCh  = buf.numberOfChannels;
    const nSmp = buf.length;
    const ch   = STATE.channel;

    if (ch === 'mix' && nCh > 1) {
        // Average all channels into mono
        const out = new Float32Array(nSmp);
        for (let c = 0; c < nCh; c++) {
            const src = buf.getChannelData(c);
            for (let i = 0; i < nSmp; i++) out[i] += src[i];
        }
        const inv = 1.0 / nCh;
        for (let i = 0; i < nSmp; i++) out[i] *= inv;
        STATE.samples = out;
    } else {
        // Clamp channel index to available channels
        const idx = (typeof ch === 'number') ? Math.min(ch, nCh - 1) : 0;
        STATE.samples = buf.getChannelData(idx).slice();
    }
}

// === === === === === === === ===
// S T F T   C O R E
// ==== ==== ==== ==== ==== ====

/**
 * Compute a Cooley-Tukey radix-2 in-place FFT on a Float32Array of length N.
 * Input is interleaved [re0, im0, re1, im1, ...].
 * N must be a power of two.
 *
 * Reference: [1] Cooley & Tukey 1965, DOI:10.1090/S0025-5718-1965-0178586-1
 *
 * @param {Float32Array} data - interleaved real/imag, length 2*N
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
            let tmp = data[2*i];   data[2*i]   = data[2*j];   data[2*j]   = tmp;
            tmp     = data[2*i+1]; data[2*i+1] = data[2*j+1]; data[2*j+1] = tmp;
        }
    }

    // Butterfly stages
    for (let len = 2; len <= n; len <<= 1) {
        const ang  = -2 * Math.PI / len;
        const wRe  = Math.cos(ang);
        const wIm  = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let curRe = 1.0, curIm = 0.0;
            for (let k = 0; k < len >> 1; k++) {
                const uRe = data[2*(i+k)];
                const uIm = data[2*(i+k)+1];
                const vRe = data[2*(i+k+len/2)]   * curRe - data[2*(i+k+len/2)+1] * curIm;
                const vIm = data[2*(i+k+len/2)]   * curIm + data[2*(i+k+len/2)+1] * curRe;
                data[2*(i+k)]         = uRe + vRe;
                data[2*(i+k)+1]       = uIm + vIm;
                data[2*(i+k+len/2)]   = uRe - vRe;
                data[2*(i+k+len/2)+1] = uIm - vIm;
                const nRe = curRe * wRe - curIm * wIm;
                curIm     = curRe * wIm + curIm * wRe;
                curRe     = nRe;
            }
        }
    }
}

/**
 * Build a window function of length N.
 * Reference: [2] Harris 1978, DOI:10.1109/PROC.1978.10837
 *
 * @param  {string}     type - 'hann' | 'hamming' | 'blackman' | 'bartlett' | 'rect'
 * @param  {number}     n    - window length
 * @returns {Float32Array}
 */
function makeWindow(type, n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const t = i / (n - 1);
        switch (type) {
            case 'hann':
                w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * t));
                break;
            case 'hamming':
                w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * t);
                break;
            case 'blackman':
                w[i] = 0.42 - 0.5 * Math.cos(2 * Math.PI * t) + 0.08 * Math.cos(4 * Math.PI * t);
                break;
            case 'bartlett':
                w[i] = 1 - Math.abs((2 * i - (n - 1)) / (n - 1));
                break;
            case 'rect':
            default:
                w[i] = 1.0;
                break;
        }
    }
    return w;
}

/**
 * Compute STFT power spectrum for a contiguous block of frames.
 * Results are written into STATE.magCache at the correct frame offsets.
 *
 * @param {number} startFrame - first frame index (inclusive)
 * @param {number} endFrame   - last frame index (exclusive)
 */
function computeChunk(startFrame, endFrame) {
    const { samples, nfft, hop, windowType } = STATE;
    const win    = makeWindow(windowType, nfft);
    const nBins  = (nfft >> 1) + 1;
    const buf    = new Float32Array(nfft * 2); // interleaved re/im

    for (let f = startFrame; f < endFrame; f++) {
        const offset = f * hop;

        // Zero-pad if the frame extends beyond the signal
        buf.fill(0);
        const copyLen = Math.min(nfft, samples.length - offset);
        if (copyLen <= 0) break;

        for (let k = 0; k < copyLen; k++) {
            buf[2*k]   = samples[offset + k] * win[k]; // real
            buf[2*k+1] = 0;                             // imag
        }

        fftInPlace(buf);

        // Store power spectrum (magnitude squared) for bins 0..nBins-1
        const cacheOff = f * nBins;
        for (let b = 0; b < nBins; b++) {
            const re = buf[2*b];
            const im = buf[2*b+1];
            STATE.magCache[cacheOff + b] = re * re + im * im;
        }
    }
}

// === === === === === === === ===
// R E N D E R   P I P E L I N E
// ==== ==== ==== ==== ==== ====

/**
 * Invalidate caches and schedule a full render pass.
 * Called whenever STFT parameters change.
 *
 * @param {boolean} recompute - if true, discard magCache and recompute from scratch
 */
function scheduleRender(recompute) {
    if (recompute) {
        const nBins     = (STATE.nfft >> 1) + 1;
        STATE.totalFrames = Math.ceil(
            (STATE.samples.length - STATE.nfft) / STATE.hop
        ) + 1;
        STATE.magCache   = new Float32Array(STATE.totalFrames * nBins);
        STATE.tileCache  = new Map();
        STATE.dirty      = true;
    } else {
        // Colormap / gain / scale change: tiles need repaint but magCache is valid
        STATE.tileCache  = new Map();
        STATE.dirty      = true;
    }

    if (STATE.rafHandle) cancelAnimationFrame(STATE.rafHandle);
    STATE.rafHandle = requestAnimationFrame(renderFrame);
}

/**
 * Single animation frame: render the currently visible canvas region.
 * Called by requestAnimationFrame; re-schedules itself if more work remains.
 */
function renderFrame() {
    STATE.rafHandle = null;
    if (!STATE.samples || STATE.rendering) return;

    STATE.rendering = true;

    const scroll      = document.getElementById('spectro-scroll');
    const canvas      = document.getElementById('spectro-canvas');
    const viewW       = scroll.clientWidth;
    const viewH       = scroll.clientHeight;

    const nBins       = (STATE.nfft >> 1) + 1;
    const totalFrames = STATE.totalFrames;

    // Canvas width: one pixel column per STFT frame
    const canvasW     = totalFrames * PX_PER_FRAME;
    canvas.width      = canvasW;
    canvas.height     = viewH;

    // Determine visible frame range from scroll position
    const scrollX     = scroll.scrollLeft;
    const firstFrame  = Math.floor(scrollX / PX_PER_FRAME);
    const lastFrame   = Math.min(
        totalFrames,
        Math.ceil((scrollX + viewW) / PX_PER_FRAME)
    );

    // Expand to chunk boundaries
    const chunkStart  = Math.floor(firstFrame / CHUNK_FRAMES) * CHUNK_FRAMES;
    const chunkEnd    = Math.min(
        totalFrames,
        (Math.ceil(lastFrame / CHUNK_FRAMES)) * CHUNK_FRAMES
    );

    // Compute any uncached chunks
    for (let c = chunkStart; c < chunkEnd; c += CHUNK_FRAMES) {
        const key = c;
        if (!STATE.tileCache.has(key) || STATE.dirty) {
            const frameEnd = Math.min(c + CHUNK_FRAMES, totalFrames);
            computeChunk(c, frameEnd);
            const tile = renderTile(c, frameEnd, nBins, viewH);
            STATE.tileCache.set(key, tile);
        }
    }

    STATE.dirty = false;

    // Paint all cached tiles to the canvas
    const ctx2d = canvas.getContext('2d');
    ctx2d.clearRect(0, 0, canvasW, viewH);
    for (const [chunkIdx, tile] of STATE.tileCache.entries()) {
        ctx2d.putImageData(tile, chunkIdx * PX_PER_FRAME, 0);
    }

    drawTimeAxis(totalFrames, canvasW);
    drawFreqAxis(nBins, viewH);

    STATE.rendering = false;
}

/**
 * Render a single tile (one chunk of frames) into an ImageData object.
 * Applies dB or linear scaling, gain, and LUT colormap.
 *
 * @param {number} startFrame
 * @param {number} endFrame
 * @param {number} nBins      - number of FFT bins (nfft/2 + 1)
 * @param {number} viewH      - canvas height in pixels
 * @returns {ImageData}
 */
function renderTile(startFrame, endFrame, nBins, viewH) {
    const lut      = STATE.luts[STATE.cmap];
    const gainLin  = Math.pow(10, STATE.gainDb / 20);
    const nFrames  = endFrame - startFrame;
    const imgData  = new ImageData(nFrames, viewH);
    const pix      = imgData.data;

    // Determine fmax bin cutoff
    const nyquist  = STATE.sampleRate / 2;
    const fmaxHz   = STATE.fmax && STATE.fmax < nyquist ? STATE.fmax : nyquist;
    const fmaxBin  = Math.min(nBins - 1, Math.floor(fmaxHz / nyquist * (nBins - 1)));
    const usedBins = fmaxBin + 1;

    for (let f = 0; f < nFrames; f++) {
        const cacheOff = (startFrame + f) * nBins;

        for (let row = 0; row < viewH; row++) {
            // Map pixel row to frequency bin (low freq at bottom)
            const bin = Math.floor((1 - row / viewH) * (usedBins - 1));
            const pow = STATE.magCache[cacheOff + bin] * gainLin * gainLin;

            let val; // normalised [0, 1]
            if (STATE.scale === 'db') {
                // Power in dB; floor at -dbRange
                const db  = 10 * Math.log10(pow + 1e-12);
                val = Math.max(0, Math.min(1, (db + STATE.dbRange) / STATE.dbRange));
            } else {
                // Linear magnitude normalised to [0, 1] via soft max
                val = Math.sqrt(Math.max(0, pow));
                val = Math.min(1, val);
            }

            const lutIdx = Math.min(255, Math.floor(val * 255)) * 3;
            const pixOff = (row * nFrames + f) * 4;
            pix[pixOff]     = lut[lutIdx];
            pix[pixOff + 1] = lut[lutIdx + 1];
            pix[pixOff + 2] = lut[lutIdx + 2];
            pix[pixOff + 3] = 255;
        }
    }

    return imgData;
}

// === === === === === === === ===
// A X E S
// ==== ==== ==== ==== ==== ====

/**
 * Draw time axis ticks and labels below the spectrogram canvas.
 *
 * @param {number} totalFrames - total number of STFT frames
 * @param {number} canvasW     - spectrogram canvas pixel width
 */
function drawTimeAxis(totalFrames, canvasW) {
    const axCanvas = document.getElementById('time-canvas');
    axCanvas.width  = canvasW;
    axCanvas.height = TIME_AXIS_H;
    const ctx = axCanvas.getContext('2d');
    ctx.clearRect(0, 0, canvasW, TIME_AXIS_H);
    ctx.fillStyle   = '#888';
    ctx.strokeStyle = '#555';
    ctx.font        = '9px Consolas, monospace';
    ctx.textAlign   = 'center';

    const totalSec  = (totalFrames * STATE.hop) / STATE.sampleRate;
    // Target roughly one tick per 100 pixels
    const nTicks    = Math.max(2, Math.floor(canvasW / 100));
    for (let t = 0; t <= nTicks; t++) {
        const frac = t / nTicks;
        const x    = Math.round(frac * canvasW);
        const sec  = frac * totalSec;
        const label = sec < 60
            ? sec.toFixed(2) + 's'
            : Math.floor(sec / 60) + 'm' + (sec % 60).toFixed(1) + 's';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, 4);
        ctx.stroke();
        ctx.fillText(label, x, 14);
    }
}

/**
 * Draw frequency axis ticks and labels to the right of the spectrogram.
 *
 * @param {number} nBins  - number of frequency bins
 * @param {number} viewH  - canvas height
 */
function drawFreqAxis(nBins, viewH) {
    const axCanvas = document.getElementById('freq-axis');
    axCanvas.width  = FREQ_AXIS_W;
    axCanvas.height = viewH;
    const ctx = axCanvas.getContext('2d');
    ctx.clearRect(0, 0, FREQ_AXIS_W, viewH);
    ctx.fillStyle   = '#888';
    ctx.strokeStyle = '#555';
    ctx.font        = '9px Consolas, monospace';
    ctx.textAlign   = 'right';

    const nyquist   = STATE.sampleRate / 2;
    const fmaxHz    = STATE.fmax && STATE.fmax < nyquist ? STATE.fmax : nyquist;
    const nTicks    = Math.max(2, Math.floor(viewH / 40));

    for (let t = 0; t <= nTicks; t++) {
        const frac  = t / nTicks;
        const y     = Math.round(frac * viewH);
        const freqHz = fmaxHz * (1 - frac);
        const label  = freqHz >= 1000
            ? (freqHz / 1000).toFixed(1) + 'k'
            : Math.round(freqHz) + '';
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(4, y);
        ctx.stroke();
        ctx.fillText(label, FREQ_AXIS_W - 6, y + 3);
    }
}

// === === === === === === === ===
// C O N T R O L   W I R I N G
// ==== ==== ==== ==== ==== ====

/**
 * Attach change handlers to all panel controls.
 * Re-render is triggered on mouseup or change (not on every mousemove).
 */
function wireControls() {
    // Helper: attach 'change' and 'mouseup' (for sliders) to an element.
    const on = (id, fn) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('change',  fn);
        el.addEventListener('mouseup', fn);
    };

    on('ctrl-nfft',    () => {
        STATE.nfft  = parseInt(document.getElementById('ctrl-nfft').value, 10);
        scheduleRender(true);
    });

    on('ctrl-hop',     () => {
        STATE.hop   = parseInt(document.getElementById('ctrl-hop').value, 10);
        scheduleRender(true);
    });

    on('ctrl-window',  () => {
        STATE.windowType = document.getElementById('ctrl-window').value;
        scheduleRender(true);
    });

    on('ctrl-scale',   () => {
        STATE.scale = document.getElementById('ctrl-scale').value;
        scheduleRender(false);
    });

    on('ctrl-dbrange', () => {
        STATE.dbRange = parseInt(document.getElementById('ctrl-dbrange').value, 10);
        document.getElementById('dbrange-val').textContent = STATE.dbRange;
        scheduleRender(false);
    });

    on('ctrl-gain',    () => {
        STATE.gainDb = parseInt(document.getElementById('ctrl-gain').value, 10);
        document.getElementById('gain-val').textContent = STATE.gainDb;
        scheduleRender(false);
    });

    on('ctrl-cmap',    () => {
        STATE.cmap  = document.getElementById('ctrl-cmap').value;
        scheduleRender(false);
    });

    on('ctrl-fmax',    () => {
        const v     = document.getElementById('ctrl-fmax').value;
        STATE.fmax  = v ? parseFloat(v) : null;
        scheduleRender(false);
    });

    on('ctrl-channel', () => {
        const v     = document.getElementById('ctrl-channel').value;
        STATE.channel = v === 'mix' ? 'mix' : parseInt(v, 10);
        if (STATE.audioBuffer) {
            applyChannelSelection(STATE.audioBuffer);
            scheduleRender(true);
        }
    });
}

/**
 * Re-render when the user scrolls horizontally into an uncached region.
 */
function wireScroll() {
    const scroll = document.getElementById('spectro-scroll');
    scroll.addEventListener('scroll', () => {
        const x = scroll.scrollLeft;
        // Only re-render if scroll crossed a chunk boundary
        const prevChunk = Math.floor(STATE.lastScrollX / (CHUNK_FRAMES * PX_PER_FRAME));
        const currChunk = Math.floor(x             / (CHUNK_FRAMES * PX_PER_FRAME));
        if (prevChunk !== currChunk) {
            STATE.lastScrollX = x;
            if (STATE.rafHandle) cancelAnimationFrame(STATE.rafHandle);
            STATE.rafHandle   = requestAnimationFrame(renderFrame);
        }
    });
}

/**
 * Show time/frequency cursor readout in the info box on mousemove over canvas.
 */
function wireCursorInfo() {
    const canvas = document.getElementById('spectro-canvas');
    canvas.addEventListener('mousemove', evt => {
        const rect   = canvas.getBoundingClientRect();
        const x      = evt.clientX - rect.left;
        const y      = evt.clientY - rect.top;

        const frame  = Math.floor(x / PX_PER_FRAME);
        const timeSec = (frame * STATE.hop) / STATE.sampleRate;

        const nyquist = STATE.sampleRate / 2;
        const fmaxHz  = STATE.fmax && STATE.fmax < nyquist ? STATE.fmax : nyquist;
        const freqHz  = fmaxHz * (1 - y / canvas.height);

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
    document.getElementById('info-sr').textContent  =
        `SR: ${buf.sampleRate} Hz`;
    document.getElementById('info-dur').textContent =
        `Duration: ${buf.duration.toFixed(3)} s`;
    document.getElementById('info-ch').textContent  =
        `Channels: ${buf.numberOfChannels}`;
    document.getElementById('fmax-hint').textContent =
        `Nyquist: ${(buf.sampleRate / 2).toLocaleString()} Hz`;

    if (window.SPECTRO_PAYLOAD.audioB64.length * 0.75 > WARN_SIZE_MB * 1024 * 1024) {
        document.getElementById('info-dur').textContent +=
            `  ⚠ large file`;
    }
}

/**
 * Update the channel selector to show only valid options for nCh channels.
 */
function updateChannelSelector(nCh) {
    const sel = document.getElementById('ctrl-channel');
    // Show Ch 2 (R) and Mix only if the file is multi-channel
    sel.options[1].disabled = nCh < 2;
    sel.options[2].disabled = nCh < 2;
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
    if (!el) return;
    el.textContent  = msg;
    el.style.display = 'flex';
}

// === === === === === === === ===
// U T I L I T I E S
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
    const u8 = base64ToUint8(b64);
    return u8.buffer;
}