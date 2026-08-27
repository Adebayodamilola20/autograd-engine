/* nabla MNIST demo — front end.
 *
 * Three jobs:
 *   1. capture a drawing
 *   2. preprocess it the way MNIST was preprocessed  <- the part that matters
 *   3. render what the network did with it
 *
 * No framework, no build step. `git clone` then `python web/server.py` and it
 * runs; there is no npm install standing between a reader and the demo.
 */

(() => {
  'use strict';

  const PAD = 280;         // on-screen drawing surface, px
  const GRID = 28;         // MNIST image size
  const BOX = 20;          // MNIST fits the digit into 20x20, then pads to 28
  const DEBOUNCE_MS = 220;

  const pad = document.getElementById('pad');
  const preview = document.getElementById('preview');
  const ctx = pad.getContext('2d', { willReadFrequently: true });
  const pctx = preview.getContext('2d', { willReadFrequently: true });

  const el = {
    placeholder: document.getElementById('placeholder'),
    clear: document.getElementById('clear'),
    random: document.getElementById('random'),
    stroke: document.getElementById('stroke'),
    verdict: document.getElementById('verdict'),
    confidence: document.getElementById('confidence'),
    runner: document.getElementById('runner'),
    timing: document.getElementById('timing'),
    bars: document.getElementById('bars'),
    layers: document.getElementById('layers'),
    activations: document.getElementById('activations'),
    sparsityNote: document.getElementById('sparsity-note'),
    statArch: document.getElementById('stat-arch'),
    statParams: document.getElementById('stat-params'),
    statAcc: document.getElementById('stat-acc'),
    toast: document.getElementById('toast'),
  };

  let drawing = false;
  let hasInk = false;
  let timer = null;
  let inFlight = false;

  // ================================================================ canvas

  function resetPad() {
    ctx.fillStyle = '#05080f';
    ctx.fillRect(0, 0, PAD, PAD);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff';
  }

  function positionOf(event) {
    // getBoundingClientRect, not offsetX: the canvas is CSS-scaled on small
    // screens, so the backing store and the displayed size differ.
    const rect = pad.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (PAD / rect.width),
      y: (event.clientY - rect.top) * (PAD / rect.height),
    };
  }

  function startStroke(event) {
    drawing = true;
    hasInk = true;
    el.placeholder.hidden = true;
    const { x, y } = positionOf(event);
    ctx.lineWidth = Number(el.stroke.value);
    ctx.beginPath();
    ctx.moveTo(x, y);
    // A dot, so a single tap registers rather than drawing nothing.
    ctx.lineTo(x + 0.01, y);
    ctx.stroke();
    pad.setPointerCapture?.(event.pointerId);
  }

  function extendStroke(event) {
    if (!drawing) return;
    const { x, y } = positionOf(event);
    ctx.lineTo(x, y);
    ctx.stroke();
    schedulePredict();
  }

  function endStroke() {
    if (!drawing) return;
    drawing = false;
    ctx.closePath();
    schedulePredict();
  }

  pad.addEventListener('pointerdown', (e) => { e.preventDefault(); startStroke(e); });
  pad.addEventListener('pointermove', extendStroke);
  pad.addEventListener('pointerup', endStroke);
  pad.addEventListener('pointercancel', endStroke);
  pad.addEventListener('pointerleave', endStroke);

  // ============================================== MNIST preprocessing
  //
  // This is the step people skip, and skipping it is why home-made digit
  // demos predict badly. MNIST was not built from raw 28x28 screenshots: each
  // digit was cropped to its ink, scaled so its longest side fits a 20x20 box,
  // and placed in a 28x28 frame with its *centre of mass* on the centre pixel.
  //
  // A network trained on that distribution has never seen a digit that fills
  // the frame or sits in a corner. Feeding it one is out-of-distribution
  // input, and it will answer confidently and wrongly. Replicating the
  // preprocessing is not polish -- it is what makes the demo work at all.

  function toGrid() {
    const source = ctx.getImageData(0, 0, PAD, PAD).data;

    // 1. Ink intensity per pixel. Drawn white on near-black, so the red
    //    channel alone is a fine proxy for intensity.
    const ink = new Float64Array(PAD * PAD);
    let minX = PAD, minY = PAD, maxX = -1, maxY = -1;

    for (let y = 0; y < PAD; y++) {
      for (let x = 0; x < PAD; x++) {
        const value = source[(y * PAD + x) * 4] / 255;
        ink[y * PAD + x] = value;
        if (value > 0.12) {           // threshold, to ignore antialiasing haze
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (maxX < 0) return null;        // nothing drawn

    // 2. Crop to the ink, scale the longest side to BOX, preserving aspect
    //    ratio. Distorting the aspect ratio would turn a 1 into a blob.
    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    const scale = BOX / Math.max(w, h);
    const outW = Math.max(1, Math.round(w * scale));
    const outH = Math.max(1, Math.round(h * scale));

    // Box-filter downsample: average the source pixels falling in each target
    // cell. Nearest-neighbour would drop thin strokes entirely.
    const small = new Float64Array(outW * outH);
    for (let ty = 0; ty < outH; ty++) {
      for (let tx = 0; tx < outW; tx++) {
        const x0 = minX + Math.floor((tx * w) / outW);
        const x1 = minX + Math.max(Math.floor(((tx + 1) * w) / outW), Math.floor((tx * w) / outW) + 1);
        const y0 = minY + Math.floor((ty * h) / outH);
        const y1 = minY + Math.max(Math.floor(((ty + 1) * h) / outH), Math.floor((ty * h) / outH) + 1);
        let sum = 0, count = 0;
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) { sum += ink[y * PAD + x]; count++; }
        }
        small[ty * outW + tx] = count ? sum / count : 0;
      }
    }

    // 3. Centre of mass of the scaled digit.
    let mass = 0, cx = 0, cy = 0;
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        const v = small[y * outW + x];
        mass += v; cx += x * v; cy += y * v;
      }
    }
    if (mass <= 0) return null;
    cx /= mass;
    cy /= mass;

    // 4. Paste into 28x28 so that the centre of mass lands on the centre.
    const grid = new Float64Array(GRID * GRID);
    const offX = Math.round(GRID / 2 - cx);
    const offY = Math.round(GRID / 2 - cy);

    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        const gx = x + offX;
        const gy = y + offY;
        if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID) {
          grid[gy * GRID + gx] = small[y * outW + x];
        }
      }
    }
    return grid;
  }

  function renderPreview(grid) {
    const image = pctx.createImageData(GRID, GRID);
    for (let i = 0; i < GRID * GRID; i++) {
      const v = Math.round(Math.min(1, Math.max(0, grid[i])) * 255);
      image.data[i * 4] = v;
      image.data[i * 4 + 1] = v;
      image.data[i * 4 + 2] = v;
      image.data[i * 4 + 3] = 255;
    }
    pctx.putImageData(image, 0, 0);
  }

  function clearPreview() {
    pctx.fillStyle = '#05080f';
    pctx.fillRect(0, 0, GRID, GRID);
  }

  // ================================================================ network

  function schedulePredict() {
    clearTimeout(timer);
    timer = setTimeout(predict, DEBOUNCE_MS);
  }

  async function predict() {
    if (!hasInk || inFlight) return;
    const grid = toGrid();
    if (!grid) return;

    renderPreview(grid);
    inFlight = true;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels: Array.from(grid) }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      render(data);
    } catch (err) {
      showToast(`Prediction failed: ${err.message}`);
    } finally {
      inFlight = false;
    }
  }

  // ================================================================ render

  function buildBars() {
    const frag = document.createDocumentFragment();
    for (let digit = 0; digit < 10; digit++) {
      const row = document.createElement('li');
      row.className = 'bar-row';
      row.dataset.digit = String(digit);
      row.innerHTML =
        `<span class="bar-label">${digit}</span>` +
        `<span class="bar-track"><span class="bar-fill"></span></span>` +
        `<span class="bar-value">0.0%</span>`;
      frag.appendChild(row);
    }
    el.bars.appendChild(frag);
  }

  function render(data) {
    const { prediction, confidence, probabilities, activations } = data;

    el.verdict.textContent = String(prediction);
    el.verdict.classList.add('live');
    el.confidence.textContent = `${(confidence * 100).toFixed(1)}% confident it's a ${prediction}`;
    el.runner.textContent =
      `Next most likely: ${data.runner_up} at ${(data.runner_up_confidence * 100).toFixed(1)}%`;
    el.timing.textContent = `${data.milliseconds.toFixed(1)} ms`;

    for (const row of el.bars.children) {
      const digit = Number(row.dataset.digit);
      const p = probabilities[digit];
      row.querySelector('.bar-fill').style.width = `${(p * 100).toFixed(2)}%`;
      row.querySelector('.bar-value').textContent = `${(p * 100).toFixed(1)}%`;
      row.classList.toggle('top', digit === prediction);
    }

    renderActivations(activations);
  }

  function renderActivations(activations) {
    el.activations.replaceChildren();

    activations.forEach((act, index) => {
      const isOutput = index === activations.length - 1;
      const block = document.createElement('div');
      block.className = 'act-block';

      const head = document.createElement('div');
      head.className = 'act-head';
      head.innerHTML =
        `<span class="act-name">layer ${index}${isOutput ? ' (logits)' : ''} &middot; ${act.size} units</span>` +
        `<span class="act-stat">${(act.zero_fraction * 100).toFixed(0)}% silent</span>`;

      const strip = document.createElement('div');
      strip.className = 'act-strip';
      const scale = Math.max(act.max, 1e-9);
      for (const value of act.preview) {
        const cell = document.createElement('span');
        const normalised = Math.max(0, value) / scale;
        cell.className = value === 0 ? 'act-cell zero' : 'act-cell';
        cell.style.height = `${Math.max(2, normalised * 100)}%`;
        strip.appendChild(cell);
      }

      block.append(head, strip);
      el.activations.appendChild(block);
    });

    const hidden = activations.slice(0, -1);
    if (hidden.length) {
      const silent = hidden.reduce((a, l) => a + l.zero_fraction, 0) / hidden.length;
      el.sparsityNote.textContent =
        `On average ${(silent * 100).toFixed(0)}% of hidden units output exactly zero for ` +
        `this digit. That is ReLU: anything negative becomes 0 and passes no gradient. ` +
        `The network recognises digits using only the units that fire.`;
    }
  }

  function renderModel(info) {
    el.statArch.textContent = info.sizes.join(' → ');
    el.statParams.textContent = info.parameters.toLocaleString();
    if (info.best_val_accuracy) {
      el.statAcc.textContent = `${(info.best_val_accuracy * 100).toFixed(2)}%`;
      el.statAcc.parentElement.querySelector('dt').textContent = 'Val accuracy';
    }

    el.layers.replaceChildren();
    for (const layer of info.layers) {
      const item = document.createElement('li');
      item.className = 'layer';
      item.innerHTML =
        `<span class="layer-shape">${layer.n_in} → ${layer.n_out}</span>` +
        `<span class="layer-params">${layer.parameters.toLocaleString()} params</span>` +
        `<span class="layer-act">${layer.activation}</span>`;
      el.layers.appendChild(item);
    }
  }

  // ================================================================ misc

  let toastTimer = null;
  function showToast(message) {
    el.toast.textContent = message;
    el.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.toast.hidden = true; }, 5000);
  }

  function clearAll() {
    resetPad();
    clearPreview();
    hasInk = false;
    el.placeholder.hidden = false;
    el.verdict.textContent = '?';
    el.verdict.classList.remove('live');
    el.confidence.textContent = 'Waiting for a digit';
    el.runner.textContent = '';
    el.timing.innerHTML = '&nbsp;';
    for (const row of el.bars.children) {
      row.querySelector('.bar-fill').style.width = '0%';
      row.querySelector('.bar-value').textContent = '0.0%';
      row.classList.remove('top');
    }
    el.activations.replaceChildren(
      Object.assign(document.createElement('p'), {
        className: 'empty',
        textContent: 'Draw a digit to see the signal move through the network.',
      })
    );
    el.sparsityNote.textContent = '';
  }

  // A few digits sketched as normalised stroke paths, so the demo can
  // demonstrate itself without the visitor drawing anything.
  const EXAMPLES = [
    [[[0.5, 0.15], [0.5, 0.85]]],                                              // 1
    [[[0.3, 0.3], [0.5, 0.15], [0.7, 0.3], [0.7, 0.5], [0.3, 0.8], [0.72, 0.8]]],  // 2
    [[[0.3, 0.2], [0.68, 0.2], [0.45, 0.48], [0.7, 0.62], [0.5, 0.82], [0.3, 0.75]]], // 3
    [[[0.62, 0.15], [0.3, 0.58], [0.75, 0.58]], [[0.62, 0.15], [0.62, 0.85]]],  // 4
    [[[0.68, 0.18], [0.34, 0.18], [0.32, 0.45], [0.6, 0.45], [0.7, 0.65], [0.5, 0.83], [0.31, 0.76]]], // 5
    [[[0.66, 0.18], [0.38, 0.42], [0.34, 0.66], [0.52, 0.82], [0.68, 0.66], [0.54, 0.52], [0.36, 0.6]]], // 6
    [[[0.3, 0.18], [0.72, 0.18], [0.45, 0.85]]],                               // 7
    [[[0.5, 0.16], [0.33, 0.32], [0.5, 0.48], [0.68, 0.66], [0.5, 0.84], [0.33, 0.66], [0.5, 0.48], [0.67, 0.32], [0.5, 0.16]]], // 8
  ];

  function drawExample(index) {
    resetPad();
    hasInk = true;
    el.placeholder.hidden = true;
    ctx.lineWidth = Number(el.stroke.value);

    const pick = Number.isInteger(index)
      ? ((index % EXAMPLES.length) + EXAMPLES.length) % EXAMPLES.length
      : Math.floor(Math.random() * EXAMPLES.length);
    const strokes = EXAMPLES[pick];
    // Jitter so repeated clicks are not identical -- it should look drawn.
    const jitter = () => (Math.random() - 0.5) * 0.03;
    for (const stroke of strokes) {
      ctx.beginPath();
      stroke.forEach(([x, y], i) => {
        const px = (x + jitter()) * PAD;
        const py = (y + jitter()) * PAD;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }
    predict();
  }

  el.clear.addEventListener('click', clearAll);
  // Wrapped, not passed directly: the click Event would arrive as `index`.
  el.random.addEventListener('click', () => drawExample());
  el.stroke.addEventListener('change', () => { if (hasInk) schedulePredict(); });

  // Keyboard: the canvas is focusable, so give it something to do.
  pad.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      drawExample();
    } else if (event.key === 'Escape' || event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault();
      clearAll();
    }
  });

  async function init() {
    buildBars();
    clearAll();
    try {
      const response = await fetch('/api/model');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      renderModel(await response.json());
    } catch (err) {
      showToast(`Could not load model info: ${err.message}`);
    }

    // `?demo` draws an example immediately. Used for documentation
    // screenshots, so the populated state is reproducible rather than
    // depending on someone drawing at the right moment. `?demo=7` picks a
    // specific example instead of a random one.
    const requested = new URLSearchParams(location.search).get('demo');
    if (requested !== null) {
      const index = Number.parseInt(requested, 10);
      drawExample(Number.isInteger(index) ? index : undefined);
    }
  }

  init();
})();
