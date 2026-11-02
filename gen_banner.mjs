// Generates header banner for QuakeSentinel (1200x400)
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const OUT = resolve("D:/quakequack/QuakeSentinel/assets");
mkdirSync(OUT, { recursive: true });

const MONO = `'Consolas','JetBrains Mono',monospace`;
const DISP = `'Arial Black',Impact,sans-serif`;
const SANS = `'Segoe UI',system-ui,sans-serif`;

function quakeBanner() {
  const W = 1200, H = 400;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0b101c"/>
    <stop offset="0.6" stop-color="#140a0c"/>
    <stop offset="1" stop-color="#1c0709"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.8" cy="0.5" r="0.6">
    <stop offset="0" stop-color="#e8381b" stop-opacity="0.30"/>
    <stop offset="0.6" stop-color="#ff5a1f" stop-opacity="0.10"/>
    <stop offset="1" stop-color="#000000" stop-opacity="0"/>
  </radialGradient>
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#ffffff" stroke-opacity="0.04" stroke-width="1"/>
  </pattern>
  <style>
    .mono { font-family: ${MONO}; font-weight: 700; }
    .disp { font-family: ${DISP}; font-weight: 900; }
    .sans { font-family: ${SANS}; }
    .blink { animation: blink 1.2s steps(1,end) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
  </style>
</defs>

<!-- BG -->
<rect width="${W}" height="${H}" fill="url(#bg)"/>
<rect width="${W}" height="${H}" fill="url(#grid)"/>
<ellipse cx="960" cy="200" rx="360" ry="240" fill="url(#glow)"/>

<!-- Top Status Line -->
<g transform="translate(60, 48)">
  <rect x="0" y="0" width="140" height="26" rx="6" fill="#e8381b" fill-opacity="0.18" stroke="#e8381b" stroke-width="1.5"/>
  <circle cx="14" cy="13" r="4" fill="#e8381b" class="blink"/>
  <text x="26" y="17" class="mono" font-size="11" letter-spacing="1.5" fill="#ff7a1a">SEISMIC TRIAGE</text>

  <text x="156" y="17" class="mono" font-size="11" letter-spacing="1.5" fill="#8a9ba8">// AERIAL DAMAGE SEGMENTATION &#183; EFFICIENTVIT &#183; RESNET34</text>
</g>

<!-- Big Title -->
<g transform="translate(56, 146)">
  <text class="disp" font-size="84" letter-spacing="-1" fill="#e8381b" opacity="0.3" x="4" y="4">QUAKESENTINEL</text>
  <text class="disp" font-size="84" letter-spacing="-1" fill="#ffffff">QUAKESENTINEL</text>
  <text class="mono" font-size="16" letter-spacing="3" fill="#ff7a1a" x="650" y="-30">QuakeViT</text>
</g>

<!-- Subtitle & Specs -->
<g transform="translate(60, 202)">
  <text class="sans" font-size="20" font-weight="600" fill="#e2e8f0">Autonomous Aerial Earthquake Damage &amp; Structural Triage</text>
  <text class="sans" font-size="15" fill="#94a3b8" y="28">High-Resolution Multi-Scale Feature Pyramid Networks for Real-Time UAV Disaster Assessment</text>

  <!-- Spec Pills -->
  <g transform="translate(0, 75)">
    <!-- Pill 1 -->
    <rect x="0" y="0" width="160" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="80" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">EfficientViT-B2</text>

    <!-- Pill 2 -->
    <rect x="172" y="0" width="145" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="244" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">ResNet34-UNet</text>

    <!-- Pill 3 -->
    <rect x="329" y="0" width="150" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="404" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">MobileNetV4</text>

    <!-- Pill 4 -->
    <rect x="491" y="0" width="190" height="32" rx="7" fill="#e8381b" fill-opacity="0.15" stroke="#e8381b" stroke-width="1.2"/>
    <text x="586" y="20" text-anchor="middle" class="mono" font-size="11" fill="#ff7a1a">3-Class Triage Head</text>
  </g>

  <!-- Metric highlight -->
  <text x="0" y="145" class="mono" font-size="12" letter-spacing="1.5" fill="#64748b">xBD EARTHQUAKE BENCHMARK &#183; ONNX &amp; TORCHSCRIPT LITE (3.5MB) EXPORTS</text>
</g>

<!-- Right Side: Graphic Visualizer -->
<g transform="translate(860, 75)">
  <!-- Outer Box -->
  <rect x="0" y="0" width="280" height="260" rx="14" fill="#0c121e" stroke="#ffffff" stroke-opacity="0.12" stroke-width="2"/>
  
  <!-- Seismograph Graphic -->
  <path d="M 20 60 H 70 L 85 30 L 100 90 L 115 45 L 130 75 L 140 60 H 260" fill="none" stroke="#e8381b" stroke-width="2.5"/>
  <circle cx="140" cy="60" r="4" fill="#ffd23f"/>

  <!-- Triage Visual simulation -->
  <!-- Intact Building (Green) -->
  <rect x="30" y="110" width="85" height="85" rx="6" fill="#7ed321" fill-opacity="0.25" stroke="#7ed321" stroke-width="2"/>
  <rect x="30" y="110" width="60" height="18" rx="3" fill="#7ed321"/>
  <text x="36" y="123" class="mono" font-size="9" fill="#0a0f1b" font-weight="900">INTACT</text>

  <!-- Damaged/Rubble (Red) -->
  <rect x="135" y="120" width="115" height="75" rx="6" fill="#e8381b" fill-opacity="0.25" stroke="#e8381b" stroke-width="2" stroke-dasharray="4 4"/>
  <rect x="135" y="120" width="68" height="18" rx="3" fill="#e8381b"/>
  <text x="141" y="133" class="mono" font-size="9" fill="#ffffff" font-weight="900">DAMAGED</text>

  <!-- Footer Tag -->
  <rect x="15" y="215" width="250" height="30" rx="6" fill="#000000" fill-opacity="0.5"/>
  <text x="25" y="234" class="mono" font-size="10" fill="#94a3b8">LATENCY: <tspan fill="#7ed321">12ms</tspan> &#183; PARAMS: <tspan fill="#00d0ff">3.5M</tspan></text>
</g>

<!-- Accent Lines -->
<rect x="0" y="396" width="${W}" height="4" fill="#e8381b"/>
</svg>`;
}

writeFileSync(resolve(OUT, "quakesentinel-banner.svg"), quakeBanner(), "utf8");
console.log("QuakeSentinel banner created!");
