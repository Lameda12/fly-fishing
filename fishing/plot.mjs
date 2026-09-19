// A dependency-free SVG line chart, for the learning curve.
//
// matplotlib is not a dependency of this repository and neither is anything
// else, so the curve is drawn here. The palette matches the arena pages.

const PALETTE = {
  bg: "#0d1011",
  panel: "#15191a",
  line: "#303738",
  ink: "#edf2ef",
  muted: "#9ca8a3",
  series: ["#62c48d", "#e8bd57", "#66a7cb", "#e1695e"],
};

const escape = (text) =>
  String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function niceMax(value) {
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  const step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return step * magnitude;
}

/**
 * One panel: a shared x axis, one or more series, a y axis from 0 to `yMax`
 * (auto when omitted).
 */
function panel({ x, y, width, height, title, xs, series, yMax, yTicks = 4, xLabel }) {
  const maxX = Math.max(1, ...xs);
  const dataMax = Math.max(
    ...series.flatMap((s) =>
      s.constant !== undefined ? [s.constant] : s.values.filter(Number.isFinite),
    ),
    0,
  );
  const top = yMax ?? niceMax(dataMax || 1);
  const px = (value) => x + (value / maxX) * width;
  const py = (value) => y + height - (Math.min(value, top) / top) * height;

  const parts = [
    `<text x="${x}" y="${y - 10}" fill="${PALETTE.ink}" font-size="13">${escape(title)}</text>`,
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${PALETTE.panel}" stroke="${PALETTE.line}"/>`,
  ];

  for (let i = 0; i <= yTicks; i++) {
    const value = (top * i) / yTicks;
    const yy = py(value);
    parts.push(
      `<line x1="${x}" y1="${yy.toFixed(1)}" x2="${x + width}" y2="${yy.toFixed(1)}" stroke="${PALETTE.line}" stroke-width="1"/>`,
      `<text x="${x - 8}" y="${(yy + 4).toFixed(1)}" fill="${PALETTE.muted}" font-size="11" text-anchor="end">${
        top >= 10 ? value.toFixed(0) : value.toFixed(2)
      }</text>`,
    );
  }

  series.forEach((entry, index) => {
    const color = entry.color ?? PALETTE.series[index % PALETTE.series.length];
    if (entry.constant !== undefined) {
      const yy = py(entry.constant).toFixed(1);
      parts.push(
        `<line x1="${x}" y1="${yy}" x2="${x + width}" y2="${yy}" stroke="${color}" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.85"/>`,
      );
      return;
    }
    const points = entry.values
      .map((value, i) => (Number.isFinite(value) ? `${px(xs[i]).toFixed(1)},${py(value).toFixed(1)}` : null))
      .filter(Boolean)
      .join(" ");
    parts.push(
      `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linejoin="round"/>`,
    );
  });

  // Legend, inside the panel so it never collides with the axis labels.
  series.forEach((entry, index) => {
    const color = entry.color ?? PALETTE.series[index % PALETTE.series.length];
    const ly = y + 16 + index * 16;
    parts.push(
      `<line x1="${x + 12}" y1="${ly}" x2="${x + 30}" y2="${ly}" stroke="${color}" stroke-width="2" ${
        entry.constant !== undefined ? 'stroke-dasharray="5 4"' : ""
      }/>`,
      `<text x="${x + 36}" y="${ly + 4}" fill="${PALETTE.muted}" font-size="11">${escape(entry.label)}</text>`,
    );
  });

  for (let i = 0; i <= 4; i++) {
    const value = (maxX * i) / 4;
    parts.push(
      `<text x="${px(value).toFixed(1)}" y="${y + height + 16}" fill="${PALETTE.muted}" font-size="11" text-anchor="middle">${value.toFixed(0)}</text>`,
    );
  }
  if (xLabel) {
    parts.push(
      `<text x="${x + width / 2}" y="${y + height + 34}" fill="${PALETTE.muted}" font-size="11" text-anchor="middle">${escape(xLabel)}</text>`,
    );
  }
  return parts.join("\n  ");
}

/**
 * The learning curve: catch rate and false-hook rate against episode, with the
 * oracle and the rate-matched random control drawn in as reference lines.
 */
export function learningCurveSvg({ points, reference = {}, title, subtitle }) {
  const width = 860;
  const height = 620;
  const plotX = 64;
  const plotWidth = width - plotX - 28;
  const xs = points.map((p) => p.episode);

  const body = [
    panel({
      x: plotX,
      y: 68,
      width: plotWidth,
      height: 210,
      title: "Catch rate (fish caught / bites)",
      xs,
      yMax: 1,
      series: [
        { label: "trained readout (greedy eval)", values: points.map((p) => p.catchRate) },
        ...(reference.oracleCatchRate !== undefined
          ? [{ label: "oracle", constant: reference.oracleCatchRate, color: PALETTE.series[2] }]
          : []),
        ...(reference.randomCatchRate !== undefined
          ? [{ label: "random control", constant: reference.randomCatchRate, color: PALETTE.series[3] }]
          : []),
      ],
    }),
    panel({
      x: plotX,
      y: 360,
      width: plotWidth,
      height: 210,
      title: "False-hook rate (snapped lines per minute)",
      xs,
      series: [
        {
          label: "trained readout (greedy eval)",
          values: points.map((p) => p.falseHooksPerMinute),
          color: PALETTE.series[1],
        },
        ...(reference.randomFalseHooksPerMinute !== undefined
          ? [
              {
                label: "random control",
                constant: reference.randomFalseHooksPerMinute,
                color: PALETTE.series[3],
              },
            ]
          : []),
      ],
      xLabel: "training episode",
    }),
  ];

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">
  <rect width="${width}" height="${height}" fill="${PALETTE.bg}"/>
  <text x="${plotX}" y="32" fill="${PALETTE.ink}" font-size="16">${escape(title)}</text>
  <text x="${plotX}" y="50" fill="${PALETTE.muted}" font-size="11">${escape(subtitle)}</text>
  ${body.join("\n  ")}
</svg>
`;
}
