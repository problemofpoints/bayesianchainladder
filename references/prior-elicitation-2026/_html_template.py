"""HTML template for the single-page interactive report.

Renders a self-contained HTML document with Plotly.js loaded from a CDN,
the analysis data embedded as a JSON blob, and JavaScript that wires up
line/view filters to four charts plus the recommendations table.

Sections of the page:
  1. Headline recommendations table (filterable by line)
  2. GLM functional-form comparison (LOO by spec, per line)
  3. CSR posteriors (overlay strip plots per line)
  4. Rho diagnostics (rho with bootstrap CI; r_d decay)
  5. Per-line descriptive figures (existing PNGs as <img>)
"""
from __future__ import annotations

import json
from textwrap import dedent


def render_html(payload: dict) -> str:
    """Build the HTML string given the analysis payload dict.

    The payload should contain:
      - lines: list[str]  — line codes in display order
      - recs:  list[dict] — one row per line with the headline recommendations
      - glm:   list[dict] — long-format GLM rows (line, snl_id, spec, loo, max_rhat, status)
      - hier:  list[dict] — long-format M4 rows
      - csr:   list[dict] — long-format CSR posterior rows
      - rho:   list[dict] — one row per line with rho + diagnostics
      - desc:  list[dict] — one row per line of descriptive summary
    """
    payload_json = json.dumps(payload, default=_json_default)

    return dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Prior Elicitation 2026 — Interactive Report</title>
          <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
          <style>
            :root {{
              --bg: #fafafa;
              --fg: #222;
              --muted: #777;
              --border: #ddd;
              --accent: #2563eb;
            }}
            * {{ box-sizing: border-box; }}
            body {{
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
              margin: 0;
              padding: 2rem;
              background: var(--bg);
              color: var(--fg);
              line-height: 1.5;
            }}
            h1, h2, h3 {{ margin-top: 1.5rem; }}
            h1 {{ border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }}
            .controls {{
              display: flex;
              gap: 1rem;
              margin: 1rem 0;
              padding: 0.75rem;
              background: white;
              border: 1px solid var(--border);
              border-radius: 6px;
              flex-wrap: wrap;
              align-items: center;
            }}
            .controls label {{ font-weight: 500; margin-right: 0.25rem; }}
            select, button {{
              padding: 0.4rem 0.7rem;
              font-size: 0.95rem;
              border: 1px solid var(--border);
              border-radius: 4px;
              background: white;
              cursor: pointer;
            }}
            button {{ background: var(--accent); color: white; border-color: var(--accent); }}
            button:hover {{ filter: brightness(0.9); }}
            table {{
              width: 100%;
              border-collapse: collapse;
              background: white;
              margin: 1rem 0;
              font-size: 0.92rem;
              overflow: hidden;
              border-radius: 4px;
            }}
            th, td {{
              padding: 0.5rem 0.75rem;
              text-align: left;
              border-bottom: 1px solid var(--border);
            }}
            th {{ background: #f0f4f8; font-weight: 600; }}
            tr:hover td {{ background: #f8fafd; }}
            .chart {{
              background: white;
              border: 1px solid var(--border);
              border-radius: 6px;
              padding: 0.5rem;
              margin: 1rem 0;
              min-height: 320px;
            }}
            .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
            @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
            footer {{
              margin-top: 3rem;
              padding-top: 1rem;
              border-top: 1px solid var(--border);
              color: var(--muted);
              font-size: 0.85rem;
            }}
            code {{ background: #eef; padding: 0.1rem 0.3rem; border-radius: 3px; }}
            .scope-note {{
              padding: 0.6rem 1rem;
              background: #fff3cd;
              border-left: 4px solid #f0ad4e;
              margin: 1rem 0;
              border-radius: 4px;
              font-size: 0.9rem;
            }}
          </style>
        </head>
        <body>
          <h1>Prior Elicitation 2026 — Interactive Report</h1>
          <p>
            Empirically-grounded prior recommendations for
            <code>BayesianChainLadderGLM</code>, <code>BayesianCSR</code>, and
            <code>CorrelatedBootstrapODPSample</code>, derived from
            Schedule P YE2024 paid-loss triangles for OLO, OLC, CAL, WC, PPAL, CMP.
          </p>
          <p>
            See <a href="design.md">design.md</a> for methodology and
            <a href="README.md">README.md</a> for the static report.
          </p>

          <div class="scope-note">
            <strong>Note:</strong> M2 (B-spline on dev) was attempted but
            excluded after smoke testing showed ~100% NUTS divergences across
            all triangles, regardless of MCMC budget. The comparison is M1
            (full categorical) vs M3 (origin spline) vs M4 (hierarchical pool).
          </div>

          <div class="controls">
            <label for="line-filter">Filter by line:</label>
            <select id="line-filter">
              <option value="ALL">All lines</option>
            </select>
          </div>

          <h2>Headline Recommendations</h2>
          <div id="recs-table"></div>

          <h2>GLM Functional-Form Comparison (LOO by spec)</h2>
          <p>Higher LOO is better. Each dot is one (snl_id, spec) fit.</p>
          <div class="chart" id="glm-loo-chart"></div>

          <h2>CSR Posteriors</h2>
          <div class="grid-2">
            <div class="chart" id="csr-logelr-chart"></div>
            <div class="chart" id="csr-gamma-chart"></div>
          </div>

          <h2>Rho — Within-Triangle Calendar-Diagonal Correlation</h2>
          <p>
            Recommended values for the <code>rho=</code> argument of
            <code>CorrelatedBootstrapODPSample</code>. Bars show the bootstrap
            point estimate; whiskers show the 80% CI.
          </p>
          <div class="chart" id="rho-chart"></div>

          <h2>Descriptive Diagnostics</h2>
          <p>Per-line age-to-age, ULR, and variance-vs-mean diagnostics.</p>
          <div id="desc-figs"></div>

          <footer>
            Generated 2026-05-09 by
            <code>references/prior-elicitation-2026/05_synthesize.py</code>.
            Charts: Plotly.js. Filters update all charts simultaneously.
          </footer>

          <script id="payload" type="application/json">{payload_json}</script>
          <script>
            const PAYLOAD = JSON.parse(document.getElementById('payload').textContent);
            const LINES = PAYLOAD.lines;
            const lineFilter = document.getElementById('line-filter');
            LINES.forEach(line => {{
              const opt = document.createElement('option');
              opt.value = line; opt.textContent = line;
              lineFilter.appendChild(opt);
            }});

            function getActiveLines() {{
              const v = lineFilter.value;
              return v === 'ALL' ? LINES : [v];
            }}

            function renderRecsTable(active) {{
              const recs = PAYLOAD.recs.filter(r => active.includes(r.line));
              if (recs.length === 0) {{
                document.getElementById('recs-table').innerHTML = '<em>No data</em>';
                return;
              }}
              const cols = Object.keys(recs[0]);
              const fmt = v => (v === null || v === undefined) ? '' : (typeof v === 'number') ? v.toFixed(3) : v;
              let html = '<table><thead><tr>' +
                cols.map(c => `<th>${{c}}</th>`).join('') + '</tr></thead><tbody>';
              recs.forEach(r => {{
                html += '<tr>' + cols.map(c => `<td>${{fmt(r[c])}}</td>`).join('') + '</tr>';
              }});
              html += '</tbody></table>';
              document.getElementById('recs-table').innerHTML = html;
            }}

            function renderGlmChart(active) {{
              const rows = PAYLOAD.glm.filter(r => active.includes(r.line) && r.status === 'ok');
              if (rows.length === 0) {{
                document.getElementById('glm-loo-chart').innerHTML = '<p style="padding:1rem;color:var(--muted)">No converged GLM fits available.</p>';
                return;
              }}
              const specs = [...new Set(rows.map(r => r.spec))].sort();
              const traces = specs.map(spec => {{
                const sub = rows.filter(r => r.spec === spec);
                return {{
                  type: 'box',
                  name: spec,
                  y: sub.map(r => r.loo),
                  x: sub.map(r => r.line),
                  boxpoints: 'all',
                  jitter: 0.5,
                  pointpos: 0,
                  marker: {{ size: 5, opacity: 0.6 }},
                  hovertemplate: 'snl_id: %{{customdata}}<br>line: %{{x}}<br>LOO: %{{y:.1f}}<extra>'+spec+'</extra>',
                  customdata: sub.map(r => r.snl_id),
                }};
              }});
              Plotly.newPlot('glm-loo-chart', traces, {{
                boxmode: 'group',
                yaxis: {{ title: 'LOO (elpd_loo, higher = better)' }},
                xaxis: {{ title: 'Line' }},
                margin: {{ t: 10 }},
                legend: {{ orientation: 'h', y: 1.1 }},
              }}, {{ displayModeBar: false, responsive: true }});
            }}

            function renderCsrCharts(active) {{
              const rows = PAYLOAD.csr.filter(r => active.includes(r.line));
              if (rows.length === 0) {{
                document.getElementById('csr-logelr-chart').innerHTML = '<p style="padding:1rem;color:var(--muted)">No converged CSR fits available.</p>';
                document.getElementById('csr-gamma-chart').innerHTML = '';
                return;
              }}
              const lines = [...new Set(rows.map(r => r.line))].sort();

              // logelr chart
              const logelrTraces = lines.map(line => {{
                const sub = rows.filter(r => r.line === line);
                return {{
                  type: 'box',
                  name: line,
                  y: sub.map(r => r.logelr_mean),
                  boxpoints: 'all',
                  jitter: 0.4,
                  pointpos: 0,
                  marker: {{ size: 5 }},
                  hovertemplate: 'snl_id: %{{customdata}}<br>logelr: %{{y:.3f}}<extra>'+line+'</extra>',
                  customdata: sub.map(r => r.snl_id),
                }};
              }});
              Plotly.newPlot('csr-logelr-chart', logelrTraces, {{
                yaxis: {{ title: 'CSR logelr (per-fit posterior mean)' }},
                xaxis: {{ title: 'Line' }},
                margin: {{ t: 10 }},
                showlegend: false,
              }}, {{ displayModeBar: false, responsive: true }});

              // gamma chart
              const gammaTraces = lines.map(line => {{
                const sub = rows.filter(r => r.line === line);
                return {{
                  type: 'box',
                  name: line,
                  y: sub.map(r => r.gamma_mean),
                  boxpoints: 'all',
                  jitter: 0.4,
                  pointpos: 0,
                  marker: {{ size: 5 }},
                  hovertemplate: 'snl_id: %{{customdata}}<br>gamma: %{{y:.4f}}<extra>'+line+'</extra>',
                  customdata: sub.map(r => r.snl_id),
                }};
              }});
              Plotly.newPlot('csr-gamma-chart', gammaTraces, {{
                yaxis: {{ title: 'CSR gamma (per-fit posterior mean)' }},
                xaxis: {{ title: 'Line' }},
                margin: {{ t: 10 }},
                showlegend: false,
              }}, {{ displayModeBar: false, responsive: true }});
            }}

            function renderRhoChart(activeLines) {{
              const rows = PAYLOAD.rho.filter(r => activeLines.includes(r.line));
              if (rows.length === 0) {{
                document.getElementById('rho-chart').innerHTML = '<p style="padding:1rem;color:var(--muted)">No rho data available.</p>';
                return;
              }}
              // Some lines may have NaN rho (e.g. OLC had no pairs). Filter those out for the chart.
              const valid = rows.filter(r => r.rho_point !== null && r.rho_point !== undefined);
              const trace = {{
                type: 'bar',
                x: valid.map(r => r.line),
                y: valid.map(r => r.rho_point),
                error_y: {{
                  type: 'data',
                  symmetric: false,
                  array: valid.map(r => (r.rho_p90 !== null && r.rho_point !== null) ? r.rho_p90 - r.rho_point : 0),
                  arrayminus: valid.map(r => (r.rho_p10 !== null && r.rho_point !== null) ? r.rho_point - r.rho_p10 : 0),
                }},
                marker: {{ color: '#2563eb' }},
                hovertemplate: 'line: %{{x}}<br>rho: %{{y:.3f}}<extra></extra>',
              }};
              Plotly.newPlot('rho-chart', [trace], {{
                yaxis: {{ title: 'rho (within-triangle, same calendar diagonal)', range: [0, 0.35] }},
                xaxis: {{ title: 'Line' }},
                margin: {{ t: 10 }},
              }}, {{ displayModeBar: false, responsive: true }});
            }}

            function renderDescFigs(activeLines) {{
              const html = activeLines.map(line =>
                `<h3>${{line}}</h3><img src="figures/01_descriptive_${{line}}.png" alt="${{line}} descriptive" style="max-width:100%;border:1px solid var(--border);">`
              ).join('');
              document.getElementById('desc-figs').innerHTML = html;
            }}

            function renderAll() {{
              const a = getActiveLines();
              renderRecsTable(a);
              renderGlmChart(a);
              renderCsrCharts(a);
              renderRhoChart(a);
              renderDescFigs(a);
            }}

            lineFilter.addEventListener('change', renderAll);
            renderAll();
          </script>
        </body>
        </html>
        """
    )


def _json_default(o):
    """JSON serializer for numpy scalars and pandas NaNs."""
    import math

    if isinstance(o, float) and math.isnan(o):
        return None
    try:
        import numpy as np

        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)
