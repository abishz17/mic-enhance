"""Self-contained local HTML report: metric table, spectrograms, audio players.

Nothing is uploaded anywhere; audio and images are embedded as data URIs.
"""

from __future__ import annotations

import base64
import html
import io
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import Audio, AudioError, load_wav
from .score import RELIABILITY, TARGET_LUFS, TARGET_TP, score_audio

SPEC_VMIN, SPEC_VMAX = -130.0, -30.0  # fixed dBFS/bin scale so variants are comparable
MAX_EMBED_SECONDS = 600.0


def _labels(dirs: list[Path]) -> dict[Path, str]:
    """Variant label per directory: the relative path as typed (e.g. `corpus`,
    `ml/out/deepfilternet3/corpus`); absolute paths get their shortest
    trailing suffix that is unique (e.g. `mic-enhance-clean/corpus`)."""
    out: dict[Path, str] = {}
    for d in dirs:
        if not d.is_absolute():
            out[d] = str(d).rstrip("/") or "."
            continue
        parts = d.parts
        for k in range(1, len(parts) + 1):
            lab = "/".join(parts[-k:])
            if all(o == d or not str(o).rstrip("/").endswith(lab) for o in dirs):
                out[d] = lab
                break
        else:
            out[d] = str(d)
    return out


def collect(inputs: list[str]) -> dict[str, list[tuple[str, Path]]]:
    """Group inputs for side-by-side comparison.

    - Only files given: one group, each file is a variant (label = file stem).
    - Directories given: files are grouped by stem across directories, so
      corpus/take1.wav, out/ours/take1.wav, ml/out/<model>/corpus/take1.wav
      compare together; the label is the shortest unique trailing dir path.
    """
    paths = [Path(p) for p in inputs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise AudioError(f"not found: {', '.join(map(str, missing))}")
    if all(p.is_file() for p in paths):
        return {"files": [(p.stem, p) for p in paths]}
    dirs = list(dict.fromkeys(p if p.is_dir() else p.parent for p in paths))
    labels = _labels(dirs)
    groups: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for p in paths:
        files = sorted(p.glob("*.wav")) if p.is_dir() else [p]
        for f in files:
            groups[f.stem].append((labels[p if p.is_dir() else p.parent], f))
    if not groups:
        raise AudioError("no .wav files found in the given inputs")
    return dict(sorted(groups.items()))


def _spectrogram_png(a: Audio, title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = a.samples
    nfft, hop = 2048, 512
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    idx = np.arange(0, len(x) - nfft + 1, hop)
    fr = np.lib.stride_tricks.sliding_window_view(x, nfft)[idx] * np.hanning(nfft)
    # scale so a full-scale sine reads ~0 dB
    S = np.abs(np.fft.rfft(fr, axis=1)) / (np.hanning(nfft).sum() / 2)
    S_db = 20 * np.log10(S + 1e-12).T
    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=100)
    ext = (0.0, len(x) / a.sr, 0.0, a.sr / 2000.0)
    im = ax.imshow(S_db, origin="lower", aspect="auto", extent=ext, vmin=SPEC_VMIN, vmax=SPEC_VMAX, cmap="magma")
    ax.set_ylabel("kHz")
    ax.set_xlabel("s")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01, label="dBFS")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _audio_uri(a: Audio) -> str | None:
    if a.duration > MAX_EMBED_SECONDS:
        return None
    buf = io.BytesIO()
    # FLAC 24-bit: lossless at the precision of the corpus, ~half the size of WAV.
    sf.write(buf, np.clip(a.samples, -1.0, 1.0), a.sr, format="FLAC", subtype="PCM_24")
    return "data:audio/flac;base64," + base64.b64encode(buf.getvalue()).decode()


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "–"
    if isinstance(v, str):
        return html.escape(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return "–"
    return f"{v:.{nd}f}"


COLUMNS: list[tuple[str, str, int]] = [
    ("voice_verdict", "Voice identity", 0),
    ("speaker_similarity", "Speaker sim (ECAPA)", 3),
    ("f0_shift_st", "F0 shift st", 2),
    ("f0_range_change_st", "F0 range &Delta; st", 2),
    ("tilt_change_db_oct", "Tilt &Delta; dB/oct", 2),
    ("words_verdict", "Words guard", 0),
    ("presence_verdict", "New sound in pauses", 0),
    ("lufs_i", "LUFS-I", 2),
    ("true_peak_dbtp", "True peak dBTP", 2),
    ("noise_floor_dbfs", "Noise floor dBFS", 1),
    ("lead_noise_dbfs", "Lead room tone dBFS", 1),
    ("snr_db", "SNR dB", 1),
    ("bandwidth_hz", "Bandwidth Hz", 0),
    ("bandwidth_limited_by", "BW limited by", 0),
    ("clipped_samples", "Clipped samples", 0),
    ("dnsmos_sig", "DNSMOS SIG", 2),
    ("dnsmos_bak", "DNSMOS BAK", 2),
    ("dnsmos_ovrl", "DNSMOS OVRL", 2),
    ("dnsmos_p808", "DNSMOS P.808", 2),
    ("pesq_wb", "PESQ-WB vs clean", 2),
    ("si_sdr_db", "SI-SDR dB vs clean", 1),
    ("estoi", "ESTOI vs clean", 3),
]

CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#ddd9d0;--card:#fff;--good:#1f7a3a;--bad:#b42318;--acc:#3451b2}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a988f;--line:#34332f;--card:#1f1f1d;--good:#5cc77d;--bad:#ff7b6b;--acc:#8fa6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text",Helvetica,Arial,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}
.mut{color:var(--mut)}.tw{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--card)}th{font-weight:600;font-size:12px;color:var(--mut)}
.good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}
.var{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:12px;margin:12px 0}
.var img{width:100%;height:auto;display:block;border-radius:4px}audio{width:100%;margin-top:8px}
.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:12px 0}label{cursor:pointer}
ul.rel{padding-left:18px}ul.rel li{margin:2px 0}code{font-size:12px}
"""

JS = """
(function(){
 const box=document.getElementById('match');
 function apply(){
  const players=[...document.querySelectorAll('audio[data-lufs]')];
  const groups={};
  players.forEach(p=>{(groups[p.dataset.group]=groups[p.dataset.group]||[]).push(p)});
  Object.values(groups).forEach(ps=>{
   const ls=ps.map(p=>parseFloat(p.dataset.lufs)).filter(Number.isFinite);
   const tgt=Math.min(...ls);
   ps.forEach(p=>{const l=parseFloat(p.dataset.lufs);
     p.volume=(box.checked&&Number.isFinite(l))?Math.min(1,Math.pow(10,(tgt-l)/20)):1;
     const t=p.parentElement.querySelector('.gain');
     if(t)t.textContent=box.checked&&Number.isFinite(l)?('playback gain '+(tgt-l).toFixed(1)+' dB'):'';});
  });
 }
 box.addEventListener('change',apply);apply();
 document.addEventListener('play',e=>{document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);
})();
"""


def build_report(
    inputs: list[str],
    out: Path,
    *,
    clean_ref: str | None = None,
    lead_silence_s: float | None = None,
    with_dnsmos: bool = True,
    downmix: bool = False,
    ref_variant: str | None = None,
    with_voice: bool = True,
    with_guard: bool = False,
) -> list[dict[str, Any]]:
    """Each group's reference (raw input) is the variant labelled `ref_variant`,
    else one labelled raw/corpus/input, else the first; every other variant is
    checked against it for voice identity (and words with `with_guard`)."""
    groups = collect(inputs)
    clean = load_wav(clean_ref, downmix=downmix) if clean_ref else None
    all_rows: list[dict[str, Any]] = []
    sections: list[str] = []
    for gi, (gname, items) in enumerate(groups.items()):
        rows: list[tuple[str, dict[str, Any], Audio]] = []
        skipped: list[str] = []
        for label, path in items:
            try:
                a = load_wav(path, downmix=downmix)
            except AudioError as e:  # one bad variant must not kill the whole report
                skipped.append(f"{label}: skipped, {e}")
                continue
            r = score_audio(a, lead_silence_s=lead_silence_s, with_dnsmos=with_dnsmos)
            if clean is not None:
                from .intrusive import intrusive

                ii = intrusive(clean.samples, a.samples, a.sr)
                r.update({"pesq_wb": ii.pesq_wb, "si_sdr_db": ii.si_sdr_db if math.isfinite(ii.si_sdr_db) else None, "estoi": ii.estoi})
            r["variant"] = label
            rows.append((label, r, a))
            all_rows.append(r)
        if not rows:
            sections.append(f"<h2>{html.escape(gname)}</h2><ul>" + "".join(f"<li class='bad'>{html.escape(x)}</li>" for x in skipped) + "</ul>")
            continue
        names = [lbl for lbl, _r, _a in rows]
        want = [ref_variant] if ref_variant else ["raw", "corpus", "input"]
        ref_i = next((names.index(n) for n in want if n in names), 0)
        ref_label, ref_row, ref_audio = rows[ref_i]
        ref_row["voice_verdict"] = "reference"
        for label, r, a in rows:
            if r is ref_row:
                continue
            if with_voice:
                from .voice import voice_check

                v = voice_check(ref_audio.samples, a.samples, a.sr)
                r.update(
                    voice_verdict=v.verdict,
                    speaker_similarity=v.speaker_similarity,
                    f0_shift_st=v.f0_shift_st,
                    f0_range_change_st=v.f0_range_change_st,
                    tilt_change_db_oct=v.tilt_change_db_oct,
                    voice_reasons=v.reasons,
                )
            if with_guard:
                from .asr_guard import guard_verdict, speech_presence_check, transcribe

                g = guard_verdict(transcribe(ref_audio.samples, a.sr), transcribe(a.samples, a.sr))
                pr = speech_presence_check(ref_audio.samples, a.samples, a.sr)
                r.update(words_verdict=g.verdict, presence_verdict=pr.verdict, guard_reasons=g.reasons + pr.reasons)
        cols = [c for c in COLUMNS if any(r.get(c[0]) is not None for _l, r, _a in rows)]
        th = "".join(f"<th>{html.escape(h)}</th>" for _k, h, _n in cols)
        trs = []
        for label, r, _a in rows:
            tds = []
            for k, _h, nd in cols:
                v = r.get(k)
                cls = ""
                if k == "lufs_i" and isinstance(v, float):
                    cls = "good" if abs(v - TARGET_LUFS) <= 1.0 else "bad"
                if k == "true_peak_dbtp" and isinstance(v, float):
                    cls = "good" if v <= TARGET_TP else "bad"
                if k == "clipped_samples" and v:
                    cls = "bad"
                if k.endswith("_verdict") and v in ("PASS", "FAIL"):
                    cls = "good" if v == "PASS" else "bad"
                tds.append(f'<td class="{cls}">{_fmt(v, nd)}</td>')
            name = f"{html.escape(label)}<br><span class='mut'>{html.escape(Path(r['file']).name)}</span>"
            trs.append(f"<tr><td>{name}</td>{''.join(tds)}</tr>")
        probs = "".join(
            f"<li class='bad'>{html.escape(label)}: {html.escape(p)}</li>"
            for label, r, _a in rows
            for p in r.get("problems", []) + r.get("voice_reasons", []) + r.get("guard_reasons", [])
        ) + "".join(f"<li class='bad'>{html.escape(x)}</li>" for x in skipped) + "".join(f"<li class='mut'>{html.escape(label)}: {html.escape(w)}</li>" for label, r, _a in rows for w in r.get("warnings", []))
        cards = []
        for label, r, a in rows:
            img = _spectrogram_png(a, f"{label} / {a.path.name}")
            uri = _audio_uri(a)
            lufs = r.get("lufs_i")
            lufs_attr = f"{lufs:.3f}" if isinstance(lufs, float) else "nan"
            player = (
                f'<audio controls preload="none" data-group="g{gi}" data-lufs="{lufs_attr}" src="{uri}"></audio><div class="mut gain"></div>'
                if uri
                else f"<p class='mut'>audio longer than {MAX_EMBED_SECONDS:.0f} s, not embedded: {html.escape(str(a.path))}</p>"
            )
            cards.append(f'<div class="var"><strong>{html.escape(label)}</strong> <span class="mut">{html.escape(str(a.path))}</span><img alt="spectrogram" src="{img}">{player}</div>')
        sections.append(
            f"<h2>{html.escape(gname)}</h2><p class='mut'>Voice identity and word checks compare each variant against <strong>{html.escape(ref_label)}</strong>.</p><div class='tw'><table><thead><tr><th>Variant</th>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
            + (f"<ul>{probs}</ul>" if probs else "")
            + "".join(cards)
        )
    from .voice import THRESHOLDS

    rel = "".join(f"<li><code>{html.escape(k)}</code>: {html.escape(v)}</li>" for k, v in RELIABILITY.items())
    rel += "".join(f"<li><code>{html.escape(k)}</code> = {v}</li>" for k, v in THRESHOLDS.items())
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mic Enhance Report</title><style>{CSS}</style></head><body><main>
<h1>Mic Enhance Report</h1>
<p class="mut">Target: {TARGET_LUFS:.0f} LUFS integrated (&plusmn;1 LU), true peak &le; {TARGET_TP:.0f} dBTP. Spectrograms share one absolute scale ({SPEC_VMIN:.0f} to {SPEC_VMAX:.0f} dBFS per bin).
{"Intrusive metrics against clean reference: " + html.escape(str(clean_ref)) if clean_ref else ""}</p>
<div class="bar"><label><input type="checkbox" id="match" checked> Loudness-match players within each group (attenuates louder variants to the quietest; louder always sounds better)</label></div>
{''.join(sections)}
<h2>How far to trust each metric</h2><ul class="rel">{rel}</ul>
</main><script>{JS}</script></body></html>"""
    out.write_text(doc, encoding="utf-8")
    return all_rows
