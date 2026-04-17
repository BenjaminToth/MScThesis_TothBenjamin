#!/usr/bin/env python3
import os
import json
import base64
import textwrap
import mne
import tempfile
from scipy.signal import welch
from fire import Fire
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import numpy as np  

"""
Annotation Colors Configuration
================================
Maps EEG artifact annotation labels to RGBA color strings. Colors are organized as:
- Single labels: distinct base colors
- Compound labels: channel-wise RGB average of the components
- Alpha fixed at 0.30 for opacity consistency
"""
ANNOT_COLORS = {
    'eyem':     'rgba( 31, 119, 180, 0.30)',  
    'musc':     'rgba(255, 127,  14, 0.30)',  
    'elec':     'rgba(148, 103, 189, 0.30)',  
    'chew':     'rgba( 44, 160,  44, 0.30)',  
    'shiv':     'rgba(214,  39,  40, 0.30)',  
    'artifact': 'rgba(127, 127, 127, 0.30)',  
    'elpp':     'rgba( 23, 190, 207, 0.30)',  

    'eyem_musc': 'rgba(143, 123,  97, 0.30)',
    'musc_elec': 'rgba(202, 115, 102, 0.30)',
    'eyem_elec': 'rgba( 90, 111, 184, 0.30)',
    'eyem_chew': 'rgba( 38, 140, 112, 0.30)',
    'chew_elec': 'rgba( 96, 132, 116, 0.30)',
    'chew_musc': 'rgba(150, 144,  29, 0.30)',
    'eyem_shiv': 'rgba(122,  79, 110, 0.30)',
    'shiv_elec': 'rgba(181,  71, 114, 0.30)',
}

"""
Annotation Patterns Configuration
==================================
Maps annotation labels to hatch patterns for visual distinction. These patterns
are overlaid on the background colors to improve accessibility and visual clarity.
"""
ANNOT_PATTERNS = {
    'eyem':     '-',
    'musc':     '|',
    'elec':     '/',
    'chew':     '\\',
    'shiv':     '.',
    'artifact': '',
    'elpp':     '.',
    'eyem_musc': '+',
    'musc_elec': 'x',
    'eyem_elec': 'x',
    'eyem_chew': 'x',
    'chew_elec': 'x',
    'chew_musc': 'x',
    'eyem_shiv': 'x',
    'shiv_elec': 'x',
}
def compute_psd(raw_or_path, t_window=1.0):
    """
    Compute power spectral density (PSD) for each channel using Welch's method.
    
    Reads an EDF file with MNE (or uses provided Raw object) and computes PSD
    for all non-annotation channels. Also computes PSD of the time-domain average
    across all channels.
    
    Parameters
    ----------
    raw_or_path : str or mne.io.Raw
        Path to EDF file or MNE Raw object
    t_window : float
        Window duration in seconds for Welch method (default: 1.0)
    
    Returns
    -------
    list[dict]
        List of dicts with keys 'label', 'f' (frequencies), 'psd' (power values)
    """
    if isinstance(raw_or_path, str):
        raw = mne.io.read_raw_edf(raw_or_path, preload=True, verbose=False)
    else:
        raw = raw_or_path
    
    annotation_channels = [ch for ch in raw.ch_names if ch.strip().upper() == 'ANOT']
    if annotation_channels:
        raw = raw.copy().drop_channels(annotation_channels)
    
    channel_data = raw.get_data()
    sampling_freq = raw.info['sfreq']
    psd_results = []
    
    for channel_idx, channel_label in enumerate(raw.ch_names):
        signal = channel_data[channel_idx]
        frequencies, power_spectral_density = welch(
            signal,
            fs=sampling_freq,
            nperseg=int(sampling_freq * t_window),
            noverlap=0
        )
        psd_results.append({
            'label': channel_label,
            'f': frequencies.tolist(),
            'psd': power_spectral_density.tolist()
        })
    
    average_signal = channel_data.mean(axis=0)
    frequencies_avg, power_avg = welch(
        average_signal,
        fs=sampling_freq,
        nperseg=int(sampling_freq * t_window),
        noverlap=0
    )
    psd_results.append({
        'label': 'average',
        'f': frequencies_avg.tolist(),
        'psd': power_avg.tolist()
    })
    return psd_results

def cyclic_fill(lst, length, default=None):
    """
    Cycle through a list to fill a target length by repeating elements.
    
    Parameters
    ----------
    lst : list, str, or None
        Input list (or single string). If None, use default value.
    length : int
        Target output length
    default : any
        Default value to use if lst is None
    
    Returns
    -------
    list
        List of length `length` with elements cycled from input
    """
    if lst is None:
        lst = default
    if isinstance(lst, str):
        lst = [lst]
    return [lst[i % len(lst)] for i in range(length)]

def edf2html(inputs, output_html, colors=['#1f77b4', '#d62728'], maxfreq=40, gains=None):
    """
    Generate an interactive HTML viewer for EDF files with annotations.
    
    Parameters
    ----------
    inputs : str, mne.io.Raw, or list
        EDF file path(s) or MNE Raw object(s)
    output_html : str
        Output HTML file path
    colors : list[str]
        List of line colors (hex) for each input. Will cycle if fewer than inputs.
    maxfreq : float
        Maximum frequency for PSD display (not currently used)
    gains : list[float] or None
        Amplification factors for each input. Defaults to [1, 1, ...]
    """
    if isinstance(inputs, (str, mne.io.BaseRaw)):
        inputs = [inputs]
    colors = cyclic_fill(colors, len(inputs))

    if gains is None:
        gains = [1] * len(inputs)
    elif not isinstance(gains, list):
        raise ValueError("gains must be a list of length equal to the number of input EDFs")
    elif len(gains) != len(inputs):
        raise ValueError("gains list must have the same length as inputs")
    gains = [float(g) for g in gains]

    base64_edfs = []
    file_names = []
    for input_item in inputs:
        if isinstance(input_item, str):
            file_path = input_item
            with open(file_path, 'rb') as f:
                edf_bytes = f.read()
            file_name = os.path.basename(file_path)
        else:
            raw = input_item
            temp_file = tempfile.NamedTemporaryFile(suffix='.edf', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            raw.export(temp_path, fmt='edf', overwrite=True)
            with open(temp_path, 'rb') as f:
                edf_bytes = f.read()
            file_name = os.path.basename(temp_path)
        base64_edfs.append(base64.b64encode(edf_bytes).decode('ascii'))
        file_names.append(file_name)

    all_psds = []

    annotations_list = []
    for input_item in inputs:
        if isinstance(input_item, str):
            raw_annotations = mne.io.read_raw_edf(input_item, preload=False, verbose=False)
        else:
            raw_annotations = input_item
        annotation_events = []
        for onset, duration, description in zip(raw_annotations.annotations.onset,
                                                 raw_annotations.annotations.duration,
                                                 raw_annotations.annotations.description):
            annotation_events.append({
                'onset': float(onset),
                'duration': float(duration),
                'description': description
            })
        annotations_list.append(annotation_events)

    gain_controls = ''
    for file_idx, (file_name, gain_value) in enumerate(zip(file_names, gains)):
        gain_controls += f'            <label for="gain{file_idx}">Gain ({file_name}):</label>\n'
        gain_controls += f'            <input type="number" id="gain{file_idx}" step="{gain_value * 0.1}" value="{gain_value}" min="0" max="999"/>\n'

    html_template = textwrap.dedent("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8"/>
        <title>EDF Viewer</title>
        <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/edfdecoder@0.1.2/dist/edfdecoder.umd.min.js" crossorigin="anonymous"></script>
        <style>
            body, html { margin:0; padding:0; height:100%; font-family:sans-serif; }
            #controls { padding:0; display:flex; gap:20px; align-items:center; padding-left: 20px; flex-wrap: wrap;}
            #plot { width:100%; height:calc(100% - 40px); }
            label, select, input, #freqDisplay {font-size:14px;}
        </style>
        </head>
        <body>
        <div id="controls">
            <label for="subsample">Subsampling:</label>
            <select id="subsample">
            <option>1</option><option>2</option><option>4</option><option>8</option>
            <option>16</option><option>32</option><option selected>64</option>
            </select>
            <span id="freqDisplay"></span>
___GAIN_CONTROLS___
            <!-- Toggles -->
            <label><input type="checkbox" id="showBg" checked> Highlight BG</label>
            <label><input type="checkbox" id="showPat" checked> Patterns</label>
        </div>

        <div id="plot"></div>

        <script>
        const base64EDFs        = ___BASE64_EDF___;
        const monoColors        = ___COLORS___;
        const fileNames         = ___FILENAMES___;
        const annotationsData   = ___ANNOTATIONS___;
        const annotColors       = ___ANNOT_COLORS___;
        const annotPatterns     = ___ANNOT_PATTERNS___;
        const N                 = base64EDFs.length;

        function base64ToArrayBuffer(b64) {
            const bin = atob(b64), buf = new Uint8Array(bin.length);
            for (let i=0; i<bin.length; i++) buf[i] = bin.charCodeAt(i);
            return buf.buffer;
        }

        function parseRecordDuration(buf) {
            const hdr = new TextDecoder('ascii').decode(new Uint8Array(buf, 0, 256));
            return parseFloat(hdr.substr(244, 8).trim());
        }

        // Parse EDFs once and cache signals and labels only
        const parsedEDFs = (function() {
            let parsed = [], offsets, chNames, maxAmplitudes = [];
            base64EDFs.forEach((b64, fileIdx) => {
                const buf    = base64ToArrayBuffer(b64),
                      recDur = parseRecordDuration(buf),
                      dec    = new edfdecoder.EdfDecoder();
                dec.setInput(buf); dec.decode();
                const edf   = dec.getOutput(),
                      nSig  = edf.getNumberOfSignals(),
                      nRec  = edf.getNumberOfRecords();

                let sigs = [], labels = [], ptps = [], baseSr = null;
                for (let i = 0; i < nSig; i++) {
                    const label = edf.getSignalLabel(i);
                    if (/annotation/i.test(label)) continue;
                    const arr = edf.getPhysicalSignalConcatRecords(i, 0, nRec);
                    if (baseSr === null) baseSr = arr.length / (nRec * recDur);
                    let mn = Infinity, mx = -Infinity;
                    arr.forEach(v => { mn = Math.min(mn, v); mx = Math.max(mx, v); });
                    ptps.push(mx - mn);
                    maxAmplitudes.push(Math.max(Math.abs(mn), Math.abs(mx)));
                    sigs.push(arr);
                    labels.push(label);
                }

                // compute raw average channel
                if (sigs.length > 0) {
                    const nPts = sigs[0].length;
                    const avgArr = Array(nPts).fill(0);
                    for (let j = 0; j < nPts; j++) {
                        let sum = 0;
                        for (let k = 0; k < sigs.length; k++) {
                            sum += sigs[k][j];
                        }
                        avgArr[j] = sum / sigs.length;
                    }
                    sigs.push(avgArr);
                    labels.push('average');
                    let avgMin = Infinity, avgMax = -Infinity;
                    avgArr.forEach(v => { avgMin = Math.min(avgMin, v); avgMax = Math.max(avgMax, v); });
                    maxAmplitudes.push(Math.max(Math.abs(avgMin), Math.abs(avgMax)));
                }

                if (fileIdx === 0) {
                    const maxAmp = Math.max(...maxAmplitudes);
                    const spacing = maxAmp * 2.5; // spacing to prevent overlap
                    offsets = labels.map((_, i) => (labels.length - 1 - i) * spacing);
                    chNames = labels;
                }
                parsed.push({ sigs: sigs, labels: labels, baseSr: baseSr });
            });
            return { parsed: parsed, offsets: offsets, chNames: chNames, maxAmplitudes: maxAmplitudes };
        })();

        function parseAnnotationChannel(description) {
            const parts = description.split(':');
            if (parts.length >= 2) {
                const channelName = parts[0].trim();
                const chIdx = parsedEDFs.chNames.findIndex(ch => ch.trim() === channelName);
                return chIdx; // -1 if not found
            }
            return -1; // applies to all channels
        }

        /* ────────────────────────────────────────────────────────────────
           Colour utilities for merging overlaps
        */
        function rgbaStringToObj(str) {
            const m = str.match(/rgba\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*([0-9.]+)\\s*\\)/i);
            if (!m) return {r:0,g:0,b:0,a:0};
            return {r:+m[1], g:+m[2], b:+m[3], a:+m[4]};
        }

        function objToRgba(o) {
            return `rgba(${Math.round(o.r)},${Math.round(o.g)},${Math.round(o.b)},${o.a.toFixed(3)})`;
        }

        function mixTwo(c1str, c2str) {
            const c1 = rgbaStringToObj(c1str), c2 = rgbaStringToObj(c2str);
            const aNew = Math.min((c1.a + c2.a) * 0.75, 1);
            const denom = (c1.a + c2.a) || 1;
            const r = (c1.r*c1.a + c2.r*c2.a) / denom;
            const g = (c1.g*c1.a + c2.g*c2.a) / denom;
            const b = (c1.b*c1.a + c2.b*c2.a) / denom;
            return objToRgba({r:r, g:g, b:b, a:aNew});
        }

        function mixMany(arr) {
            if (arr.length === 1) return arr[0];
            return arr.reduce((acc, c) => mixTwo(acc, c));
        }

        function patternFgColor(bgColorStr){
            const o = rgbaStringToObj(bgColorStr);
            const newA = (1 + o.a) / 2;
            return `rgba(${o.r},${o.g},${o.b},${newA})`;
        }
        /* ──────────────────────────────────────────────────────────────── */

        function redraw() {
            const sliderHeight = 0.12;
            const gd = document.getElementById('plot');
            // Capture both x and y ranges before any changes
            const prevXRange = (gd.layout && gd.layout.xaxis1 && gd.layout.xaxis1.range)
                ? gd.layout.xaxis1.range.slice() : null;
            const prevYRange = (gd.layout && gd.layout.yaxis1 && gd.layout.yaxis1.range)
                ? gd.layout.yaxis1.range.slice() : null;

            // collect per-file gain values
            const eegScales = [];
            for (let fi = 0; fi < N; fi++) {
                eegScales.push(+document.getElementById('gain'+fi).value);
            }

            // toggles
            const showBg  = document.getElementById('showBg').checked;
            const showPat = document.getElementById('showPat').checked;

            // compute y-range for main plot
            const offsetsArr = parsedEDFs.offsets;
            const padVal  = offsetsArr.length > 1
                ? Math.abs(offsetsArr[0] - offsetsArr[1])
                : (offsetsArr[0] || 1);
            const yLow    = Math.min(...offsetsArr) - padVal;
            const yHigh   = Math.max(...offsetsArr) + padVal;

            // Calculate channel spacing for annotation boxes
            const channelSpacing = offsetsArr.length > 1 
                ? Math.abs(offsetsArr[0] - offsetsArr[1])
                : padVal;
            const channelHeight = channelSpacing * 0.8;

            /* ────────────────────────────────────────────────────────────
               Build per-channel interval lists, merge for background,
               then overlay individual pattern rectangles.
            */
            let intervalsPerChannel = new Map();
            parsedEDFs.chNames.forEach((_, ci) => intervalsPerChannel.set(ci, []));

            annotationsData.forEach((annotsList) => {
                if (!annotsList || annotsList.length === 0) return;
                annotsList.forEach(evt => {
                    const t0 = evt.onset, t1 = evt.onset + evt.duration;
                    const channelIdx = parseAnnotationChannel(evt.description);
                    const annotationType = evt.description.includes(':')
                        ? evt.description.split(':')[1]
                        : evt.description;
                    const fillColor = annotColors[annotationType] || 'rgba(128,128,128,0.5)';
                    if (channelIdx >= 0) {
                        intervalsPerChannel.get(channelIdx).push({t0, t1, color:fillColor, desc:evt.description, aType:annotationType});
                    } else { // applies to all channels
                        parsedEDFs.chNames.forEach((_, ci) => {
                            intervalsPerChannel.get(ci).push({t0, t1, color:fillColor, desc:evt.description, aType:annotationType});
                        });
                    }
                });
            });

            function mergeIntervals(list) {
                if (list.length === 0) return [];
                let events = [];
                list.forEach(iv => {
                    events.push({t:iv.t0, type:'start', color:iv.color, desc:iv.desc, aType:iv.aType});
                    events.push({t:iv.t1, type:'end',   color:iv.color, desc:iv.desc, aType:iv.aType});
                });
                events.sort((a,b)=>a.t-b.t || (a.type==='end') - (b.type==='end'));
                let active = [], segments = [], lastT = null;
                for (const ev of events) {
                    if (lastT !== null && ev.t > lastT && active.length > 0) {
                        const segColor = mixMany(active.map(a=>a.color));
                        const segDesc  = active.map(a=>a.desc).join(' & ');
                        const uniqTypes = [...new Set(active.map(a=>a.aType))];
                        segments.push({t0:lastT, t1:ev.t, color:segColor, desc:segDesc, types:uniqTypes});
                    }
                    if (ev.type === 'start') {
                        active.push({color:ev.color, desc:ev.desc, aType:ev.aType});
                    } else {
                        const idx = active.findIndex(a=>a.color===ev.color && a.desc===ev.desc && a.aType===ev.aType);
                        if (idx >= 0) active.splice(idx,1);
                    }
                    lastT = ev.t;
                }
                return segments;
            }

            let highlightTraces = [];
            intervalsPerChannel.forEach((ivList, chIdx) => {
                const segs = mergeIntervals(ivList);
                const yCenter = parsedEDFs.offsets[chIdx];
                const yBot = yCenter - channelHeight / 2;
                const yTop = yCenter + channelHeight / 2;

                segs.forEach(seg => {
                    // 1) Background rectangle
                    if (showBg) {
                        highlightTraces.push({
                            x:[seg.t0, seg.t1, seg.t1, seg.t0],
                            y:[yBot, yBot, yTop, yTop],
                            fill:'toself',
                            fillcolor:seg.color,
                            line:{width:0},
                            hoverinfo:'text',
                            hovertext:seg.desc,
                            showlegend:false,
                            xaxis:'x1',
                            yaxis:'y1',
                            type:'scatter',
                            mode:'lines'
                        });
                    }

                    // 2) Pattern overlays
                    if (showPat) {
                        seg.types.forEach(tp => {
                            const pat = annotPatterns[tp] || '.';
                            const baseCol = annotColors[tp] || 'rgba(0,0,0,0.6)';
                            const fg  = patternFgColor(baseCol);
                            highlightTraces.push({
                                x:[seg.t0, seg.t1, seg.t1, seg.t0],
                                y:[yBot, yBot, yTop, yTop],
                                fill:'toself',
                                fillcolor:'rgba(0,0,0,0.001)',
                                fillpattern:{
                                    shape: pat,
                                    size: 8,
                                    solidity: 0.35,
                                    fgcolor: fg
                                },
                                line:{width:0},
                                hoverinfo:'skip',
                                showlegend:false,
                                xaxis:'x1',
                                yaxis:'y1',
                                type:'scatter',
                                mode:'lines'
                            });
                        });
                    }
                });
            });

            // --- generate legend entries for present annotation types ---
            let annotLegendTraces = [];
            let annotTypes = new Set();
            annotationsData.forEach(annotsList => {
                if (!annotsList) return;
                annotsList.forEach(evt => {
                    const annotationType = evt.description.includes(':')
                        ? evt.description.split(':')[1]
                        : evt.description;
                    annotTypes.add(annotationType);
                });
            });
            annotTypes.forEach(type => {
                const patShape = annotPatterns[type] || '.';
                const baseCol  = annotColors[type] || 'rgba(128,128,128,0.5)';
                const patFg    = patternFgColor(baseCol);
                annotLegendTraces.push({
                    x:[0,1,1,0],
                    y:[0,0,1,1],
                    type:'scatter',
                    mode:'lines',
                    fill:'toself',
                    fillcolor: baseCol,
                    fillpattern:{
                        shape: patShape,
                        size: 8,
                        solidity: 0.35,
                        fgcolor: patFg
                    },
                    line:{width:1, color: baseCol},
                    name: type,
                    showlegend: true,
                    hoverinfo: 'skip'
                });
            });

            // --- EEG traces ---
            let eegTraces = [];
            parsedEDFs.parsed.forEach((edfObj, fileIdx) => {
                const { sigs } = edfObj;
                const factor = +document.getElementById('subsample').value;
                const sr     = edfObj.baseSr / factor;
                document.getElementById('freqDisplay').textContent =
                    `Sampling rate: ${sr.toFixed(2)} Hz`;
                sigs.forEach((arr,i) => {
                    const sub = arr.filter((_,idx)=>idx%factor===0),
                          t   = sub.map((_,k)=> k/sr);
                    eegTraces.push({
                        x: t,
                        y: sub.map(v => v*eegScales[fileIdx] + parsedEDFs.offsets[i]),
                        mode:'lines',
                        line:{color:monoColors[fileIdx],width:1},
                        name: 'EEG' + (fileNames.length > 1 ? ` ${fileIdx+1}` : ''),
                        showlegend:(fileIdx===0 && i===0),
                        xaxis:'x1', yaxis:'y1'
                    });
                });
            });

            const legendY = 0.98;
            let layout = {
                dragmode: 'zoom',
                uirevision: 'true',  // This preserves UI state including zoom
                legend:{
                    orientation:'v',
                    yanchor:'top',
                    y: legendY,
                    xanchor:'right',
                    x: 0.99,
                    font:{size:14, color:'#000'},
                    bgcolor:'rgba(255,255,255,0.75)',
                    borderwidth:0,
                    itemclick:false,
                    itemdoubleclick:false
                },
                margin:{t:20,l:80,r:40,b:0},
                xaxis1:{
                    domain:[0,1],
                    title:'Time (s)',
                    rangeslider:{visible:true, thickness:sliderHeight},
                    anchor:'y1',
                    ...(prevXRange ? {range: prevXRange, autorange: false} : {})
                },
                yaxis1:{
                    domain: [0,1],
                    tickmode: 'array',
                    tickvals: parsedEDFs.offsets,
                    ticktext: parsedEDFs.chNames,
                    showgrid: false,
                    zeroline: false,
                    anchor: 'x1',
                    fixedrange: false,
                    ...(prevYRange ? {range: prevYRange} : {range: [yLow, yHigh]})
                }
            };

            Plotly.react(
                'plot',
                annotLegendTraces.concat(highlightTraces, eegTraces),
                layout,
                {responsive:true, displayModeBar:true, scrollZoom:true}
            );
        }

        document.getElementById('subsample').addEventListener('change', redraw);
        for (let fi = 0; fi < N; fi++) {
            document.getElementById('gain'+fi).addEventListener('change', redraw);
        }
        document.getElementById('showBg').addEventListener('change', redraw);
        document.getElementById('showPat').addEventListener('change', redraw);
        document.addEventListener('DOMContentLoaded', redraw);
        </script>
        </body>
        </html>
    """)

    rendered_html = (
        html_template
        .replace("___GAIN_CONTROLS___", gain_controls)
        .replace("___BASE64_EDF___", json.dumps(base64_edfs))
        .replace("___COLORS___", json.dumps(colors))
        .replace("___FILENAMES___", json.dumps(file_names))
        .replace("___PSD___", json.dumps(all_psds))
        .replace("___ANNOTATIONS___", json.dumps(annotations_list))
        .replace("___ANNOT_COLORS___", json.dumps(ANNOT_COLORS))
        .replace("___ANNOT_PATTERNS___", json.dumps(ANNOT_PATTERNS))
    )

    with open(output_html, 'w') as f:
        f.write(rendered_html)

def tests():
    """
    Some test cases for the plotter function edf2html
    """
    os.makedirs('tests', exist_ok=True)
    edf2html(
        ['ica_inputs/ZE-010-065-348_cut_200-800.edf'],
        'tests/helicopter.html'
        )
    edf2html(
        ['ica_inputs/ZE-010-065-348_cut_200-800.edf', 'ica_outputs/ZE-010-065-348_ICA_cut_200-800.edf'],
        'tests/helicopter_ICA.html'
        )
    raw = mne.io.read_raw_edf('ica_inputs/ZE-010-065-348_cut_700-1000.edf', preload=True, verbose=False)
    raw = raw.notch_filter(10.5, notch_widths=1.5)
    edf2html(
        ['ica_inputs/ZE-010-065-348_cut_700-1000.edf', raw],
        'tests/helicopter_notch.html'
        )

def main(inputs:str=None, output=None, colors=['#1f77b4', '#d62728'], gains=None, test=False, threads:int=32):
    """
    CLI entry-point.

    If --test is provided, runs the internal test suite.
    Otherwise, requires --inputs and --output to generate an HTML viewer.
    """
    if test:
        tests()

    if inputs is None or output is None:
        if not test:
            raise ValueError(
                "Both --inputs and --output are required unless --test is set"
            )
    else:
        tasks = []
        if os.path.isdir(inputs):
            indir = inputs
            for root, _, files in os.walk(indir):
                rel_root = os.path.relpath(root, indir)
                out_root = output if rel_root == '.' else os.path.join(output, rel_root)
                for fname in files:
                    if fname.lower().endswith('.edf'):
                        inpath  = os.path.join(root, fname)
                        outname = os.path.splitext(fname)[0] + '.html'
                        outpath = os.path.join(out_root, outname)
                        os.makedirs(out_root, exist_ok=True)
                        tasks.append((inpath, outpath))
        else:
            indir = os.path.dirname(inputs)
            fname = os.path.basename(inputs)
            if not fname.lower().endswith('.edf'):
                raise ValueError("Input file must have .edf extension")
            os.makedirs(output, exist_ok=True)
            inpath  = os.path.join(indir, fname)
            outname = os.path.splitext(fname)[0] + '.html'
            outpath = os.path.join(output, outname)
            tasks.append((inpath, outpath))

        if not tasks:
            print("No .edf files found to process.")
            return

        if len(tasks) > 1 and threads and threads > 1:
            print(f"⚙️ Parallel processing {len(tasks)} file(s) with {threads} thread(s)...")
            with ThreadPoolExecutor(max_workers=threads) as ex, tqdm(total=len(tasks), desc="Rendering", unit="file") as pbar:
                fut_to_task = {
                    ex.submit(edf2html, inpath, outpath, colors=colors, gains=gains): (inpath, outpath)
                    for (inpath, outpath) in tasks
                }
                for fut in as_completed(fut_to_task):
                    inpath, outpath = fut_to_task[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"Failed {inpath} -> {outpath}: {e}")
                    finally:
                        pbar.update(1)
        else:
            for (inpath, outpath) in tqdm(tasks, desc="Rendering", unit="file"):
                try:
                    edf2html(inpath, outpath, colors=colors, gains=gains)
                except Exception as e:
                    print(f"Failed {inpath} -> {outpath}: {e}")

if __name__ == "__main__":
    Fire(main)