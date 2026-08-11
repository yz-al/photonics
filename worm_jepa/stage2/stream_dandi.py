"""
stream_dandi.py -- reproduce the bridge's neural-data cache directly from DANDI
via HTTP range requests (no full download of the multi-GB NWB files).

Three sources, matched to the three data regimes the bridge composes:

  000776  Flavell  freely-moving whole-brain + behavior       -> flav_worms/
  000981  chemosensory: sensory+command, aversive stimulus    -> chemo_worms/
  000541  microfluidic: sensory+command, attractive stimulus  -> chemo541_worms/

Each NWB is HDF5; we open it over HTTP with fsspec + h5py and read only the
small datasets we need. Cached as .npz per worm. Idempotent (skips existing).

    from stream_dandi import stream_all
    stream_all("/path/to/cache")
"""
import os, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")

DANDI = "https://api.dandiarchive.org/api/dandisets/%s/versions/draft/assets/"


def _assets(dsid):
    import requests
    out, url = [], DANDI % dsid + "?page_size=100"
    while url:
        r = requests.get(url, timeout=60).json()
        out += r["results"]; url = r.get("next")
    return [a for a in out if a["path"].endswith(".nwb")]


def _s3(dsid, asset_id):
    import requests
    return requests.head(DANDI % dsid + "%s/download/" % asset_id,
                         allow_redirects=True, timeout=60).url


def stream_flavell(cache, limit=None):
    """000776: traces (T,neurons), names, velocity, reversal, behavior channels."""
    import requests, h5py, fsspec
    out = os.path.join(cache, "flav_worms"); os.makedirs(out, exist_ok=True)
    fs = fsspec.filesystem("http"); nwb = _assets("000776"); done = 0
    CH = ["angular_velocity", "head_curvature", "body_curvature", "pumping"]
    for a in (nwb[:limit] if limit else nwb):
        wid = a["path"].split("/")[0]; fn = os.path.join(out, wid + ".npz")
        if os.path.exists(fn):
            done += 1; continue
        try:
            h = h5py.File(fs.open(_s3("000776", a["asset_id"]), block_size=512 * 1024), "r")
            proc = h["processing"]
            ca = proc["CalciumActivity"] if "CalciumActivity" in proc else proc["NeuralActivity"]
            # traces: (neurons, T) or (T, neurons); normalize to (T, neurons)
            tr = None
            for key in ("ActivityTraces/activity", "TracesRaw/data", "activity"):
                try:
                    tr = ca[key][:]; break
                except Exception:
                    pass
            names = None
            for key in ("ActivityTraces/neuron", "neuron_ids", "NeuronIDs/labels"):
                try:
                    names = [s.decode() if isinstance(s, bytes) else str(s) for s in ca[key][:]]; break
                except Exception:
                    pass
            if tr.shape[0] == len(names):
                tr = tr.T
            B = proc["Behavior"]
            vel = B["velocity/velocity/data"][:]
            rev = B["reversal_events/reversal_events/data"][:] if "reversal_events" in B else \
                (vel < 0).astype(float)
            rec = dict(tr=tr, vel=vel, rev=rev, names=np.array(names, dtype=object))
            for c in CH:
                try:
                    rec["beh_" + c] = B["%s/%s/data" % (c, c)][:]
                except Exception:
                    rec["beh_" + c] = np.full(len(vel), np.nan)
            h.close(); np.savez(fn, **rec); done += 1
            print("[flav %d] %s tr%s" % (done, wid, tr.shape), flush=True)
        except Exception as e:
            print("flav %s FAIL %s" % (wid, str(e)[:70]), flush=True)
    return done


def _stream_chemo(dsid, folder, cache, seconds_stim, limit=None):
    """000981 / 000541: activity (neurons,T), names, stimulus id+frame, rate.
    000981 keeps stimulus under processing/CalciumActivity/StimulusInfo (frame
    indices); 000541 keeps it under intervals/chemical_stimuli (seconds)."""
    import requests, h5py, fsspec
    out = os.path.join(cache, folder); os.makedirs(out, exist_ok=True)
    fs = fsspec.filesystem("http"); nwb = _assets(dsid); done = 0
    for a in (nwb[:limit] if limit else nwb):
        wid = a["path"].split("/")[0]; fn = os.path.join(out, wid + ".npz")
        if os.path.exists(fn):
            done += 1; continue
        try:
            h = h5py.File(fs.open(_s3(dsid, a["asset_id"]), block_size=262144), "r")
            ca = h["processing/CalciumActivity"]
            if seconds_stim:  # 000541
                sr = ca["SignalRawFluor/SignalCalciumImResponseSeries"]
                data = sr["data"][:]                 # (T, neurons)
                rate = float(sr["starting_time"].attrs.get("rate", 4.0))
                act = data.T
                names = [s.decode() if isinstance(s, bytes) else str(s)
                         for s in ca["NeuronIDs/labels"][:]]
                ci = h["intervals/chemical_stimuli"]
                sid = [s.decode() if isinstance(s, bytes) else str(s) for s in ci["stimulus"][:]]
                st = (ci["start_time"][:] * rate).astype(float)
            else:            # 000981
                act = ca["ActivityTraces/activity"][:]     # (neurons, T)
                names = [s.decode() if isinstance(s, bytes) else str(s)
                         for s in ca["ActivityTraces/neuron"][:]]
                si = ca["StimulusInfo"]
                sid = [s.decode() if isinstance(s, bytes) else str(s) for s in si["data"][:]]
                st = si["timestamps"][:].astype(float)
                rate = np.nan
                try:
                    cis = h["acquisition/CalciumImageSeries"]
                    if "starting_time" in cis and "rate" in cis["starting_time"].attrs:
                        rate = float(cis["starting_time"].attrs["rate"])
                except Exception:
                    pass
            h.close()
            np.savez(fn, act=act, names=np.array(names, dtype=object),
                     sid=np.array(sid, dtype=object), st=st,
                     rate=np.array([rate]), start=np.array([0.0]))
            done += 1
            print("[%s %d] %s act%s stim%d" % (dsid, done, wid, act.shape, len(sid)), flush=True)
        except Exception as e:
            print("%s %s FAIL %s" % (dsid, wid, str(e)[:70]), flush=True)
    return done


def stream_all(cache, limit=None):
    n1 = stream_flavell(cache, limit)
    n2 = _stream_chemo("000981", "chemo_worms", cache, seconds_stim=False, limit=limit)
    n3 = _stream_chemo("000541", "chemo541_worms", cache, seconds_stim=True, limit=limit)
    print("cached: flavell=%d chemo981=%d chemo541=%d -> %s" % (n1, n2, n3, cache))


if __name__ == "__main__":
    import sys
    stream_all(sys.argv[1] if len(sys.argv) > 1 else "./data/bridge_cache")
