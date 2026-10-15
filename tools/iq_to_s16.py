#!/usr/bin/env python3
"""
iq_to_s16.py - turn an SDR IQ capture of VHS RF (e.g. an SDRPlay RSP1B
recording from SDRconnect) into real 16-bit samples that vhs-decode can read.

Input formats
  .wav            2-channel I/Q WAV, int16 or float32 (SDRuno, SDRconnect, SDR#)
  raw files       --format cf32  complex float32 (GQRX, GNU Radio)
                  --format cs16  interleaved int16 I,Q
                  --format cu8   interleaved uint8 I,Q (RTL-SDR style)

Pipeline (overlap-save per block, so block edges are seamless)
  1. remove the zero-IF DC spike with a gentle high-pass
  2. interpolate x2
  3. shift by +fc so the real-world 0 Hz lands at 0 Hz, keep the real part
  4. resample to --out-rate (default 20 MSPS)
  5. scale to int16 and write a .s16 file

The SDR centre frequency is read from the file name when it ends in e.g.
"_3450000HZ.wav" (SDRconnect / SDRuno style); otherwise pass --fc.

Examples
  python iq_to_s16.py SDRconnect_IQ_..._3450000HZ.wav -o tape.s16
  python iq_to_s16.py capture.wav --fc 3.9e6 -o tape.s16
  python iq_to_s16.py capture.cf32 --format cf32 --fs 8e6 --fc 3.9e6 -o tape.s16
  python iq_to_s16.py capture.wav --check     # spectrum only, no output
then
  vhs-decode --pal --tf vhs -f 20 tape.s16 tape      (check `vhs-decode --help` for exact flags)

Needs numpy and scipy. matplotlib is optional (--plot).
"""

import argparse
import re
import struct
import sys
from fractions import Fraction
from math import ceil

import numpy as np
from scipy import signal


# ----------------------------------------------------------------------------
# Input readers
# ----------------------------------------------------------------------------

def parse_wav(path):
    """Minimal RIFF/RF64 WAV parser. Returns (dtype, channels, fs, data_offset, data_bytes)."""
    with open(path, "rb") as f:
        riff = f.read(12)
        if riff[:4] not in (b"RIFF", b"RF64") or riff[8:12] != b"WAVE":
            raise ValueError("not a WAV file")
        is_rf64 = riff[:4] == b"RF64"
        ds64_data_size = None
        fmt = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csize = struct.unpack("<4sI", hdr)
            if cid == b"ds64":
                body = f.read(csize)
                _riff_size, data_size = struct.unpack_from("<QQ", body, 0)
                ds64_data_size = data_size
            elif cid == b"fmt ":
                body = f.read(csize)
                tag, ch, fs, _br, _ba, bits = struct.unpack_from("<HHIIHH", body, 0)
                if tag == 0xFFFE and csize >= 26:  # WAVE_FORMAT_EXTENSIBLE
                    tag = struct.unpack_from("<H", body, 24)[0]
                fmt = (tag, ch, fs, bits)
            elif cid == b"data":
                data_offset = f.tell()
                data_bytes = csize
                if is_rf64 and csize == 0xFFFFFFFF and ds64_data_size is not None:
                    data_bytes = ds64_data_size
                # header can overstate the size if the recorder didn't finalise it
                f.seek(0, 2)
                data_bytes = min(data_bytes, f.tell() - data_offset)
                break
            else:
                f.seek(csize + (csize & 1), 1)
        else:
            raise ValueError("no data chunk")
    if fmt is None:
        raise ValueError("no fmt chunk")
    tag, ch, fs, bits = fmt
    if tag == 1 and bits == 16:
        dtype = np.dtype("<i2")
    elif tag == 3 and bits == 32:
        dtype = np.dtype("<f4")
    elif tag == 1 and bits == 8:
        dtype = np.dtype("u1")
    else:
        raise ValueError(f"unsupported WAV format tag={tag} bits={bits}")
    return dtype, ch, fs, data_offset, data_bytes


def open_input(path, fmt, fs):
    """Returns (memmap[n,2] of raw samples, converter function, fs)."""
    if fmt == "wav":
        dtype, ch, fs_wav, off, nbytes = parse_wav(path)
        if ch != 2:
            raise ValueError(f"WAV has {ch} channels, need 2 (I,Q)")
        n = nbytes // (dtype.itemsize * 2)
        mm = np.memmap(path, dtype=dtype, mode="r", offset=off, shape=(n, 2))
        fs = fs if fs else fs_wav
        if dtype == np.dtype("<i2"):
            conv = lambda a: a.astype(np.float32) / 32768.0
        elif dtype == np.dtype("u1"):
            conv = lambda a: (a.astype(np.float32) - 127.5) / 127.5
        else:
            conv = lambda a: a.astype(np.float32)
        return mm, conv, fs
    if not fs:
        raise ValueError("--fs is required for raw input")
    if fmt == "cf32":
        mm = np.memmap(path, dtype="<f4", mode="r")
        mm = mm[: (mm.size // 2) * 2].reshape(-1, 2)
        return mm, (lambda a: a.astype(np.float32)), fs
    if fmt == "cs16":
        mm = np.memmap(path, dtype="<i2", mode="r")
        mm = mm[: (mm.size // 2) * 2].reshape(-1, 2)
        return mm, (lambda a: a.astype(np.float32) / 32768.0), fs
    if fmt == "cu8":
        mm = np.memmap(path, dtype="u1", mode="r")
        mm = mm[: (mm.size // 2) * 2].reshape(-1, 2)
        return mm, (lambda a: (a.astype(np.float32) - 127.5) / 127.5), fs
    raise ValueError(f"unknown format {fmt}")


# ----------------------------------------------------------------------------
# DSP
# ----------------------------------------------------------------------------

class Converter:
    def __init__(self, fs, fc, out_rate, swap_iq=False, hp_hz=5000.0):
        self.fs = float(fs)
        self.fc = float(fc)
        self.fs2 = 2.0 * self.fs
        self.swap_iq = swap_iq
        r = Fraction(out_rate).limit_denominator(1000) / Fraction(self.fs2).limit_denominator(1000)
        self.p2, self.q2 = r.numerator, r.denominator   # out = fs2 * p2/q2
        self.out_rate = self.fs2 * self.p2 / self.q2
        self.sos = signal.butter(2, hp_hz, "highpass", fs=self.fs, output="sos")

    def process(self, raw, n0):
        """raw: float32 [n,2] block whose first sample has global index n0.
        Returns real float32 samples at out_rate."""
        i = raw[:, 0]
        q = raw[:, 1]
        if self.swap_iq:
            i, q = q, i
        # 1. DC / LO-leak removal
        i = i - i.mean()
        q = q - q.mean()
        i = signal.sosfilt(self.sos, i).astype(np.float32)
        q = signal.sosfilt(self.sos, q).astype(np.float32)
        # 2. interpolate x2 (real and imag separately)
        i2 = signal.resample_poly(i, 2, 1).astype(np.float32)
        q2 = signal.resample_poly(q, 2, 1).astype(np.float32)
        # 3. shift up by fc and take the real part: Re{(i+jq) e^{+j2*pi*fc*n/fs2}}
        n = np.arange(len(i2), dtype=np.float64) + 2.0 * n0
        ph = 2.0 * np.pi * ((self.fc / self.fs2) * n % 1.0)
        r = (i2 * np.cos(ph) - q2 * np.sin(ph)).astype(np.float32)
        # 4. resample to the output rate
        if (self.p2, self.q2) != (1, 1):
            r = signal.resample_poly(r, self.p2, self.q2).astype(np.float32)
        return r


def band_db(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    if not m.any():
        return float("nan")
    return 10.0 * np.log10(pxx[m].mean() + 1e-30)


def spectrum_report(y, fs, plot_path=None):
    f, pxx = signal.welch(y, fs=fs, nperseg=8192)
    bands = [
        ("0.4-0.9 MHz  colour-under (expect a peak near 0.63)", 0.4e6, 0.9e6),
        ("1.2-1.6 MHz  HiFi L (if HiFi deck)", 1.2e6, 1.6e6),
        ("1.6-2.0 MHz  HiFi R", 1.6e6, 2.0e6),
        ("2.0-3.5 MHz  luma lower sideband", 2.0e6, 3.5e6),
        ("3.5-5.0 MHz  luma FM carrier (expect the main hump)", 3.5e6, 5.0e6),
        ("5.0-7.0 MHz  luma upper sideband", 5.0e6, 7.0e6),
        ("7.0-9.0 MHz  should be low; high here means --swap-iq", 7.0e6, 9.0e6),
    ]
    ref = band_db(f, pxx, 3.5e6, 5.0e6)
    print("\nSpectrum of converted output (dB relative to the 3.5-5.0 MHz band):")
    for name, lo, hi in bands:
        if lo < fs / 2:
            print(f"  {band_db(f, pxx, lo, hi) - ref:7.1f} dB  {name}")
    m = (f > 0.3e6) & (f < 1.2e6)
    if m.any():
        print(f"  strongest line between 0.3 and 1.2 MHz: {f[m][np.argmax(pxx[m])]/1e6:.3f} MHz")
    m = (f > 3.0e6) & (f < 6.0e6)
    if m.any():
        print(f"  strongest line between 3.0 and 6.0 MHz: {f[m][np.argmax(pxx[m])]/1e6:.3f} MHz")
    if plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.semilogy(f / 1e6, pxx)
            plt.xlabel("MHz")
            plt.ylabel("PSD")
            plt.title("Converted output spectrum")
            plt.grid(True, which="both", alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=120)
            print(f"  plot written to {plot_path}")
        except ImportError:
            print("  (matplotlib not installed, no plot)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="output .s16 file (16-bit signed real samples)")
    ap.add_argument("--format", default="wav", choices=["wav", "cf32", "cs16", "cu8"])
    ap.add_argument("--fs", type=float, default=None, help="input sample rate in Hz (read from WAV if omitted)")
    ap.add_argument("--fc", type=float, default=None,
                    help="SDR centre frequency in Hz, e.g. 3.9e6 (default: from the file name)")
    ap.add_argument("--out-rate", type=float, default=20e6, help="output sample rate in Hz (default 20e6)")
    ap.add_argument("--swap-iq", action="store_true", help="use if the spectrum comes out mirrored")
    ap.add_argument("--gain", type=float, default=None, help="fixed scale to int16 (default: auto from first block)")
    ap.add_argument("--seconds", type=float, default=None, help="only convert the first N seconds")
    ap.add_argument("--check", action="store_true", help="spectrum report only, write nothing")
    ap.add_argument("--plot", default=None, help="write spectrum plot PNG here")
    args = ap.parse_args()

    if args.fc is None:
        m = re.search(r"_(\d+)hz(?:\.[^.]*)?$", args.input, re.IGNORECASE)
        if not m:
            ap.error("--fc is required (no centre frequency like _3450000HZ in the file name)")
        args.fc = float(m.group(1))

    mm, conv, fs = open_input(args.input, args.format, args.fs)
    n_total = mm.shape[0]
    if args.seconds:
        n_total = min(n_total, int(args.seconds * fs))
    cv = Converter(fs, args.fc, args.out_rate, swap_iq=args.swap_iq)
    print(f"input: {args.input}  fs={fs/1e6:.4f} MSPS  fc={args.fc/1e6:.4f} MHz  "
          f"samples={n_total} ({n_total/fs:.1f} s)")
    print(f"output rate: {cv.out_rate/1e6:.4f} MSPS  (band 0 to {fs/1e6:.2f} MHz usable; "
          f"RF covers {(args.fc-fs/2)/1e6:.2f} to {(args.fc+fs/2)/1e6:.2f} MHz)")
    if args.fc < fs / 2:
        print("note: fc < fs/2, so RF below %.2f MHz is captured twice (mirror) and comes out ~6 dB hot"
              % ((fs / 2 - args.fc) / 1e6))

    # block geometry: input block M and overlap K, both multiples of q2 so output indices are exact
    q = cv.q2
    K = 32768 - (32768 % q)
    M = (1 << 20) - ((1 << 20) % q)
    R = Fraction(cv.p2, cv.q2) * 2  # output samples per input sample

    out = None if args.check else open(args.output or "out.s16", "wb")
    scale = args.gain
    clipped = 0
    written = 0
    check_blocks = []
    nblocks = ceil(n_total / M)
    for b in range(nblocks):
        start = b * M
        stop = min(start + M, n_total)
        ext_start = max(0, start - K)
        ext_stop = min(n_total, stop + K)
        raw = conv(np.asarray(mm[ext_start:ext_stop]))
        y = cv.process(raw, ext_start)
        o0 = int((start - ext_start) * R)
        o1 = o0 + int(ceil((stop - start) * R))
        y = y[o0:o1]
        if scale is None:
            peak = np.percentile(np.abs(y), 99.99)
            scale = 0.6 * 32767.0 / max(peak, 1e-9)
            print(f"auto gain: x{scale:.1f} (99.99th percentile of first block -> 60% of full scale)")
        ys = y * scale
        clipped += int(np.count_nonzero(np.abs(ys) > 32767))
        if len(check_blocks) < 4:
            check_blocks.append(ys.astype(np.float32))
        if out is not None:
            out.write(np.clip(ys, -32768, 32767).astype("<i2").tobytes())
            written += len(ys)
        if b % 10 == 0 or b == nblocks - 1:
            print(f"  block {b+1}/{nblocks}", end="\r", flush=True)
        if args.check and len(check_blocks) >= 4:
            break
    print()
    if out is not None:
        out.close()
        print(f"wrote {written} samples ({written/cv.out_rate:.1f} s) to {args.output or 'out.s16'}; "
              f"clipped {clipped} samples" + ("  <- lower --gain" if clipped > written * 1e-5 else ""))
    spectrum_report(np.concatenate(check_blocks), cv.out_rate, args.plot)
    if out is not None:
        print("\nnext:  vhs-decode --pal --tf vhs -f %g %s tape\n       (then ld-analyse tape.tbc, or tbc-video-export tape.tbc)"
              % (cv.out_rate / 1e6, args.output or "out.s16"))


if __name__ == "__main__":
    main()
