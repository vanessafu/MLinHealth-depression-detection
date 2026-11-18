"""只下载 DAIC-WOZ（300–492）每个 zip 里的 XXX_TRANSCRIPT.csv（含 Ellie 的话），不下载音频。
用 HTTP Range 读 zip 的目录和这一个成员（每个 session 只传几十 KB）。

  python scripts/data/fetch_daicwoz_transcripts.py --out $HPCVAULT/daicwoz_transcripts
"""
import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"


class HttpFile(io.RawIOBase):
    """只读、可 seek 的远程文件（每次 read 发一个 Range 请求）。"""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sids", nargs="*", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = urllib.request.urlopen(BASE, timeout=60).read().decode()
    sids = sorted({int(m) for m in re.findall(r'href="(\d+)_P\.zip"', idx)})
    if a.sids:
        sids = [s for s in sids if s in set(a.sids)]
    ok, fail = 0, []
    for sid in sids:
        dst = out / f"{sid}_TRANSCRIPT.csv"
        if dst.exists() and dst.stat().st_size > 0:
            ok += 1
            continue
        try:
            z = zipfile.ZipFile(io.BufferedReader(HttpFile(f"{BASE}{sid}_P.zip"), buffer_size=1 << 16))
            name = next(n for n in z.namelist() if n.endswith("TRANSCRIPT.csv"))
            dst.write_bytes(z.read(name))
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail.append((sid, repr(e)))
            print("FAIL", sid, e, flush=True)
    print(f"{ok}/{len(sids)} transcripts -> {out}; failed {len(fail)}")


if __name__ == "__main__":
    main()
