#!/bin/bash
# 下载并解压 E-DAIC 的 data/ 到 $HPCVAULT/data/XXX_P/（可断点续跑：已解压的 session 跳过）
#   nohup bash scripts/data/download_edaic.sh > $HPCVAULT/data/download.log 2>&1 &
# 每个包：下载到 .part -> 校验大小 -> 解压 -> 删 tar.gz；失败的记到 failed.txt，重跑脚本即可补上
set -uo pipefail
BASE=https://dcapswoz.ict.usc.edu/wwwedaic
OUT=${OUT:-$HPCVAULT/data}
JOBS=${JOBS:-4}
mkdir -p "$OUT"
cd "$OUT"
curl -s "$BASE/?dir=data" | grep -oE 'data/[0-9]+_P\.tar\.gz' | sort -u > files.txt
echo "$(date) $(wc -l < files.txt) archives -> $OUT (jobs=$JOBS)"
: > failed.txt

one() {
    f=$1; name=$(basename "$f" .tar.gz)
    [ -f "$name/.done" ] && return 0
    want=$(curl -sI "$BASE/$f" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
    for try in 1 2 3; do
        curl -s --retry 5 --retry-delay 10 -C - -o "$name.tar.gz.part" "$BASE/$f"
        have=$(stat -c %s "$name.tar.gz.part" 2>/dev/null || echo 0)
        [ -n "$want" ] && [ "$have" = "$want" ] && break
        echo "$(date +%T) $name size $have != $want (try $try)"
    done
    if [ "$have" != "$want" ]; then echo "$f" >> failed.txt; return 1; fi
    rm -rf "$name" && tar xzf "$name.tar.gz.part" && touch "$name/.done" && rm "$name.tar.gz.part" \
        && echo "$(date +%T) ok $name" || { echo "$(date +%T) EXTRACT FAIL $name"; echo "$f" >> failed.txt; }
}
export -f one; export BASE
xargs -a files.txt -P "$JOBS" -I{} bash -c 'one {}'
n=$(ls -d [0-9]*_P/.done 2>/dev/null | wc -l)
echo "$(date) done: $n/$(wc -l < files.txt) sessions; failed: $(wc -l < failed.txt) (see failed.txt, rerun to retry)"
