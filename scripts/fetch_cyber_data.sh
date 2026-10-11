#!/usr/bin/env bash
# Fetch the cyber domain's sources onto the data root, at the versions pinned in rlm_rl/data.py (domain "cyber").
#
#     bash scripts/fetch_cyber_data.sh            # from Git Bash on Windows (data root D:\RLM-data)
#     RLM_DATA_DIR=/mnt/d/RLM-data bash scripts/fetch_cyber_data.sh    # from WSL
#
# What it fetches, ~140 GB in all (splunk attack_data's Git LFS content is the unknown part):
#   - OTRF Security-Datasets (MIT)                      -> github/OTRF__Security-Datasets
#   - splunk attack_data (Apache-2.0, Git LFS)          -> github/splunk__attack_data
#   - EVTX-ATTACK-SAMPLES (GPL-3.0, eval only)          -> github/sbousseaden__EVTX-ATTACK-SAMPLES
#   - AIT-LDS v2, 8 zips, 131 GB (CC-BY-NC-SA-4.0, eval only) -> zenodo/ait-lds-v2
# Each repo is checked out at its pinned commit, and each zip is checked against Zenodo's md5. It can be rerun:
# finished pieces are skipped and partial zips resume. Nothing is opened or run; afterwards, rebuild the manifest
# with `python scripts/build_manifest.py`. Zenodo serves each file at about 0.6 MB/s, so the zips take many hours.
set -uo pipefail

ROOT="${RLM_DATA_DIR:-/d/RLM-data}"
LOG="$ROOT/logs/fetch-cyber.log"
mkdir -p "$ROOT/github" "$ROOT/zenodo/ait-lds-v2" "$ROOT/logs"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

clone() {  # url, dir under the data root, pinned commit
    local url=$1 dir="$ROOT/$2" sha=$3
    if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then say "have $2 at ${sha:0:12}"; return; fi
    [ -d "$dir/.git" ] || git clone -q --depth 1 "$url" "$dir" || { say "FAILED clone $2"; return 1; }
    if [ "$(git -C "$dir" rev-parse HEAD)" != "$sha" ]; then
        git -C "$dir" fetch -q --depth 1 origin "$sha" && git -C "$dir" checkout -q "$sha" || { say "FAILED checkout $2"; return 1; }
    fi
    git -C "$dir" lfs pull 2>/dev/null || true
    say "fetched $2 at ${sha:0:12}: $(git -C "$dir" ls-files | wc -l) files, $(git -C "$dir" ls-files --deleted | wc -l) missing"
}

clone https://github.com/OTRF/Security-Datasets.git github/OTRF__Security-Datasets d9d40ef123d2c87d5d3df28c96bcab4f0faccc87
clone https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES.git github/sbousseaden__EVTX-ATTACK-SAMPLES 4ceed2f4706daf601c212a8f91c113dd85349a2c
clone https://github.com/splunk/attack_data.git github/splunk__attack_data 4389fa7a4e74a7c083c0fbe448fa2c7c32378981

zip_md5() {
    case $1 in
        russellmitchell.zip) echo 78e9b7d169b438f03816019b52ab3ff7 ;; santos.zip) echo fbd2663a41c83f345aa08a7fb11a6c6c ;;
        fox.zip) echo b6cd133398fbed6f3e7434dafbc45756 ;; harrison.zip) echo 4fbe3dd17a194776b2851116f8b51380 ;;
        wardbeck.zip) echo ff524634e64c7779def2eb804e8c52ac ;; shaw.zip) echo 872bbec7ffe12f14e06364e328a96088 ;;
        wheeler.zip) echo e3f5605d867fcc644f93ae6c736cdb30 ;; wilson.zip) echo 952dd37cb262ef0054ad6d5371bdbe70 ;;
    esac
}
fetch_zip() {
    local name=$1 out="$ROOT/zenodo/ait-lds-v2/$1"
    [ -f "$out.ok" ] && { say "have $name"; return; }
    curl -sS -L -C - --retry 10 --retry-delay 30 -o "$out" "https://zenodo.org/api/records/5789064/files/$name/content" \
        || { say "FAILED download $name (rerun to resume)"; return 1; }
    if [ "$(md5sum "$out" | cut -c1-32)" = "$(zip_md5 "$name")" ]; then touch "$out.ok"; say "fetched $name, md5 ok"
    else say "BAD md5 for $name: delete it and rerun"; fi
}
export -f fetch_zip zip_md5 say
export ROOT LOG
printf '%s\n' russellmitchell.zip santos.zip fox.zip harrison.zip wardbeck.zip shaw.zip wheeler.zip wilson.zip \
    | xargs -P 3 -I{} bash -c 'fetch_zip {}'
say "done; now run: python scripts/build_manifest.py"
