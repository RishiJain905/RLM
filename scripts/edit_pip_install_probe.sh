#!/bin/sh
# Red-team probe (step-9.5 B2): how is the repository's package installed in a derived Python
# image's site-packages — a real copy of the source (an answer for bug tasks), or an editable
# install pointing back at the checkout (which is /work/repo in the grader and the episode)?
# Usage: docker run --rm -i --network none --user <uid> --entrypoint /bin/sh <image> -s < this
# $1 = package name (default: arrow).
set -eu

PKG="${1:-arrow}"

SP=$(python3 -c 'import site,sys; sys.stdout.write(site.getsitepackages()[0])')
echo "site-packages: $SP"

echo "--- entries named after the package:"
ls "$SP" | grep -i "$PKG" || echo "(none)"

echo "--- is it importable, and from where:"
python3 -c "import $PKG, sys; print('$PKG ->', getattr($PKG, '__file__', 'no __file__'))" 2>&1 || true

echo "--- dist-info RECORD's first lines (does it point at /testbed?):"
DI=$(ls -d "$SP"/${PKG}*.dist-info "$SP"/${PKG}*.egg-info 2>/dev/null | head -1 || true)
echo "dist-info: $DI"
[ -n "$DI" ] && head -6 "$DI/RECORD" 2>/dev/null || true

echo "--- any .pth or __editable__ entry naming it:"
grep -l -s "$PKG" "$SP"/*.pth 2>/dev/null | head -5 || echo "(no .pth mentions)"
echo "--- direct_url / editable markers:"
[ -n "$DI" ] && cat "$DI/direct_url.json" 2>/dev/null || true
[ -n "$DI" ] && ls "$DI" | head -10 || true