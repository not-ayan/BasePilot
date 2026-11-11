# Current Clash price digits

`clash_digits.png` is a binary atlas of digits cropped from the anonymized menu
fixtures named in `clash_digits.json`. Columns are 0 through 9; each 40x40 tile
contains one normalized variant. Empty tiles are ignored. There are no player
names, tags, village images, or complete screenshots in this atlas.

`python tools/build_digit_atlas.py` rebuilds both assets from the manually checked
price annotations in that script. It requires the normal app dependencies and
Tesseract only to identify menu boundaries; digit labels are not produced by OCR.

The reader uses connected glyph boundaries, per-digit shape scores and a margin
against the second choice. It rejects clipped, ambiguous and nonnumeric shapes.
The current-card parser uses it for white and red inventory prices; only a fresh
affordable card and the independently checked purchase dialog can authorize a
purchase. Other font layouts may remain unknown until representative captures
are added. The packaged self-test checks every digit class is present.

rc6 also uses this atlas for native confirmation prices. A higher-confidence
shape match with two matching text passes can resolve a split HUD OCR vote;
strong contradictions remain unknown, and live spending budgets still need
consecutive-frame agreement.
