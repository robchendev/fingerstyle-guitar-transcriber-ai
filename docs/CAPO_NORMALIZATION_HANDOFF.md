# Eddie tuning/capo correction handoff

## Purpose

`C:\Users\robertchen\Downloads\eddie-capo-normalization-review.html` is the
source-bound review UI for the 203 Eddie van der Meer pairs. It separates the
40 standard-tuning entries, 49 Open C-family entries and 114 remaining entries.
Every video link opens in a new tab.

The page autosaves in browser local storage. That browser state is not stored
in the original file. Before handing the review to an implementation agent,
click **Save review into HTML** and replace the original file with the
downloaded snapshot. The snapshot embeds a version-3
`eddie-capo-normalization-review` payload in
`#embedded-review-state`. **Export handoff JSON** writes the same payload as a
separate backup.

Each pair has one mutually exclusive decision:

- `correct`: the existing GP tuning and full capo match the filmed setup.
- `compensated`: the existing GP was normalized and needs the recorded tuning
  and/or capo shifts applied.

For a compensated entry, `overallSemitonesShifted` is `tuningShift +
capoShift`. A pure compensation correction should normally be zero. For
example, moving the capo down one fret produces `-1`; raising every open
string by one semitone returns the total to zero.

## Future implementation request

When the owner supplies the completed HTML snapshot or exported JSON and asks
to apply it, implement a fail-closed importer and correction command. Do not
manually edit files based only on titles or visual inspection of the page.

1. Parse only schema version 3 with kind
   `eddie-capo-normalization-review`. For HTML, read the JSON from
   `#embedded-review-state`.
2. Report and stop if the snapshot contains no decisions. If any of the 203
   entries is undecided, report the exact IDs and require the owner to confirm
   whether a partial application is intended.
3. Join records by stable `tab-NNNN` ID to `data\pairs.json`; require
   `performerId == "eddie-van-der-meer"`.
4. Extract the current six-string tuning and full capo from
   `data\pairs\<id>\raw.gp`. Require them to equal `originalTuning` and
   `originalCapo` from the review payload. String order is 6 to 1. Any mismatch
   means the review is stale and must stop that entry.
5. Recompute `correctedTuning`, `correctedCapo` and
   `overallSemitonesShifted` from the integer shifts. Never trust redundant
   values without checking them. Require MIDI pitches in range and capo
   `0..24`.
6. A compensated entry with a nonzero overall shift requires explicit
   per-entry owner confirmation before editing. A nonzero shift changes
   sounding pitch and may require more than metadata correction.
7. Before changing each source GP, run:

   ```powershell
   .\.venv\Scripts\python.exe -m scripts.prepare_training_data --workspace data invalidate --id tab-NNNN --reason "Owner-reviewed filmed tuning/capo correction"
   ```

   Invalidation must happen while the currently prepared source still matches
   its recorded hashes.
8. Edit only `data\pairs\<id>\raw.gp` as the owned source. Change the fixed
   six-string tuning and full capo metadata while preserving notes, frets,
   voices, measures, tempo, meter and all other content. Do not directly patch
   `normalized.gp`, canonical JSON, prepared bundles or frozen releases.
9. Re-open every edited GP through the project GP inspector. Require the
   extracted tuning/capo to equal the requested correction, no partial capo,
   no tuning/capo automation and no newly introduced warnings. Flag free text
   that mentions an old tuning or capo for human review rather than rewriting
   it silently.
10. Write an audit report containing the checklist hash, pair ID, title,
    before/after GP hashes, original/corrected tuning and capo, both shifts and
    the overall shift. Keep the untouched source backup until the rebuilt
    corpus is accepted.
11. Re-run preparation and human score/alignment review for affected pairs,
    then create a new immutable release version. Existing frozen releases must
    remain unchanged. New downbeat-conditioned training must use the rebuilt
    release.

Do not transpose notes or frets merely because tuning/capo metadata changed.
For the expected zero-overall-shift corrections, the same fret retains the
same sounding pitch. Any nonzero-overall-shift case is a separate musical
change and must be reviewed explicitly.
