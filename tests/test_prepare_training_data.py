from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import ANY, patch
from uuid import uuid4
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf

from scripts import transcriber
from scripts.dataset_io import read_json, sha256
from scripts.audio_tools import AcquisitionError, executable
from scripts.dataset_release import candidate_digest, validate_release
from scripts.percussion_supervision import percussion_annotation_coverage
from scripts.prepare_training_data import ARTIFACTS, CODE_FILES, automatic_candidate, extract_score, main, parser, release_dataset, rules_template, selection, source_paths, workspace_path
from scripts.score_alignment import ScoreClock
from scripts.transcriber_audio import FeatureConfig
from scripts.transcriber_data import TrainingDataset
from scripts.transcriber_model import ModelConfig
from tests.test_gp_events import musical_score, note_xml
from tests.test_gp_normalization import archive_bytes


ROOT = Path(__file__).resolve().parents[1]


def fresh_score(title, shift=0, *, pickup=False, unknown=False, legend=False):
    root = musical_score()
    root.find("./Score/Title").text = title
    root.find("./MasterTrack/Automations/Automation/Value").text = "60 2"
    if pickup:
        ET.SubElement(root.find("MasterTrack"), "Anacrusis")
    for name in ("MasterBars", "Bars", "Voices", "Beats", "Notes", "Rhythms"):
        root.find(name).clear()
    root.find("Rhythms").append(ET.fromstring('<Rhythm id="0"><NoteValue>Quarter</NoteValue></Rhythm>'))
    for measure in range(4 + int(legend)):
        master = ET.SubElement(root.find("MasterBars"), "MasterBar")
        ET.SubElement(master, "Time").text = "4/4"
        ET.SubElement(master, "Bars").text = str(measure)
        if measure == 4:
            ET.SubElement(ET.SubElement(master, "Section"), "Text").text = "Instructions"
        bar = ET.SubElement(root.find("Bars"), "Bar", id=str(measure))
        ET.SubElement(bar, "Voices").text = f"{measure} -1 -1 -1"
        voice = ET.SubElement(root.find("Voices"), "Voice", id=str(measure))
        ET.SubElement(voice, "Beats").text = " ".join(str(measure * 4 + offset) for offset in range(4))
        for offset in range(4):
            identifier = str(measure * 4 + offset)
            beat = ET.SubElement(root.find("Beats"), "Beat", id=identifier)
            ET.SubElement(beat, "Rhythm", ref="0")
            if measure < 4:
                ET.SubElement(beat, "Notes").text = identifier
                root.find("Notes").append(note_xml(identifier, fret=(measure * 4 + offset + shift) % 8))
            if unknown and identifier == "3":
                ET.SubElement(beat, "FreeText").text = "???"
            if legend and identifier in ("0", "16"):
                ET.SubElement(beat, "FreeText").text = "*" if identifier == "0" else "Tap side of guitar *"
    return root


def fresh_audio(path, shift=0, *, rate=8000):
    samples = np.zeros((17 * rate, 2), dtype=np.float64)
    for index in range(16):
        start = round((.4 + index) * rate)
        local = np.arange(round(.8 * rate)) / rate
        frequency = 440 * 2 ** ((42 + (index + shift) % 8 - 69) / 12)
        tone = .35 * np.sin(2 * np.pi * frequency * local) * np.exp(-4 * local)
        samples[start:start + len(tone), 0] += tone
        samples[start:start + len(tone), 1] -= tone
    sf.write(path, samples, rate, subtype="PCM_24")
    return samples


class TrainingPreparationTests(unittest.TestCase):
    def test_fresh_generic_conventions_do_not_label_unknown_short_text_as_percussion(self):
        self.add("pair-A", unknown=True)
        self.run_cli("prepare", "--accept-conventions")
        labels = read_json(self.workspace / "pairs" / "pair-A" / "canonical.json")
        source_provenance = labels["provenance"]["rawCanonicalProvenance"]
        self.assertEqual(source_provenance["notationConventionsId"], "fingerstyle-v1")
        self.assertFalse(source_provenance["shortTextIsPercussiveHit"])
        self.assertEqual(labels["targets"]["gestures"], [])
        self.assertTrue(any(row["reason"] == "uninterpreted_annotation" for row in labels["review"]["unresolvedGestures"]))

    def test_explicit_generic_conventions_preserve_o_x_and_ghost_x_labels(self):
        score = fresh_score("synthetic-percussion")
        for identifier in ("0", "2"):
            note = score.find(f"./Notes/Note[@id='{identifier}']")
            ET.SubElement(ET.SubElement(note.find("Properties"), "Property", name="Muted"), "Enable")
        ET.SubElement(score.find("./Notes/Note[@id='0']"), "AntiAccent").text = "Normal"
        ET.SubElement(score.find("./Beats/Beat[@id='1']"), "FreeText").text = "O"
        gp, audio = self.inputs / "percussion.gp", self.inputs / "percussion.wav"
        gp.write_bytes(archive_bytes(score))
        fresh_audio(audio)
        self.run_cli("add", "--id", "percussion", "--group", "source-group", "--gp", str(gp), "--audio", str(audio))
        self.run_cli("prepare", "--accept-conventions")
        labels = read_json(self.workspace / "pairs" / "percussion" / "canonical.json")
        self.assertEqual({row["technique"] for row in labels["targets"]["gestures"]}, {"wrist_thump", "thumb_slap", "percussive_hit"})
        self.assertTrue(all(row["source"]["interpretationRuleId"].startswith("fingerstyle-v1-")
                            for row in labels["provenance"]["sourceGestures"]))

    def test_explicit_pair_order_is_preserved_for_repeatable_release_windows(self):
        self.add("pair-A")
        self.add("pair-B", 2)
        self.assertEqual([pair["id"] for pair in selection(self.workspace, ["pair-B", "pair-A"])], ["pair-B", "pair-A"])

    def test_cli_emits_utf8_notation_even_with_ascii_console_environment(self):
        self.add("pair-A", title="Notation \u25b2")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.prepare_training_data", "--workspace", str(self.workspace), "status"],
            cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "ascii"}, capture_output=True, check=True,
        )
        self.assertEqual(json.loads(result.stdout.decode("utf-8"))["pairs"][0]["title"], "Notation \u25b2")

    def setUp(self):
        self.root = ROOT / f".training-preparation-tests-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / ".gitignore").write_text("*\n", encoding="ascii")
        self.workspace = self.root / "workspace"
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.run_cli("init")

    def run_cli(self, *arguments, error=None, authorize_release=True):
        if arguments[0] == "release" and authorize_release:
            arguments = (*arguments, "--reviewer", "synthetic-release-owner", "--authorize-release")
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(["--workspace", str(self.workspace), *arguments])
        if error is not None:
            self.assertEqual(code, 2, output.getvalue())
            self.assertIn(error, errors.getvalue())
            return errors.getvalue()
        self.assertEqual(code, 0, errors.getvalue())
        return json.loads(output.getvalue())

    def add(self, identifier, shift=0, *, group=None, pickup=False, unknown=False, legend=False, rate=8000, title=None):
        gp, audio = self.inputs / f"{identifier}.gp", self.inputs / f"{identifier}.wav"
        gp.write_bytes(archive_bytes(fresh_score(identifier, shift, pickup=pickup, unknown=unknown, legend=legend)))
        fresh_audio(audio, shift, rate=rate)
        self.run_cli("add", "--id", identifier, "--group", group or f"group-{identifier}", "--performer", f"independent artist {shift}", "--gp", str(gp), "--audio", str(audio), *(["--title", title] if title is not None else []))
        return gp, audio

    def prepare_two(self, *, unknown=False):
        self.add("pair-A", pickup=True, unknown=unknown)
        self.add("pair-B", 2)
        return self.run_cli("prepare", "--accept-conventions")

    def approve(self, identifier, *, exclude=False, end="16.4", split=None):
        args = [
            "review", "--id", identifier, "--reviewer", "synthetic-reviewer",
            "--anchor", "1=0.4", "--anchor", "2=4.4", "--anchor", "3=8.4",
            "--anchor", "4=12.4", "--anchor", f"end={end}",
            "--approve-range", f"0.4:{end}", "--split", split or ("validation" if identifier == "pair-B" else "train"),
            "--authorize-use", "--confirm-pitch", "--confirm-notation", "--confirm-grouping",
            "--approve-experimental-ranges", "--acknowledge-uncertainty",
        ]
        if exclude:
            args += ["--exclude-range", "7:7.5"]
        return self.run_cli(*args)

    def release(self, version="v1"):
        return self.run_cli("release", "--version", version, "--validation-group", "group-pair-B")

    def test_propose_cli_forwards_selection_and_streams_progress_without_mutating_sources(self):
        self.add("pair-A")
        before = {path: sha256(path) for path in self.workspace.rglob("*") if path.is_file()}

        def propose(workspace, name, *, validation_ids, progress):
            progress("Proposal: synthetic progress.")
            return {"trainingReady": False, "proposalPath": str(workspace / "proposals" / f"{name}.json")}

        for identifiers in (["pair-B"], ["pair-C", "pair-B"]):
            options = ["--validation-ids", *identifiers]
            output, errors = StringIO(), StringIO()
            with self.subTest(options=options), patch("scripts.dataset_proposal.propose_dataset", side_effect=propose) as proposal, patch("scripts.prepare_training_data.prepare_pair", side_effect=AssertionError("A proposal must not reprepare sources")), patch("scripts.prepare_training_data.release_dataset", side_effect=AssertionError("A proposal must not activate a release")), redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(main(["--workspace", str(self.workspace), "propose", "--name", "representative", *options]), 0, errors.getvalue())
            proposal.assert_called_once_with(self.workspace, "representative", validation_ids=identifiers, progress=ANY)
            progress, _, document = output.getvalue().partition("\n")
            self.assertEqual(progress, "Proposal: synthetic progress.")
            self.assertFalse(json.loads(document)["trainingReady"])
            self.assertEqual(before, {path: sha256(path) for path in self.workspace.rglob("*") if path.is_file()})
        self.assertNotIn("dataset_proposal.py", CODE_FILES)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(["--workspace", str(self.workspace), "propose", "--name", "missing-split"])

    def test_release_api_rejects_scalar_unordered_empty_and_duplicate_group_inputs(self):
        for groups in (
            None, "group-pair-B", b"group-pair-B", {"group-pair-B"}, {"group-pair-B": True},
            [], (), [""], [" "], [None], [12], [["nested"]], ["group-pair-B", "group-pair-B"],
        ):
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, "Validation group"):
                release_dataset(self.workspace, "v1", groups, reviewer="synthetic-owner", authorize_release=True)
        self.run_cli(
            "release", "--version", "v1", "--validation-group", "group-pair-B",
            "--validation-group", "group-pair-B", error="must be unique",
        )
        self.assertFalse((self.workspace / "releases").exists())

    def test_requested_validation_groups_must_exist_in_the_explicit_pair_selection(self):
        self.add("pair-A")
        self.add("pair-B", 2)
        self.add("pair-C", 4)
        for group in ("group-pair-C", "unregistered-group"):
            with self.subTest(group=group):
                self.run_cli(
                    "release", "--version", "v1", "--ids", "pair-A", "pair-B",
                    "--validation-group", "group-pair-B", "--validation-group", group,
                    error="present in the explicitly selected pairs",
                )
        self.assertFalse((self.workspace / "releases").exists())

    def test_repeated_validation_groups_release_and_load_independent_recordings_in_one_dataset(self):
        self.add("pair-A")
        self.add("pair-B", 2)
        self.add("pair-C", 4)
        self.run_cli("prepare", "--accept-conventions")
        self.add("unselected", 6)
        for identifier in ("pair-A", "pair-B", "pair-C"):
            self.approve(identifier)
        arguments = (
            "release", "--version", "multi", "--ids", "pair-B", "pair-A", "pair-C",
            "--validation-group", "group-pair-C", "--validation-group", "group-pair-B",
        )
        self.run_cli(*arguments, error="reviewed split differs")
        self.assertFalse((self.workspace / "releases").exists())
        self.approve("pair-C", split="validation")
        source_reviews = {identifier: sha256(self.workspace / "pairs" / identifier / "review.json") for identifier in ("pair-A", "pair-B", "pair-C")}
        released = self.run_cli(*arguments)
        manifest_path = Path(released["manifestPath"])
        manifest, records, _ = validate_release(manifest_path)
        self.assertEqual(manifest["kind"], "local-training-dataset")
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["validationGroups"], ["group-pair-C", "group-pair-B"])
        self.assertEqual(manifest["releaseAuthorization"]["validationGroups"], manifest["validationGroups"])
        self.assertNotIn("validationGroup", manifest)
        self.assertNotIn("validationGroup", manifest["releaseAuthorization"])
        self.assertEqual(
            [(entry["id"], entry["groupId"], entry["split"]) for entry, _ in records],
            [("pair-B", "group-pair-B", "validation"), ("pair-A", "group-pair-A", "train"), ("pair-C", "group-pair-C", "validation")],
        )
        features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
        dataset = TrainingDataset(manifest_path, "validation", features, ModelConfig(n_mels=16), root=self.workspace)
        self.assertEqual(len(dataset), manifest["counts"]["windowsBySplit"]["validation"])
        self.assertEqual({dataset[index]["metadata"]["windowId"].split(":")[0] for index in range(len(dataset))}, {"pair-B", "pair-C"})
        frozen = {path: sha256(path) for path in manifest_path.parent.rglob("*") if path.is_file()}
        self.run_cli(*arguments)
        self.assertEqual(frozen, {path: sha256(path) for path in frozen})
        self.assertEqual(source_reviews, {identifier: sha256(self.workspace / "pairs" / identifier / "review.json") for identifier in source_reviews})
        self.assertEqual([path.name for path in (self.workspace / "releases").iterdir()], ["multi"])

    def test_completeness_confirmation_requires_reviewer_and_grants_no_other_approval(self):
        self.add("pair-A")
        self.run_cli("prepare", "--accept-conventions")
        directory = self.workspace / "pairs" / "pair-A"
        before = {name: sha256(directory / name) for name in ARTIFACTS}
        self.run_cli("review", "--id", "pair-A", "--confirm-percussion-completeness", error="--reviewer")
        self.assertFalse((directory / "review.json").exists())
        report = self.run_cli("review", "--id", "pair-A", "--reviewer", "source-owner", "--confirm-percussion-completeness")
        approval = report["approval"]
        self.assertTrue(approval["percussionAnnotationsComplete"])
        self.assertEqual(approval["reviewer"], "source-owner")
        self.assertFalse(any(approval.get(field) for field in ("authorizedUse", "recordingAndTargetPitchConfirmed", "notationReviewed", "approveExperimentalRangesAndSplit", "groupingConfirmed", "uncertaintyAcknowledged")))
        self.assertEqual(self.run_cli("status")["pairs"][0]["status"], "needs-review")
        self.assertEqual(before, {name: sha256(directory / name) for name in ARTIFACTS})
        state = read_json(directory / "preparation.json")
        self.assertEqual(state["inputs"]["implementation"]["percussion_supervision.py"], sha256(ROOT / "scripts" / "percussion_supervision.py"))
        self.assertEqual(read_json(directory / "review.json")["preparationSha256"], candidate_digest(state))

    def test_candidate_range_and_source_revisions_invalidate_completeness(self):
        self.add("pair-A")
        self.run_cli("prepare", "--accept-conventions")
        directory = self.workspace / "pairs" / "pair-A"
        for changed in (
            ("--anchor", "2=4.6"), ("--approve-range", "1:16.4"),
            ("--exclude-range", "7:7.5"), ("--split", "validation"),
        ):
            self.approve("pair-A")
            self.run_cli("review", "--id", "pair-A", "--reviewer", "source-owner", "--confirm-percussion-completeness")
            report = self.run_cli("review", "--id", "pair-A", "--reviewer", "source-owner", *changed)
            with self.subTest(changed=changed):
                self.assertNotIn("percussionAnnotationsComplete", report["approval"])
        self.run_cli("review", "--id", "pair-A", "--reviewer", "source-owner", "--confirm-percussion-completeness")
        (directory / "raw.gp").write_bytes(archive_bytes(fresh_score("owner-selected revision")))
        self.run_cli("review", "--id", "pair-A", "--reviewer", "source-owner", "--confirm-percussion-completeness", error="invalidate with a reason")
        self.run_cli("invalidate", "--id", "pair-A", "--reason", "synthetic source revision")
        self.run_cli("prepare", "--ids", "pair-A")
        self.assertNotIn("percussionAnnotationsComplete", self.run_cli("review", "--id", "pair-A")["approval"])
        self.assertTrue(read_json(directory / "history.json")[-1]["review"]["approval"]["percussionAnnotationsComplete"])

    def test_release_projects_only_confirmed_sources_and_preserves_old_release(self):
        self.prepare_two(unknown=True)
        self.approve("pair-A", exclude=True)
        self.approve("pair-B")
        old_release = Path(self.release()["manifestPath"]).parent
        frozen = {path: sha256(path) for path in old_release.rglob("*") if path.is_file()}
        musical = {path: sha256(path) for identifier in ("pair-A", "pair-B") for name in ARTIFACTS for path in [self.workspace / "pairs" / identifier / name]}
        self.run_cli("review", "--id", "pair-A", "--reviewer", "source-owner", "--confirm-percussion-completeness")
        released = self.release("v2")
        manifest, records, _ = validate_release(released["manifestPath"])
        for entry, payload in records:
            approved = entry["id"] == "pair-A"
            scope = next(item for item in manifest["releaseAuthorization"]["selectedScope"] if item["id"] == entry["id"])
            self.assertEqual(scope.get("percussionAnnotationsComplete"), True if approved else None)
            coverage = percussion_annotation_coverage(payload["canonical"], payload["candidate"], payload["normalization"]) if approved else None
            if approved:
                self.assertTrue(coverage)
                self.assertFalse(any(left <= 3.4 < right for left, right in coverage))
            for window in payload["windows"]:
                targets = window["targets"]
                self.assertEqual(targets["negativePercussionSupervision"], approved)
                if approved:
                    left, right = window["startSample"] / entry["sampleRate"], window["stopSampleExclusive"] / entry["sampleRate"]
                    self.assertEqual(targets["percussionAnnotationCoverage"], [[max(a, left) - left, min(b, right) - left] for a, b in coverage if max(a, left) < min(b, right)])
                else:
                    self.assertNotIn("percussionAnnotationCoverage", targets)
        self.assertEqual(frozen, {path: sha256(path) for path in frozen})
        self.assertEqual(musical, {path: sha256(path) for path in musical})
        validate_release(old_release / "manifest.json")

    def test_optional_title_is_retained_in_registry_binding_status_and_review(self):
        title = "A Readable Song (Fingerstyle)"
        self.add("sample-a", title=title)
        self.add("sample-b", 2)
        pairs = read_json(self.workspace / "pairs.json")["pairs"]
        self.assertEqual(pairs[0]["title"], title)
        self.assertNotIn("title", pairs[1])
        blocked = self.run_cli("status")["pairs"]
        self.assertEqual((blocked[0]["id"], blocked[0]["title"], blocked[0]["status"]), ("sample-a", title, "blocked"))
        self.assertNotIn("title", blocked[1])
        self.run_cli("prepare", "--accept-conventions")
        directory = self.workspace / "pairs" / "sample-a"
        self.assertEqual(read_json(directory / "preparation.json")["inputs"]["pair"], pairs[0])
        self.assertEqual(self.run_cli("status")["pairs"][0]["title"], title)
        review = self.run_cli("review", "--id", "sample-a")
        self.assertEqual((review["id"], review["title"]), ("sample-a", title))
        self.assertEqual(read_json(directory / "review-report.json")["title"], title)
        self.assertNotIn("title", self.run_cli("review", "--id", "sample-b"))
        self.approve("sample-a")
        reviewed = self.run_cli("status")["pairs"][0]
        self.assertEqual((reviewed["title"], reviewed["status"]), (title, "reviewed"))

    def test_supplied_titles_must_be_nonempty_strings(self):
        gp, audio = self.add("pair-A")
        self.run_cli("add", "--id", "pair-B", "--group", "group-B", "--gp", str(gp), "--audio", str(audio), "--title", " ", error="title must be a nonempty string")
        self.assertFalse((self.workspace / "pairs" / "pair-B").exists())
        registry = self.workspace / "pairs.json"
        document = read_json(registry)
        for title in ("", " ", None, 12, []):
            with self.subTest(title=title):
                document["pairs"][0]["title"] = title
                registry.write_text(json.dumps(document), encoding="utf-8")
                self.run_cli("status", error="title must be a nonempty string")

    def test_optional_preparation_provenance_is_opaque_and_does_not_rerun_dsp(self):
        self.add("pair-A")
        self.run_cli("prepare", "--accept-conventions")
        directory = self.workspace / "pairs" / "pair-A"
        path = directory / "preparation.json"
        state = read_json(path)
        inputs, artifacts = state["inputs"], state["artifacts"]
        state["provenance"] = {"archivedEvidencePath": "unavailable\\historical-evidence.zip", "originalSourceFirstSample": 12345}
        path.write_text(json.dumps(state), encoding="utf-8")
        with patch("scripts.prepare_training_data.automatic_candidate", side_effect=AssertionError("Existing preparations must not rerun DSP")):
            self.assertEqual(self.run_cli("prepare")[0]["status"], "reused")
            self.approve("pair-A")
            self.assertEqual(self.run_cli("status")["pairs"][0]["status"], "reviewed")
        self.assertEqual(read_json(path)["inputs"], inputs)
        self.assertEqual({name: sha256(directory / name) for name in ARTIFACTS}, artifacts)
        self.assertEqual(read_json(directory / "review.json")["preparationSha256"], candidate_digest(state))

    def test_default_workspace_and_nonempty_directory_guard(self):
        self.assertEqual(parser().parse_args(["init"]).workspace, ROOT / "data")
        with patch("scripts.prepare_training_data.ROOT", self.root):
            self.assertEqual(workspace_path(self.root / "data"), self.root / "data")
            self.assertEqual(workspace_path(self.root / "datasets" / "custom"), self.root / "datasets" / "custom")
            for name in ("pairs", "releases", "archive"):
                with self.assertRaises(ValueError):
                    workspace_path(self.root / "data" / name / "nested")
        nonempty = self.root / "nonempty"
        nonempty.mkdir()
        asset = nonempty / "do-not-adopt.gp"
        asset.write_bytes(b"owner material")
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            self.assertEqual(main(["--workspace", str(nonempty), "init"]), 2)
        self.assertEqual(asset.read_bytes(), b"owner material")
        self.assertFalse((nonempty / "pairs.json").exists())

    def test_owned_imports_are_independent_and_flac_is_not_duplicated(self):
        gp = self.inputs / "original.gp"
        audio = self.inputs / "original.flac"
        gp.write_bytes(archive_bytes(fresh_score("owned raw score")))
        fresh_audio(audio)
        pair = self.run_cli("add", "--id", "pair-A", "--group", "group-A", "--gp", str(gp), "--audio", str(audio))
        self.assertEqual(pair["gpPath"], "pairs\\pair-A\\raw.gp")
        self.assertEqual(pair["audioPath"], "pairs\\pair-A\\trimmed.flac")
        owned_gp, owned_audio = source_paths(self.workspace, pair)
        before = {path: (sha256(path), path.stat().st_mtime_ns) for path in (owned_gp, owned_audio)}
        self.assertEqual(sha256(gp), sha256(owned_gp))
        self.assertEqual(sha256(audio), sha256(owned_audio))
        self.run_cli("prepare", "--accept-conventions")
        self.approve("pair-A")
        review = sha256(owned_gp.parent / "review.json")
        gp.write_bytes(b"external source has changed")
        audio.write_bytes(b"external audio has changed")
        self.assertEqual(self.run_cli("prepare")[0]["status"], "reused")
        self.assertEqual(sha256(owned_gp.parent / "review.json"), review)
        self.assertEqual(before, {path: (sha256(path), path.stat().st_mtime_ns) for path in before})
        self.assertEqual(list(owned_gp.parent.glob("*.flac")), [owned_audio])
        self.assertFalse(list(owned_gp.parent.glob("trimmed-source.*")))
        for field, relative in (
            ("gpPath", "pairs\\pair-A\\normalized.gp"),
            ("gpPath", "pairs\\other\\raw.gp"),
            ("audioPath", "pairs\\other\\trimmed.flac"),
            ("gpPath", str(gp)),
        ):
            with self.subTest(field=field, relative=relative), self.assertRaises(ValueError):
                source_paths(self.workspace, {**pair, field: relative})
        self.run_cli("add", "--id", "another", "--group", "group-B", "--gp", str(owned_gp), "--audio", str(owned_audio), error="another pair")

    def test_fine_override_uses_last_musical_bar_and_binds_the_rule(self):
        self.add("pair-A", legend=True)
        directory = self.workspace / "pairs" / "pair-A"
        score = fresh_score("explicitly reviewed Fine", legend=True)
        bars = score.find("MasterBars")
        ET.SubElement(ET.SubElement(bars[0], "Directions"), "Target").text = "Segno"
        ET.SubElement(ET.SubElement(bars[3], "Directions"), "Jump").text = "DaSegnoAlFine"
        gp = directory / "raw.gp"
        gp.write_bytes(archive_bytes(score))
        original = (sha256(gp), gp.stat().st_mtime_ns)
        self.run_cli("prepare", "--accept-conventions", error="Fine")
        rules_path = directory / "rules.json"
        rules = read_json(rules_path)
        rules["fineTarget"] = "last-musical-measure"
        rules_path.write_text(json.dumps(rules), encoding="utf-8")
        self.run_cli("prepare")
        events = read_json(directory / "events.json")
        self.assertEqual([visit["measureIndex"] for visit in events["playback"]["measureVisits"]], [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(read_json(directory / "normalization.json")["linearMeasureCount"], 8)
        self.assertEqual(read_json(directory / "preparation.json")["inputs"]["rulesSha256"], candidate_digest(rules))
        self.assertEqual((sha256(gp), gp.stat().st_mtime_ns), original)
        rules["fineTarget"] = None
        rules_path.write_text(json.dumps(rules), encoding="utf-8")
        self.run_cli("prepare", error="invalidate with a reason")

    def test_linked_sources_and_changed_imports_never_register_a_pair(self):
        gp, audio = self.inputs / "raw.gp", self.inputs / "raw.flac"
        gp.write_bytes(archive_bytes(fresh_score("immutable import")))
        fresh_audio(audio)
        alias = self.inputs / "alias.gp"
        alias.hardlink_to(gp)
        self.run_cli("add", "--id", "pair-A", "--group", "group-A", "--gp", str(alias), "--audio", str(audio), error="aliases")
        alias.unlink()
        copyfile = shutil.copyfile

        def changed_source(source, destination):
            result = copyfile(source, destination)
            if source == gp:
                gp.write_bytes(b"concurrent source revision")
            return result

        with patch("scripts.prepare_training_data.shutil.copyfile", side_effect=changed_source):
            self.run_cli("add", "--id", "pair-A", "--group", "group-A", "--gp", str(gp), "--audio", str(audio), error="changed during import")
        self.assertEqual(read_json(self.workspace / "pairs.json")["pairs"], [])
        self.assertEqual(list((self.workspace / "pairs").iterdir()), [])

    def test_fixed_text_confirmation_never_bypasses_partial_capo_or_retuning(self):
        score = fresh_score("fixed full capo")
        ET.SubElement(score.find("./Beats/Beat"), "FreeText").text = "Fixed tuning and full capo"
        gp = self.inputs / "text.gp"
        gp.write_bytes(archive_bytes(score))
        rules = rules_template()
        with self.assertRaisesRegex(ValueError, "capo_or_tuning_text"):
            extract_score(gp, {"id": "text"}, rules)
        rules["confirmFixedTuningCapoText"] = True
        self.assertEqual(extract_score(gp, {"id": "text"}, rules)[0]["instrument"]["capoFret"], 2)
        score.find(".//Staff/Properties/Property[@name='PartialCapoFret']/Fret").text = "2"
        score.find(".//Staff/Properties/Property[@name='PartialCapoStringFlags']/Bitset").text = "001111"
        gp.write_bytes(archive_bytes(score))
        with self.assertRaisesRegex(ValueError, "partial_capo_active"):
            extract_score(gp, {"id": "text"}, rules)
        score.find(".//Staff/Properties/Property[@name='PartialCapoFret']/Fret").text = "0"
        score.find(".//Staff/Properties/Property[@name='PartialCapoStringFlags']/Bitset").text = "000000"
        ET.SubElement(ET.SubElement(score.find("./MasterTrack/Automations"), "Automation"), "Type").text = "Tuning"
        gp.write_bytes(archive_bytes(score))
        with self.assertRaisesRegex(ValueError, "tuning_or_capo_automation"):
            extract_score(gp, {"id": "text"}, rules)

    def test_automatic_plateau_is_reviewable_and_requires_source_bound_acceptance(self):
        from types import SimpleNamespace

        self.prepare_two()
        directory = self.workspace / "pairs" / "pair-A"
        labels, normalization = read_json(directory / "canonical.json"), read_json(directory / "normalization.json")
        result = {"reference_indices": np.arange(5), "audio_indices": np.array([0, 1, 1, 2, 3]), "local_costs": np.zeros(5), "diagnostics": {}}
        with patch("scripts.prepare_training_data.audio_features", return_value=SimpleNamespace(times=np.arange(4, dtype=float))), patch("scripts.prepare_training_data.reference_features", return_value=SimpleNamespace(times=np.arange(5, dtype=float))), patch("scripts.prepare_training_data.align_first_attack", return_value=result):
            candidate = automatic_candidate(labels, normalization, directory / "trimmed.flac")
        self.assertEqual(candidate["method"], "first-attack-dtw")
        self.assertEqual([point["clipSeconds"] for point in candidate["denseMapping"]], [0., 1., 1., 2., 3.])
        self.assertEqual(candidate["timingRisks"], [{"kind": "score-time-plateau", "clipSeconds": 1., "referenceSecondsStart": 1., "referenceSecondsEnd": 2.}])
        self.run_cli("invalidate", "--id", "pair-A", "--reason", "synthetic plateau candidate")
        with patch("scripts.prepare_training_data.automatic_candidate", return_value=candidate):
            self.run_cli("prepare", "--ids", "pair-A")
        self.approve("pair-B")
        args = ("review", "--id", "pair-A", "--reviewer", "synthetic-reviewer", "--approve-range", "0:2", "--authorize-use", "--confirm-pitch", "--confirm-notation", "--confirm-grouping", "--approve-experimental-ranges")
        report = self.run_cli(*args)
        self.assertEqual(report["timingRisks"], candidate["timingRisks"])
        self.run_cli("release", "--version", "plateau", "--validation-group", "group-pair-B", error="acknowledge-uncertainty")
        self.run_cli("review", "--id", "pair-A", "--reviewer", "synthetic-reviewer", "--acknowledge-uncertainty")
        _, records, _ = validate_release(self.release("plateau")["manifestPath"])
        self.assertEqual(records[0][1]["candidate"]["denseMapping"], candidate["denseMapping"])
        (directory / "raw.gp").write_bytes(archive_bytes(fresh_score("changed owned source")))
        self.run_cli("review", "--id", "pair-A", error="invalidate with a reason")

    def test_fresh_pairs_real_normalization_alignment_review_and_portable_release(self):
        prepared = self.prepare_two(unknown=True)
        self.assertTrue(all(item["status"] == "prepared" for item in prepared))
        # The ordinary, informative one-track fixture actually exercises the DSP matcher.
        self.assertTrue(any(item["alignment"] == "first-attack-dtw" for item in prepared), prepared)
        original_hashes = {path: sha256(path) for path in self.inputs.iterdir()}
        for identifier in ("pair-A", "pair-B"):
            directory = self.workspace / "pairs" / identifier
            source, source_rate = sf.read(self.inputs / f"{identifier}.wav", dtype="int32", always_2d=True)
            copy, copy_rate = sf.read(directory / "trimmed.flac", dtype="int32", always_2d=True)
            self.assertEqual(source_rate, copy_rate)
            np.testing.assert_array_equal(source, copy)
            self.assertEqual(copy.shape[1], 2)
            self.assertLessEqual(np.max(np.abs(copy[:, 0] + copy[:, 1])), 256)
            self.assertEqual(sha256(directory / "raw.gp"), sha256(self.inputs / f"{identifier}.gp"))
            normalization = read_json(directory / "normalization.json")
            self.assertEqual(normalization["linearMeasureCount"], 4)
            self.assertTrue(normalization["normalizedTempoEvents"])
        self.approve("pair-A", exclude=True)
        self.approve("pair-B")
        report = self.run_cli("review", "--id", "pair-A", "--cue", "1", "--cue", "first-attack")
        self.assertEqual(report["barsAndFirstAttack"][0]["ordinal"], 1)
        self.assertEqual(report["barsAndFirstAttack"][0]["displayedBar"], 0)
        self.assertEqual(report["approvedClipRanges"], [[.4, 6.5], [8., 16.4]])
        self.assertEqual(len(report["cues"]), 2)
        excerpt = report["cues"][0]
        plain, rate = sf.read(excerpt["plainPath"], dtype="int32", always_2d=True)
        original, _ = sf.read(self.workspace / "pairs" / "pair-A" / "trimmed.flac", dtype="int32", always_2d=True)
        start = round(excerpt["excerptStartSeconds"] * rate)
        np.testing.assert_array_equal(plain, original[start:start + len(plain)])
        released = self.release()
        manifest_path = Path(released["manifestPath"])
        manifest, records, _ = validate_release(manifest_path)
        self.assertTrue(manifest["trainingReady"])
        self.assertTrue(all(manifest["counts"]["windowsBySplit"].values()))
        self.assertEqual({path: sha256(path) for path in original_hashes}, original_hashes)
        unresolved = records[0][1]["canonical"]["review"]["unresolvedGestures"]
        self.assertTrue(unresolved)
        self.assertTrue(all(item["labelMask"]["gesture"] is False for item in unresolved))
        self.assertFalse(records[0][1]["canonical"]["targets"]["gestures"])
        for entry, payload in records:
            directory = self.workspace / "pairs" / entry["id"]
            self.assertEqual(payload["canonical"], read_json(directory / "canonical.json"))
            self.assertEqual(payload["normalization"], read_json(directory / "normalization.json"))
            self.assertEqual(payload["approval"]["sourceGpSha256"], sha256(self.inputs / f"{entry['id']}.gp"))
            self.assertEqual(
                set(payload["downbeatConditioning"]),
                {"pickupPresent", "scoreQuarter", "clipSeconds", "source"},
            )
            self.assertGreaterEqual(payload["downbeatConditioning"]["clipSeconds"], 0)
            self.assertFalse(payload["windows"][0]["targets"]["negativePercussionSupervision"])
            for window in payload["windows"]:
                a, b = window["startSample"] / entry["sampleRate"], window["stopSampleExclusive"] / entry["sampleRate"]
                self.assertTrue(any(left <= a < b <= right for left, right in payload["approval"]["approvedClipRanges"]))
        moved = self.root / "moved-release"
        shutil.copytree(manifest_path.parent, moved)
        shutil.rmtree(self.workspace)
        shutil.rmtree(self.inputs)
        validate_release(moved / "manifest.json")
        with redirect_stdout(StringIO()):
            self.assertEqual(transcriber.main(["preflight", "--data-root", str(moved), "--manifest", "manifest.json"]), 0)
        preflight = read_json(moved / "runs" / "preflight.json")
        self.assertFalse(preflight["trainingRun"])

    def test_rerun_revision_invalidates_approval_without_changing_frozen_release(self):
        self.prepare_two()
        self.approve("pair-A")
        self.approve("pair-B")
        first = Path(self.release()["manifestPath"]).parent
        frozen = {str(path.relative_to(first)): sha256(path) for path in first.rglob("*") if path.is_file()}
        original = {path: (sha256(path), path.stat().st_mtime_ns) for identifier in ("pair-A", "pair-B") for name in ARTIFACTS for path in [self.workspace / "pairs" / identifier / name]}
        rerun = self.run_cli("prepare")
        self.assertTrue(all(item["status"] == "reused" for item in rerun))
        self.assertEqual(original, {path: (sha256(path), path.stat().st_mtime_ns) for path in original})
        self.release()
        gp = self.workspace / "pairs" / "pair-A" / "raw.gp"
        gp.write_bytes(archive_bytes(fresh_score("deliberate new revision", pickup=True)))
        self.run_cli("prepare", "--ids", "pair-A", error="invalidate with a reason")
        self.run_cli("review", "--id", "pair-A", error="invalidate with a reason")
        self.run_cli("release", "--version", "v2", "--validation-group", "group-pair-B", error="invalidate with a reason")
        self.run_cli("invalidate", "--id", "pair-A", "--reason", "synthetic owner-selected new GP revision")
        self.run_cli("prepare", "--ids", "pair-A")
        self.assertFalse((self.workspace / "pairs" / "pair-A" / "review.json").exists())
        self.run_cli("release", "--version", "v2", "--validation-group", "group-pair-B", error="missing source/notation/use/group/range approval")
        self.approve("pair-A")
        self.run_cli("release", "--version", "v1", "--validation-group", "group-pair-B", error="NEW version")
        self.release("v2")
        self.assertEqual(frozen, {str(path.relative_to(first)): sha256(path) for path in first.rglob("*") if path.is_file()})
        validate_release(first / "manifest.json")
        self.assertEqual(len(read_json(self.workspace / "pairs" / "pair-A" / "history.json")), 1)
        self.assertFalse(any(path.name.startswith(".preparing-") or path.name.endswith(".building") for path in self.workspace.rglob("*")))

    def test_sparse_anchor_confirmation_is_not_range_or_split_approval(self):
        self.prepare_two()
        self.run_cli("review", "--id", "pair-A", "--reviewer", "synthetic-reviewer", "--anchor", "1=0.4", "--anchor", "end=16.4", "--confirm-notation")
        self.approve("pair-B")
        self.run_cli("release", "--version", "v1", "--validation-group", "group-pair-B", error="missing source/notation/use/group/range approval")
        self.run_cli("review", "--id", "pair-A", "--reviewer", "synthetic-reviewer", "--approve-experimental-ranges", error="requires explicit")
        self.approve("pair-A")
        self.run_cli("review", "--id", "pair-A", "--reviewer", "synthetic-reviewer", "--anchor", "2=4.6")
        self.run_cli("release", "--version", "v1", "--validation-group", "group-pair-B", error="missing source/notation/use/group/range approval")
        self.run_cli("review", "--id", "pair-A", "--reviewer", "synthetic-reviewer", "--anchor", "3=2.0", error="strictly")

    def test_failed_automatic_alignment_can_be_completed_with_manual_anchors(self):
        self.add("pair-A", rate=96000)
        self.add("pair-B", 2)
        prepared = self.run_cli("prepare", "--accept-conventions")
        self.assertEqual(prepared[0]["alignment"], "needs-manual-anchors")
        self.assertIn("48000", prepared[0]["alignmentError"])
        self.run_cli("review", "--id", "pair-A", "--reviewer", "synthetic-reviewer", "--approve-range", ".4:16.4", error="at least two")
        self.approve("pair-A")
        self.approve("pair-B")
        _, records, _ = validate_release(self.release()["manifestPath"])
        self.assertEqual(records[0][0]["sampleRate"], 96000)
        self.assertEqual(records[0][1]["candidate"]["method"], "manual-anchors-linear-nominal-time")

    def test_source_specific_rules_need_no_internal_hash_authoring(self):
        self.add("pair-A", legend=True)
        self.run_cli("prepare", "--ids", "pair-A", error="accept-conventions")
        rules_path = self.workspace / "pairs" / "pair-A" / "rules.json"
        rules = read_json(rules_path)
        rules["rules"] = [{
            "id": "source-side-tap", "technique": "body_tap", "attributes": {"location": "side"},
            "evidenceBeatIds": ["m4:v0:b0"], "match": {"tokens": ["*"]},
            "consumedTokens": ["*"], "consumesDeadNotes": False,
        }]
        rules_path.write_text(json.dumps(rules), encoding="utf-8")
        self.run_cli("prepare", "--ids", "pair-A", "--accept-conventions")
        labels = read_json(self.workspace / "pairs" / "pair-A" / "canonical.json")
        self.assertEqual(labels["targets"]["gestures"][0]["technique"], "percussive_hit")
        self.assertEqual(len(labels["measureVisits"]), 4)
        self.assertTrue(any(beat["referenceOnly"] for beat in read_json(self.workspace / "pairs" / "pair-A" / "notation.json")["beats"]))

    def test_full_capo_confirmation_handles_only_inactive_stale_metadata(self):
        self.add("pair-A")
        directory = self.workspace / "pairs" / "pair-A"
        gp = directory / "raw.gp"
        root = fresh_score("reviewed full capo")
        properties = root.find("./Tracks/Track/Staves/Staff/Properties")
        for name, child, text in (("PartialCapoFret", "Fret", "4"), ("PartialCapoStringFlags", "Bitset", "000000")):
            prop = properties.find(f"Property[@name='{name}']")
            if prop is None:
                prop = ET.SubElement(properties, "Property", name=name)
                ET.SubElement(prop, child)
            prop.find(child).text = text
        gp.write_bytes(archive_bytes(root))
        original = sha256(gp)
        self.run_cli("prepare", "--accept-conventions", error="inconsistent_partial_capo_metadata")
        rules_path = directory / "rules.json"
        rules = read_json(rules_path)
        rules["confirmFullCapoMetadata"] = True
        rules_path.write_text(json.dumps(rules), encoding="utf-8")
        self.run_cli("prepare")
        self.assertEqual(sha256(gp), original)
        self.run_cli("invalidate", "--id", "pair-A", "--reason", "synthetic active partial capo")
        properties.find("Property[@name='PartialCapoStringFlags']/Bitset").text = "001111"
        gp.write_bytes(archive_bytes(root))
        self.run_cli("prepare", error="partial_capo_active")

    def test_missing_tempo_and_unsafe_identity_never_get_musical_defaults(self):
        gp, audio = self.add("pair-A")
        score = fresh_score("no initial tempo")
        score.find("./MasterTrack/Automations").clear()
        (self.workspace / "pairs" / "pair-A" / "raw.gp").write_bytes(archive_bytes(score))
        self.run_cli("prepare", "--accept-conventions", error="tempo")
        self.assertFalse((self.workspace / "pairs" / "pair-A" / "preparation.json").exists())
        self.run_cli("add", "--id", "CON", "--group", "new", "--gp", str(gp), "--audio", str(audio), error="device")
        self.run_cli("add", "--id", "PAIR-a", "--group", "new", "--gp", str(gp), "--audio", str(audio), error="already exists")

    def test_group_or_duplicate_recording_cannot_cross_splits(self):
        self.prepare_two()
        self.approve("pair-A", split="validation")
        self.approve("pair-B")
        self.run_cli("release", "--version", "v1", "--validation-group", "group-pair-B", error="reviewed split differs")
        self.approve("pair-A")
        self.run_cli("invalidate", "--id", "pair-B", "--reason", "duplicate recording synthetic check")
        shutil.copyfile(self.workspace / "pairs" / "pair-A" / "trimmed-source.wav", self.workspace / "pairs" / "pair-B" / "trimmed-source.wav")
        self.run_cli("prepare", "--ids", "pair-B")
        self.approve("pair-B")
        self.run_cli("release", "--version", "v1", "--validation-group", "group-pair-B", error="Identical audio")
        self.assertFalse((self.workspace / "releases" / "v1").exists())

    def test_compressed_audio_preserves_decoded_pcm_and_existing_flac_bytes(self):
        try:
            ffmpeg = executable("ffmpeg")
            executable("ffprobe")
        except AcquisitionError as error:
            self.skipTest(str(error))
        gp, wave = self.inputs / "pair-A.gp", self.inputs / "pair-A.wav"
        gp.write_bytes(archive_bytes(fresh_score("pair-A")))
        fresh_audio(wave, rate=32000)
        other_gp, other = self.inputs / "pair-B.gp", self.inputs / "pair-B.wav"
        other_gp.write_bytes(archive_bytes(fresh_score("pair-B", 2)))
        fresh_audio(other, 2)
        mp3, flac = wave.with_suffix(".mp3"), other.with_suffix(".flac")
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-n", "-i", str(wave), "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True, capture_output=True)
        samples, rate = sf.read(other, dtype="int32", always_2d=True)
        sf.write(flac, samples, rate, subtype="PCM_24")
        for identifier, score, audio in (("pair-A", gp, mp3), ("pair-B", other_gp, flac)):
            self.run_cli("add", "--id", identifier, "--group", f"group-{identifier}", "--gp", str(score), "--audio", str(audio))
        original = {path: sha256(path) for path in (mp3, flac, wave, other)}
        self.run_cli("prepare", "--accept-conventions")
        converted = self.workspace / "pairs" / "pair-A" / "trimmed.flac"

        def pcm(path):
            return subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-f", "s24le", "-c:a", "pcm_s24le", "pipe:1"], capture_output=True, check=True).stdout

        self.assertEqual(pcm(mp3), pcm(converted))
        self.assertEqual(sha256(flac), sha256(self.workspace / "pairs" / "pair-B" / "trimmed.flac"))
        info = sf.info(converted)
        self.assertEqual((info.samplerate, info.channels, info.frames), (32000, 2, 17 * 32000))
        self.assertEqual(original, {path: sha256(path) for path in original})

    def test_float_pcm_is_not_silently_quantized(self):
        self.add("pair-A")
        audio = self.workspace / "pairs" / "pair-A" / "trimmed-source.wav"
        sf.write(audio, np.ones((8000, 2)) * .1, 8000, subtype="FLOAT")
        self.run_cli("prepare", "--accept-conventions", error="silent quantization")
        self.assertFalse((self.workspace / "pairs" / "pair-A" / "preparation.json").exists())

    def test_release_preserves_complete_tempo_ramp_clock_and_half_open_targets(self):
        self.add("pair-A")
        gp = self.workspace / "pairs" / "pair-A" / "raw.gp"
        self.add("pair-B", 2)
        root = fresh_score("tempo ramp source")
        root.find("./MasterTrack/Automations/Automation/Linear").text = "true"
        root.find("./MasterTrack/Automations").append(ET.fromstring(
            "<Automation><Type>Tempo</Type><Bar>1</Bar><Position>0</Position>"
            "<Value>90 2</Value><Linear>false</Linear></Automation>"
        ))
        gp.write_bytes(archive_bytes(root))
        self.run_cli("prepare", "--accept-conventions")
        self.approve("pair-A")
        self.approve("pair-B")
        _, records, _ = validate_release(self.release()["manifestPath"])
        payload = records[0][1]
        normalization = payload["normalization"]
        self.assertEqual(normalization["durationQuarter"], [16, 1])
        self.assertEqual(len(normalization["normalizedTempoEvents"]), 2)
        for event in normalization["normalizedTempoEvents"]:
            self.assertTrue({"positionRatio", "offsetQuarter", "quarterBpm", "bpm", "beatUnit", "linear"} <= event.keys())
        clock = ScoreClock(payload["canonical"], normalization)
        mapping = payload["candidate"]["denseMapping"]
        for point in mapping:
            self.assertEqual(point["scoreQuarter"], clock.quarter_at(point["referenceSeconds"]))
        for a, b in zip(mapping, mapping[1:]):
            self.assertLess(a["clipSeconds"], b["clipSeconds"])
            self.assertLess(a["referenceSeconds"], b["referenceSeconds"])
        first_window = payload["windows"][0]
        stop = first_window["stopSampleExclusive"] / records[0][0]["sampleRate"]
        self.assertTrue(all(note["proposedOnsetClipSeconds"] < stop for note in first_window["targets"]["notes"]))
        self.assertNotIn(8.4, [note["proposedOnsetClipSeconds"] for note in first_window["targets"]["notes"]])

    def test_release_authorizes_complete_scope_and_finalizes_split_neutral_reviews(self):
        self.prepare_two()
        for identifier in ("pair-A", "pair-B"):
            self.run_cli(
                "review", "--id", identifier, "--reviewer", "synthetic-source-reviewer",
                "--anchor", "1=0.4", "--anchor", "end=16.4", "--approve-range", "0.4:16.4",
                "--authorize-use", "--confirm-pitch", "--confirm-notation", "--confirm-grouping",
                "--approve-experimental-ranges", "--acknowledge-uncertainty",
            )
            source_review = read_json(self.workspace / "pairs" / identifier / "review.json")
            self.assertNotIn("split", source_review["approval"])
        self.run_cli("release", "--version", "v1", "--validation-group", "group-pair-B", authorize_release=False, error="--authorize-release")
        manifest, records, _ = validate_release(self.release()["manifestPath"])
        authorization = manifest["releaseAuthorization"]
        self.assertEqual(manifest["validationGroups"], ["group-pair-B"])
        self.assertEqual(authorization["validationGroups"], ["group-pair-B"])
        self.assertNotIn("validationGroup", manifest)
        self.assertNotIn("validationGroup", authorization)
        self.assertEqual([(item["id"], item["split"]) for item in authorization["selectedScope"]], [("pair-A", "train"), ("pair-B", "validation")])
        self.assertEqual(authorization["sha256"], candidate_digest({key: value for key, value in authorization.items() if key != "sha256"}))
        for entry, payload in records:
            self.assertEqual(payload["approval"]["groupId"], entry["groupId"])
            self.assertEqual(payload["approval"]["split"], entry["split"])
            self.assertEqual(payload["approval"]["releaseAuthorizationSha256"], authorization["sha256"])
            self.assertEqual(payload["approval"]["releaseReviewer"], "synthetic-release-owner")
            self.assertNotIn("split", read_json(self.workspace / "pairs" / entry["id"] / "review.json")["approval"])


if __name__ == "__main__":
    unittest.main()
