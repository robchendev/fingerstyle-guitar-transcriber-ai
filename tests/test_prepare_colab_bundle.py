import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from scripts.prepare_colab_bundle import build_colab_bundle


class ColabBundleTests(unittest.TestCase):
    def test_builds_portable_mac_archive_with_gpu_device_and_stage_families(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "data" / "releases" / "corrected" / "manifest.json"
            gp = root / "data" / "pairs" / "song" / "raw.gp"
            audio = root / "data" / "releases" / "corrected" / "audio" / "song.flac"
            video = root / "runs" / "video-evidence" / "sources" / "song" / "source.mkv"
            reuse = root / "runs" / "video-evidence" / "hands" / "song" / "hands.json"
            hand_model = root / "runs" / "video-evidence" / "models" / "hand_landmarker.task"
            detector = root / "downloads" / "best.pt"
            for path, content in (
                (release, "{}"), (gp, "gp"), (audio, "audio"), (video, "video"),
                (reuse, "{}"), (reuse.with_name("hands.npz"), "arrays"),
                (hand_model, "hands"), (detector, "detector"),
                (root / "data" / "pairs.json", "{}"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            batch = root / "runs" / "source-batch.json"
            batch.parent.mkdir(parents=True, exist_ok=True)
            batch.write_text(json.dumps({
                "schemaVersion": 1,
                "kind": "paired-preparation-batch",
                "workspace": r"C:\old\data",
                "releaseManifest": r"C:\old\data\releases\old\manifest.json",
                "sourceMapping": "obsolete",
                "records": [{
                    "id": "song", "groupId": "group", "split": "train",
                    "gp": str(gp), "audio": str(audio), "video": str(video),
                    "videoReceipt": "obsolete", "pluckingScreenSide": "left",
                    "reuse": {"hands": str(reuse), "bundle": "obsolete"},
                }],
            }))
            output = root / "colab-input.tar"

            def bundle_creator(_root, destination):
                destination.write_bytes(b"git bundle")

            report = build_colab_bundle(
                batch, release, detector, output,
                hand_model=hand_model, fretboard_device="0", workers=1,
                root=root, git_bundle_creator=bundle_creator,
            )
            self.assertEqual(report["records"], 1)
            self.assertEqual(report["fretboardDevice"], "0")
            with tarfile.open(output) as archive:
                names = set(archive.getnames())
                expected = {
                    "fingerstyle.bundle",
                    "runs/colab-batch.json",
                    "runs/fretboard-detector-platform/best.pt",
                    "runs/video-evidence/models/hand_landmarker.task",
                    "runs/video-evidence/hands/song/hands.json",
                    "runs/video-evidence/hands/song/hands.npz",
                    "runs/video-evidence/sources/song/source.mkv",
                    "data/releases/corrected/manifest.json",
                    "data/releases/corrected/audio/song.flac",
                    "data/pairs/song/raw.gp",
                }
                self.assertTrue(expected <= names)
                manifest = json.load(io.TextIOWrapper(
                    archive.extractfile("runs/colab-batch.json"),
                    encoding="utf-8",
                ))
            self.assertEqual(manifest["videoPython"], "__COLAB_PYTHON__")
            self.assertEqual(manifest["fretboardDevice"], "0")
            self.assertEqual(manifest["workers"], 1)
            self.assertNotIn("sourceMapping", manifest)
            self.assertNotIn("videoReceipt", manifest["records"][0])
            self.assertNotIn("bundle", manifest["records"][0]["reuse"])
            self.assertEqual(
                manifest["records"][0]["video"],
                "runs/video-evidence/sources/song/source.mkv",
            )

    def test_refuses_overwrite_and_invalid_device(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing.tar"
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                build_colab_bundle(
                    "missing", "missing", "missing", output, root=root,
                )


if __name__ == "__main__":
    unittest.main()
