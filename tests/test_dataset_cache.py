import importlib
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class DatasetCachePathTests(unittest.TestCase):
    def test_dataset_cache_dir_uses_hf_home(self):
        import common.paths as paths

        with TemporaryDirectory() as tmp:
            old_hf_home = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = tmp
            try:
                paths = importlib.reload(paths)
                self.assertEqual(
                    paths.dataset_cache_dir("MagicDec"),
                    Path(tmp) / "datasets" / "fast_infer_text_sum" / "MagicDec",
                )
            finally:
                if old_hf_home is None:
                    os.environ.pop("HF_HOME", None)
                else:
                    os.environ["HF_HOME"] = old_hf_home
                importlib.reload(paths)


if __name__ == "__main__":
    unittest.main()
