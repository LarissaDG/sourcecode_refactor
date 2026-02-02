import os
import zipfile
import tempfile
import unittest
import pandas as pd

from src.utils.file_operations import unzip_file, load_csv, save_csv


class TestFileOperations(unittest.TestCase):

    def test_unzip_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "test.zip")
            unzip_dir = os.path.join(tmpdir, "unzipped")
            os.makedirs(unzip_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("test.txt", "hello")

            unzip_file(zip_path, unzip_dir)

            self.assertTrue(os.path.exists(os.path.join(unzip_dir, "test.txt")))

    def test_load_save_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2]})
            csv_path = os.path.join(tmpdir, "test.csv")
            output_path = os.path.join(tmpdir, "out.csv")

            df.to_csv(csv_path, index=False)

            loaded = load_csv(csv_path)
            self.assertTrue(loaded.equals(df))

            save_csv(df, output_path)
            self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
