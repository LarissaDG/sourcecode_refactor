import unittest
import pandas as pd

from src.preprocessing.columns import initialize_columns, add_score_columns


class TestScores(unittest.TestCase):

    def test_initialize_columns(self):
        df = pd.DataFrame({"filename": ["a.jpg"]})
        df = initialize_columns(df)

        self.assertIn("Description", df.columns)
        self.assertIn("Total aesthetic score", df.columns)
        self.assertEqual(df["Description"].iloc[0], "")

    def test_add_score_columns(self):
        df = pd.DataFrame({
            "Creativity": [1, 2],
            "Color": [3, 4]
        })

        df = add_score_columns(df, ["Creativity", "Color"])

        self.assertIn("Total Score", df.columns)
        self.assertEqual(df["Total Score"].tolist(), [4, 6])
        self.assertEqual(df["Avg Score"].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
