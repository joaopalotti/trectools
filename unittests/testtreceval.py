import unittest
import numpy as np
from trectools import TrecRun, TrecQrel, TrecEval


class TestTrecEval(unittest.TestCase):

    def setUp(self):

        run1 = TrecRun("./unittests/files/r4.run")
        qrels1 = TrecQrel("./unittests/files/qrel1.txt")

        run2 = TrecRun("./unittests/files/input.uic0301")
        qrels2 = TrecQrel("./unittests/files/robust03_cs_qrels.txt")

        # Contains the first 30 documents for the first 10 topics in input.uic0301
        run3 = TrecRun("./unittests/files/input.uic0301_top30")

        # All documents retrieved are not relevant to qrel1.txt
        no_rel_run = TrecRun("./unittests/files/r5.run")
        # All documents retrieved have no labels
        no_labels_run = TrecRun("./unittests/files/r6.run")

        self.common_topics = ["303", "307", "310", "314", "320", "322", "325", "330", "336", "341"]
        self.teval1 = TrecEval(run1, qrels1)
        self.teval2 = TrecEval(run2, qrels2)
        self.teval3 = TrecEval(run3, qrels2)
        self.teval_no_rel = TrecEval(no_rel_run, qrels1)
        self.teval_no_labels = TrecEval(no_labels_run, qrels1)

    def tearDown(self):
        pass

    def test_get_reciprocal_rank(self):

        value = self.teval1.get_reciprocal_rank(depth=1000, trec_eval=True)
        self.assertAlmostEqual(value, 0.5000, places=4)

        value = self.teval2.get_reciprocal_rank(depth=1000, trec_eval=True)
        self.assertAlmostEqual(value, 0.6466, places=4)

        values1 = self.teval2.get_reciprocal_rank(depth=30, per_query=True, trec_eval=True).loc[self.common_topics].values
        values2 = self.teval3.get_reciprocal_rank(depth=1000, per_query=True, trec_eval=True).loc[self.common_topics].values
        for v1, v2 in zip(values1, values2):
            self.assertAlmostEqual(v1, v2, places=4)

        results = self.teval2.get_reciprocal_rank(depth=1000, trec_eval=True, per_query=True)
        correct_results = [0.0017, 0.1429, 0.3333]
        values = map(float, results.loc[["378","650","624"]].values)
        for v, c in zip(values, correct_results):
            self.assertAlmostEqual(v, c, places=4)

    def test_get_map(self):

        value = self.teval1.get_map(depth=1000, trec_eval=True)
        self.assertAlmostEqual(value, 0.2685, places=4)

        value = self.teval2.get_map(depth=1000, trec_eval=True)
        self.assertAlmostEqual(value, 0.2396, places=4)

        values1 = self.teval2.get_map(depth=30, per_query=True, trec_eval=True).loc[self.common_topics].values
        values2 = self.teval3.get_map(depth=1000, per_query=True, trec_eval=True).loc[self.common_topics].values
        for v1, v2 in zip(values1, values2):
            self.assertAlmostEqual(v1, v2, places=4)

        results = self.teval2.get_map(depth=1000, trec_eval=True, per_query=True)
        correct_results = [0.4926, 0.2808, 0.2335]
        values = map(float, results.loc[["622", "609", "320"]].values)
        for v, c in zip(values, correct_results):
            self.assertAlmostEqual(v, c, places=4)

    def test_get_precision(self):
        value = self.teval1.get_precision(depth=1000, trec_eval=True)
        self.assertAlmostEqual(value, 0.0010, places=4)

        value = self.teval2.get_precision(depth=1000, trec_eval=True)
        self.assertAlmostEqual(value, 0.0371, places=4)

        values1 = self.teval2.get_precision(depth=30, per_query=True, trec_eval=True).loc[self.common_topics].values
        values2 = self.teval3.get_precision(depth=30, per_query=True, trec_eval=True).loc[self.common_topics].values
        for v1, v2 in zip(values1, values2):
            self.assertAlmostEqual(v1, v2, places=4)

        values1 = self.teval2.get_precision(depth=500, per_query=True, trec_eval=True).loc[self.common_topics].values
        values2 = self.teval3.get_precision(depth=500, per_query=True, trec_eval=True).loc[self.common_topics].values
        for v1, v2 in zip(values1, values2):
            self.assertNotAlmostEqual(float(v1), float(v2), places=4)

        results = self.teval2.get_precision(depth=30, trec_eval=True, per_query=True)
        correct_results = [0.1333, 0.0333, 0.5333]
        values = map(float, results.loc[["607", "433", "375"]].values)
        for v, c in zip(values, correct_results):
            self.assertAlmostEqual(v, c, places=4)

    def test_get_recall(self):
        value = self.teval1.get_recall(depth=1000, trec_eval=True)
        self.assertAlmostEqual(value, 0.3889, places=4)

        expected_results = {
            5: 0.0858,
            10: 0.1314,
            15: 0.1686,
            20: 0.1968,
            30: 0.2530,
            100: 0.4243,
            200: 0.5323,
            500: 0.6569,
            1000: 0.7518
        }
        for recall_at in expected_results.keys():
            value = self.teval2.get_recall(depth=recall_at, trec_eval=True)
            self.assertAlmostEqual(value, expected_results[recall_at], places=4)
        
        values1 = self.teval2.get_recall(depth=30, per_query=True, trec_eval=True).loc[self.common_topics].values
        values2 = self.teval3.get_recall(depth=30, per_query=True, trec_eval=True).loc[self.common_topics].values
        for v1, v2 in zip(values1, values2):
            self.assertAlmostEqual(v1, v2, places=4)

        values1 = self.teval2.get_recall(depth=500, per_query=True, trec_eval=True).loc[self.common_topics].values
        values2 = self.teval3.get_recall(depth=500, per_query=True, trec_eval=True).loc[self.common_topics].values
        for v1, v2 in zip(values1, values2):
            self.assertNotAlmostEqual(float(v1), float(v2), places=4)

        results = self.teval2.get_recall(depth=30, trec_eval=True, per_query=True)
        correct_results = [0.3636, 0.0769, 0.2000, 0.1000]
        values = map(float, results.loc[["607", "433", "375", "303"]].values)
        for v, c in zip(values, correct_results):
            self.assertAlmostEqual(v, c, places=4)

    def test_no_relevant_retrieved(self):
        result = self.teval_no_rel.evaluate_all(per_query=True)

        # All of these should be 0
        metrics = []
        depths = [5, 10, 15, 20, 30, 100, 200, 500, 100]
        metrics += [f"NDCG_{i}" for i in depths]
        metrics += [f"P_{i}" for i in depths]
        metrics += [f"R_{i}" for i in depths]
        metrics += ["map", "Rprec", "num_rel_ret"]
        # recip_rank could arguably be included

        metrics = result.data[result.data["metric"].isin(metrics)]

        self.assertTrue(np.allclose(metrics["value"].values.astype(np.float32), 0))

    def test_no_labels_retrieved(self):
        result = self.teval_no_labels.evaluate_all(per_query=True)

        # All of these should be 0
        metrics = []
        depths = [5, 10, 15, 20, 30, 100, 200, 500, 100]
        metrics += [f"NDCG_{i}" for i in depths]
        metrics += [f"P_{i}" for i in depths]
        metrics += [f"R_{i}" for i in depths]
        metrics += ["map", "Rprec", "num_rel_ret"]
        # recip_rank could arguably be included

        metrics = result.data[result.data["metric"].isin(metrics)]
        self.assertTrue(np.allclose(metrics["value"].values.astype(np.float32), 0))




if __name__ == '__main__':
    unittest.main()
