from trectools import TrecRes, TrecRun, TrecQrel
from scipy.stats import norm
import pandas as pd
import numpy as np


class TrecEval:

    def __init__(self, run, qrels):
        """
            TrecEval performs the retrieval system evaluation.

            Params
            -------
            run: an object of type TrecRun
            qrels: an object of type TrecQrel

            Returns
            --------
            None

        """
        if not isinstance(run, TrecRun):
            raise TypeError('"run" should be a TrecRun object')

        if not isinstance(qrels, TrecQrel):
            raise TypeError('"qrels" should be a TrecQrel object')

        self.run = run
        self.qrels = qrels

        self.GMEAN_MIN = .00001 # To have the same behavior as trec_eval

    def get_runid(self):
        return self.run.get_filename()

    def evaluate_all(self, per_query=False):
        """
            Runs all evaluation metrics as the default trec_eval tool.

            Params
            -------
            per_query: If True, runs the evaluation per query. Default = False

            Returns
            --------
            An TrecRes object

        """
        run_id = self.run.get_runid()
        results_per_query = []

        if per_query:

            bpref_pq = self.get_bpref(depth=1000, per_query=True, trec_eval=True).reset_index()
            bpref_pq["metric"] = "bpref"
            bpref_pq.rename(columns={"Bpref@1000":"value"}, inplace=True)
            results_per_query.append(bpref_pq)

            for v in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
                precision_per_query = self.get_precision(depth=v, per_query=True, trec_eval=True).reset_index()
                precision_per_query["metric"] = "P_%d" % (v)
                precision_per_query.rename(columns={"P@%d" % (v): "value"}, inplace=True)
                results_per_query.append(precision_per_query)

                recall_per_query = self.get_recall(depth=v, per_query=True, trec_eval=True).reset_index()
                recall_per_query["metric"] = "R_%d" % (v)
                recall_per_query.rename(columns={"R@%d" % (v): "value"}, inplace=True)
                results_per_query.append(recall_per_query)


                ndcg_per_query = self.get_ndcg(depth=v, per_query=True, trec_eval=True).reset_index()
                ndcg_per_query["metric"] = "NDCG_%d" % (v)
                ndcg_per_query.rename(columns={"NDCG@%d" % (v): "value"}, inplace=True)
                results_per_query.append(ndcg_per_query)

            map_pq = self.get_map(depth=1000, per_query=True, trec_eval=True).reset_index()
            map_pq["metric"] = "map"
            map_pq.rename(columns={"MAP@1000":"value"}, inplace=True)
            results_per_query.append(map_pq)

            num_ret = self.get_retrieved_documents(per_query=True).reset_index()
            num_ret["metric"] = "num_ret"
            num_ret.rename(columns={"docid":"value"}, inplace=True)
            results_per_query.append(num_ret)

            num_rel = self.get_relevant_documents(per_query=True).reset_index()
            num_rel["metric"] = "num_rel"
            num_rel.rename(columns={"relevant_per_query":"value"}, inplace=True)
            results_per_query.append(num_rel)

            num_rel_ret = self.get_relevant_retrieved_documents(per_query=True).reset_index()
            num_rel_ret["metric"] = "num_rel_ret"
            num_rel_ret.rename(columns={"rel":"value"}, inplace=True)
            results_per_query.append(num_rel_ret)

            rprec = self.get_rprec(per_query=True).reset_index()
            rprec["metric"] = "Rprec"
            rprec.rename(columns={"RPrec@1000":"value"}, inplace=True)
            results_per_query.append(rprec)

            recip_rank = self.get_reciprocal_rank(per_query=True).reset_index()
            recip_rank["metric"] = "recip_rank"
            recip_rank.rename(columns={"recip_rank@1000":"value"}, inplace=True)
            results_per_query.append(recip_rank)

        ps = {}
        re = {}
        ndcg = {}
        for v in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
            ps[v] = self.get_precision(depth=v, per_query=False, trec_eval=True)
            re[v] = self.get_recall(depth=v, per_query=False, trec_eval=True)
            ndcg[v] = self.get_ndcg(depth=v, per_query=False, trec_eval=True)
        map_ = self.get_map(depth=10000, per_query=False, trec_eval=True)
        gm_map_ = self.get_geometric_map(depth=10000, trec_eval=True)
        bpref_ = self.get_bpref(depth=1000, per_query=False, trec_eval=True)
        rprec_ = self.get_rprec(depth=1000, per_query=False, trec_eval=True)
        recip_rank_ = self.get_reciprocal_rank(depth=1000, per_query=False, trec_eval=True)

        rows = [
            {"metric": "runid", "query": "all", "value": run_id},
            {"metric": "num_ret", "query": "all", "value": self.get_retrieved_documents(per_query=False)},
            {"metric": "num_rel", "query": "all", "value": self.get_relevant_documents(per_query=False)},
            {"metric": "num_rel_ret", "query": "all", "value": self.get_relevant_retrieved_documents(per_query=False)},
            {"metric": "num_q", "query": "all", "value": len(self.run.topics())},
            {"metric": "map", "query": "all", "value": map_},
            {"metric": "gm_map", "query": "all", "value": gm_map_},
            {"metric": "bpref", "query": "all", "value": bpref_},
            {"metric": "Rprec", "query": "all", "value": rprec_},
            {"metric": "recip_rank", "query": "all", "value": recip_rank_},
            {"metric": "P_5", "query": "all", "value": ps[5]},
            {"metric": "P_10", "query": "all", "value": ps[10]},
            {"metric": "P_15", "query": "all", "value": ps[15]},
            {"metric": "P_20", "query": "all", "value": ps[20]},
            {"metric": "P_30", "query": "all", "value": ps[30]},
            {"metric": "P_100", "query": "all", "value": ps[100]},
            {"metric": "P_200", "query": "all", "value": ps[200]},
            {"metric": "P_500", "query": "all", "value": ps[500]},
            {"metric": "P_1000", "query": "all", "value": ps[1000]},
            {"metric": "R_5", "query": "all", "value": re[5]},
            {"metric": "R_10", "query": "all", "value": re[10]},
            {"metric": "R_15", "query": "all", "value": re[15]},
            {"metric": "R_20", "query": "all", "value": re[20]},
            {"metric": "R_30", "query": "all", "value": re[30]},
            {"metric": "R_100", "query": "all", "value": re[100]},
            {"metric": "R_200", "query": "all", "value": re[200]},
            {"metric": "R_500", "query": "all", "value": re[500]},
            {"metric": "R_1000", "query": "all", "value": re[1000]},
            {"metric": "NDCG_5", "query": "all", "value": ndcg[5]},
            {"metric": "NDCG_10", "query": "all", "value": ndcg[10]},
            {"metric": "NDCG_15", "query": "all", "value": ndcg[15]},
            {"metric": "NDCG_20", "query": "all", "value": ndcg[20]},
            {"metric": "NDCG_30", "query": "all", "value": ndcg[30]},
            {"metric": "NDCG_100", "query": "all", "value": ndcg[100]},
            {"metric": "NDCG_200", "query": "all", "value": ndcg[200]},
            {"metric": "NDCG_500", "query": "all", "value": ndcg[500]},
            {"metric": "NDCG_1000", "query": "all", "value": ndcg[1000]},
        ]

        # TODO: iprec_at_recall_LEVEL is missing from the default trec_eval metrics

        rows = pd.DataFrame(rows)
        if len(results_per_query) > 0:
            results_per_query = pd.concat(results_per_query)
            rows = pd.concat((results_per_query, rows), sort=True).reset_index(drop=True)

        res = TrecRes()
        res.data = rows
        res.runid = run_id

        return res

    def get_retrieved_documents(self, per_query=False):
        """
            Returns the number retrieved documents

            Params
            -------
            per_query: If True, runs the evaluation per query. Default = False

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, num_retrieved_docs)
            else: returns the total number of retrieved documents for all queries.

        """
        retrieved = self.run.run_data.groupby("query")["docid"].count()
        if per_query:
            return retrieved
        return retrieved.sum()

    def get_relevant_documents(self, per_query=False):
        """
            Returns the number retrieved documents.

            Params
            -------
            per_query: If True, runs the evaluation per query. Default = False

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, nrelevant_per_query)
            else: returns the total number of relevant documents for all queries.

        """
        qrels = self.qrels.qrels_data.copy()
        qrels["relevant_per_query"] = qrels["rel"] > 0
        total_rel_per_query = qrels.groupby("query")["relevant_per_query"].sum().astype(int)

        if per_query:
            return total_rel_per_query
        return total_rel_per_query.sum()

    def get_relevant_retrieved_documents(self, per_query=False):
        """
            Returns the number relevant documents among the retrieved ones.

            Params
            -------
            per_query: If True, runs the evaluation per query. Default = False

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, num_rel_ret_docs)
            else: returns the total number of relevant retrieved documents for all queries.

        """
        merged = pd.merge(self.run.run_data[["query","docid"]], self.qrels.qrels_data[["query","docid","rel"]])

        result = merged[merged["rel"]>0].groupby("query")["rel"].count()

        if per_query:
            return result
        return result.sum()

    def get_unjudged(self, depth=10, per_query=False, trec_eval=True):
        """
        :param depth:
        :param per_query:
        :param trec_eval:
        :return:
        """

        label = "UNJ@%d" % (depth)

        if trec_eval:
            trecformat = self.run.run_data.sort_values(["query", "score", "docid"], ascending=[True,False,False]).reset_index()
            topX = trecformat.groupby("query")[["query","docid"]].head(depth)
        else:
            topX = self.run.run_data.groupby("query")[["query","docid"]].head(depth)

        # check number of queries
        nqueries = len(self.qrels.topics())

        selection = pd.merge(topX, self.qrels.qrels_data[["query","docid","rel"]], how="left")
        selection[label] = selection["rel"].isnull()

        unjX_per_query = selection[["query", label]].groupby("query").sum().astype(int) / depth

        if per_query:
            """ This will return a pandas dataframe with ["query", "UNJ@X"] values """
            return unjX_per_query
        return (unjX_per_query.sum() / nqueries)[label]

    def get_reciprocal_rank(self, depth=1000, per_query=False, trec_eval=True, removeUnjudged=False):
        """
            Calculates the reciprocal rank of the first relevant retrieved document.

            Params
            -------
            per_query: If True, runs the evaluation per query. Default = False
            depth: the evaluation depth. Default = 1000
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.
            removeUnjudged: set to True if you want to remove the unjudged documents before calculating this metric.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, recip_rank@X)
            else: returns a float value representing the recip_rank@depth.

        """

        label = "recip_rank@%d" % (depth)

        run = self.run.run_data
        qrels = self.qrels.qrels_data

        # check number of queries
        nqueries = len(self.run.topics())

        if removeUnjudged:
            onlyjudged = pd.merge(run, qrels[["query","docid","rel"]], how="left")
            onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
            run = onlyjudged[["query","q0","docid","rank","score","system"]]

        if trec_eval:
            trecformat = self.run.run_data.sort_values(["query", "score", "docid"], ascending=[True,False,False]).reset_index()
            topX = trecformat.groupby("query")[["query","docid","score"]].head(depth)
        else:
            topX = self.run.run_data.groupby("query")[["query","docid","score"]].head(depth)

        # Make sure that rank position starts by 1
        topX["rank"] = 1
        topX["rank"] = topX.groupby("query")["rank"].cumsum()

        relevant_docs = qrels[qrels.rel > 0]
        selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left")
        # converting query to category makes it explicit when using groupby.
        # This way we end up with a group even if no relevant documents are found for a query.
        selection["query"] = pd.Categorical(selection["query"])
        selection = selection[~selection["rel"].isnull()].groupby("query").first().copy()
        selection[label] = 1.0 / selection["rank"]
        recip_rank_per_query = selection[[label]]

        if per_query:
            """ This will return a pandas dataframe with ["query", "recip_rank@depth"] values """
            return recip_rank_per_query

        if recip_rank_per_query.empty:
            return 0.0

        return (recip_rank_per_query.sum() / nqueries)[label]

    def get_geometric_map(self, depth=1000, trec_eval=True):
        """
            The Geometric Mean Average Precision is the same as measured by MAP (mean average precision) on individual topics,\n
            but the geometric mean is used on over the results of each topic.
            Note that as done in the original trec_eval, the Geometric Map is only reported in the summary over all topics, not
            for individual topics.

            Params
            -------
            depth: the evaluation depth. Default = 1000
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.

            Returns
            --------
            The Geometric Mean Average Precision for all topics. Topics with MAP = 0 are replaced by MAP = GMEAN_MIN (default = .00001)
        """
        from scipy.stats.mstats import gmean
        maps = self.get_map(depth=depth, trec_eval=trec_eval, per_query=True)
        maps = maps.replace(0.0, self.GMEAN_MIN)
        return gmean(maps)[0]


    def get_map(self, depth=1000, per_query=False, trec_eval=True):
        """
            The Mean Average Precision.\n

            Params
            -------
            depth: the evaluation depth. Default = 1000
            per_query: If True, runs the evaluation per query. Default = False
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, MAP@X)
            else: returns a float value representing the MAP@deph.
        """
        #ToDo: missing option to remove unjuged documents.

        label = "MAP@%d" % (depth)

        # We only care for binary evaluation here:
        relevant_docs = self.qrels.qrels_data[self.qrels.qrels_data.rel > 0].copy()
        relevant_docs["rel"] = 1

        if trec_eval:
            trecformat = self.run.run_data.sort_values(["query", "score", "docid"], ascending=[True,False,False]).reset_index()
            topX = trecformat.groupby("query")[["query","docid","score"]].head(depth)
        else:
            topX = self.run.run_data.groupby("query")[["query","docid","score"]].head(depth)

        # check number of queries
        nqueries = len(self.run.topics())

        # Make sure that rank position starts by 1
        topX["rank"] = 1
        topX["rank"] = topX.groupby("query")["rank"].cumsum()
        topX["discount"] = 1. / np.log2(topX["rank"]+1)

        # Keep only documents that are relevant (rel > 0)
        selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left")

        selection["rel"] = selection.groupby("query")["rel"].cumsum()
        # contribution of each relevant document
        selection[label] = selection["rel"] / selection["rank"]

        # MAP is the sum of individual's contribution
        map_per_query = selection[["query", label]].groupby("query").sum()
        relevant_docs[label] = relevant_docs["rel"]
        nrel_per_query = relevant_docs[["query",label]].groupby("query").sum()
        map_per_query = map_per_query / nrel_per_query

        if per_query:
            return map_per_query

        if map_per_query.empty:
            return 0.0

        return (map_per_query.sum() / nqueries)[label]


    def get_recall(self, depth=1000, per_query=False, trec_eval=True, removeUnjudged=False):
        """
            Calculates the Recall.

            Params
            -------
            depth: the evaluation depth. Default = 1000
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.
            per_query: If True, runs the evaluation per query. Default = False
            removeUnjudged: set to True if you want to remove the unjudged documents before calculating this metric.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, Recall@d)
            else: returns a float value representing the Recall.
        """
        label = "Recall@%d" % (depth)

        run = self.run.run_data
        qrels = self.qrels.qrels_data

        if removeUnjudged:
            onlyjudged = pd.merge(run, qrels[["query","docid","rel"]], how="left")
            onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
            run = onlyjudged[["query","q0","docid","rank","score","system"]]

        # Select only topX documents per query
        topX = run.groupby("query")[["query","docid","score"]].head(depth)

        # Make sure that rank position starts by 1
        topX["rank"] = 1
        topX["rank"] = topX.groupby("query")["rank"].cumsum()
        topX["discount"] = 1. / np.log2(topX["rank"]+1)

        # Keep only documents that are relevant (rel > 0)
        relevant_docs = qrels[qrels.rel > 0]
        selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left")
        selection = selection[~selection["rel"].isnull()]

        relevant_per_query = {}
        recall_per_query = []
        for _, i in relevant_docs.iterrows():
            if i['query'] not in relevant_per_query:
                relevant_per_query[i['query']] = set()

            relevant_per_query[i['query']].add(i['docid'])

        for query in relevant_per_query.keys():
            retrieved_docs = selection[selection['query'] == query]['docid'].unique()
            retrieved_relevant_docs = [i for i in retrieved_docs if i in relevant_per_query[query]]
            recall_per_query += [{'query': query, label: len(retrieved_relevant_docs)/ len(relevant_per_query[query])}]

        recall_per_query = pd.DataFrame(recall_per_query)
        if per_query:
            return recall_per_query
        else:
            return recall_per_query[label].mean()


    def get_rprec(self, depth=1000, per_query=False, trec_eval=True, removeUnjudged=False):
        """
            The Precision at R, where R is the number of relevant documents for a topic.

            Params
            -------
            depth: the evaluation depth. Default = 1000
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.
            per_query: If True, runs the evaluation per query. Default = False
            removeUnjudged: set to True if you want to remove the unjudged documents before calculating this metric.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, RPrec)
            else: returns a float value representing the RPrec.
        """
        label = "RPrec@%d" % (depth)

        run = self.run.run_data
        qrels = self.qrels.qrels_data

        # check number of queries
        nqueries = len(self.run.topics())

        if removeUnjudged:
            onlyjudged = pd.merge(run, qrels[["query","docid","rel"]], how="left")
            onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
            run = onlyjudged[["query","q0","docid","rank","score","system"]]

        if trec_eval:
            trecformat = self.run.run_data.sort_values(["query", "score", "docid"], ascending=[True,False,False]).reset_index()
            topX = trecformat.groupby("query")[["query","docid","score"]].head(depth)
        else:
            topX = self.run.run_data.groupby("query")[["query","docid","score"]].head(depth)

        # gets the number of relevant documents per query
        n_relevant_docs = self.get_relevant_documents(per_query = True)

        # Gets only the top R documents per topic:
        topX = topX.groupby("query").apply(lambda x: x.head(n_relevant_docs.loc[x.name])).reset_index(drop=True)

        relevant_docs = qrels[qrels.rel > 0]
        selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left")
        selection = selection[~selection["rel"].isnull()]

        rprec_per_query = selection.groupby("query")["docid"].count() / n_relevant_docs
        rprec_per_query.name = label
        rprec_per_query = rprec_per_query.reset_index().set_index("query")

        if per_query:
            return rprec_per_query

        if rprec_per_query.empty:
            return 0.0

        return (rprec_per_query.sum() / nqueries)[label]


    def get_ndcg(self, depth=1000, per_query=False, trec_eval=True, removeUnjudged=False):
        """
            Calculates the normalized discounted cumulative gain (NDCG).

            Params
            -------
            depth: the evaluation depth. Default = 1000
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.
            per_query: If True, runs the evaluation per query. Default = False
            removeUnjudged: set to True if you want to remove the unjudged documents before calculating this metric.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, NDCG@d)
            else: returns a float value representing the RPrec.
        """

        label = "NDCG@%d" % (depth)

        run = self.run.run_data
        qrels = self.qrels.qrels_data

        # check number of queries
        nqueries = len(self.run.topics())

        if removeUnjudged:
            onlyjudged = pd.merge(run, qrels[["query","docid","rel"]], how="left")
            onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
            run = onlyjudged[["query","q0","docid","rank","score","system"]]

        # Select only topX documents per query
        topX = run.groupby("query")[["query","docid","score"]].head(depth)

        # Make sure that rank position starts by 1
        topX["rank"] = 1
        topX["rank"] = topX.groupby("query")["rank"].cumsum()
        topX["discount"] = 1. / np.log2(topX["rank"]+1)

        # Keep only documents that are relevant (rel > 0)
        relevant_docs = qrels[qrels.rel > 0]
        selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left")
        selection = selection[~selection["rel"].isnull()]

        # Calculate DCG
        if trec_eval:
            selection[label] = (selection["rel"]) * selection["discount"]
        else:
            selection[label] = (2**selection["rel"] - 1.0) * selection["discount"]

        # Calculate IDCG
        perfect_ranking = relevant_docs.sort_values(["query","rel"], ascending=[True,False]).reset_index(drop=True)
        perfect_ranking = perfect_ranking.groupby("query").head(depth)

        perfect_ranking["rank"] = 1
        perfect_ranking["rank"] = perfect_ranking.groupby("query")["rank"].cumsum()
        perfect_ranking["discount"] = 1. / np.log2(perfect_ranking["rank"]+1)
        if trec_eval:
            perfect_ranking[label] = (perfect_ranking["rel"]) * perfect_ranking["discount"]
        else:
            perfect_ranking[label] = (2**perfect_ranking["rel"] - 1.0) * perfect_ranking["discount"]

        # DCG is the sum of individual's contribution
        dcg_per_query = selection[["query", label]].groupby("query").sum()
        idcg_per_query = perfect_ranking[["query",label]].groupby("query").sum()
        ndcg_per_query = dcg_per_query / idcg_per_query

        if per_query:
            return ndcg_per_query

        if ndcg_per_query.empty:
            return 0.0

        return (ndcg_per_query.sum() / nqueries)[label]


    def get_bpref(self, depth=1000, per_query=False, trec_eval=True):
        """
            Calculates the binary preference (BPREF).

            Params
            -------
            depth: the evaluation depth. Default = 1000
            per_query: If True, runs the evaluation per query. Default = False
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, NDCG@d)
            else: returns a float value representing the RPrec.

        """

        label = "Bpref@%d" % (depth)

        # check number of queries
        nqueries = len(self.run.topics())

        qrels = self.qrels.qrels_data.copy()
        run = self.run.run_data

        # number of relevant and non-relevant documents per query:
        qrels["is_rel_per_query"] = qrels["rel"] > 0
        total_rel_per_query = qrels.groupby("query")["is_rel_per_query"].sum()
        total_nrel_per_query = qrels.groupby("query")["is_rel_per_query"].count() - qrels.groupby("query")["is_rel_per_query"].sum()
        total_rel_per_query.name = "rels_per_query"

        # Denominator is the minimal of the two dataframes. Using 'where' clause as a 'min'
        # denominator = min(total_rel_per_query, total_nrel_per_query)
        denominator = total_rel_per_query.where(total_rel_per_query < total_nrel_per_query, total_nrel_per_query)
        denominator.name = "denominator"

        merged = pd.merge(run, qrels[["query","docid","rel"]], how="left")

        if trec_eval:
            merged.sort_values(["query", "score", "docid"], ascending=[True,False,False], inplace=True)

        # We explicitly remove unjudged documents
        merged = merged[~merged.rel.isnull()]

        # Select only topX documents per query
        merged = merged.groupby("query")[["query","docid","rel"]].head(depth)

        merged["is_nrel"] = merged["rel"] == 0
        merged["nrel_so_far"] = merged.groupby("query")["is_nrel"].cumsum()

        merged = pd.merge(merged, total_rel_per_query.reset_index(), on="query", how="left")
        merged = pd.merge(merged, denominator.reset_index(), on="query", how="left")

        merged[label] = (1.0 - (1.0 * merged[["nrel_so_far","rels_per_query"]].min(axis=1) / merged["denominator"])) / merged["rels_per_query"]

        # Accumulates scores only for relevant documents retrieved
        merged = merged[~merged["is_nrel"]]

        bpref_per_query = merged[["query", label]].groupby("query").sum()

        if per_query:
            """ This will return a pandas dataframe with ["query", "P@X"] values """
            return bpref_per_query
        return (bpref_per_query.sum() / nqueries)[label]


    def get_ubpref(self, other_qrels, per_query=False, trec_eval=True, normalization_factor = 1.0, depth=1000):
        """
            other_qrels: the qrels for other dimensions, i.e., understandability or trustworthiness
        """

        if not isinstance(other_qrels, TrecQrel):
            raise TypeError('"other_qrels" should be a TrecQrel object')

        label = "uBpref@%d" % (depth)

        # check number of queries
        nqueries = len(self.run.topics())

        qrels = self.qrels.qrels_data.copy()
        other = other_qrels.qrels_data.copy()
        other["rel"] = other["rel"] * normalization_factor
        run = self.run.run_data

        # number of relevant and non-relevant documents per query:
        qrels["is_rel_per_query"] = qrels["rel"] > 0
        total_rel_per_query = qrels.groupby("query")["is_rel_per_query"].sum()
        total_nrel_per_query = qrels.groupby("query")["is_rel_per_query"].count() - qrels.groupby("query")["is_rel_per_query"].sum()
        total_rel_per_query.name = "rels_per_query"

        # Denominator is the minimal of the two dataframes. Using 'where' clause as a 'min'
        # denominator = min(total_rel_per_query, total_nrel_per_query)
        denominator = total_rel_per_query.where(total_rel_per_query < total_nrel_per_query, total_nrel_per_query)
        denominator.name = "denominator"

        merged = pd.merge(run, qrels[["query","docid","rel"]], how="left").merge(other, on=["query","docid"], suffixes=("","_other"))

        if trec_eval:
            merged.sort_values(["query", "score", "docid"], ascending=[True,False,False], inplace=True)

        # We explicitly remove unjudged documents
        merged = merged[~merged.rel.isnull()]

        # Select only topX documents per query
        merged = merged.groupby("query")[["query","docid","rel","rel_other"]].head(depth)

        merged["is_nrel"] = merged["rel"] == 0
        merged["nrel_so_far"] = merged.groupby("query")["is_nrel"].cumsum()

        merged = pd.merge(merged, total_rel_per_query.reset_index(), on="query", how="left")
        merged = pd.merge(merged, denominator.reset_index(), on="query", how="left")

        merged[label] = (1.0 - (1.0 * merged[["nrel_so_far","rels_per_query"]].min(axis=1) / merged["denominator"])) * merged["rel_other"] / merged["rels_per_query"]

        # Accumulates scores only for relevant documents retrieved
        merged = merged[~merged["is_nrel"]]

        ubpref_per_query = merged[["query", label]].groupby("query").sum()

        if per_query:
            """ This will return a pandas dataframe with ["query", "P@X"] values """
            return ubpref_per_query
        return (ubpref_per_query.sum() / nqueries)[label]

    def get_precision(self, depth=1000, per_query=False, trec_eval=True, removeUnjudged=False):
        """
            Calculates the binary precision at depth d (P@d).

            Params
            -------
            depth: the evaluation depth. Default = 1000
            per_query: If True, runs the evaluation per query. Default = False
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.
            removeUnjudged: set to True if you want to remove the unjudged documents before calculating this metric.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, P@d)
            else: returns a float value representing the RPrec.

        """
        label = "P@%d" % (depth)

        # check number of queries
        nqueries = len(self.run.topics())

        qrels = self.qrels.qrels_data
        run = self.run.run_data

        merged = pd.merge(run[["query", "docid", "score"]], qrels[["query","docid","rel"]], how="left")

        if trec_eval:
            merged.sort_values(["query", "score", "docid"], ascending=[True,False,False], inplace=True)

        if removeUnjudged:
            merged = merged[~merged.rel.isnull()]

        topX = merged.groupby("query")[["query","docid","rel"]].head(depth)
        topX[label] = topX["rel"] > 0
        pX_per_query = topX[["query", label]].groupby("query").sum().astype(int) / depth

        if per_query:
            """ This will return a pandas dataframe with ["query", "P@X"] values """
            return pX_per_query
        return (pX_per_query.sum() / nqueries)[label]
    
    def get_recall(self, depth=1000, per_query=False, trec_eval=True, removeUnjudged=False):
        """
            Calculates the binary recall at depth d (R@d).

            Params
            -------
            depth: the evaluation depth. Default = 1000
            per_query: If True, runs the evaluation per query. Default = False
            trec_eval: set to True if result should be the same as trec_eval, e.g., sort documents by score first. Default = True.
            removeUnjudged: set to True if you want to remove the unjudged documents before calculating this metric.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, P@d)
            else: returns a float value representing the RPrec.

        """
        label = "R@%d" % (depth)

        # check number of queries
        nqueries = len(self.run.topics())

        qrels = self.qrels.qrels_data
        run = self.run.run_data

        merged = pd.merge(run[["query", "docid", "score"]], qrels[["query","docid","rel"]], how="left")

        if trec_eval:
            merged.sort_values(["query", "score", "docid"], ascending=[True,False,False], inplace=True)

        if removeUnjudged:
            merged = merged[~merged.rel.isnull()]

        n_relevant_docs = self.get_relevant_documents(per_query = True) # pd.Series with the number of relevant documents per query

        topX = merged.groupby("query")[["query","docid","rel"]].head(depth)
        topX[label] = topX["rel"] > 0
        rX_per_query = topX[["query", label]].groupby("query").sum().astype(int).div(n_relevant_docs, axis=0) # pd.DataFrame divided by pd.Series
        # TODO: check if div need operations like how='left'

        if per_query:
            """ This will return a pandas dataframe with ["query", "R@X"] values """
            return rX_per_query
        return (rX_per_query.sum() / nqueries)[label]



    def get_rbp(self, p=0.8, depth=1000, per_query=False, binary_topical_relevance=True, average_ties=True, removeUnjudged=False):
        """
            Calculates the rank-bias precision at depth d (RBP@d) with persistece paramter p.

            Params
            -------
            p: persistence parameter. Default = .80
            binary_topical_relevance: If True, document relevance is binarized. Default = True.
            average_ties: ToDo --- Missing documentation. Default = True.

            depth: the evaluation depth. Default = 1000
            per_query: If True, runs the evaluation per query. Default = False
            removeUnjudged: set to True if you want to remove the unjudged documents before calculating this metric.

            Returns
            --------
            if per_query == True: returns a pandas dataframe with two cols (query, RBP(p)@d)
            else: returns a pair of float values representing (RBP(p)@d, RBPResiduals).

        """
        label = "RBP(%.2f)@%d" % (p, depth)

        run = self.run.run_data
        qrels = self.qrels.qrels_data

        # check number of queries
        nqueries = len(self.run.topics())

        if removeUnjudged:
            onlyjudged = pd.merge(run, qrels[["query","docid","rel"]], how="left")
            onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
            run = onlyjudged[["query","q0","docid","rank","score","system"]]

        # Select only topX documents per query
        topX = run.groupby("query")[["query","docid","score"]].head(depth)

        # Make sure that rank position starts by 1
        topX["rank"] = 1
        topX["rank"] = topX.groupby("query")["rank"].cumsum()

        # Calculate RBP based on rank of documents
        topX[label] = (1.0-p) * (p) ** (topX["rank"]-1)

        # Average ties if required:
        if average_ties:
            topX["score+1"] = topX["score"].shift(1)
            topX["ntie"] = topX["score"] != topX["score+1"]
            topX["grps"] = topX["ntie"].cumsum()
            averages = topX[[label, "grps"]].groupby("grps")[label].mean().reset_index().rename(columns={label: "avgs"})
            topX = pd.merge(averages, topX)
            topX[label] = topX["avgs"]
            for k in ["score","score+1","ntie","grps","avgs"]:
                del topX[k]

        # Residuals:
        residuals = pd.merge(topX, qrels[["query","docid","rel"]], how="left")
        residuals.loc[residuals.rel.isnull(),"rel"] = 1 # Transform non judged docs into relevant ones
        residuals = residuals[residuals["rel"] > 0]

        # Keep only documents that are relevant (rel > 0)
        relevant_docs = qrels[qrels.rel > 0]
        selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left")
        selection = selection[~selection["rel"].isnull()]

        if not binary_topical_relevance:
            selection[label] = selection[label] * selection["rel"]

        # RBP is the sum of individual's contribution
        rbp_per_query = selection[["query", label]].groupby("query").sum()
        rbp_res_per_query = residuals[["query", label]].groupby("query").sum()

        if per_query:
            return rbp_per_query, rbp_res_per_query - rbp_per_query + p**depth

        if rbp_per_query.empty:
            return (np.nan, np.nan)

        return (rbp_per_query.sum() / nqueries)[label], (rbp_res_per_query.sum() / nqueries)[label]  + p** depth - (rbp_per_query.sum() / nqueries)[label]

    def get_urbp(self, additional_qrel, strategy="direct_multiplication", normalization_factor = 1.0, p=0.8, depth=1000,
            per_query=False, binary_topical_relevance=True, average_ties=True, removeUnjudged=False):
        """
            uRBP is the modification of RBP to cope with other dimentions of relevation.
            The important parameters are:
                * p: same as RBP(p)
                * depth: the depth per topic/query that we should look at when evaluation
                * strategy: one of:
                    - direct_multiplication: simply will multiply the RBP value of a document by the additional_qrel["rel"] for that document
                    - TODO (dictionary transformation)
                * normalization_factor: a value which will be multiplied to the addtional_qrel["rel"] value. Use it to transform a 0-1 scale into a 0-100 (with normalization_factor = 100). Default: 1.0

        """

        label = "uRBP(%.2f)@%d" % (p, depth)

        # check number of queries
        nqueries = len(self.qrels.topics())

        run = self.run.run_data
        qrels = self.qrels.qrels_data

        if removeUnjudged:
            onlyjudged = pd.merge(run, qrels[["query","docid","rel"]], how="left")
            onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
            run = onlyjudged[["query","q0","docid","rank","score","system"]]

        # Select only topX documents per query
        topX = run.groupby("query")[["query","docid","score"]].head(depth)

        # Make sure that rank position starts by 1
        topX["rank"] = 1
        topX["rank"] = topX.groupby("query")["rank"].cumsum()

        # Calculate RBP based on rank of documents
        topX[label] = (1.0-p) * (p) ** (topX["rank"]-1)

        # Average ties if required:
        if average_ties:
            topX["score+1"] = topX["score"].shift(1)
            topX["ntie"] = topX["score"] != topX["score+1"]
            topX["grps"] = topX["ntie"].cumsum()
            averages = topX[[label,"grps"]].groupby("grps")[label].mean().reset_index().rename(columns={label: "avgs"})
            topX = pd.merge(averages, topX)
            topX[label] = topX["avgs"]
            for k in ["score","score+1","ntie","grps","avgs"]:
                del topX[k]

        # Keep only documents that are relevant (rel > 0)
        relevant_docs = qrels[qrels.rel > 0]
        selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left").\
                                merge(additional_qrel.qrels_data, on=["query","docid"], suffixes=("","_other"))
        selection = selection[~selection["rel"].isnull()]

        if strategy == "direct_multiplication":
            selection[label] = selection[label] * selection["rel_other"] * normalization_factor

        if not binary_topical_relevance:
            selection[label] = selection[label] * selection["rel"]

        # RBP is the sum of individual's contribution
        rbp_per_query = selection[["query", label]].groupby("query").sum()

        if per_query:
            """ This will return a pandas dataframe with ["query", "RBP"] values """
            return rbp_per_query

        if rbp_per_query.empty:
            return 0.0

        return (rbp_per_query.sum() / nqueries)[label]

    def get_alpha_urbp(self, additional_qrel, goals, strategy="direct_multiplication", normalization_factor = 1.0, p=0.8, depth=1000, per_query=False, binary_topical_relevance=True, average_ties=True):

        """
            alphaURBP is the modification of uRBP to cope with various profiles defined using alpha.
            The important parameters are:
                * p: same as RBP(p)
                * depth: the depth per topic/query that we should look at when evaluation
                * goals: a dictionary like {query: [goal,var]}
                * strategy: one of:
                    - direct_multiplication: simply will multiply the RBP value of a document by the additional_qrel["rel"] for that document
                    - TODO (dictionary transformation)
                * normalization_factor: a value which will be multiplied to the addtional_qrel["rel"] value. Use it to transform a 0-1 scale into a 0-100 (with normalization_factor = 100). Default: 1.0

        """
        if not isinstance(additional_qrel, TrecQrel):
            raise TypeError('"additional_qrel" should be a TrecQrel object')

        label = "auRBP(%.2f)@%d" % (p, depth)

        # Select only topX documents per query
        topX = self.run.run_data.groupby("query")[["query","docid","score"]].head(depth)

        # check number of queries
        nqueries = len(self.qrels.topics())

        # Make sure that rank position starts by 1
        topX["rank"] = 1
        topX["rank"] = topX.groupby("query")["rank"].cumsum()

        # Calculate RBP based on rank of documents
        topX[label] = (1.0-p) * (p) ** (topX["rank"]-1)

        # Average ties if required:
        if average_ties:
            topX["score+1"] = topX["score"].shift(1)
            topX["ntie"] = topX["score"] != topX["score+1"]
            topX["grps"] = topX["ntie"].cumsum()
            averages = topX[[label,"grps"]].groupby("grps")[label].mean().reset_index().rename(columns={label: "avgs"})
            topX = pd.merge(averages, topX)
            topX[label] = topX["avgs"]
            for k in ["score","score+1","ntie","grps","avgs"]:
                del topX[k]

        # Keep only documents that are relevant (rel > 0)
        relevant_docs = self.qrels.qrels_data[self.qrels.qrels_data.rel > 0]
        selection = pd.merge(topX, relevant_docs[["query","docid","rel"]], how="left").\
                                merge(additional_qrel.qrels_data, on=["query","docid"], suffixes=("","_other"))
        selection = selection[~selection["rel"].isnull()]

        # Transform dictionary into dataframe
        goals = pd.DataFrame.from_dict(goals, orient='index').reset_index()
        goals.columns = ["query", "mean", "var"]

        def normvalue(value, goal, var):
            return norm.pdf(value, goal, var) * 100. / norm.pdf(goal, goal, var)

        # TODO: now I am forcing the queries to be integer. Need to find a better way to cope with different data types
        selection["query"] = selection["query"].astype(int)
        goals["query"] = goals["query"].astype(int)

        selection = pd.merge(selection, goals)
        selection["rel_other"] = selection[["rel_other", "mean", "var"]].\
                                    apply(lambda x: normvalue(x["rel_other"], x["mean"], x["var"]), axis=1)

        if strategy == "direct_multiplication":
            selection[label] = selection[label] * selection["rel_other"] * normalization_factor

        if not binary_topical_relevance:
            selection[label] = selection[label] * selection["rel"]

        # RBP is the sum of individual's contribution
        rbp_per_query = selection[["query", label]].groupby("query").sum()

        if per_query:
            """ This will return a pandas dataframe with ["query", "RBP"] values """
            return rbp_per_query

        if rbp_per_query.empty:
            return 0.0

        return (rbp_per_query.sum() / nqueries)[label]
