#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Item Retrieval & Hit@K / NDCG@K Evaluation
- Loads user_prediction.json (predicted item types from LLM)
- Embeds predicted item texts via OpenAI text-embedding-3-large
- Retrieves top-K nearest items from gpt_feat.npy (aligned with item2id.json)
- Checks whether the held-out test item (last item in each user's inter.json sequence) is retrieved
"""

import argparse
import json
import os
import time
import pickle
from math import log2
from typing import Dict, List, Tuple

import numpy as np
import requests


# -------------------------
# Helpers
# -------------------------

def ensure_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key


def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / np.clip(norm, eps, None)


def topk_by_similarity(
    query: np.ndarray,
    base: np.ndarray,
    k: int,
    sim_metric: str = "dot",
    degree_weights = None,
    exclude_indices: np.ndarray = None,
) -> List[Tuple[int, float]]:
    """Return top-k indices and similarity scores.
    Dot-product retrieval is used for ranking.
    Ranking uses similarity * degree_weights (if provided).
    """
    # query: (d,), base: (n, d)
    sim_metric = (sim_metric or "dot").strip().lower()
    if sim_metric not in {"dot"}:
        sim_metric = "dot"

    sim_scores = base @ query  # (n,)

    valid_mask = np.ones(len(sim_scores), dtype=bool)
    if exclude_indices is not None and len(exclude_indices) > 0:
        valid_mask[np.asarray(exclude_indices, dtype=np.int64)] = False

    if degree_weights is not None:
        rank_scores = sim_scores * degree_weights
    else:
        rank_scores = sim_scores

    rank_scores = np.where(valid_mask, rank_scores, -np.inf)
    valid_count = int(valid_mask.sum())
    if valid_count <= 0:
        return []

    k_eff = min(k, valid_count)
    if k_eff >= len(rank_scores):
        idx = np.argsort(rank_scores)[::-1][:k_eff]
    else:
        idx_part = np.argpartition(rank_scores, -k_eff)[-k_eff:]
        idx = idx_part[np.argsort(rank_scores[idx_part])[::-1]]

    return [(int(i), float(sim_scores[i])) for i in idx]


# -------------------------
# OpenAI Embedding
# -------------------------

def get_batch_embeddings(texts: List[str], model: str = "text-embedding-3-large") -> np.ndarray:
    api_key = ensure_openai_key()
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "input": texts,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if resp.status_code == 200:
        data = resp.json()["data"]
        data_sorted = sorted(data, key=lambda x: x["index"])
        emb = [np.array(item["embedding"], dtype=np.float32) for item in data_sorted]
        return np.stack(emb, axis=0)
    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")


def embed_texts(texts: List[str], model: str = "text-embedding-3-large", batch_size: int = 100) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            emb = get_batch_embeddings(batch, model=model)
            for t, e in zip(batch, emb):
                out[t] = e
        except Exception as e:
            print(f"[WARN] batch {i//batch_size+1} failed: {e}")
            # fallback single
            for t in batch:
                try:
                    single = get_batch_embeddings([t], model=model)[0]
                    out[t] = single
                    time.sleep(0.2)
                except Exception as e2:
                    print(f"[WARN] embed fail '{t}': {e2}")
        time.sleep(0.5)
    return out


def _safe_model_tag(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model)


def load_embedding_cache(cache_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"[WARN] failed to read cache {cache_path}: {e}")
        return {}

    if not isinstance(obj, dict):
        return {}

    cache: Dict[str, np.ndarray] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1 and arr.size > 0:
            cache[k] = arr
    return cache


def save_embedding_cache(cache_path: str, cache: Dict[str, np.ndarray]) -> None:
    serializable = {k: np.asarray(v, dtype=np.float32) for k, v in cache.items()}
    with open(cache_path, "wb") as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)


# -------------------------
# CoRRe pipeline
# -------------------------

def run(
    dataset: str,
    topk_list: List[int],
    model: str = "text-embedding-3-large",
    degree_alpha: float = 0.0,
    mask_train_items: bool = True,
) -> None:
    base_dir = os.path.join("data", dataset)
    cache_dir = base_dir
    os.makedirs(cache_dir, exist_ok=True)
    emb_cache_path = os.path.join(cache_dir, f"query_emb_cache_{_safe_model_tag(model)}.pkl")
    
    user_pred_path = os.path.join(base_dir, "generated_intent.json")
    gpt_feat_path = os.path.join(base_dir, "refined_itemfeat.npy")
    itemsid_path = os.path.join(base_dir, "item2id.json")
    title_path = os.path.join(base_dir, "title.pickle")
    inter_path = os.path.join(base_dir, "inter.json")

    print(f"[LOAD] dataset={dataset}")
    print(f" user predictions: {user_pred_path}")
    print(f" gpt features: {gpt_feat_path}")
    with open(user_pred_path, "r", encoding="utf-8") as f:
        user_predictions = json.load(f)
    gpt_feat = np.load(gpt_feat_path)
    with open(itemsid_path, "r", encoding="utf-8") as f:
        itemsid: Dict[str, int] = json.load(f)
    with open(inter_path, "r", encoding="utf-8") as f:
        user_inter: Dict[str, List[str]] = json.load(f)

    title_map: Dict[str, str] = {}
    if os.path.exists(title_path):
        with open(title_path, "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, dict):
                title_map = obj

    index_to_item = {v: k for k, v in itemsid.items()}
    gpt_feat = gpt_feat.astype(np.float32)
    sim_metric = "dot"

    gpt_feat_norm = l2_normalize(gpt_feat, axis=1)

    # degree weights (item popularity)
    degree_weights = None
    if degree_alpha != 0.0:
        degree = np.zeros(len(itemsid), dtype=np.float32)
        for seq in user_inter.values():
            # Exclude each user's held-out test item (last interaction) from degree counting.
            for it in seq[:-1]:
                idx = itemsid.get(it)
                if idx is not None:
                    degree[idx] += 1.0
        degree_weights = np.power(np.maximum(degree, 1.0), degree_alpha)

    # Per-user train-sequence mask (exclude train items from retrieval candidates)
    user_train_item_indices: Dict[str, np.ndarray] = {}
    if mask_train_items:
        masked_counts = []
        for uid, seq in user_inter.items():
            if len(seq) <= 1:
                continue
            idxs = {
                itemsid[it]
                for it in seq[:-1]
                if it in itemsid
            }
            if idxs:
                arr = np.array(list(idxs), dtype=np.int64)
                user_train_item_indices[uid] = arr
                masked_counts.append(len(arr))
        avg_masked = float(np.mean(masked_counts)) if masked_counts else 0.0
        print(
            f"[MASK] enabled: users_with_mask={len(user_train_item_indices)}, "
            f"avg_masked_train_items={avg_masked:.2f}"
        )
    else:
        print("[MASK] disabled: train-sequence items are allowed in retrieval candidates")

    # collect per-user recommendation sentence
    all_queries = set()
    user_to_query = {}
    for uid, data in user_predictions.items():
        sent = (data.get("recommendation_sentence") or "").strip()
        if not sent:
            # backward compatibility: if old format exists, convert list -> one sentence
            preds = data.get("predicted_items")
            if isinstance(preds, list) and preds:
                sent = "The user is likely to interact with or purchase " + ", ".join([str(x).strip() for x in preds[:3]]) + " next."
            else:
                sent = (data.get("raw_response") or "").strip()
        if sent:
            user_to_query[uid] = sent
            all_queries.add(sent)

    print(f"[INFO] users with recommendation sentence: {len(user_to_query)}")
    print(f"[INFO] unique recommendation sentences: {len(all_queries)}")

    # embed recommendation sentence per user (cache + missing only)
    query_list = list(all_queries)
    query_emb_dict = load_embedding_cache(emb_cache_path)
    before_cache_count = len(query_emb_dict)

    missing_queries = [q for q in query_list if q not in query_emb_dict]
    if missing_queries:
        print(f"[EMBED] cache hit: {len(query_list) - len(missing_queries)} / {len(query_list)}")
        print(f"[EMBED] requesting OpenAI for missing queries: {len(missing_queries)}")
        new_emb = embed_texts(missing_queries, model=model, batch_size=100)
        query_emb_dict.update(new_emb)
        save_embedding_cache(emb_cache_path, query_emb_dict)
        print(f"[EMBED] cache updated: +{len(new_emb)} (from {before_cache_count} to {len(query_emb_dict)})")
    else:
        print(f"[EMBED] all queries loaded from cache: {len(query_list)}/{len(query_list)}")

    query_emb_dict = {q: query_emb_dict[q] for q in query_list if q in query_emb_dict}
    print(f"[EMBED] success: {len(query_emb_dict)}/{len(query_list)}")

    # Store all results for different topk values
    all_summaries = []

    # Process each topk value
    for topk in topk_list:
        print(f"\n{'='*60}")
        print(f"Processing Top-K = {topk}")
        print(f"{'='*60}")
        
        # retrieval
        retrieval_results = {}
        for idx, (uid, sentence) in enumerate(user_to_query.items(), 1):
            if idx % 200 == 0:
                print(f"[RETRIEVE K={topk}] {idx}/{len(user_to_query)} users")

            if sentence not in query_emb_dict:
                retrieval_results[uid] = {
                    "source_items": user_predictions.get(uid, {}).get("source_items", []),
                    "recommendation_sentence": sentence,
                    "retrieved_items": [],
                }
                continue

            q_raw = query_emb_dict[sentence]
            if sim_metric == "dot":
                q = l2_normalize(q_raw.reshape(1, -1), axis=1)[0]
                base_mat = gpt_feat
            else:
                q = l2_normalize(q_raw.reshape(1, -1), axis=1)[0]
                base_mat = gpt_feat_norm

            topk_results = topk_by_similarity(
                q,
                base_mat,
                topk,
                sim_metric=sim_metric,
                degree_weights=degree_weights,
                exclude_indices=user_train_item_indices.get(uid),
            )
            similar_items = []
            for item_idx, score in topk_results:
                itm = index_to_item.get(item_idx)
                if itm is not None:
                    similar_items.append({
                        "item_id": itm,
                        "similarity": score,
                        "index": item_idx,
                    })

            retrieval_results[uid] = {
                "source_items": user_predictions.get(uid, {}).get("source_items", []),
                "recommendation_sentence": sentence,
                "retrieved_items": [
                    {
                        "predicted_sentence": sentence,
                        "similar_items": similar_items,
                    }
                ],
            }

        print(f"[RETRIEVE K={topk}] completed: {len(retrieval_results)} users")

        # evaluation: test item = last item in inter sequence
        evaluation_results = []
        for uid, data in retrieval_results.items():
            inter_seq = user_inter.get(uid, [])
            if len(inter_seq) == 0:
                continue
            test_item = inter_seq[-1]
            test_item_title = title_map.get(test_item, test_item)

            ranked_items: List[str] = []
            seen_items = set()
            for pred_block in data["retrieved_items"]:
                for sim in pred_block["similar_items"]:
                    iid = sim["item_id"]
                    if iid in seen_items:
                        continue
                    seen_items.add(iid)
                    ranked_items.append(iid)

            rank_position = None
            if test_item in seen_items:
                rank_position = ranked_items.index(test_item) + 1

            hit = 1 if rank_position is not None and rank_position <= topk else 0
            ndcg = (1.0 / log2(rank_position + 1)) if rank_position is not None and rank_position <= topk else 0.0

            ranked_titles = [title_map.get(item_id, item_id) for item_id in ranked_items]

            evaluation_results.append({
                "user_id": uid,
                "test_item": test_item,
                "test_item_title": test_item_title,
                "num_retrieved_items": len(ranked_items),
                "hit": hit,
                "ndcg": ndcg,
                "rank": rank_position,
                "ranked_titles": ranked_titles,
                "retrieved_sample": ranked_titles[:10],
                "is_hit": bool(hit),
            })

        # stats
        total_users = len(evaluation_results)
        hit_sum = sum(r["hit"] for r in evaluation_results)
        ndcg_sum = sum(r["ndcg"] for r in evaluation_results)
        avg_unique_retrieved = float(np.mean([r["num_retrieved_items"] for r in evaluation_results])) if evaluation_results else 0.0

        # average retrieved count per user (non-unique, sum of per-predicted retrieval sizes)
        per_user_retrieved_counts = []
        for uid, data in retrieval_results.items():
            total_retrieved = 0
            for pred_block in data["retrieved_items"]:
                total_retrieved += len(pred_block.get("similar_items", []))
            per_user_retrieved_counts.append(total_retrieved)
        avg_total_retrieved = float(np.mean(per_user_retrieved_counts)) if per_user_retrieved_counts else 0.0

        print(f"\n{'='*60}")
        print(f"EVALUATION - Top-K = {topk}")
        print(f"{'='*60}")
        if total_users == 0:
            print("No users evaluated (check inputs)")
        else:
            print(f"Users evaluated: {total_users}")
            print(f"Hit@{topk}: {hit_sum / total_users:.4f} ({hit_sum}/{total_users})")
            print(f"NDCG@{topk}: {ndcg_sum / total_users:.4f}")
            print(f"Avg unique retrieved items/user: {avg_unique_retrieved:.2f}")
            print(f"Avg total retrieved items/user: {avg_total_retrieved:.2f}")
        print(f"{'='*60}")

        # save results for this topk
        alpha_tag = str(degree_alpha).replace(".", "p")
        out_retrieval = os.path.join(base_dir, f"retrieval_results_top{topk}_{sim_metric}_alpha{alpha_tag}.json")
        out_eval = os.path.join(base_dir, f"evaluation_results_top{topk}_{sim_metric}_alpha{alpha_tag}.json")

        
        summary = {
            "dataset": dataset,
            "top_k_per_predicted_item": topk,
            "similarity_metric": sim_metric,
            "min_similarity_threshold": 0.0,
            "mask_train_items": mask_train_items,
            "total_users": total_users,
            "hit_rate": hit_sum / total_users if total_users else 0.0,
            "ndcg": ndcg_sum / total_users if total_users else 0.0,
            "avg_retrieved_items_per_user": avg_unique_retrieved,
            "avg_total_retrieved_items_per_user": avg_total_retrieved,
            "note": "User-level sentence embedding retrieval. Test item = last item in inter.json sequence; retrieved items below threshold are dropped; optional train-sequence masking applied",
        }
        all_summaries.append(summary)

        # Save combined summary
        alpha_tag = str(degree_alpha).replace(".", "p")


    # Print final comparison table
    print(f"User predictions: {user_pred_path}")
    print(f"Gpt features: {gpt_feat_path}")
    print(f"Similarity metric: {sim_metric}")
    print(f"Degree alpha: {degree_alpha}")
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY - Hit/NDCG Comparison")
    print(f"{'='*60}")
    print(f"{'Top-K':<10} {'Hit Rate':<15} {'NDCG':<15} {'Avg Retrieved':<15}")
    print(f"{'-'*60}")
    for s in all_summaries:
        print(f"{s['top_k_per_predicted_item']:<10} {s['hit_rate']:<15.4f} {s['ndcg']:<15.4f} {s['avg_retrieved_items_per_user']:<15.2f}")
    print(f"{'='*60}")


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Item Retrieval & Hit@K / NDCG@K Evaluation")
    p.add_argument("--dataset", type=str, default="sports", help="Dataset name under /data/{dataset}")
    p.add_argument("--topk", type=int, nargs="+", default=[10, 20, 30, 50], help="Top-K values to evaluate (space-separated, e.g., --topk 20 40 60)")
    p.add_argument("--model", type=str, default="text-embedding-3-large", help="OpenAI embedding model")
    p.add_argument("--degree_alpha", type=float, default=0.0, help="Multiply similarity by degree^alpha; default 0.0")
    p.add_argument("--mask_train_items", action="store_true", default=True, help="Exclude each user train-sequence items (inter.json[:-1]) from retrieval candidates")
    p.add_argument("--no_mask_train_items", action="store_false", dest="mask_train_items", help="Disable train-sequence masking in retrieval")
    return p.parse_args()


def main():
    args = parse_args()

    topk_list = args.topk if isinstance(args.topk, list) else [args.topk]
    run(
        dataset=args.dataset,
        topk_list=topk_list,
        model=args.model,
        degree_alpha=args.degree_alpha,
        mask_train_items=args.mask_train_items,
    )


if __name__ == "__main__":
    main()
