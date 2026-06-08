#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_interactions(path: str) -> Dict[str, List[str]]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def build_mappings(user2items: Dict[str, List[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
	user_ids = list(user2items.keys())
	item_ids: List[str] = []
	for items in user2items.values():
		item_ids.extend(items)
	item_ids = list(dict.fromkeys(item_ids))
	user2idx = {u: i for i, u in enumerate(user_ids)}
	item2idx = {it: i for i, it in enumerate(item_ids)}
	return user2idx, item2idx


def load_mapping(path: str) -> Dict[str, int]:
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Invalid mapping format: {path}")
	return {str(k): int(v) for k, v in data.items()}


def split_train_val_test(
	user2items: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
	train_user2items: Dict[str, List[str]] = {}
	val_items: Dict[str, str] = {}
	test_items: Dict[str, str] = {}
	for u, items in user2items.items():
		# Keep all users for train history construction:
		# train history always excludes only the last interaction (seq[:-1]).
		train_user2items[u] = items[:-1] if len(items) >= 1 else []
		# Validation/test targets are only available when sequence length >= 2.
		if len(items) >= 2:
			val_items[u] = items[-2]
			test_items[u] = items[-1]
	return train_user2items, val_items, test_items


def build_item_item_edge_index_rtr(
	user2items: Dict[str, List[str]],
	item2idx: Dict[str, int],
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
	rows: List[int] = []
	cols: List[int] = []
	uidx = 0
	for _, items in user2items.items():
		uniq_items = [it for it in dict.fromkeys(items) if it in item2idx]
		if not uniq_items:
			continue
		for it in uniq_items:
			rows.append(uidx)
			cols.append(item2idx[it])
		uidx += 1

	if not rows:
		empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
		empty_w = torch.empty((0,), dtype=torch.float32, device=device)
		return empty_idx, empty_w

	try:
		import scipy.sparse as sp
	except ImportError as e:
		raise ImportError("rt_r graph construction requires scipy. Please install scipy.") from e

	num_users = uidx
	num_items = len(item2idx)
	data = np.ones(len(rows), dtype=np.float32)
	user_item = sp.coo_matrix((data, (rows, cols)), shape=(num_users, num_items), dtype=np.float32).tocsr()
	item_item = (user_item.T @ user_item).tocoo()

	mask = item_item.row != item_item.col
	ii_rows = item_item.row[mask]
	ii_cols = item_item.col[mask]
	ii_vals = item_item.data[mask].astype(np.float32)

	if ii_rows.size == 0:
		empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
		empty_w = torch.empty((0,), dtype=torch.float32, device=device)
		return empty_idx, empty_w

	edge_index = torch.tensor(np.stack([ii_rows, ii_cols]), dtype=torch.long, device=device)
	edge_weight = torch.tensor(ii_vals, dtype=torch.float32, device=device)
	return edge_index, edge_weight


def normalize_adj(
	edge_index: torch.Tensor,
	num_nodes: int,
	edge_weight: Optional[torch.Tensor] = None,
) -> torch.sparse.FloatTensor:
	if edge_weight is None:
		values = torch.ones(edge_index.size(1), device=edge_index.device)
	else:
		values = edge_weight
	adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
	deg = torch.sparse.sum(adj, dim=1).to_dense()
	deg_norm = torch.pow(deg + 1e-8, -0.5)
	deg_norm[deg_norm == float("inf")] = 0.0
	row, col = adj.indices()
	adj_values = adj.values()
	norm_values = deg_norm[row] * adj_values * deg_norm[col]
	norm_adj = torch.sparse_coo_tensor(adj.indices(), norm_values, (num_nodes, num_nodes))
	return norm_adj.coalesce()


def compute_item_interaction_degree(
	train_user2items: Dict[str, List[str]],
	item2idx: Dict[str, int],
	device: torch.device,
) -> torch.Tensor:
	degree = torch.zeros(len(item2idx), dtype=torch.float32, device=device)
	for items in train_user2items.values():
		for it in items:
			idx = item2idx.get(it)
			if idx is not None:
				degree[idx] += 1.0
	return degree


def aggregate_layer_embeddings(
	embs: List[torch.Tensor],
) -> torch.Tensor:
	return embs[-1]


def propagate_from_gpt_feat(
	train_user2items: Dict[str, List[str]],
	item2idx: Dict[str, int],
	gpt_feat_path: str,
	num_layers: int,
	device: torch.device,
	fusion_alpha: float = 0.5,
	apply_degree_multiply: bool = True,
	degree_alpha: float = 0.0,
) -> np.ndarray:
	num_items = len(item2idx)
	feat = np.load(gpt_feat_path)
	if feat.ndim != 2:
		raise ValueError(f"gpt_feat must be 2D, got shape={feat.shape}")
	if feat.shape[0] != num_items:
		raise ValueError(
			f"gpt_feat rows ({feat.shape[0]}) must match num_items ({num_items})."
		)

	edge_index, edge_weight = build_item_item_edge_index_rtr(
		user2items=train_user2items,
		item2idx=item2idx,
		device=device,
	)
	norm_adj = normalize_adj(edge_index, num_items, edge_weight=edge_weight)

	item_emb = torch.tensor(feat, dtype=torch.float32, device=device)
	base_item_emb = item_emb
	embs = [item_emb]
	with torch.no_grad():
		for _ in range(num_layers):
			item_emb = torch.sparse.mm(norm_adj, item_emb)
			embs.append(item_emb)
		prop_item_emb = aggregate_layer_embeddings(embs=embs)
		if not (0.0 <= fusion_alpha <= 1.0):
			raise ValueError(f"fusion_alpha must be in [0, 1], got {fusion_alpha}")
		item_final = fusion_alpha * base_item_emb + (1.0 - fusion_alpha) * prop_item_emb
		if apply_degree_multiply:
			# Normalize propagated/fused item embeddings before degree scaling.
			item_final = F.normalize(item_final, p=2, dim=1, eps=1e-12)
			item_degree = compute_item_interaction_degree(
				train_user2items=train_user2items,
				item2idx=item2idx,
				device=device,
			)
			item_degree = torch.pow(torch.clamp(item_degree, min=1.0), degree_alpha)
			item_final = item_final * item_degree.view(-1, 1)
	return item_final.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Propagate GPT item embeddings with co-purchase graph")
	parser.add_argument("--dataset", type=str, default="sports")
	parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num_layers", type=int, default=1)
	parser.add_argument(
		"--fusion_alpha",
		type=float,
		default=0.5,
		help="Final residual blend: alpha*text + (1-alpha)*prop",
	)
	degree_scale_group = parser.add_mutually_exclusive_group()
	degree_scale_group.add_argument(
		"--apply_degree_multiply",
		dest="apply_degree_multiply",
		action="store_true",
		default=True,
		help="Multiply each item embedding by item degree after propagation/fusion (default)",
	)
	degree_scale_group.add_argument(
		"--no-apply_degree_multiply",
		dest="apply_degree_multiply",
		action="store_false",
		help="Disable item degree scaling",
	)
	parser.add_argument(
		"--degree_alpha",
		type=float,
		default=0.05,
		help="Exponent alpha for degree scaling d^alpha",
	)
	parser.add_argument("--gpt_feat_path", type=str, default="gpt_feat.npy", help="Path to gpt_feat.npy")
	parser.add_argument("--gpt_feat_out", type=str, default="refined_itemfeat.npy", help="Path to save propagated embeddings")
	
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.gpt_feat_path:
		args.gpt_feat_path = os.path.join(args.data_dir, args.dataset, "gpt_feat.npy")
	if not args.gpt_feat_out:
		args.gpt_feat_out = os.path.join(args.data_dir, args.dataset, "refined_itemfeat.npy")
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	data_path = os.path.join(args.data_dir, args.dataset, "inter.json")
	if not os.path.exists(data_path):
		raise FileNotFoundError(f"inter.json not found: {data_path}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Refinement] device={device}")
	print(f"[Refinement] loading interactions: {data_path}")
	print(f"[Refinement] loading features: {args.gpt_feat_path}")

	user2items = load_interactions(data_path)
	train_user2items, _, _ = split_train_val_test(user2items)

	user_map_path = os.path.join(args.data_dir, args.dataset, "user2id.json")
	item_map_path = os.path.join(args.data_dir, args.dataset, "item2id.json")
	if os.path.exists(user_map_path) and os.path.exists(item_map_path):
		_ = load_mapping(user_map_path)
		item2idx = load_mapping(item_map_path)
		print(f"[Refinement] loaded mappings from {user_map_path} and {item_map_path}")
	else:
		_, item2idx = build_mappings(train_user2items)
		print("[Refinement] mapping files missing; built item mapping from train interactions")

	out_base_path = Path(args.gpt_feat_out)
	out_base_path.parent.mkdir(parents=True, exist_ok=True)

	item_final = propagate_from_gpt_feat(
		train_user2items=train_user2items,
		item2idx=item2idx,
		gpt_feat_path=args.gpt_feat_path,
		num_layers=args.num_layers,
		device=device,
		fusion_alpha=args.fusion_alpha,
		apply_degree_multiply=args.apply_degree_multiply,
		degree_alpha=args.degree_alpha,
	)
	np.save(str(out_base_path), item_final)
	print(f"[Refinement] saved propagated item embeddings to {out_base_path}")


if __name__ == "__main__":
	main()
