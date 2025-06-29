# -*- coding: utf-8 -*-
"""
KEGG Drug–Drug Interaction Checker (Streamlit App) - 改善版
-----------------------------------------------------------
日本語（ブランド名・一般名いずれも可）の医薬品名を入力すると、KEGG REST `/ddi` API を使って
相互作用を検索し、結果を表形式で表示する簡易チェッカーです。

改善点:
- 不明な薬品名に対して類似する薬品名の候補を表示
- より厳密なマッチング制御

### 実行方法
```bash
streamlit run kegg_ddi_checker_improved.py
```
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from difflib import get_close_matches

import pandas as pd
import requests
import streamlit as st

# =============================================================================
# 定数
# =============================================================================
CSV_PATH = Path(__file__).with_name("kegg_drug_list_merged_ja_v1.1.csv")
API_BASE = "https://rest.kegg.jp/ddi/"
INTERACTION_JP = {"CI": "禁忌", "P": "併用注意", "CI,P": "禁忌・併注", "P,CI": "禁忌・併注"}
TIMEOUT = 20

# =============================================================================
# 正規化ユーティリティ
# =============================================================================
JP_CHAR = re.compile("[一-龠ぁ-ゔァ-ヴー]")
REMOVE_CHARS = re.compile(r"[\s・･\-‐–—／/®™\(\)（）\[\]【】,、，.．｡･:：'\"`]+")


def _clean_alias(name: str) -> str:
    """CSV の raw 文字列を前処理（(JAN) など除去）"""
    return name.replace("(JAN)", "").replace("(JP)", "").strip()


def _normalize(key: str) -> str:
    """マッチング用キー：不要文字を削除して小文字化"""
    return REMOVE_CHARS.sub("", key).lower()

# =============================================================================
# CSV → マッピング
# =============================================================================

def _find_col(df: pd.DataFrame, cand: set[str]):
    for col in df.columns:
        if col.strip().lower() in cand:
            return col
    return None


@st.cache_data(show_spinner=False)
def load_maps(csv_path: Path = CSV_PATH):
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path, dtype=str).fillna("")

    col_id = _find_col(df, {"kegg_id", "kegg"})
    col_jp = _find_col(df, {"drug_name_jp", "drug_jp", "drug_name_j"})
    col_jp0 = _find_col(df, {"drug_name_0_jp", "drug_name0_jp", "drug_brand", "drug_name0"})
    col_en = _find_col(df, {"drug_name", "drug_en", "name"})

    if not col_id or not col_jp:
        st.error("CSV に kegg_id / drug_name_jp 列が見つかりませんでした。")
        st.stop()

    jp2id: dict[str, str] = {}
    id2jp: dict[str, str] = {}
    id2jp0: dict[str, str] = {}
    # 正規化前の元の名前を保持（候補表示用）
    norm2original: dict[str, str] = {}

    for _, row in df.iterrows():
        kid = row[col_id].strip()
        if not kid:
            continue

        jp_raw = _clean_alias(row[col_jp])
        jp0_raw = _clean_alias(row[col_jp0]) if col_jp0 else ""
        en_raw = _clean_alias(row[col_en]) if col_en else ""

        aliases = [a.strip() for a in re.split(r"[;/]", jp_raw) if a.strip()]
        aliases0 = [a.strip() for a in re.split(r"[;/]", jp0_raw) if a.strip()]
        aliases.extend(aliases0)
        if en_raw:
            aliases.append(en_raw)

        if not aliases:
            continue

        for al in set(aliases):
            norm_key = _normalize(al)
            jp2id[norm_key] = kid
            # 正規化前の名前を保存（表示用）
            if norm_key not in norm2original or len(al) < len(norm2original[norm_key]):
                norm2original[norm_key] = al

        rep = next((a for a in aliases if JP_CHAR.search(a)), None) or aliases[0]
        id2jp[kid] = rep
        id2jp0[kid] = jp0_raw or rep

    return jp2id, id2jp, id2jp0, norm2original

# =============================================================================
# KEGG REST /ddi
# =============================================================================

def _call_ddi(endpoint: str):
    url = API_BASE + endpoint
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text.split("\n")
    except Exception as e:
        st.error(f"KEGG API Error: {e}")
        return None


def _strip_prefix(kid: str) -> str:
    return kid.replace("dr:", "").replace("cpd:", "").strip()


def fetch_ddi(ids: list[str]):
    if len(ids) < 2:
        return [], []

    recs = _call_ddi("+".join(ids))
    if recs is None:
        recs = []
        for a, b in combinations(ids, 2):
            r = _call_ddi(f"{a}+{b}")
            if r:
                recs.extend(r)

    ddi: list[tuple[str, str, str, str]] = []
    ids_in_results: set[str] = set()
    for line in recs:
        sp = line.split("\t")
        if len(sp) >= 4:
            a_id = _strip_prefix(sp[0])
            b_id = _strip_prefix(sp[1])
            ids_in_results.update([a_id, b_id])
            ddi.append((a_id, b_id, sp[2], sp[3]))

    ids_not_in_results = [i for i in ids if i not in ids_in_results]
    return ddi, ids_not_in_results

# =============================================================================
# 入力名 → KEGG ID（改善版）
# =============================================================================

def find_candidates(unknown_name: str, jp2id: dict[str, str], norm2original: dict[str, str], 
                   max_candidates: int = 5) -> list[tuple[str, float]]:
    """
    不明な薬品名に対して類似する候補を検索
    
    Returns:
        list of (original_name, similarity_score) tuples
    """
    key = _normalize(unknown_name)
    
    # 1. 部分一致の候補を探す
    partial_matches = []
    if len(key) >= 2:  # 2文字以上の場合のみ
        for norm_key, original in norm2original.items():
            if key in norm_key or norm_key in key:
                # 文字列の長さの差をスコアとして使用（差が小さいほど良い）
                length_diff = abs(len(key) - len(norm_key))
                similarity = 1.0 / (1.0 + length_diff)
                partial_matches.append((original, similarity))
    
    # 2. 類似度マッチングで候補を探す
    close_matches = get_close_matches(key, jp2id.keys(), n=max_candidates, cutoff=0.6)
    similarity_matches = []
    for match in close_matches:
        # difflib の類似度を計算
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, key, match).ratio()
        original_name = norm2original.get(match, match)
        similarity_matches.append((original_name, similarity))
    
    # 3. 候補を統合してスコアでソート
    all_candidates = {}
    for name, score in partial_matches + similarity_matches:
        if name in all_candidates:
            all_candidates[name] = max(all_candidates[name], score)
        else:
            all_candidates[name] = score
    
    # スコアの高い順にソート
    sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_candidates[:max_candidates]


def resolve_input_with_suggestions(names: list[str], jp2id: dict[str, str], 
                                 norm2original: dict[str, str], strict_mode: bool = False):
    """
    入力名をKEGG IDに変換（候補提案付き）
    
    Returns:
        ids: マッチしたKEGG IDのリスト
        unknown_with_candidates: 不明な薬品名とその候補のリスト
        id_display: KEGG ID -> 入力名のマッピング
    """
    ids: list[str] = []
    unknown_with_candidates: list[tuple[str, list[tuple[str, float]]]] = []
    id_display: dict[str, str] = {}

    for raw in names:
        key = _normalize(raw)
        
        # 1. 完全一致を試す
        hit = jp2id.get(key)
        
        if not hit and not strict_mode:
            # 2. 部分一致を試す（3文字以上の場合のみ）
            if len(key) >= 3:
                partial = [v for k, v in jp2id.items() if key in k]
                if len(partial) == 1:  # 一意な部分一致の場合のみ採用
                    hit = partial[0]
        
        if hit:
            ids.append(hit)
            id_display[hit] = raw
        else:
            # 候補を検索
            candidates = find_candidates(raw, jp2id, norm2original)
            unknown_with_candidates.append((raw, candidates))

    return ids, unknown_with_candidates, id_display

# =============================================================================
# Streamlit UI
# =============================================================================

st.title("KEGG 医薬品相互作用チェック β版")

jp2id_map, id2jp_map, id2jp0_map, norm2original_map = load_maps()

# セッション状態の初期化
if 'selected_drugs' not in st.session_state:
    st.session_state.selected_drugs = {}

# ---- コールバック：入力をクリア ------------------------------------------------

def clear_input():
    st.session_state["drug_input"] = ""
    st.session_state.selected_drugs = {}

# ---- 入力ウィジェット --------------------------------------------------------
user_text = st.text_input(
    "医薬品名をスペース区切りで入力 (例: クラビット カンデサルタン ピロカルピン)",
    key="drug_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    st.button("クリア", type="secondary", on_click=clear_input)
with col2:
    strict_mode = st.checkbox("厳密モード（完全一致のみ）", value=True)

# 入力なしなら終了
if not user_text.strip():
    st.stop()

# --------------------------
# 入力処理
# --------------------------
raw_list = user_text.strip().split()
kegg_ids, unknown_with_candidates, disp_map = resolve_input_with_suggestions(
    raw_list, jp2id_map, norm2original_map, strict_mode
)

# --- 不明な薬品名の候補表示 ---
if unknown_with_candidates:
    st.warning("データベースにない薬剤が見つかりました。以下から選択するか、入力を修正してください。")
    
    for unknown_name, candidates in unknown_with_candidates:
        if candidates:
            st.subheader(f"「{unknown_name}」の候補:")
            
            # 候補を選択可能にする
            selected = st.radio(
                f"{unknown_name}の代替候補を選択",
                options=["選択しない"] + [f"{name} (類似度: {score:.2f})" for name, score in candidates],
                key=f"select_{unknown_name}",
                index=0
            )
            
            if selected != "選択しない":
                # 選択された候補から薬品名を抽出
                selected_name = selected.split(" (類似度:")[0]
                selected_key = _normalize(selected_name)
                if selected_key in jp2id_map:
                    selected_id = jp2id_map[selected_key]
                    st.session_state.selected_drugs[unknown_name] = (selected_id, selected_name)
        else:
            st.error(f"「{unknown_name}」に類似する薬品が見つかりませんでした。")

# 選択された候補を追加
for orig_name, (selected_id, selected_name) in st.session_state.selected_drugs.items():
    if selected_id not in kegg_ids:
        kegg_ids.append(selected_id)
        disp_map[selected_id] = f"{selected_name} (←{orig_name})"

# --- 同じ KEGG ID の薬剤が複数入力されていた場合を検出 ---
id_counts = {}
for kid in kegg_ids:
    id_counts[kid] = id_counts.get(kid, 0) + 1
same_id_list = [id2jp_map.get(k, k) for k, v in id_counts.items() if v > 1]

if same_id_list:
    st.info("同一KEGG IDの薬剤が入力されました: " + ", ".join(same_id_list))

# 最終的に選択された薬剤の表示
if kegg_ids:
    st.success(f"検索対象薬剤 ({len(kegg_ids)}個): " + 
               ", ".join([disp_map.get(kid, id2jp_map.get(kid, kid)) for kid in kegg_ids]))

if len(kegg_ids) < 2:
    st.info("相互作用をチェックするには2つ以上の薬剤が必要です。")
    st.stop()

# --------------------------
# 相互作用検索
# --------------------------
with st.spinner("相互作用を検索中..."):
    ddi, no_ddi_ids = fetch_ddi(kegg_ids)

# --------------------------
# 相互作用なし → ここで終了
# --------------------------
if not ddi:
    st.success("相互作用は検出されませんでした。")
    st.stop()

# 入力薬のうち相互作用行に現れなかったものを参考表示
if no_ddi_ids:
    st.info(
        "相互作用が見つからなかった薬剤: "
        + ", ".join(disp_map.get(x, id2jp_map.get(x, x)) for x in no_ddi_ids)
    )

# --------------------------
# 結果テーブル
# --------------------------
out_rows = []
for a, b, code, mech in ddi:
    name_a = disp_map.get(a, id2jp_map.get(a, a))
    name_b = disp_map.get(b, id2jp_map.get(b, b))
    code_disp = INTERACTION_JP.get(code, code)
    out_rows.append(
        {
            "薬剤A": name_a,
            "薬剤B": name_b,
            "相互作用": code_disp,
            "機序": mech,
        }
    )

st.subheader("⚠️ 相互作用リスト ⚠️")
st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

# =============================================================================
# =============================================================================