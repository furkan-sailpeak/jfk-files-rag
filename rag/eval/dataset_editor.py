"""Live editor for the eval datasets in rag/eval/datasets/.

    streamlit run rag/eval/dataset_editor.py

Pick a dataset (dataset_{project}_{YYYY-MM-DD}.json) in the sidebar or create
one. Every dataset has the same columns (see store.py). Edits are written to
the file immediately; changes made to the file from outside (another editor,
git checkout, a script) show up within ~2s.
"""
import os
from datetime import date

import pandas as pd
import store
import streamlit as st

st.set_page_config(page_title="Eval datasets", page_icon=":material/fact_check:", layout="wide")
ss = st.session_state
STATUSES = ["All", "Validated", "Not validated"]


# --- State ---------------------------------------------------------------------
def _mtime():
    return os.path.getmtime(ss.path)


def open_dataset(path):
    ss.path = path
    reload()


def reload():
    ss.rows = store.load(ss.path)
    ss.mtime = _mtime()
    ss.version = ss.get("version", 0) + 1  # new widget keys -> fresh table/form


def write(rows, message):
    ss.rows = store.save(rows, ss.path)
    ss.mtime = _mtime()
    ss.flash = message


# --- Sidebar: choose or create a dataset -------------------------------------------
datasets = store.list_datasets()
with st.sidebar:
    st.header("Datasets")
    if datasets:
        current = ss.get("path") if ss.get("path") in datasets else datasets[0]
        picked = st.selectbox(
            "Dataset", datasets, index=datasets.index(current), key="dataset",
            format_func=lambda p: "{} · {}".format(*store.parse_name(p)),
        )
        if picked != ss.get("path"):
            open_dataset(picked)
    else:
        st.info("No datasets yet. Create one below.", icon=":material/info:")

    with st.expander("New dataset", icon=":material/add:", expanded=not datasets):
        with st.form("new_dataset", border=False):
            project = st.text_input("Project name", placeholder="e.g. jfk", key="new_project")
            day = st.date_input("Date", date.today(), key="new_date")
            copy = st.toggle("Copy rows from the open dataset", key="new_copy", disabled=not datasets)
            if st.form_submit_button("Create", icon=":material/add:", type="primary", key="new_submit"):
                try:
                    path = store.create(project, day, ss.rows if copy and datasets else None)
                except (ValueError, FileExistsError) as e:
                    st.error(str(e))
                else:
                    ss.flash = f"Created {path.name}"
                    ss.pop("dataset", None)
                    open_dataset(path)
                    st.rerun()
    st.caption(f"Folder: `{store.DATASETS_DIR}`")

if "path" not in ss:
    st.stop()


@st.fragment(run_every=2)
def watch_file():
    if not ss.path.exists():
        ss.pop("path")
        st.rerun()
    if _mtime() != ss.mtime:
        reload()
        ss.flash = "Reloaded: the file changed on disk"
        st.rerun()


watch_file()
if msg := ss.pop("flash", None):
    st.toast(msg, icon=":material/save:")

rows, v, name = ss.rows, ss.version, ss.path.name
project, day = store.parse_name(ss.path)

# --- Header ------------------------------------------------------------------
st.title(f":material/fact_check: {project}")
st.caption(f"`{name}` · {day} · every edit saves to the file · outside changes show up within 2s")

with st.container(horizontal=True):
    st.metric("Questions", len(rows), border=True)
    st.metric("Validated", f"{sum(r['validated'] for r in rows)} / {len(rows)}", border=True)
    st.metric("With expected sources", sum(bool(r["expected_sources"]) for r in rows), border=True)

# --- Filters -----------------------------------------------------------------
cats = st.pills("Category", store.CATEGORIES, selection_mode="multi", key=f"f-cat-{name}")
with st.container(horizontal=True, vertical_alignment="bottom"):
    status = st.segmented_control("Status", STATUSES, default="All", key=f"f-status-{name}") or "All"
    search = st.text_input("Search", placeholder="Words in the question, answer or file names", key=f"f-search-{name}")


def visible(r):
    if cats and r["category"] not in cats:
        return False
    if status != "All" and r["validated"] != (status == "Validated"):
        return False
    files = " ".join(s["file"] for s in r["expected_sources"])
    return search.lower() in f"{r['question']} {r['ground_truth_answer']} {files}".lower()


# --- Table -------------------------------------------------------------------
# Streamlit stores table edits as a delta on the data it was first given, so
# that base must stay frozen for as long as the key is the same. The key
# changes with the dataset, the file version and the filters, never with our
# own saves.
key = f"editor-{name}-{v}-{hash((tuple(cats), status, search))}"
if ss.get("editor_key") != key:
    ss.editor_key = key
    ss.base = [dict(r) for r in rows if visible(r)]
shown = ss.base

table = pd.DataFrame(
    [{
        **r,
        "expected_sources": [store.source_text(s) for s in r["expected_sources"]],
        "first_source": store.source_url(r["expected_sources"][0]) if r["expected_sources"] else None,
    } for r in shown],
    columns=[*store.FIELDS[:-1], "first_source", "validated"],
).astype({"validated": bool})

edited = st.data_editor(
    table,
    key=key,
    num_rows="dynamic",
    hide_index=True,
    height=600,
    alt=f"Questions of the {project} dataset with expected sources and validation status",
    column_config={
        "id": st.column_config.TextColumn("ID", disabled=True, width="small"),
        "question": st.column_config.TextColumn("Question", width="medium", required=True),
        "category": st.column_config.SelectboxColumn("Category", options=store.CATEGORIES, width="small", required=True),
        "ground_truth_answer": st.column_config.TextColumn("Ground truth answer", width="large"),
        "expected_sources": st.column_config.ListColumn(
            "Expected sources", width="medium",
            help="Pages the answer should cite, one per item, e.g. `104-10015-10001.pdf p.12`",
        ),
        "first_source": st.column_config.LinkColumn(
            "Open", disabled=True, width="small", display_text="Open",
            help="Opens the first expected source at its page. All links are in the review panel below.",
        ),
        "validated": st.column_config.CheckboxColumn("Validated", default=False, width="small"),
    },
)
st.caption(f"Showing {len(shown)} of {len(rows)} · add a row at the bottom, select rows and press Delete to remove")

edited_rows, bad = [], []
for r in edited.drop(columns="first_source").to_dict("records"):
    r = {k: ("" if not isinstance(val, (list, bool)) and pd.isna(val) else val) for k, val in r.items()}
    try:
        r["expected_sources"] = [store.parse_source(s) for s in (r["expected_sources"] or [])]
    except ValueError as e:
        bad.append(f"{r['id'] or 'new row'}: {e}")
        continue
    edited_rows.append(store.normalize(r))

if bad:
    st.error("Not saved, fix these sources first:\n\n" + "\n".join(f"- {b}" for b in bad), icon=":material/error:")
else:
    # Merge the visible subset back into the full list: rows edited in place
    # keep their position, rows deleted from the view are dropped, new rows are
    # appended once they have a question (a half-filled new row stays put).
    by_id = {r["id"]: r for r in edited_rows if r["id"]}
    shown_ids = {r["id"] for r in shown}
    merged = [by_id.get(r["id"], r) for r in rows if r["id"] in by_id or r["id"] not in shown_ids]
    new = [r for r in edited_rows if not r["id"] and r["question"]]
    if merged + new != rows:
        write(merged + new, f"Saved to {name}")
        if new:
            reload()  # show the generated ids
        st.rerun()

# --- Review panel ----------------------------------------------------------------
st.subheader("Review a question")
ids = [r["id"] for r in rows if visible(r)]
if not ids:
    st.info("No questions match the filters.", icon=":material/filter_alt_off:")
    st.stop()

by_row = {r["id"]: r for r in rows}
pick = st.selectbox("Question", ids, key=f"pick-{name}",
                    format_func=lambda i: f"{i} · {by_row[i]['question']}")
cur = by_row[pick]

with st.container(border=True):
    st.markdown("**Expected sources** (open each page to check the answer)")
    if cur["expected_sources"]:
        with st.container(horizontal=True):
            for s in cur["expected_sources"]:
                st.link_button(store.source_text(s), store.source_url(s), icon=":material/description:")
    else:
        st.caption("None: no citation is expected for this question.")

with st.form(f"one-{name}-{pick}-{v}", border=True):
    q = st.text_input("Question", cur["question"], key=f"q-{pick}-{v}")
    cat = st.selectbox("Category", store.CATEGORIES, key=f"cat-{pick}-{v}",
                       index=store.CATEGORIES.index(cur["category"]) if cur["category"] in store.CATEGORIES else None)
    gt = st.text_area("Ground truth answer", cur["ground_truth_answer"], height=180, key=f"gt-{pick}-{v}")
    current_src = [store.source_text(s) for s in cur["expected_sources"]]
    src = st.multiselect(
        "Expected sources", current_src, default=current_src, accept_new_options=True, key=f"src-{pick}-{v}",
        help="Type a new one as `104-10015-10001.pdf p.12` and press Enter",
    )
    ok = st.checkbox("Validated: answer and sources checked", cur["validated"], key=f"ok-{pick}-{v}")
    if st.form_submit_button("Save", icon=":material/save:", type="primary", key=f"save-{v}"):
        try:
            sources = [store.parse_source(s) for s in src]
        except ValueError as e:
            st.error(str(e), icon=":material/error:")
        else:
            write([{**r, "question": q, "category": cat or "", "ground_truth_answer": gt,
                    "expected_sources": sources, "validated": ok} if r["id"] == pick else r for r in rows],
                  f"Saved {pick}")
            reload()
            st.rerun()
