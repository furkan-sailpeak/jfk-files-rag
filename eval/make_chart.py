"""
Thesis headline visual: JFK-RAG vs. plain GPT-5.4 on 24 like-for-like
evaluation questions. Top row shows per-category breakdown on the two
headline metrics (hallucination, completeness). Bottom row collapses
everything into a single composite score comparing the two systems.

Writes comparison_chart.html.
"""
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
OUT = HERE / "comparison_chart.html"

CATS = ["factual", "biographical", "analytical", "partial evidence"]

HALLUC_RAG = [0.00, 0.00, 0.17, 0.17]
HALLUC_GPT = [0.50, 0.50, 0.83, 0.67]

COMPL_RAG = [0.57, 0.54, 0.57, 0.56]
COMPL_GPT = [0.46, 0.73, 0.40, 0.34]

# For the composite score (all normalised 0–1, higher is better).
FAITH_RAG = [0.62, 0.83, 0.75, 0.74]     # RAG-side grounding
CORR_GPT  = [0.66, 0.83, 0.59, 0.70]     # GPT-side correctness-vs-corpus
CLARITY_RAG = [4.50, 4.33, 4.17, 4.33]   # 1–5
CLARITY_GPT = [5.00, 4.83, 4.17, 4.50]

RAG_COLOR = "#2563eb"
GPT_COLOR = "#f97316"


def mean(xs):
    return sum(xs) / len(xs)


def add_category_group(fig, col, rag_vals, gpt_vals, fmt):
    if fmt == "pct":
        rag_txt = [f"{int(round(v*100))}%" for v in rag_vals]
        gpt_txt = [f"{int(round(v*100))}%" for v in gpt_vals]
    else:
        rag_txt = [f"{v:.2f}" for v in rag_vals]
        gpt_txt = [f"{v:.2f}" for v in gpt_vals]
    showlegend = col == 1
    fig.add_trace(
        go.Bar(x=CATS, y=rag_vals, name="JFK-RAG",
               marker_color=RAG_COLOR, text=rag_txt, textposition="outside",
               showlegend=showlegend,
               hovertemplate="RAG · %{x}: %{y:.2f}<extra></extra>"),
        row=1, col=col,
    )
    fig.add_trace(
        go.Bar(x=CATS, y=gpt_vals, name="GPT-5.4 (no retrieval)",
               marker_color=GPT_COLOR, text=gpt_txt, textposition="outside",
               showlegend=showlegend,
               hovertemplate="GPT · %{x}: %{y:.2f}<extra></extra>"),
        row=1, col=col,
    )


def main():
    # Composite score components (all normalised 0–1, higher is better).
    # Equal-weight mean of four axes: coverage, grounded-ness, correctness, clarity.
    compl_r, compl_g = mean(COMPL_RAG), mean(COMPL_GPT)
    nonhall_r, nonhall_g = 1 - mean(HALLUC_RAG), 1 - mean(HALLUC_GPT)
    correct_r, correct_g = mean(FAITH_RAG), mean(CORR_GPT)
    clarity_r, clarity_g = mean(CLARITY_RAG) / 5, mean(CLARITY_GPT) / 5

    components = [
        ("Completeness", compl_r, compl_g),
        ("Non-hallucination", nonhall_r, nonhall_g),
        ("Faithfulness / Correctness", correct_r, correct_g),
        ("Clarity (÷5)", clarity_r, clarity_g),
    ]
    overall_r = mean([c[1] for c in components])
    overall_g = mean([c[2] for c in components])

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "Hallucination rate by category (lower is better)",
            "Completeness by category (higher is better)",
            f"Overall composite score (equal-weight mean of four axes) — "
            f"<b><span style='color:{RAG_COLOR}'>RAG {overall_r:.2f}</span></b> "
            f"vs <b><span style='color:{GPT_COLOR}'>GPT {overall_g:.2f}</span></b>",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
        row_heights=[0.55, 0.45],
    )

    # top row — per-category breakdowns
    add_category_group(fig, 1, HALLUC_RAG, HALLUC_GPT, "pct")
    add_category_group(fig, 2, COMPL_RAG,  COMPL_GPT,  "score")

    # bottom row — composite score. Four components + overall at the right.
    labels = [c[0] for c in components] + ["<b>OVERALL</b>"]
    rag_bottom = [c[1] for c in components] + [overall_r]
    gpt_bottom = [c[2] for c in components] + [overall_g]

    fig.add_trace(
        go.Bar(x=labels, y=rag_bottom, name="JFK-RAG",
               marker_color=RAG_COLOR,
               text=[f"{v:.2f}" for v in rag_bottom], textposition="outside",
               showlegend=False,
               hovertemplate="RAG · %{x}: %{y:.2f}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=gpt_bottom, name="GPT-5.4 (no retrieval)",
               marker_color=GPT_COLOR,
               text=[f"{v:.2f}" for v in gpt_bottom], textposition="outside",
               showlegend=False,
               hovertemplate="GPT · %{x}: %{y:.2f}<extra></extra>"),
        row=2, col=1,
    )

    # axes
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", row=1, col=1,
                     title_text="% of answers", gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(range=[0, 1.12], row=1, col=2,
                     title_text="mean score (0–1)", gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(range=[0, 1.12], row=2, col=1,
                     title_text="normalised score (0–1)", gridcolor="#e5e7eb", zeroline=False)
    fig.update_xaxes(tickangle=0, ticks="outside")

    # spacing / style for subplot titles
    for ann in fig["layout"]["annotations"]:
        ann["yanchor"] = "bottom"
        ann["font"] = dict(size=14, color="#111827", family="Inter, sans-serif")

    fig.update_layout(
        title=dict(
            text="<b>Corpus-grounded RAG vs. plain GPT-5.4 — JFK-archive evaluation</b>",
            x=0.5, xanchor="center", y=0.97, yanchor="top",
            font=dict(size=20),
        ),
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Helvetica, sans-serif", size=13, color="#111827"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
        height=900,
        width=1500,
        margin=dict(t=140, b=110, l=70, r=40),
    )

    fig.add_annotation(
        text=("24 questions across 4 categories (6 each) · judge: gpt-5.4-mini · "
              "out-of-scope excluded (behavioural split, not comparable) · "
              "composite = mean(completeness, 1−hallucination, faithfulness/correctness, clarity÷5)"),
        xref="paper", yref="paper", x=0.5, y=-0.11,
        xanchor="center", yanchor="top",
        showarrow=False,
        font=dict(size=11, color="#6b7280", family="Inter, sans-serif"),
    )

    fig.write_html(OUT, include_plotlyjs="cdn")
    png_out = OUT.with_suffix(".png")
    try:
        fig.write_image(png_out, width=1500, height=900, scale=2)
        print(f"wrote {png_out}")
    except Exception as e:
        print(f"(png export skipped: {e})")
    print(f"wrote {OUT}")
    print(f"overall: RAG {overall_r:.3f}   GPT {overall_g:.3f}")


if __name__ == "__main__":
    main()
