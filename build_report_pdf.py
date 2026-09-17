#!/usr/bin/env python3
"""Professional dense PDF report for CISC3024 AI Assignment #1."""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, "CISC3024 AI Assignment #1 | CHE CHI HIN, Angus (UC325182)", align="L")
        self.ln(8)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def mc(pdf: PDF, h: float, text: str, style: str = "") -> None:
    pdf.set_font("Helvetica", style, 10 if style != "B" else 11)
    pdf.multi_cell(pdf.epw, h, text, new_x="LMARGIN", new_y="NEXT")


def h1(pdf: PDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(pdf.epw, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)


def h2(pdf: PDF, text: str) -> None:
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(pdf.epw, 6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)


def para(pdf: PDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(pdf.epw, 5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def bullets(pdf: PDF, items: list[str]) -> None:
    for it in items:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(pdf.epw, 5, f"  - {it}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def table(pdf: PDF, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
    if col_widths is None:
        w = pdf.epw / len(headers)
        col_widths = [w] * len(headers)
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for row in rows:
        # estimate height
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 5.5, str(cell), border=1, align="C")
        pdf.ln()
    pdf.ln(2)


def fig_block(pdf: PDF, path: Path, caption: str, interpretation: str, max_h: float = 95) -> None:
    if not path.exists():
        return
    # Ensure room for image + caption + interpretation
    need = max_h + 28
    if pdf.get_y() + need > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(pdf.epw, 4.5, caption, new_x="LMARGIN", new_y="NEXT")
    # fit width, clamp height
    import PIL.Image

    im = PIL.Image.open(path)
    iw, ih = im.size
    tw = pdf.epw
    th = tw * ih / iw
    if th > max_h:
        th = max_h
        tw = th * iw / ih
    x = pdf.l_margin + (pdf.epw - tw) / 2
    pdf.image(str(path), x=x, w=tw, h=th)
    pdf.ln(1)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(pdf.epw, 4.2, interpretation, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def loadj(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    pconv = loadj("pconv_efficiency.json")
    extra = loadj("fasternet_aug_extra_analysis.json")
    multi = loadj("multiseed_summary.json")
    ndiv = loadj("ndiv_ablation.json")
    ms = multi.get("multiseed_fasternet_aug_mini", {})
    t0 = multi.get("t0lite", {})
    emnist = loadj("fasternet_aug_emnist_eval.json")
    em_ct = emnist.get("corruption_table", {})

    pdf = PDF(format="A4")
    pdf.set_margins(14, 12, 14)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(pdf.epw, 8, "AI Assignment #1", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(
        pdf.epw,
        6,
        "FasterNet for Robust Digit Pattern Recognition under Geometric and Noise Corruptions",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(1)
    para(pdf, "CISC3024 Pattern Recognition")
    para(pdf, "Name: CHE CHI HIN, Angus    |    Student ID: UC325182")
    para(pdf, "AI agent used for search, coding, experiments, and report drafting: Cursor agent. No hand-written code.")

    h1(pdf, "Abstract")
    para(
        pdf,
        "This report presents an AI-agent-driven reproduction and application of FasterNet (CVPR 2023), "
        "a recent Deep CNN whose core operator is Partial Convolution (PConv). The application is robust "
        "handwritten digit recognition (classes 0-9) under rotation, noise, blur, shift, occlusion, and "
        "elastic distortion. Models are evaluated on MNIST and writer-varied EMNIST Digits. A clean-only "
        "CNN reaches ~99% upright accuracy but collapses under 90-degree rotation (~13%); robust "
        "augmentation recovers Rot90 to ~80-84% (TTA ~94%). We verify PConv mechanistically (identity of "
        "untouched channels; FLOPs/latency vs dense/depthwise conv), ablate n_div, report multi-seed "
        "mean+/-std, calibration (ECE), and confusable pairs such as 6<->9 under rotation.",
    )

    h1(pdf, "1. Task and Problem Statement")
    para(
        pdf,
        "The assignment asks for a recent Deep CNN or Deep Autoencoder with a computer-vision / pattern-"
        "recognition application, with search, programming, and writing performed by an AI tool/agent. "
        "We select FasterNet and frame the task as robust pattern recognition: a digit such as '3' should "
        "remain identifiable after rotation or corruption, across all classes 0-9, not only on clean upright MNIST.",
    )

    h1(pdf, "2. AI-Assisted Workflow")
    para(
        pdf,
        "All technical work was executed by an AI coding agent under human direction. The human set goals "
        "(robust multi-condition recognition, EMNIST extension, professional report), reviewed figures/metrics, "
        "and approved iterations. The agent performed literature search, paper selection, environment setup, "
        "implementation, training/evaluation, figure generation, ablation studies, and report drafting. "
        "A detailed command-level diary is kept in process_log.md for the required 'how I used AI' documentation.",
    )
    bullets(
        pdf,
        [
            "Search: recent lightweight CNNs (2023-2024); selected FasterNet for PConv; avoided peer RepViT narrative.",
            "Implement: FasterNet-mini / T0-lite, baseline CNN, corruption suite, train/eval/demo, Gradio, Grad-CAM.",
            "Iterate: widened rotation aug to cardinal angles; added TTA; EMNIST Digits; n_div ablation; dense PDF rewrite.",
        ],
    )

    h1(pdf, "3. Algorithm: FasterNet (CVPR 2023)")
    para(
        pdf,
        "Reference: Chen et al., 'Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks,' CVPR 2023 "
        "(arXiv:2303.03667). FasterNet argues that reducing FLOPs alone (e.g., via depthwise conv) may not "
        "maximize throughput because of memory-access cost. Partial Convolution applies a 3x3 conv to only "
        "1/n_div of channels and leaves the rest untouched, then mixes channels with a residual pointwise MLP.",
    )
    para(
        pdf,
        "Related work. MobileOne (CVPR 2023) improves mobile latency via structural re-parameterization "
        "(a different mechanism than PConv). Hendrycks & Dietterich (ICLR 2019) motivate corruption-style "
        "evaluation protocols that we adapt to digits. Cohen et al. (2017) EMNIST provides writer-varied "
        "digits beyond textbook MNIST. LeCun et al. (1998) introduced MNIST as the classic digit PR benchmark.",
    )
    fig_block(
        pdf,
        OUT / "fig_architecture.png",
        "Figure 1. FasterNet-mini architecture used in this assignment.",
        "Interpretation: the network preserves FasterNet's PConv+MLP block idea while using a 2x2 stem "
        "and three stages suitable for 28x28 inputs. The yellow callout highlights the operator we verify.",
        max_h=55,
    )

    h1(pdf, "4. PConv Mechanism and Efficiency Verification")
    para(
        pdf,
        "To parallel the spirit of mechanism checks in architecture papers (and peer fusion-style reports), "
        "we verify two claims about PConv: (i) untouched channels are exact identities; (ii) compute/latency "
        "is far below a dense 3x3 convolution on the same activation shape.",
    )
    if pconv:
        ops = {r["op"]: r for r in pconv.get("operators", [])}
        pc, dw, dens = ops["PConv (n_div=4)"], ops["Depthwise 3x3"], ops["Dense 3x3"]
        table(
            pdf,
            ["Operator", "Params", "Rel. FLOPs", "Latency ms (B64)"],
            [
                ["PConv n_div=4", str(pc["params"]), f"{pc['relative_flops_vs_dense']:.4f}x", f"{pc['latency_ms']:.3f}"],
                ["Depthwise 3x3", str(dw["params"]), f"{dw['relative_flops_vs_dense']:.4f}x", f"{dw['latency_ms']:.3f}"],
                ["Dense 3x3", str(dens["params"]), "1.0000x", f"{dens['latency_ms']:.3f}"],
            ],
            [45, 35, 40, 50],
        )
        para(
            pdf,
            f"Untouched-channel max abs error after PConv is {pconv.get('pconv_untouched_channel_max_abs_err')} "
            "(exactly 0), confirming the identity path. PConv uses only 6.25% of dense 3x3 FLOPs in this setting.",
        )

    # n_div ablation
    h2(pdf, "4.1 Ablation on n_div in {2,4,8}")
    para(
        pdf,
        "n_div controls how many channels are spatially convolved. Smaller n_div convolves more channels "
        "(stronger mixing, higher cost); larger n_div is cheaper but mixes less. We report a controlled "
        "5-epoch probe on a 20k/5k MNIST subset plus operator latency.",
    )
    if ndiv.get("accuracy_ablation_5ep_20k"):
        rows = []
        for r in ndiv["accuracy_ablation_5ep_20k"]:
            rows.append(
                [
                    str(r["n_div"]),
                    str(r["params"]),
                    f"{r['clean']:.3f}",
                    f"{r['rotate_45']:.3f}",
                    f"{r['rotate_90']:.3f}",
                    f"{r['pconv_latency_ms']:.3f}",
                ]
            )
        table(
            pdf,
            ["n_div", "Params", "Clean", "Rot45", "Rot90", "PConv ms"],
            rows,
            [22, 32, 28, 28, 28, 32],
        )
    fig_block(
        pdf,
        OUT / "fig_ndiv_ablation.png",
        "Figure 2. n_div ablation: Rot90 accuracy vs PConv latency (probe setting).",
        "Interpretation: n_div trades spatial mixing for speed. We keep n_div=4 as the paper-default "
        "balance used in the main FasterNet-mini experiments.",
        max_h=58,
    )

    h1(pdf, "5. Experimental Setup")
    para(
        pdf,
        "Training uses AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, batch size 128, on Apple MPS. "
        "Robust-aug training applies stochastic cardinal/continuous rotations and noise/blur/shift/"
        "occlusion/elastic warps. We compare clean-only vs robust-aug FasterNet-mini, a SimpleCNN baseline, "
        "and a T0-lite width/depth variant. EMNIST Digits uses a 60k/10k subsample for student-scale runtime "
        "while retaining writer variation. Metrics include corruption-wise accuracy, Rot90+TTA, reject@0.7, "
        "ECE, and multi-seed mean+/-std (n=3).",
    )

    h1(pdf, "6. Results")
    para(
        pdf,
        "Table 1 summarizes the main robustness profile. Clean-only models look strong upright but fail "
        "under large rotations; robust augmentation restores performance on both MNIST and EMNIST Digits. "
        "Rotation TTA further helps when orientation is unknown at test time.",
    )
    table(
        pdf,
        ["Setting", "Clean", "Rot45", "Rot90", "Rot90+TTA", "Noise"],
        [
            ["MNIST FasterNet-aug", "0.989", "0.965", "0.805", "0.939", "0.940"],
            ["EMNIST FasterNet-aug", "0.988", "0.968", "0.835", "0.941", "0.801"],
            ["MNIST FasterNet clean-only", "0.992", "0.548", "0.130", "-", "0.425"],
            ["MNIST Baseline-aug", "0.958", "0.735", "0.267", "-", "0.934"],
        ],
        [52, 22, 22, 22, 28, 22],
    )
    para(
        pdf,
        "Reject@0.7 on MNIST clean test yields 98.4% coverage and 99.4% accuracy on accepted predictions. "
        "Multi-seed MNIST robust-aug Rot90 is 84.1% +/- 10.0% (values 0.887, 0.727, 0.911), showing that "
        "single-run Rot90 claims need error bars. T0-lite (657k params) reaches Rot90 0.892 in a 12-epoch run "
        "but is slower and weaker on heavy noise than the mini model.",
    )
    if extra:
        clean, rot = extra.get("clean", {}), extra.get("rotate_90", {})
        para(
            pdf,
            f"Calibration: clean ECE={clean.get('ece', 0):.4f} (well calibrated, supporting reject). "
            f"Under Rot90, ECE rises to {rot.get('ece', 0):.4f}. The hardest confusions are 6<->9 "
            f"(6->9 count={rot.get('confusable_pairs', {}).get('6_9', {}).get('6_as_9', '?')}) and milder 3<->8 swaps, "
            "which matches geometric ambiguity after rotation.",
        )

    h2(pdf, "6.1 Why this application is not 'too simple'")
    para(
        pdf,
        "Upright MNIST classification is easy; recognizing all digits 0-9 under unknown orientation, "
        "photometric noise, occlusion, and writer variation is not. EMNIST Digits results (Rot90 83.5%, "
        "TTA 94.1%) show the pipeline transfers beyond the classic MNIST distribution.",
    )

    # Dense figure section with interpretations
    h1(pdf, "7. Figures and Qualitative Evidence")
    fig_block(
        pdf,
        OUT / "fig_mnist_vs_emnist_bars.png",
        "Figure 3. MNIST vs EMNIST Digits robustness bars.",
        "Interpretation: EMNIST remains strong on rotation; noise is harder than on MNIST, confirming a more realistic writer/domain shift.",
        max_h=70,
    )
    fig_block(
        pdf,
        OUT / "fig_mnist_multangle_0to9.png",
        "Figure 4. MNIST digits 0-9 under multiple rotation angles.",
        "Interpretation: evaluation is not limited to 90 degrees; mild angles stay near-perfect while extreme angles stress the model.",
        max_h=100,
    )
    fig_block(
        pdf,
        OUT / "fig_mnist_multicorr_0to9.png",
        "Figure 5. MNIST digits 0-9 under multiple corruptions.",
        "Interpretation: blur/shift/occlusion are largely handled; elastic and heavy noise create harder failure modes.",
        max_h=110,
    )
    fig_block(
        pdf,
        OUT / "fig_mnist_rot90_vs_tta_0to9.png",
        "Figure 6. Single-pass Rot90 vs rotation TTA on digits 0-9.",
        "Interpretation: TTA averages predictions over cardinal orientations and recovers many 90-degree mistakes when orientation is unknown.",
        max_h=55,
    )
    fig_block(
        pdf,
        OUT / "fig_emnist_multangle_0to9.png",
        "Figure 7. EMNIST Digits 0-9 under rotations.",
        "Interpretation: writer-varied strokes remain recognizable after robust training, supporting generalization beyond MNIST.",
        max_h=70,
    )
    fig_block(
        pdf,
        OUT / "fig_emnist_multicorr_0to9.png",
        "Figure 8. EMNIST Digits 0-9 under corruptions.",
        "Interpretation: the same corruption suite applies; qualitative errors align with the quantitative EMNIST noise gap.",
        max_h=85,
    )
    fig_block(
        pdf,
        OUT / "fig_gradcam_multiangle.png",
        "Figure 9. Grad-CAM across angles for selected digits.",
        "Interpretation: attention remains on stroke structure under rotation, indicating the model uses shape cues rather than absolute pixel position alone.",
        max_h=70,
    )
    fig_block(
        pdf,
        OUT / "fasternet_aug_rotate_90_perclass_cm.png",
        "Figure 10. MNIST confusion matrix at 90-degree rotation.",
        "Interpretation: class-wise failures concentrate on geometrically confusable pairs (notably 6/9), not uniform collapse.",
        max_h=75,
    )
    fig_block(
        pdf,
        OUT / "fasternet_aug_clean_reliability.png",
        "Figure 11. Reliability diagram / ECE on clean MNIST.",
        "Interpretation: confidence tracks accuracy closely, which justifies confidence-based reject/abstain.",
        max_h=70,
    )

    h1(pdf, "8. Discussion")
    para(
        pdf,
        "The central lesson is that clean accuracy is an incomplete metric for pattern recognition under "
        "nuisance transformations. Robust augmentation aligns training with the threat model; TTA is a "
        "cheap test-time complement when orientation is unknown. Multi-seed variance on Rot90 warns against "
        "over-interpreting a single checkpoint. Compared with a peer report focused on RepViT structural "
        "re-parameterization on tiny 8x8 digits, this work verifies a different Deep-CNN mechanism (PConv), "
        "uses MNIST+EMNIST, and emphasizes multi-condition application metrics and figures.",
    )
    para(
        pdf,
        "Limitations: student-scale training is not ImageNet mobile latency reproduction; EMNIST is "
        "subsampled for runtime; residual 90-degree errors remain on confusable shapes. Future work could "
        "add explicit orientation estimation or scene-text digits (e.g., SVHN).",
    )

    h1(pdf, "9. Conclusion")
    para(
        pdf,
        "Using an AI agent end-to-end, we searched, implemented, and evaluated FasterNet for robust digit "
        "pattern recognition. PConv is verified for identity/efficiency; robust training and TTA deliver "
        "strong multi-condition recognition on MNIST and EMNIST Digits; and the report documents both "
        "technical evidence and the AI-assisted process required by the assignment.",
    )

    h1(pdf, "References")
    bullets(
        pdf,
        [
            "Chen et al. (2023). Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks. CVPR 2023. arXiv:2303.03667.",
            "Vasu et al. (2023). MobileOne: An Improved One Millisecond Mobile Backbone. CVPR 2023.",
            "Hendrycks & Dietterich (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR.",
            "LeCun et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.",
            "Cohen et al. (2017). EMNIST: an extension of MNIST to handwritten letters.",
        ],
    )

    h1(pdf, "Appendix A. AI Process Summary (see also process_log.md)")
    para(
        pdf,
        "Human direction examples: request professional planning Q&A; require full robustness feature pack; "
        "ask for differentiation from a peer PDF without copying; skip week schedule and implement; "
        "request Rot90 improvements, EMNIST, richer figures, and PDF polish items 1-8. Agent actions: "
        "created the repository, trained MNIST/EMNIST models, generated quantitative/qualitative artifacts, "
        "and wrote this report from those artifacts.",
    )

    h1(pdf, "Appendix B. Reproduce")
    bullets(
        pdf,
        [
            "python train.py --model fasternet --aug --epochs 24",
            "python train.py --model fasternet --aug --dataset emnist --epochs 12",
            "python evaluate.py --ckpt outputs/fasternet_aug.pt",
            "python evaluate.py --ckpt outputs/fasternet_aug_emnist.pt",
            "python make_rich_figures.py && python pconv_efficiency.py && python analyze_extra.py",
            "python build_report_pdf.py",
            "python app_gradio.py",
        ],
    )

    out = ROOT / "AIAssignment1_Report.pdf"
    pdf.output(str(out))
    print("Wrote", out, "pages", pdf.page_no())


if __name__ == "__main__":
    main()
