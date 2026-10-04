#!/usr/bin/env python3
"""Génère un carrousel PDF LinkedIn (format document) — projet Black–Scholes."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "01-black-scholes" / "report" / "figures"
OUT = Path(__file__).resolve().parent / "Black-Scholes_LinkedIn.pdf"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Format proche d'un carrousel LinkedIn (portrait)
W, H = 190, 240  # mm


class Slide(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format=(W, H))
        self.set_auto_page_break(auto=False)
        self.add_font("DejaVu", "", FONT)
        self.add_font("DejaVu", "B", FONT_B)
        self.set_margins(14, 14, 14)

    def bg(self, color=(248, 246, 242)) -> None:
        self.set_fill_color(*color)
        self.rect(0, 0, W, H, "F")

    def accent_bar(self, color=(20, 70, 90)) -> None:
        self.set_fill_color(*color)
        self.rect(0, 0, 6, H, "F")

    def footer_brand(self) -> None:
        self.set_font("DejaVu", "", 8)
        self.set_text_color(110, 110, 110)
        self.set_xy(14, H - 12)
        self.cell(0, 6, "Portfolio maths appliquées L3/M1  ·  github.com/hamelayen-create/maths-appliquees-l3-m1")


def title_slide(pdf: Slide) -> None:
    pdf.add_page()
    pdf.bg((18, 55, 72))
    pdf.set_text_color(245, 240, 230)
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_xy(16, 36)
    pdf.cell(0, 8, "PROJET MATHS APPLIQUÉES  ·  FINANCE QUANTITATIVE")
    pdf.set_font("DejaVu", "B", 28)
    pdf.set_xy(16, 58)
    pdf.multi_cell(W - 32, 12, "Black–Scholes\nformule fermée, EDP\net Monte Carlo")
    pdf.set_font("DejaVu", "", 12)
    pdf.set_xy(16, 118)
    pdf.multi_cell(
        W - 32,
        7,
        "Un même prix d'option calculé de 3 façons indépendantes\n"
        "pour montrer que la théorie, le numérique et la simulation\n"
        "se rejoignent vraiment.",
    )
    pdf.set_font("DejaVu", "B", 11)
    pdf.set_xy(16, 160)
    pdf.set_text_color(200, 220, 210)
    pdf.multi_cell(W - 32, 7, "Niveau L3 / M1  ·  Python · NumPy · SciPy\nRepo open source sur GitHub")
    pdf.set_text_color(160, 180, 175)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_xy(16, H - 14)
    pdf.cell(0, 6, "Document LinkedIn  ·  Rayane Hamel")


def bullets_slide(pdf: Slide, title: str, lines: list[str], fig: Path | None = None) -> None:
    pdf.add_page()
    pdf.bg()
    pdf.accent_bar()
    pdf.set_text_color(20, 55, 70)
    pdf.set_font("DejaVu", "B", 18)
    pdf.set_xy(16, 18)
    pdf.multi_cell(W - 28, 9, title)
    pdf.set_draw_color(20, 70, 90)
    pdf.set_line_width(0.4)
    pdf.line(16, 38, W - 16, 38)

    y = 46
    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(35, 35, 35)
    for line in lines:
        pdf.set_xy(18, y)
        pdf.multi_cell(W - 36 if fig is None else W - 36, 6.2, f"•  {line}")
        y = pdf.get_y() + 3

    if fig is not None and fig.exists():
        # place figure in lower half
        max_w = W - 32
        max_h = H - y - 28
        pdf.image(str(fig), x=16, y=min(y + 4, H - 90), w=max_w, h=min(max_h, 78))
    pdf.footer_brand()


def results_slide(pdf: Slide) -> None:
    pdf.add_page()
    pdf.bg()
    pdf.accent_bar()
    pdf.set_text_color(20, 55, 70)
    pdf.set_font("DejaVu", "B", 18)
    pdf.set_xy(16, 18)
    pdf.cell(0, 10, "Résultat clé : 3 méthodes, 1 prix")

    rows = [
        ("Formule fermée", "10.4506", "référence exacte"),
        ("EDP Crank–Nicolson", "10.4530", "erreur ~ 0.002"),
        ("Monte Carlo (N=2e5)", "10.476 ± 0.046", "IC 95 %"),
    ]
    y = 48
    for name, val, note in rows:
        pdf.set_fill_color(232, 240, 242)
        pdf.rect(16, y, W - 32, 28, "F")
        pdf.set_xy(20, y + 4)
        pdf.set_font("DejaVu", "B", 12)
        pdf.set_text_color(20, 55, 70)
        pdf.cell(0, 7, name)
        pdf.set_xy(20, y + 13)
        pdf.set_font("DejaVu", "B", 16)
        pdf.set_text_color(15, 90, 80)
        pdf.cell(70, 8, val)
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 8, note)
        y += 34

    pdf.set_xy(16, y + 4)
    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(
        W - 32,
        6,
        "Ce n'est pas trois codes séparés : c'est une preuve numérique que "
        "l'EDP, l'espérance risque-neutre et la formule fermée décrivent le même objet.",
    )
    pdf.footer_brand()


def cta_slide(pdf: Slide) -> None:
    pdf.add_page()
    pdf.bg((18, 55, 72))
    pdf.set_text_color(245, 240, 230)
    pdf.set_font("DejaVu", "B", 22)
    pdf.set_xy(16, 40)
    pdf.multi_cell(W - 32, 10, "Ce que tu trouveras\ndans le repo")
    pdf.set_font("DejaVu", "", 12)
    pdf.set_xy(16, 80)
    items = [
        "Rapport pédagogique ~15 pages (théorie + expériences)",
        "Code Python testé (formule, EDP, Monte Carlo)",
        "Figures de convergence et Grecs",
        "8 autres projets : Markowitz, SVI, EDP, Schrödinger, PageRank, ML, crypto…",
    ]
    for it in items:
        pdf.set_x(16)
        pdf.multi_cell(W - 32, 8, f"→  {it}")
        pdf.ln(2)

    pdf.set_font("DejaVu", "B", 12)
    pdf.set_xy(16, 175)
    pdf.set_text_color(180, 220, 200)
    pdf.multi_cell(
        W - 32,
        7,
        "Lien dans le premier commentaire\n"
        "github.com/hamelayen-create/maths-appliquees-l3-m1",
    )
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(160, 180, 175)
    pdf.set_xy(16, H - 20)
    pdf.cell(0, 6, "Like / commente / partage si ça t'est utile — feedback bienvenu.")


def main() -> None:
    pdf = Slide()
    title_slide(pdf)
    bullets_slide(
        pdf,
        "Le problème en une slide",
        [
            "Prix d'un call européen sans arbitrage.",
            "Hypothèse : sous-jacent en mouvement brownien géométrique.",
            "Résultat théorique : EDP de Black–Scholes + formule fermée.",
            "En pratique : on doit aussi savoir calculer numériquement.",
        ],
        FIG / "07_model_diagram.png",
    )
    bullets_slide(
        pdf,
        "Ce que j'ai construit",
        [
            "Formule fermée call/put + Grecs (Δ, Γ, Vega).",
            "EDP résolue en Crank–Nicolson (grille log-spot).",
            "Monte Carlo avec variables antithétiques.",
            "Comparaison d'erreurs, convergence et parité call–put.",
        ],
        FIG / "10_methods_comparison.png",
    )
    results_slide(pdf)
    bullets_slide(
        pdf,
        "Ce que ça démontre",
        [
            "Relier une EDS, une EDP et une espérance Monte Carlo.",
            "Lire une erreur numérique (biais vs variance).",
            "Écrire du code de pricing testé, pas seulement des formules.",
            "Documenter proprement pour un portfolio GitHub.",
        ],
        FIG / "04_mc_convergence.png",
    )
    cta_slide(pdf)
    pdf.output(OUT)
    print(f"OK  {OUT}")


if __name__ == "__main__":
    main()
