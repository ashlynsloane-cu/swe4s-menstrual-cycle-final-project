"""Check that version-controlled figures use a consistent directory layout."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]



def test_figures_are_grouped_by_aim():
    """Both aims have a discoverable, version-controlled figure directory."""
    assert (ROOT / "figures/README.md").is_file()
    assert (ROOT / "figures/aim1/README.md").is_file()
    assert (ROOT / "figures/aim2/README.md").is_file()
    assert not (ROOT / "docs/figures/aim2").exists()

    for extension in ("png", "pdf", "svg"):
        figure = ROOT / f"figures/aim1/aim1_distance_boxes.{extension}"
        assert figure.is_file()

    for stem in (
        "figure1_near_term_personalization",
        "figure2_longitudinal_mechanisms",
        "figure3_personalization_thresholds",
    ):
        for extension in ("png", "pdf"):
            assert (ROOT / f"figures/aim2/{stem}.{extension}").is_file()


def test_readme_links_to_both_aims():
    """Repository entry point references both sets of exported figures."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "figures/aim1/aim1_distance_boxes.png" in readme
    assert "figures/aim2/figure1_near_term_personalization.png" in readme
    assert "figures/aim2/figure2_longitudinal_mechanisms.png" in readme
    assert "figures/aim2/figure3_personalization_thresholds.png" in readme
    assert "docs/figures/aim2" not in readme
