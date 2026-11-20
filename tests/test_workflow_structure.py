"""Check opt-in targets; Snakemake dry-run is a separate validation."""

from pathlib import Path
import re


def test_default_target_is_aim1_not_aim2():
    snakefile = (
        Path(__file__).resolve().parents[1] / "workflow/Snakefile"
    ).read_text(encoding="utf-8")
    rules = re.findall(r"(?m)^rule (\w+):", snakefile)
    assert rules[0] == "all"
    for name in (
        "aim2", "aim2_primary", "aim2_longitudinal", "aim2_simulation",
        "primary_models", "longitudinal_models", "oracle_recenter",
        "simulation_components", "repeatability", "simulation_depth",
        "simulation_sensitivity", "simulation_variance",
        "simulation_precision", "figure1", "figure2", "figure3",
        "aim1_assumption_audit", "audit_aim1",
    ):
        assert rules.count(name) == 1
    # This is not a DAG validation; execute 'snakemake --dry-run' separately.
    default_rule = snakefile.split("rule all:", 1)[1].split("\nrule ", 1)[0]
    assert "AIM1_OUTPUTS" in default_rule
    assert "FIGURE" not in default_rule
    assert "directory(\"results/aim2/centered_primary\")" in snakefile
    assert "directory(\"results/aim2/longitudinal_folds\")" in snakefile
    assert "PYTENSOR_FLAGS='cxx='" in snakefile
