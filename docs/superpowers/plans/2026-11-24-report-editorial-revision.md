# Final Report Editorial Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the full LaTeX report to meet the requested academic style, document implemented algorithms and bonuses accurately, and keep implementation distinct from experimental results.

**Architecture:** Edit the existing six report section files and preserve the current chapter structure. Use the assignment specification, model source, notebooks, and saved experiment results as the factual sources. Verify LaTeX output and scan narrative prose separately from formulas, tables, and code listings.

**Tech Stack:** LaTeX, Python standard-library audit snippets, `latexmk`.

**Spec:** User request in the current conversation and `extra/Spesifikasi Tugas Besar 2 IF3170 Inteligensi Artifisial 2026-2027.md`.

## Global Constraints

- Preserve exact quantitative results, citations, equations, and model names unless a source check proves a discrepancy.
- Keep foreign technical terms in English and italicize where appropriate. Use “from scratch” in place of “implementasi dari nol.”
- Explanatory prose must contain no em dash, semicolon, or colon. Exempt code, formulas, labels, URLs, and table structure.
- Do not invent results, methods, citations, or unimplemented features.
- Keep implementation and theory in Chapter 3. Keep measured outcomes and their interpretation in Chapter 4.
- Remove the duplicate KNN 2:1 subsection and consolidate that observation in the existing cost-sensitive results discussion.

## Review Focus

- Equation and code punctuation must not be mistaken for narrative punctuation. Verify with prose-aware extraction and a manual check.
- Bonus completion claims must match the repository and assignment text, especially the Logistic Regression animation.
- Solver descriptions must match the actual solver used by each final model, not visualization-only or legacy alternatives.
- Keep existing report title and section naming conventions intact.
- Compilation must not leave unresolved references or new LaTeX errors.

---

### Task 1: Audit and revise report prose

**Files:**
- Modify: `docs/laporan-akhir/sections/01-pendahuluan.tex`
- Modify: `docs/laporan-akhir/sections/02-landasan-teori.tex`
- Modify: `docs/laporan-akhir/sections/03-implementasi-algoritma.tex`
- Modify: `docs/laporan-akhir/sections/04-hasil-eksperimen-dan-analisis.tex`
- Modify: `docs/laporan-akhir/sections/05-kesimpulan-dan-saran.tex`
- Modify: `docs/laporan-akhir/sections/06-lampiran.tex`

- [ ] Scan all narrative for em dashes, semicolons, colons, stock transitions, contrast templates, and literal “implementasi dari nol.”
- [ ] Revise flagged prose without changing numeric claims, technical English terms, citations, or report hierarchy.
- [ ] Review neighboring paragraphs for repeated sentence rhythm and unsupported causal transitions.

### Task 2: Explain optimization implementations

**Files:**
- Modify: `docs/laporan-akhir/sections/03-implementasi-algoritma.tex`

- [ ] Expand DTL cost-complexity pruning, SVM Modified Finite Newton and Nyström, Logistic Regression training method, and Naive Bayes KDB plus Adam weighting with concise theory and code snippets.
- [ ] Cite only sources already present in `dafpus.bib`.
- [ ] Keep the implementation discussion free of experimental outcome claims.

### Task 3: Separate optimization and cost results

**Files:**
- Modify: `docs/laporan-akhir/sections/04-hasil-eksperimen-dan-analisis.tex`

- [ ] Reorganize optimization discussion so Chapter 4 contains measured outcomes and evidence-based analysis only.
- [ ] Keep cost-sensitive method and decision rule in Chapter 3. Keep cost comparison tables and outcomes in Chapter 4.
- [ ] Remove the duplicate KNN 2:1 subsection and add its distinct result context to the shared cost analysis without conflating protocols.

### Task 4: Complete bonus explanations and checklist

**Files:**
- Modify: `docs/laporan-akhir/sections/03-implementasi-algoritma.tex`
- Modify: `docs/laporan-akhir/sections/04-hasil-eksperimen-dan-analisis.tex`

- [ ] Add concise SVM and Bayesian Network code snippets explaining how their visualizations are produced.
- [ ] Add a bonus-specification table based on the assignment with a checkbox for repository implementation status.
- [ ] Verify Logistic Regression animation status against its final notebook and distinguish any repository gap from uncertain recollection.

### Task 5: Qualitative conclusion and final verification

**Files:**
- Modify: `docs/laporan-akhir/sections/05-kesimpulan-dan-saran.tex`

- [ ] Replace numerical conclusion and recommendations with contextual qualitative statements, retaining separate sections.
- [ ] Compile the report with `latexmk` and verify no unresolved references or fatal errors.
- [ ] Run `git diff --check` and repeat the prose-only punctuation and pattern scan.
