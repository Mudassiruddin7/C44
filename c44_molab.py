import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import os
    import sys
    import urllib.request
    import zipfile

    import marimo as mo

    # Downloads the C44 code from GitHub (once per session) and makes it importable.
    _dst = os.path.abspath("C44")
    if not os.path.exists(_dst):
        _data = urllib.request.urlopen("https://codeload.github.com/Mudassiruddin7/C44/zip/main").read()
        zipfile.ZipFile(io.BytesIO(_data)).extractall(".")
        os.rename("C44-main", _dst)
    if _dst not in sys.path:
        sys.path.insert(0, _dst)
    from c44 import molab

    return mo, molab


@app.cell
def _(mo):
    mo.md("""
# C44 on molab
Before running: attach the RTX PRO 6000 GPU, and add `HF_TOKEN` in the **Secrets** panel.
Press the buttons in order; wait for each to finish.
""")
    return


@app.cell
def _(mo):
    check_btn = mo.ui.run_button(label="0 · Check (1 min)")
    setup_btn = mo.ui.run_button(label="1 · Setup (20-45 min)")
    pilot_btn = mo.ui.run_button(label="2 · Pilot, 6 scenes (30-45 min)")
    eval_btn = mo.ui.run_button(label="3 · Evaluate (10-15 min)")
    full_btn = mo.ui.run_button(label="4 · Full, 150 scenes (6-9 h)")
    mo.vstack([check_btn, setup_btn, pilot_btn, eval_btn, full_btn])
    return check_btn, eval_btn, full_btn, pilot_btn, setup_btn


@app.cell
def _(check_btn, mo, molab):
    mo.stop(not check_btn.value)
    molab.run("check")
    return


@app.cell
def _(mo, molab, setup_btn):
    mo.stop(not setup_btn.value)
    molab.run("setup")
    return


@app.cell
def _(mo, molab, pilot_btn):
    mo.stop(not pilot_btn.value)
    molab.run("pilot")
    return


@app.cell
def _(eval_btn, mo, molab):
    mo.stop(not eval_btn.value)
    molab.run("evaluate")
    return


@app.cell
def _(full_btn, mo, molab):
    mo.stop(not full_btn.value)
    molab.run("full")
    return


if __name__ == "__main__":
    app.run()
