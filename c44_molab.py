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
    import shutil
    import time
    _dst = os.path.abspath("C44")
    shutil.rmtree(_dst, ignore_errors=True)       # always fetch the latest code
    shutil.rmtree("C44-main", ignore_errors=True)
    _data = urllib.request.urlopen(f"https://codeload.github.com/Mudassiruddin7/C44/zip/main?t={time.time()}").read()
    zipfile.ZipFile(io.BytesIO(_data)).extractall(".")
    os.rename("C44-main", _dst)
    for _m in [m for m in sys.modules if m == "c44" or m.startswith("c44.")]:
        del sys.modules[_m]                        # drop stale code already imported in this kernel
    if _dst not in sys.path:
        sys.path.insert(0, _dst)
    from c44 import molab

    return mo, molab, os


@app.cell
def _(mo):
    form = (
        mo.md(
            """
            **Hugging Face token** (hf_...) {token}

            **Scenes** (6 = pilot, 150 = full run) {n}

            **Optional: private HF dataset repo to keep results if molab restarts** (`yourname/c44-w2v`, the token
            needs write access; leave empty to just download the zip at the end) {repo}
            """
        )
        .batch(
            token=mo.ui.text(kind="password", full_width=True),
            n=mo.ui.number(1, 400, value=6, step=1),
            repo=mo.ui.text(full_width=True),
        )
        .form(submit_button_label="Start the whole pipeline", clear_on_submit=False)
    )
    form
    return (form,)


@app.cell
def _(form, mo, molab, os):
    # ONE cell runs everything after the button:
    # check -> setup -> smoke test -> build -> evaluate -> robustness -> resolution control -> final analysis -> zip
    mo.stop(form.value is None, mo.md("Fill in the token and press **Start the whole pipeline**."))
    _v = form.value
    mo.stop(not _v["token"].strip(), mo.md("**The token is empty.** Paste it and press Start again."))
    os.environ["HF_TOKEN"] = _v["token"].strip()
    if _v["repo"].strip():
        os.environ["C44_HF_REPO"] = _v["repo"].strip()
    _n = int(_v["n"])
    print("=== 1/7 check ===")
    molab.run("check")
    print("=== 2/7 setup (about 10 min) ===")
    molab.run("setup")
    print(f"=== 3/7 smoke test + build {_n} scenes (resumes from the HF repo if there is one) ===")
    clips = molab.run("pilot", n_scenes=_n)
    print(clips.groupby(["cell", "status"]).size().to_string())
    print("=== 4/7 evaluate ===")
    print(molab.run("evaluate").to_string(index=False))
    print("=== 5/7 robustness: recompression + calibrated blur/noise attacks ===")
    molab.run("robust", n_scenes=_n)
    print("=== 6/7 resolution control: supersampled renders ===")
    molab.run("control", n_scenes=_n)
    print("=== 7/7 final analysis ===")
    zip_path = molab.run("analyse")
    print("ALL DONE - use the download button under this cell")
    return (zip_path,)


@app.cell
def _(mo, zip_path):
    mo.stop(zip_path is None)
    _data = open(zip_path, "rb").read()
    mo.vstack([
        mo.md("### Done. Download the results (tables, analysis, scores, features)"),
        mo.download(data=_data, filename="c44_results.zip", mimetype="application/zip", label="Download c44_results.zip"),
    ])
    return


if __name__ == "__main__":
    app.run()
