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
    # Runs everything automatically after the button: check -> setup -> smoke test -> build -> evaluate -> robustness.
    mo.stop(form.value is None, mo.md("Fill in the token and press **Start the whole pipeline**."))
    _v = form.value
    mo.stop(not _v["token"].strip(), mo.md("**The token is empty.** Paste it and press Start again."))
    os.environ["HF_TOKEN"] = _v["token"].strip()
    if _v["repo"].strip():
        os.environ["C44_HF_REPO"] = _v["repo"].strip()
    _n = int(_v["n"])
    print("=== 1/5 check ===")
    molab.run("check")
    print("=== 2/5 setup (about 10 min) ===")
    molab.run("setup")
    print(f"=== 3/5 smoke test + build {_n} scenes ===")
    clips = molab.run("pilot", n_scenes=_n)
    print(clips.to_string(index=False))
    print("=== 4/5 evaluate ===")
    results = molab.run("evaluate")
    print(results.to_string(index=False))
    print("=== 5/5 robustness (recompression + camera-simulating attacks) ===")
    robust = molab.run("robust")
    print("ALL DONE - download c44_results.zip from the file browser")
    robust
    return


if __name__ == "__main__":
    app.run()
