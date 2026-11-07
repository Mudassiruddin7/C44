import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import os
    import shutil
    import sys
    import time
    import urllib.request
    import zipfile

    import marimo as mo

    # Downloads the C44 code from GitHub (fresh on every run, 3 attempts) and makes it importable.
    _dst = os.path.abspath("C44")
    shutil.rmtree(_dst, ignore_errors=True)
    shutil.rmtree("C44-main", ignore_errors=True)
    _data = None
    for _try in range(3):
        try:
            _data = urllib.request.urlopen(
                f"https://codeload.github.com/Mudassiruddin7/C44/zip/main?t={time.time()}", timeout=60).read()
            break
        except Exception as _e:
            print(f"code download attempt {_try + 1} failed: {_e!r}")
            time.sleep(5)
    assert _data, "could not download the code from GitHub"
    zipfile.ZipFile(io.BytesIO(_data)).extractall(".")
    os.rename("C44-main", _dst)
    for _m in [m for m in sys.modules if m == "c44" or m.startswith("c44.")]:
        del sys.modules[_m]                        # drop stale code already imported in this kernel
    if _dst not in sys.path:
        sys.path.insert(0, _dst)
    from c44 import molab

    return mo, molab, os


@app.cell
def _(mo, molab):
    _text = """
        **Steps 7-8 only** (physics features + CLIP/NPR embeddings, then the final analysis). The data, attacks and
        scores from steps 1-6 are read from your HF repo. Code version: `VERSION_HERE`

        **Hugging Face token** (hf_..., needs write access to the repo) {token}

        **Private HF dataset repo with the earlier work** {repo}
        """.replace("VERSION_HERE", molab.VERSION)
    form = (
        mo.md(_text)
        .batch(
            token=mo.ui.text(kind="password", full_width=True),
            repo=mo.ui.text(value="mudassiruddin21/c44-w2v", full_width=True),
        )
        .form(submit_button_label="Run steps 7 and 8", clear_on_submit=False)
    )
    form
    return (form,)


@app.cell
def _(form, mo, molab, os):
    # ONE cell: light setup (about 2 min) -> pull clips + results from HF (no geometry) -> physics + embeddings -> analysis -> zip
    mo.stop(form.value is None, mo.md("Fill in the token and press **Run steps 7 and 8**."))
    _v = form.value
    mo.stop(not _v["token"].strip(), mo.md("**The token is empty.** Paste it and press the button again."))
    mo.stop(not _v["repo"].strip(), mo.md("**The repo is empty.** It holds the earlier work."))
    os.environ["HF_TOKEN"] = _v["token"].strip()
    os.environ["C44_HF_REPO"] = _v["repo"].strip()
    print("=== 1/4 check ===")
    molab.run("check")
    print("=== 2/4 light setup + pull clips and results from HF (no geometry) ===")
    molab.run("lite")
    print("=== 3/4 physics features + CLIP / NPR embeddings (resumes; progress is pushed to HF every 100 clips) ===")
    molab.run("physics")
    print("=== 4/4 final analysis ===")
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
