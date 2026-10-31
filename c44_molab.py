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

    N_SCENES = 6  # pilot. Set 150 for the full benchmark (6-9 h).

    # Downloads the C44 code from GitHub (once per session) and makes it importable.
    _dst = os.path.abspath("C44")
    if not os.path.exists(_dst):
        _data = urllib.request.urlopen("https://codeload.github.com/Mudassiruddin7/C44/zip/main").read()
        zipfile.ZipFile(io.BytesIO(_data)).extractall(".")
        os.rename("C44-main", _dst)
    if _dst not in sys.path:
        sys.path.insert(0, _dst)
    from c44 import molab

    return N_SCENES, mo, molab, os


@app.cell
def _(mo):
    token_box = mo.ui.text(kind="password", label="Paste your Hugging Face token (hf_...) and press Enter",
                           full_width=True)
    token_box
    return (token_box,)


@app.cell
def _(N_SCENES, mo, molab, os, token_box):
    # Runs the whole pipeline automatically as soon as a token is present: check -> setup -> build -> evaluate.
    mo.stop(not token_box.value.strip(), mo.md("Waiting for the token above. Everything runs automatically after."))
    os.environ["HF_TOKEN"] = token_box.value.strip()
    print("=== 1/4 check ===")
    molab.run("check")
    print("=== 2/4 setup (20-45 min) ===")
    molab.run("setup")
    print(f"=== 3/4 build {N_SCENES} scenes ===")
    clips = molab.run("pilot", n_scenes=N_SCENES)
    print(clips.to_string(index=False))
    print("=== 4/4 evaluate ===")
    results = molab.run("evaluate")
    print("ALL DONE - download c44_results.zip from the file browser")
    results
    return


if __name__ == "__main__":
    app.run()
