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
    code_ok, code_msg = _data is not None, "could not download the code from GitHub (3 attempts): check molab's internet access and press Run again."
    molab = None
    if _data is not None:
        zipfile.ZipFile(io.BytesIO(_data)).extractall(".")
        os.rename("C44-main", _dst)
        for _m in [m for m in sys.modules if m == "c44" or m.startswith("c44.")]:
            del sys.modules[_m]                        # drop stale code already imported in this kernel
        if _dst not in sys.path:
            sys.path.insert(0, _dst)
        from c44 import molab
        _have = str(getattr(molab, "VERSION", "unknown"))
        code_ok = _have.startswith("v18") and hasattr(molab, "LAST_UPLOAD")
        code_msg = (f"The code on GitHub is version `{_have}`, which does not contain the weakness stages yet (they are in version v18). "
                    "Push the latest code to github.com/Mudassiruddin7/C44 (branch main) and run this cell again. Nothing was started.")
    return code_msg, code_ok, mo, molab, os


@app.cell
def _(code_msg, code_ok, mo, molab):
    mo.stop(not code_ok, mo.md(f"**Cannot start.** {code_msg}"))
    _text = """
        # C44 weakness notebook

        Attacks the weak side of the paper, **CPU work on the clips you already have on Hugging Face** (only the CLIP / NPR
        embeddings of the new clips use the GPU). Code version: `VERSION_HERE`

        1. **Operator families** the renderer lacks (temporal noise reduction, chroma / luma denoise, sharpening, rolling
           shutter, lens optics), applied to real **and** fake clips: the material for a leave-one-family-out test of the defences.
        2. **Attacker ladder** L1 (the paper's attacker) -> L2 (+ lens, rolling shutter) -> L3 (+ ISP after the noise): detectability
           versus attacker effort.
        3. **Joint cues** (CPS-3) that couple blur, noise and motion inside the frame.
        4. Paired statistics (BCa scene bootstrap, Holm), tables, a figure, one zip.

        The work runs as a **detached background job**: refreshing this page does not stop it - press the button again and it re-attaches.
        Everything is resumable: clips and features are pushed to your HF repo every 25-100 items, so even a recycled container costs little. Expect about 3-4 hours for 150 scenes (6 scenes = a 20-minute smoke test).
        The results zip is uploaded to the repo as `weak/c44_weak_results_latest.zip` (and a time-stamped copy) and is also offered below.

        **Hugging Face token** (hf_..., write access to the repo) {token}

        **Private HF dataset repo with the main run and where results go** {repo}

        **Scenes** (150 = full, 6 = smoke test) {n}
        """.replace("VERSION_HERE", molab.VERSION)
    form = (
        mo.md(_text)
        .batch(
            token=mo.ui.text(kind="password", full_width=True),
            repo=mo.ui.text(value="mudassiruddin21/c44-w2v", full_width=True),
            n=mo.ui.number(1, 400, value=150, step=1),
        )
        .form(submit_button_label="Run the weakness experiments", clear_on_submit=False)
    )
    form
    return (form,)


@app.cell
def _(code_ok, form, mo, molab, os):
    # ONE cell runs everything after the button:
    # check -> light setup + pull -> operator families + ladder -> features -> embeddings -> analysis -> zip -> HF upload
    mo.stop(not code_ok, mo.md("Cannot start: see the message above."))
    mo.stop(form.value is None, mo.md("Fill in the token and press **Run the weakness experiments**."))
    _v = form.value
    mo.stop(not _v["token"].strip(), mo.md("**The token is empty.** Paste it and press the button again."))
    mo.stop(not _v["repo"].strip(), mo.md("**The repo is empty.** It holds the main run and receives the results."))
    os.environ["HF_TOKEN"] = _v["token"].strip()
    os.environ["C44_HF_REPO"] = _v["repo"].strip()
    _n = int(_v["n"])
    failures = []

    def _step(label, stage, fatal=True, **kw):
        print(f"\n=== {label} ===", flush=True)
        try:
            return molab.run(stage, **kw)
        except Exception as e:                                    # noqa: BLE001 - report, keep the work already pushed to HF
            msg = f"{label} FAILED: {str(e)[-1800:]}"
            print("!!! " + msg, flush=True)
            failures.append(msg)
            if fatal:
                raise
            return None

    zip_path = None
    try:
        _step("1/3 check", "check", fatal=False)
        if molab.run("weak_job_running"):
            print("\nA weakness job is already running in this session (the notebook was refreshed). Re-attaching to it; nothing is restarted.")
        else:
            _step("2/3 light setup + pull the clips and results you already have from HF (resumes whatever was pushed before)", "weak_lite")
        zip_path = _step(f"3/3 detached job for {_n} scenes: operator-family clips + attacker ladder -> features -> embeddings -> analysis -> zip -> HF "
                         "(survives a notebook refresh; progress is pushed to HF every 25-100 items)", "weak_job", n_scenes=_n)
        print("\nALL DONE")
    except Exception:
        print("\nSTOPPED. Everything finished so far is on your HF repo; press the button again to resume (or to re-attach if the job is still alive).")
    return failures, zip_path


@app.cell
def _(failures, mo, molab, zip_path):
    mo.stop(zip_path is None and not failures, mo.md(""))
    _items = []
    if failures:
        _items.append(mo.md("### Problems\n\n" + "\n\n".join(f"```\n{f}\n```" for f in failures)))
    if zip_path is not None:
        from pathlib import Path
        _summary = Path(molab.config.work_dir()) / "results" / "weak" / "summary.md"
        _txt = _summary.read_text(encoding="utf-8") if _summary.exists() else "(no summary written)"
        _items.append(mo.md("### Hugging Face upload\n\n" + "\n\n".join(f"- {m}" for m in (molab.LAST_UPLOAD or ["no upload was attempted"]))))
        _items.append(mo.md("### Done. Download the results"))
        _items.append(mo.download(data=open(zip_path, "rb").read(), filename="c44_weak_results.zip", mimetype="application/zip",
                                  label="Download c44_weak_results.zip"))
        _items.append(mo.accordion({"Summary of the findings (summary.md)": mo.md(_txt[:20000])}))
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
