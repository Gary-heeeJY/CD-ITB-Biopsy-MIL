# GitHub release checklist

- [ ] Put no WSIs, H5 files, labels, private split files, feature tensors, checkpoints, or downloaded weights in the commit.
- [ ] Keep all local paths relative to `data/`, `weights/`, and `outputs/`.
- [ ] Confirm that the reported seed is `421`.
- [ ] Confirm that CD=`1`, ITB/TB=`0`, and NO=`2`.
- [ ] Confirm that MUSK extraction uses `ms_aug=False`.
- [ ] Confirm that MUSK, UNI2-h, CONCH, and ResNet50 extraction all apply the shared Macenko preprocessing.
- [ ] Confirm that external CORAL is run with `scripts/apply_coral.py` and writes a separate `_CORAL` directory.
- [ ] Run `python scripts/validate_release.py`.
- [ ] Run `python -m compileall -q framework scripts feature_extraction`.
- [ ] Run repository tests in the release environment.
- [ ] Verify that the upstream citations and licenses in `docs/MODEL_SOURCES.md` are retained.
- [ ] Verify `CITATION.cff` and add the final manuscript DOI when available.
- [ ] Add the project license selected by the authors before making the repository public.
