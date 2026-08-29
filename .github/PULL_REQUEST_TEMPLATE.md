## What this changes

<!-- One or two sentences. If it fixes something that would have been invisible in the
output, say so explicitly. -->

## Does it touch a reported number?

- [ ] No — documentation, tooling, or tests only
- [ ] Yes — the pipeline was rerun and the documents rebuilt

If yes, which profile produced the committed results? `results/MANIFEST.json` must say
`FULL` for anything the paper is built from.

## Protocol

- [ ] `docs/PROTOCOL.md` is unchanged
- [ ] `docs/PROTOCOL.md` §10 has a new amendment row with a date and a reason

## Checks

- [ ] `make test` passes
- [ ] `make docs` builds all three documents
- [ ] `./verify.sh` reports ALL CHECKS PASSED
- [ ] No number was typed by hand into any `.tex` file
