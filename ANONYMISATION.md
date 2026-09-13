# Anonymous review build

This repository is a review-only snapshot prepared for a double-blind
submission. It is intentionally detached from the development repository and
must not be merged back as a source of release history.

## Removed from this snapshot

- all original Git objects, branches, tags, remotes, commit authors, and dates;
- author names, affiliations, direct contact details, and personal accounts;
- the public preprint link and the manuscript PDF;
- checkpoint and dataset URLs whose namespaces reveal contributor identities;
- promotional video material containing author credits;
- optional robot assets and license files whose vendor name reveals an author
  affiliation;
- ignored checkpoints, MANO files, local datasets, logs, caches, and machine
  paths.

The optional robot-retargeting integration is retained under neutral internal
names so that the surrounding Viewer interface remains inspectable, but its
vendor-specific runtime and assets are not included during review.

## Preserved on purpose

Third-party source trees retain their upstream copyright notices, licenses,
names, and citations. These identify independent upstream projects, not the
submission authors, and removing them would violate attribution requirements.
Public benchmark and dataset names are also retained where technically needed.

## Checks before publication

Run the following from the repository root:

```bash
python scripts/privacy_audit.py --strict
git log --format='%an <%ae>'
git remote -v
```

The release should contain one anonymous root commit. Its only remote should be
the anonymous review repository. Binary samples and figures require visual
review in addition to automated text scans.
