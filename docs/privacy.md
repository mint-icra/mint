# Privacy and release checklist

Egocentric video can expose faces, screens, documents, reflections, voices,
locations, device identifiers, and bystanders. A public-dataset label does not
remove these risks.

## Source release checklist

- Run `python scripts/privacy_audit.py --strict`.
- Review Git history, not only the current tree, for deleted credentials.
- Rotate any credential that has ever entered a tracked file.
- Remove personal paths, cloud resource IDs, private registries, and hostnames.
- Keep weights, MANO files, logs, caches, and experiment trackers ignored.
- Confirm third-party notices and pin reviewed revisions.

## Sample release workflow

1. Confirm in writing that the dataset license permits redistribution of the
   exact clip.
2. Create a review manifest with `redistribution_approved: true` and
   `privacy_reviewed: true` for each input.
3. Add normalized static blur boxes for screens, documents, labels, and other
   persistent regions.
4. Run `scripts/prepare_samples.py`; it removes audio and metadata, limits
   duration and resolution, normalizes names, and applies the masks.
5. Review every frame of the transformed output at normal speed and by
   scrubbing. Automated blur is not proof of anonymization.
6. Set `post_transform_reviewed` to true only after the final review and obtain
   a second reviewer for sensitive scenes.

## Review manifest example

```json
{
  "clips": [
    {
      "file": "licensed/source/clip.mp4",
      "dataset": "epic-kitchens",
      "redistribution_approved": true,
      "privacy_reviewed": true,
      "normalized_masks": [[0.72, 0.02, 0.25, 0.16]]
    }
  ]
}
```

Each mask is `[x, y, width, height]` in normalized image coordinates. Prefer a
larger mask when motion, rolling shutter, or cropping can reveal the boundary.

## Incident response

If private material is published, disable distribution, preserve an internal
incident record, rotate exposed credentials, purge the material from repository
history and release assets, and notify affected parties according to applicable
policy and law.

