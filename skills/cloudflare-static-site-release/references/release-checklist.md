# Release evidence checklist

Record these items for every release:

- Source revision and generated-output hash
- Test, crawler, syntax, and dry-run results
- Staging URL and verifier result
- Custom-domain DNS, certificate, origin, Worker, and Pages inventory
- Approved migration and rollback owner
- Previous known-good version ID
- Production deployment version ID and URL
- Production verifier result
- Desktop/mobile, light/dark, and locale browser QA
- Rollback result and post-rollback verifier result, when used

Wrangler command shape:

```sh
WRANGLER_LOG_PATH=/tmp/site-wrangler.log bun wrangler <command> --config /absolute/path/to/wrangler.jsonc
```

Use `--env staging` for staging and `--env=""` for the top-level production environment when the config defines named environments.
