# Publishing the npm launcher

The `npm` job in `.github/workflows/publish-packages.yml` publishes the `lilbee`
package on every release, after the release binaries are attached. It
authenticates with npm trusted publishing: the registry accepts the job's OIDC
token, so there is no npm token in the repo and nothing to rotate. Provenance
is attested automatically.

## Versions

Release tags carry a build suffix that npm rejects, so the tag maps to a semver
prerelease: `v0.6.90b423` becomes `0.6.90-b423`. The full tag also goes into the
`lilbee.release` field, which is how the launcher finds the release assets.
`node packages/lilbee/tools/stamp-release.mjs <tag>` writes both fields and
prints the npm version. CI runs it; nothing is committed back.

## One-time setup

1. Create an npmjs.com account, or log into the existing one. Turn on 2FA.

2. Claim the name with one manual publish. Trusted publishing is configured on
   a package that already exists, so this has to come first. From this branch,
   with the version still at `0.6.90`:

   ```bash
   npm login
   cd packages/lilbee && npm publish --access public
   ```

3. On npmjs.com, open the `lilbee` package → Settings → Trusted Publisher →
   GitHub Actions. Fill in:

   - Organization or user: `tobocop2`
   - Repository: `lilbee`
   - Workflow filename: `publish-packages.yml`
   - Environment: leave empty

   The workflow filename must match exactly. Renaming or moving the publish job
   to another file breaks the publish until this is updated.

4. Turn the job on:

   ```bash
   gh variable set NPM_TRUSTED_PUBLISHING -R tobocop2/lilbee --body enabled
   ```

That is all of it. No tokens, no expiry.

Until the variable is set, the `npm` job runs the tests, logs why it is not
publishing, and passes. Releases do not break.

## Notes

- Trusted publishing needs npm 11.5.1 or newer. The job installs `npm@latest`
  and prints the version.
- OIDC needs `id-token: write` on the job. The `npm` job sets it.
- CI publishes `0.6.90-b423`-style prereleases. npm still moves the `latest`
  tag to them, so `npm install lilbee` gets the newest release, but a `^0.6.90`
  range does not match a prerelease.
- To republish a tag by hand, dispatch the workflow with only this channel:
  `gh workflow run publish-packages.yml -f tag=v0.6.90b423 -f channels=npm`.
