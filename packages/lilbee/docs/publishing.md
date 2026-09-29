# Publishing the npm launcher

The `npm` job in `.github/workflows/publish-packages.yml` publishes the `lilbee`
package on every release, after the release binaries are attached. It
authenticates with npm trusted publishing: the registry accepts the job's OIDC
token, so there is no npm token in the repo and nothing to rotate. Provenance
is attested automatically.

## Versions

CI publishes `packages/lilbee/package.json` exactly as committed. The launcher
downloads the latest GitHub release at run time, so npm needs a new version
only when the launcher itself changes.

To ship a new npm version, bump `version` in a PR. In the same PR, move
`lilbee.release` to a recent tested release. The launcher uses that release
when it cannot look up the latest one. If the version is already on npm, the
job skips the publish.

## One-time setup

1. Create an npmjs.com account, or log into the existing one. Turn on 2FA.

2. The `lilbee` package already exists on npm, so trusted publishing can be
   configured on it. A manual publish needs `npm login` first:

   ```bash
   npm login
   cd packages/lilbee && npm publish --access public
   ```

3. On npmjs.com, open the `lilbee` package → Settings → Trusted Publisher →
   GitHub Actions. Fill in:

   - Organization or user: `tobocop2`
   - Repository: `lilbee`
   - Workflow filename: `publish-packages.yml`
   - Environment: `npm`

   The workflow filename and environment name must match exactly — the job is
   pinned to `environment: npm` the same way the PyPI job is pinned to
   `pypi`. Renaming either side breaks the publish until they agree.

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
- To run the job by hand, dispatch the workflow with only this channel:
  `gh workflow run publish-packages.yml -f tag=v0.6.90b423 -f channels=npm`.
  It publishes the committed version, and only if that version is not on npm
  yet.
