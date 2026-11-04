# Publishing the npm launcher

The `npm` job in `.github/workflows/publish-packages.yml` publishes the `lilbee`
package on every release, after the release binaries are attached. It
authenticates with npm trusted publishing: the registry accepts the job's OIDC
token, so there is no npm token in the repo and nothing to rotate. Provenance
is attested automatically.

## Versions

The npm version is the lilbee version. `scripts/release.sh` and
`scripts/promote_release.sh` set `version` in `package.json` and
`package-lock.json`, and `lilbee.release` in `package.json`, in the same commit
that sets the lilbee version. Nobody edits them by hand.

The launcher runs exactly `lilbee.release`. It downloads that release on first
use and never looks up a newer one.

`scripts/release_version.sh npm-version` maps the lilbee version to semver:

| lilbee version | npm version | `lilbee.release` |
| --- | --- | --- |
| `0.6.90b691` | `0.6.90-beta.691` | `v0.6.90b691` |
| `0.6.90` | `0.6.90` | `v0.6.90` |

The beta number is a separate identifier (`beta.691`), so semver sorts it as a
number: `beta.1000` is newer than `beta.691`. Dev builds (`.devN`) are never
published to npm. A version with no mapping (`a`, `rc`, `post`) stops the release.

The `npm` job checks out the tag and stops when `package.json` does not carry
the npm version of that tag. It publishes with `--tag latest`, because npm 11
refuses a prerelease publish without a dist-tag. If the version is already on
npm, the job skips the publish.

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
  `gh workflow run publish-packages.yml -f tag=<tag> -f channels=npm`.
  It publishes the version the tag carries, and only if that version is not on
  npm yet.
