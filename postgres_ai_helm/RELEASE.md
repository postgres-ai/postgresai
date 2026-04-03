# Helm chart release process

This document describes how to create and publish releases of the PostgresAI monitoring helm chart.

## Automated release workflow

The helm chart uses an automated release process triggered by git tags. When you push a tag matching the pattern `helm-v*.*.*`, a GitLab CI/CD pipeline automatically:

1. Extracts the version from the tag
2. Updates `Chart.yaml` with the new version
3. Updates helm dependencies
4. Lints the chart
5. Packages the chart into a `.tgz` file
6. Creates a GitLab release with release notes
7. Attaches the packaged chart to the release

## Creating a release

### Method 1: Using the release script (recommended)

The easiest way to create a release is using the provided script:

```bash
cd postgres_ai_helm
./release.sh <VERSION>   # e.g.: ./release.sh 0.14.0
```

The script will:
- Validate the version format
- Update `Chart.yaml`
- Commit the changes
- Create and push a git tag
- Trigger the automated release pipeline

### Method 2: Manual release

If you prefer to create the release manually:

```bash
# 1. Update Chart.yaml
cd postgres_ai_helm
# Edit Chart.yaml and update the version field

# 2. Commit the change
git add Chart.yaml
git commit -m "chore(helm): bump chart version to 0.14.0"

# 3. Create and push the tag
git tag -a helm-v0.14.0 -m "Helm chart release 0.14.0"
git push origin HEAD
git push origin helm-v0.14.0
```

## Version numbering

Follow semantic versioning for helm chart releases:

- **Major version** (X.0.0): Breaking changes, incompatible API changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, backward compatible

Examples:
- `0.14.0`: New feature release
- `0.14.1`: Bug fix release
- `1.0.0`: First stable release

## Tag naming conventions

The helm chart release pipeline uses the `helm-v*.*.*` tag pattern (e.g. `helm-v0.14.0`).

**Why `helm-v` instead of `v`?**

The org standard is `vX.Y.Z`, but the `helm-v` prefix is an explicit exception: it prevents the
`cli:npm:publish` and `docker:publish:images` CI jobs from firing on helm chart tags. Those jobs
trigger on any `$CI_COMMIT_TAG`, so a bare `v0.14.0` tag would also attempt an npm/Docker publish
with the chart version, which is incorrect.

This exception is documented in `.gitlab-ci.yml` above the `release-helm-chart` job.

## Monitoring the release

After pushing a tag:

1. Go to the GitLab CI/CD pipelines page
2. Find the pipeline for your tag
3. Monitor the progress and check for any errors
4. Once complete, verify the release at the releases page

GitLab URLs:
- **Pipelines**: https://gitlab.com/postgres-ai/postgresai/-/pipelines
- **Releases**: https://gitlab.com/postgres-ai/postgresai/-/releases

## Release artifacts

Each release includes:

- **Packaged chart**: `postgres-ai-monitoring-chart.tgz`
- **Release notes**: Automatically generated with installation instructions
- **Source code**: Snapshot of the repository at the tag

## Using the released chart

Users can install the chart from the GitLab release:

```bash
# Download the chart package (replace <VERSION> with the desired release, e.g. 0.14.0)
curl -LO https://gitlab.com/postgres-ai/postgresai/-/releases/helm-v<VERSION>/downloads/postgres-ai-monitoring-chart.tgz

# Install
helm install postgres-ai-monitoring postgres-ai-monitoring-chart.tgz

# Or with custom values
helm install postgres-ai-monitoring postgres-ai-monitoring-chart.tgz -f custom-values.yaml
```

## GitLab CI/CD configuration

The release automation is defined in `.gitlab-ci.yml` at the repository root.

The release pipeline uses GitLab's native `release:` keyword with `CI_JOB_TOKEN`, so no additional tokens or variables are required.

## Setting up a helm repository (optional)

For easier distribution, you can set up a helm repository using GitLab Pages:

1. Create a `.gitlab-ci.yml` job to publish to pages
2. Users can then add your helm repository:
   ```bash
   helm repo add postgres-ai https://postgres-ai.gitlab.io/postgresai
   helm repo update
   helm install postgres-ai-monitoring postgres-ai/postgres-ai-monitoring
   ```

Example GitLab Pages job:

```yaml
pages:
  stage: deploy
  script:
    - mkdir -p public/charts
    - cp postgres_ai_helm/*.tgz public/charts/ || true
    - helm repo index public/charts --url https://postgres-ai.gitlab.io/postgresai/charts
  artifacts:
    paths:
      - public
  only:
    - tags
```

## Troubleshooting

### Release pipeline fails

1. Check the GitLab CI/CD logs for specific errors
2. Common issues:
   - Linting errors: Run `helm lint .` locally to check
   - Dependency issues: Run `helm dependency update` locally

### Tag already exists

If you need to recreate a release:

```bash
# Delete the tag locally and remotely
git tag -d helm-v0.14.0
git push origin :refs/tags/helm-v0.14.0

# Delete the GitLab release manually via the web UI
# Then create the tag again
git tag -a helm-v0.14.0 -m "Helm chart release 0.14.0"
git push origin helm-v0.14.0
```

### Version mismatch

Ensure the version in `Chart.yaml` matches your intended release version before creating the tag. The automated pipeline will override it, but it's better to keep them in sync.

## Best practices

1. **Test locally first**: Always test the chart locally before releasing
   ```bash
   helm dependency update
   helm lint .
   helm install test-release . --dry-run --debug
   ```

2. **Run unit and integration tests**: Verify release logic before creating a release
   ```bash
   ./test-release-logic.sh
   ./test-release.sh
   ```

3. **Update documentation**: Update `INSTALLATION_GUIDE.md` and other docs as needed

4. **Write meaningful commit messages**: Follow the project's commit conventions

5. **Review changes**: Check what's changed since the last release
   ```bash
   git log helm-v0.13.0..HEAD --oneline
   ```

6. **Communicate**: Announce releases to users via release notes or community channels

## Rollback procedure

If a release has issues:

1. Create a new patch release with the fix
2. Or, edit the release on GitLab and add a warning to the description
3. Never delete a published release (breaks user installations)

## CI/CD pipeline stages

The GitLab CI/CD pipeline has two stages defined in `.gitlab-ci.yml` at the repository root.

### 1. Validate stage — `validate-helm-chart` job

**Trigger conditions:**
- Merge requests when any of these paths change: `postgres_ai_helm/**/*` or `.gitlab-ci.yml`
- Pushes to the default branch when the same paths change

**Steps:**
- Updates helm dependencies (`helm dependency update`)
- Lints the chart (`helm lint .`)
- Renders templates (`helm template test-release .`)
- Packages the chart (`helm package .`)

**Artifacts:** `postgres_ai_helm/*.tgz` (expires after 1 week)

**Required CI/CD variables:** None (uses `CI_JOB_TOKEN` implicitly via GitLab runner)

### 2. Release stage — `release-helm-chart` job

**Trigger conditions:**
- Tag push matching the pattern `helm-v[0-9]+\.[0-9]+\.[0-9]+` (e.g., `helm-v1.2.3`)
- Tags with fewer or more than three version segments are ignored

**Steps:**
- Extracts version from tag (`CI_COMMIT_TAG` with `helm-v` prefix removed)
- Updates `Chart.yaml` with the new version
- Packages the chart
- Creates a GitLab release via the native `release:` keyword (no extra tokens required)
- Attaches the packaged chart as a release asset

**Artifacts:** `postgres-ai-monitoring-chart.tgz` (expires after 30 days)

**Required CI/CD variables:** None — the job uses `CI_JOB_TOKEN` (provided automatically by GitLab) for release creation via the `release:` keyword.

## Additional resources

- [Helm chart documentation](https://helm.sh/docs/topics/charts/)
- [Semantic versioning](https://semver.org/)
- [GitLab CI/CD documentation](https://docs.gitlab.com/ee/ci/)
- [GitLab Release API](https://docs.gitlab.com/ee/api/releases/)
