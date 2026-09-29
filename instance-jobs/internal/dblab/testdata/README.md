# Captured engine replies

Real response bodies, captured off the shared rig's engine
(`v4.2.0-20260911-0242`, edition `community`) rather than written by hand — an
invented YAML string would only prove that the code handles the string its
author imagined.

| file                      | endpoint                  | `Content-Type`                                                   |
|---------------------------|---------------------------|------------------------------------------------------------------|
| `admin_config.yaml.golden`| `GET /admin/config.yaml`  | `application/yaml; charset=utf-8`                                |
| `admin_config.json.golden`| `GET /admin/config`       | `application/json; charset=utf-8`                                |
| `metrics.txt.golden`      | `GET /metrics`            | `text/plain; version=0.0.4; charset=utf-8; escaping=underscores` |

`admin_config.yaml.golden` is 15789 bytes, sha256 `34079181e90f59e0…`.

**The first two are the pair that matters.** They are siblings on the same
engine, differing only in the suffix, and one is JSON while the other is not —
so whether a reply is wrapped has to be decided by the REPLY and never by the
path. `metrics.txt.golden` is here to say out loud that `/admin/config.yaml`
is not the only non-JSON endpoint.

All secrets are masked by the engine itself (`DefaultConfigMask`): every
`verificationToken`, `password` and `orgKey` in the capture reads `"****"`.

The `.golden` suffix is load-bearing twice over: it keeps `check-yaml` from
parsing a captured body as repository configuration (the real one has duplicate
merge keys and fails), and `.pre-commit-config.yaml` excludes this path from
`end-of-file-fixer` and `trailing-whitespace` — a hook that tidies a capture
turns a byte-for-byte assertion into a reading of the hook's own output.
