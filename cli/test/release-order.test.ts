import { expect, test } from "bun:test";
import { readFileSync } from "fs";
import { resolve } from "path";
import yaml from "js-yaml";

// UAT 2026-10-01: a dev tag published the CLI to npm before its Docker images
// existed, so `pgai mon local-install` (which pulls images of the CLI's version)
// failed for whoever installed it first. npm publishes only after the images.
const ci = yaml.load(readFileSync(resolve(import.meta.dir, "..", "..", ".gitlab-ci.yml"), "utf8")) as Record<string, any>;

test("npm publish waits for the Docker images of the same tag", () => {
  expect(ci["cli:npm:publish"].needs).toEqual([{ job: "docker:publish:images", artifacts: false }]);
  // Same tags, so the job it waits for always exists in the pipeline.
  expect(ci["cli:npm:publish"].rules).toEqual(ci["docker:publish:images"].rules);
});
