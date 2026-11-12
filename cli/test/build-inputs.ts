import { resolve } from "node:path";
import buildInputs from "../build-inputs.json";

/** Absolute paths of the build inputs outside cli/ (build-inputs.json is the one place they are named). */
const repoRoot = resolve(import.meta.dir, "..", "..");
export const METRICS_YML = resolve(repoRoot, buildInputs.metricsYml);
export const REPORT_SCHEMAS_DIR = resolve(repoRoot, buildInputs.reportSchemas);
export const DEMO_INSTANCES = resolve(repoRoot, buildInputs.demoInstances);
