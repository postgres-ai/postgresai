// Copies the demo target list into the package root, where `mon local-install --demo` finds it.
import { copyFileSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { basename, dirname, resolve } from "node:path";

const cliDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { demoInstances } = JSON.parse(readFileSync(resolve(cliDir, "build-inputs.json"), "utf8"));
copyFileSync(resolve(cliDir, "..", demoInstances), resolve(cliDir, basename(demoInstances)));
