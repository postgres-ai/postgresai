import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const scriptsDir = dirname(fileURLToPath(import.meta.url));
const cliDir = resolve(scriptsDir, "..");
const { reportSchemas } = JSON.parse(readFileSync(resolve(cliDir, "build-inputs.json"), "utf8"));
const sourceDir = resolve(cliDir, "..", reportSchemas);
const targetDir = resolve(cliDir, "schemas");

rmSync(targetDir, { recursive: true, force: true });
mkdirSync(targetDir, { recursive: true });

const schemaFiles = readdirSync(sourceDir)
  .filter((file) => file.endsWith(".schema.json"))
  .sort();

if (schemaFiles.length === 0) {
  throw new Error(`No JSON Schemas found in ${sourceDir}`);
}

for (const file of schemaFiles) {
  cpSync(resolve(sourceDir, file), resolve(targetDir, file));
}

console.log(`Copied ${schemaFiles.length} JSON Schemas to ${targetDir}`);
