import * as fs from "fs";
import * as path from "path";
import * as os from "os";

/**
 * Configuration object structure
 */
export interface Config {
  apiKey: string | null;
  baseUrl: string | null;
  storageBaseUrl: string | null;
  orgId: number | null;
  defaultProject: string | null;
  /** Docker Compose project name for monitoring stack */
  projectName: string | null;
  /** Epoch ms when the occasional feedback tip was last shown (see lib/feedback.ts) */
  feedbackTipLastShownAt?: number | null;
}

/**
 * Wire prefix of a global (user-scoped) token — see platform-all #629.
 *
 * A global token reaches every org its owner belongs to, so it carries no
 * org_id of its own and every org-scoped command must name one explicitly.
 */
export const GLOBAL_TOKEN_PREFIX = "pai_global_";

/**
 * Whether an API key is a global token.
 *
 * Derived from the token itself rather than stored as a separate `tokenKind`
 * field, deliberately. The key can arrive from `--api-key`, from
 * `PGAI_API_KEY`, or from the config file, and only the last of those could
 * carry a stored kind — a stored flag would therefore be wrong exactly when the
 * key is overridden, which is the case automation uses most. The prefix travels
 * with the credential, so it is always right.
 */
export function isGlobalTokenValue(apiKey: string | null | undefined): boolean {
  return typeof apiKey === "string" && apiKey.trim().startsWith(GLOBAL_TOKEN_PREFIX);
}

/** Whether the resolved config's API key is a global token. */
export function isGlobalToken(cfg: Pick<Config, "apiKey">): boolean {
  return isGlobalTokenValue(cfg.apiKey);
}

/**
 * Get the user-level config directory path
 * @returns Path to ~/.config/postgresai
 */
export function getConfigDir(): string {
  const configHome = process.env.XDG_CONFIG_HOME || path.join(os.homedir(), ".config");
  return path.join(configHome, "postgresai");
}

/**
 * Get the user-level config file path
 * @returns Path to ~/.config/postgresai/config.json
 */
export function getConfigPath(): string {
  return path.join(getConfigDir(), "config.json");
}

/**
 * Get the legacy project-local config file path
 * @returns Path to .pgwatch-config in current directory
 */
export function getLegacyConfigPath(): string {
  return path.resolve(process.cwd(), ".pgwatch-config");
}

/**
 * Read configuration from file
 * Tries user-level config first, then falls back to legacy project-local config
 * @returns Configuration object with apiKey, baseUrl, orgId
 */
export function readConfig(): Config {
  const config: Config = {
    apiKey: null,
    baseUrl: null,
    storageBaseUrl: null,
    orgId: null,
    defaultProject: null,
    projectName: null,
    feedbackTipLastShownAt: null,
  };

  // Try user-level config first
  const userConfigPath = getConfigPath();
  if (fs.existsSync(userConfigPath)) {
    try {
      const content = fs.readFileSync(userConfigPath, "utf8");
      const parsed = JSON.parse(content);
      config.apiKey = parsed.apiKey ?? null;
      config.baseUrl = parsed.baseUrl ?? null;
      config.storageBaseUrl = parsed.storageBaseUrl ?? null;
      config.orgId = parsed.orgId ?? null;
      config.defaultProject = parsed.defaultProject ?? null;
      config.projectName = parsed.projectName ?? null;
      config.feedbackTipLastShownAt = parsed.feedbackTipLastShownAt ?? null;
      return config;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`Warning: Failed to read config from ${userConfigPath}: ${message}`);
    }
  }

  // Fall back to legacy project-local config
  const legacyPath = getLegacyConfigPath();
  if (fs.existsSync(legacyPath)) {
    try {
      const stats = fs.statSync(legacyPath);
      if (stats.isFile()) {
        const content = fs.readFileSync(legacyPath, "utf8");
        const lines = content.split(/\r?\n/);
        for (const line of lines) {
          const match = line.match(/^api_key=(.+)$/);
          if (match) {
            config.apiKey = match[1].trim();
            break;
          }
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`Warning: Failed to read legacy config from ${legacyPath}: ${message}`);
    }
  }

  return config;
}

/**
 * Write configuration to user-level config file
 * @param config - Configuration object with apiKey, baseUrl, orgId
 */
export function writeConfig(config: Partial<Config>): void {
  const configDir = getConfigDir();
  const configPath = getConfigPath();

  // Ensure config directory exists
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true, mode: 0o700 });
    // Create-only, same as the file mode below: a pre-existing 0755 config dir
    // keeps it. The file itself is 0600 so the token is not disclosed, but a
    // group-writable dir would let another user substitute it.
    try {
      fs.chmodSync(configDir, 0o700);
    } catch {
      /* best-effort; the directory is usable either way */
    }
  }

  // Read existing config and merge
  let existingConfig: Record<string, unknown> = {};
  if (fs.existsSync(configPath)) {
    try {
      const content = fs.readFileSync(configPath, "utf8");
      existingConfig = JSON.parse(content);
    } catch (err) {
      // Ignore parse errors, will overwrite
    }
  }

  const mergedConfig = {
    ...existingConfig,
    ...config,
  };

  // Write config file with restricted permissions
  fs.writeFileSync(configPath, JSON.stringify(mergedConfig, null, 2) + "\n", {
    mode: 0o600,
  });
  // `mode` above only applies when the file is CREATED. A config that already
  // exists with looser permissions keeps them while receiving an API key --
  // which under #327 can be a global token, the widest credential we issue.
  //
  // Best-effort: chmod raises EPERM on a file owned by another UID (a shared CI
  // image, a container running as a different user), where the write itself can
  // still succeed. Warn rather than throw -- refusing to save the credential
  // would be a worse outcome than saving it with the permissions it already had,
  // and the user needs to be told either way.
  try {
    fs.chmodSync(configPath, 0o600);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(
      `Warning: could not restrict permissions on ${configPath}: ${message}`,
    );
    console.error("         It may be readable by other users on this machine.");
  }
}

/**
 * Delete specific keys from configuration
 * @param keys - Array of keys to delete (e.g., ['apiKey'])
 */
export function deleteConfigKeys(keys: string[]): void {
  const configPath = getConfigPath();
  if (!fs.existsSync(configPath)) {
    return;
  }

  try {
    const content = fs.readFileSync(configPath, "utf8");
    const config: Record<string, unknown> = JSON.parse(content);

    for (const key of keys) {
      delete config[key];
    }

    fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n", {
      mode: 0o600,
    });
    fs.chmodSync(configPath, 0o600);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`Warning: Failed to update config: ${message}`);
  }
}

/**
 * Check if config file exists
 * @returns True if config exists
 */
export function configExists(): boolean {
  return fs.existsSync(getConfigPath()) || fs.existsSync(getLegacyConfigPath());
}
