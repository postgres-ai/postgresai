import { pbkdf2Sync, createHmac, createHash, randomBytes } from "crypto";

/**
 * Build a PostgreSQL SCRAM-SHA-256 password verifier from a plaintext password.
 *
 * The returned string is what Postgres stores in pg_authid.rolpassword. Passing
 * it verbatim to `ALTER USER ... PASSWORD '<verifier>'` makes Postgres store it
 * as-is (it recognizes the `SCRAM-SHA-256$` prefix), so the cleartext password
 * never reaches the server — it stays out of pg_stat_activity, the server log,
 * and pg_stat_statements. See RFC 5802 / RFC 7677 and PostgreSQL's
 * src/common/scram-common.c.
 *
 * Format: SCRAM-SHA-256$<iterations>:<base64(salt)>$<base64(StoredKey)>:<base64(ServerKey)>
 *
 * Only printable ASCII is accepted until PostgreSQL-compatible SASLprep is
 * implemented. Generated base64url passwords are within this range.
 */
export const SCRAM_DEFAULT_ITERATIONS = 4096;
const SCRAM_SALT_LENGTH = 16; // bytes
const SCRAM_KEY_LENGTH = 32; // SHA-256 digest length

export function scramSha256Verifier(
  password: string,
  opts?: { salt?: Buffer; iterations?: number }
): string {
  if (!/^[\x20-\x7e]+$/.test(password)) {
    throw new Error("Monitoring password must contain only printable ASCII characters");
  }
  const iterations = opts?.iterations ?? SCRAM_DEFAULT_ITERATIONS;
  const salt = opts?.salt ?? randomBytes(SCRAM_SALT_LENGTH);

  // SaltedPassword = Hi(password, salt, i) = PBKDF2-HMAC-SHA256, dkLen = hash len.
  const saltedPassword = pbkdf2Sync(
    Buffer.from(password, "utf8"),
    salt,
    iterations,
    SCRAM_KEY_LENGTH,
    "sha256"
  );

  // ClientKey = HMAC(SaltedPassword, "Client Key"); StoredKey = H(ClientKey).
  const clientKey = createHmac("sha256", saltedPassword).update("Client Key").digest();
  const storedKey = createHash("sha256").update(clientKey).digest();
  // ServerKey = HMAC(SaltedPassword, "Server Key").
  const serverKey = createHmac("sha256", saltedPassword).update("Server Key").digest();

  return (
    `SCRAM-SHA-256$${iterations}:${salt.toString("base64")}` +
    `$${storedKey.toString("base64")}:${serverKey.toString("base64")}`
  );
}
