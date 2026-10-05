import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// `mon targets add` takes the DB URL from PGAI_DB_URL when it is not in argv,
// where `ps` and the sudo log would show the password. Same as prepare-db and
// local-install.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
const secret = "S3cretPw-env-91c4";
const url = `postgresql://monitor:${secret}@db.example:5432/app`;
const argvSecret = "ArgvPw-222";
const other = `postgresql://monitor:${argvSecret}@other.example:5432/app`;
let dir: string, projectDir: string, env: Record<string, string>;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/targets-add-env-`);
  projectDir = `${dir}/project`;
  for (const path of [projectDir, `${dir}/bin`, `${dir}/home`, `${dir}/xdg`]) mkdirSync(path);
  writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
  writeFileSync(`${projectDir}/instances.yml`, "");
  // Logs the args and the environment of every docker call.
  writeFileSync(`${dir}/bin/docker`, `#!/bin/sh\nprintf '%s\\n' "$*" >> "${dir}/docker.log"\nenv >> "${dir}/docker.log"\nexit 0\n`);
  chmodSync(`${dir}/bin/docker`, 0o755);
  writeFileSync(`${dir}/docker.log`, "");
  env = { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: projectDir };
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function run(args: string[], extra: Record<string, string> = {}) {
  const result = Bun.spawnSync([process.execPath, cli, "mon", "targets", ...args], { cwd: dir, env: { ...env, ...extra }, timeout: 60000 });
  return { exitCode: result.exitCode, out: result.stdout.toString() + result.stderr.toString() };
}
const instances = () => readFileSync(`${projectDir}/instances.yml`, "utf8");
const docker = () => readFileSync(`${dir}/docker.log`, "utf8");

test("targets add reads the DB URL from PGAI_DB_URL, with the name as the only argument", () => {
  const { exitCode, out } = run(["add", "app"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(out).toContain("Using PGAI_DB_URL (user monitor)");
  expect(out).toContain("Monitoring target 'app' added");
  expect(instances()).toContain(`conn_str: ${url}`);
  expect(instances()).toContain("name: app");
  expect(out).not.toContain(secret);
  // docker and compose run, without the URL in their args or environment.
  expect(docker()).toContain("compose");
  expect(docker()).not.toContain(secret);
  expect(docker()).not.toContain("PGAI_DB_URL");
});

test("the user of an accepted PGAI_DB_URL is shown decoded", () => {
  const { exitCode, out } = run(["add", "app"], { PGAI_DB_URL: url.replace("monitor:", "ops%2Bmon:") });
  expect(exitCode).toBe(0);
  expect(out).toContain("Using PGAI_DB_URL (user ops+mon)");
  expect(out).not.toContain(secret);
});

test("targets add reads PGAI_DB_URL with no argument and uses the default name", () => {
  const { exitCode, out } = run(["add"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(out).toContain("Monitoring target 'db-example-app' added");
  expect(instances()).toContain(`conn_str: ${url}`);
  expect(out).not.toContain(secret);
});

test.each(["region=west", "Prod.db_1-A"])("%p is a name", (name) => {
  const { exitCode, out } = run(["add", name], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(instances()).toContain(`name: ${name}`);
  expect(instances()).toContain(`conn_str: ${url}`);
  expect(out).not.toContain(secret);
});

// As `targets add <url> "$NAME"`: a script running `targets add "$NAME"` with
// NAME empty gets the default name, and the name is trimmed.
test.each([
  ["", "db-example-app"],
  [" ", "db-example-app"],
  ["my-db ", "my-db"],
  [" my-db", "my-db"],
])("with PGAI_DB_URL set, the argument %p is the name %p", (arg, name) => {
  const { exitCode, out } = run(["add", arg], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(out).toContain(`Monitoring target '${name}' added`);
  expect(instances()).toContain(`name: ${name}`);
  expect(out).not.toContain(secret);
});

test("a URL in argv still wins over PGAI_DB_URL", () => {
  const { exitCode, out } = run(["add", other, "app"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(instances()).toContain(`conn_str: ${other}`);
  expect(instances()).not.toContain(secret);
  expect(out).not.toContain(secret);
  // PGAI_DB_URL is unused here, and still kept from docker and compose.
  expect(docker()).toContain("compose");
  expect(docker()).not.toContain(secret);
  expect(docker()).not.toContain("PGAI_DB_URL");
});

test("a URL alone in argv wins over PGAI_DB_URL and gets the default name", () => {
  const { exitCode, out } = run(["add", other], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(out).toContain("Monitoring target 'other-example-app' added");
  expect(out).not.toContain("PGAI_DB_URL");
  expect(instances()).toContain(`conn_str: ${other}`);
  expect(instances()).toContain("name: other-example-app");
  expect(instances()).not.toContain(secret);
  expect(out).not.toContain(secret);
  expect(docker()).toContain("compose");
  expect(docker()).not.toContain(secret);
  expect(docker()).not.toContain("PGAI_DB_URL");
});

test("an upper-case scheme in argv is a connection string, refused as without PGAI_DB_URL", () => {
  const { exitCode, out } = run(["add", other.replace("postgresql", "POSTGRESQL")], { PGAI_DB_URL: url });
  expect(exitCode).toBe(1);
  expect(out).toContain("Invalid connection string format");
  expect(out).not.toContain("pass only the target name");
  expect(out).not.toContain(argvSecret);
  expect(instances()).toBe("");
});

test("two arguments with PGAI_DB_URL set: the first is the connection string", () => {
  const { exitCode, out } = run(["add", "foo", "bar"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(1);
  expect(out).toContain("Invalid connection string format");
  expect(instances()).toBe("");
});

// Each would otherwise be saved as the name of the PGAI_DB_URL target, and its
// password printed and written to instances.yml and the node_name label.
test.each([
  `host=other.example user=monitor password=${argvSecret} dbname=app`,
  `HOST=other.example USER=monitor PASSWORD=${argvSecret}`,
  `Host=other.example;Username=monitor;Password=${argvSecret};Database=app`,
  `password=${argvSecret}`,
  `Pwd=${argvSecret}`,
  `monitor:${argvSecret}@other.example:5432/app`,
  `other.example:5432/app?password=${argvSecret}`,
  `postgresql:other.example/app?password=${argvSecret}`,
  `postgresql:/other.example/app?password=${argvSecret}`,
  `postgresql+ssl://monitor:${argvSecret}@other.example:5432/app`,
  ` ${other}`,
  `host=other.example password = ${argvSecret}`,
  `team@prod-${argvSecret}`,
  // No password, still not a name.
  "Host=other.example;Database=app",
  "other.example:5432/app",
  "host=other.example user=monitor dbname=app",
  "прод",
])("with PGAI_DB_URL set, a lone argument that is not a plain name is refused: %p", (arg) => {
  const { exitCode, out } = run(["add", arg], { PGAI_DB_URL: url });
  expect(exitCode).toBe(1);
  expect(out).toContain("PGAI_DB_URL is set: pass only the target name (ASCII letters, digits, '.', '_', '=', '-'; no password= or pwd=)");
  expect(out).not.toContain(argvSecret);
  expect(out).not.toContain(secret);
  expect(instances()).toBe("");
});

test("an empty PGAI_DB_URL is no connection string", () => {
  const { exitCode, out } = run(["add"], { PGAI_DB_URL: "" });
  expect(exitCode).toBe(1);
  expect(out).toContain("Connection string required");
  expect(instances()).toBe("");
});

test("an invalid PGAI_DB_URL is refused, named, and not printed", () => {
  const { exitCode, out } = run(["add", "app"], { PGAI_DB_URL: `monitor:${secret}@db.example:5432/app` });
  expect(exitCode).toBe(1);
  expect(out).toContain("PGAI_DB_URL");
  expect(out).toContain("Invalid connection string format");
  expect(out).not.toContain(secret);
  expect(instances()).toBe("");
});

test("a PGAI_DB_URL that does not parse is refused, and its password not printed", () => {
  const { exitCode, out } = run(["add", "app"], { PGAI_DB_URL: `postgresql://monitor:${secret}@db.example:54x2/app` });
  expect(exitCode).toBe(1);
  expect(out).toContain("Invalid connection string format");
  expect(out).not.toContain(secret);
  expect(instances()).toBe("");
});

const FORMAT = "Invalid connection string format: use postgresql://user:password@host[:port]/database";
const NO_PASSWORD = "Invalid connection string format: put the password in the URL: postgresql://user:password@host[:port]/database";
const IPV6 = "Invalid connection string format: an IPv6 address is not supported as the host; use a host name";
const ENCODE = "Invalid connection string format: percent-encode the user name and the password (all but ASCII letters, digits, '-', '.', '_' and '~'), and an '@' in the database name or the query";
const USER_COLON = "Invalid connection string format: percent-encode the user name and the password one at a time, with a raw ':' between them";
const CONTROL = "Invalid connection string format: remove the control character (such as a CR from a CRLF file), or percent-encode it";

// pgx and WHATWG split the user info at the last '@': no part of the password
// may end up in the default name (and the node_name label).
test.each([
  "postgresql://monitor:Ab@Tail-91c4@db.example:5432/app",
  "postgres://monitor:Ab@Tail-91c4@db.example:5432/app",
])("the default name takes the host after the last '@': %p", (dbUrl) => {
  const { exitCode, out } = run(["add"], { PGAI_DB_URL: dbUrl });
  expect(exitCode).toBe(0);
  expect(out).toContain("Monitoring target 'db-example-app' added");
  expect(instances()).toContain("name: db-example-app");
  expect(instances()).toContain("node_name: db-example-app");
  expect(out).not.toContain("Tail-91c4");
});

const pwHead = "PwHead-5d1", pwTail = "PwTail-9f3";

// What the percent-encode error asks for.
test("a password with '@', '/', '?' and '#' percent-encoded gets the default name", () => {
  const { exitCode, out } = run(["add"], { PGAI_DB_URL: `postgresql://monitor:Ab%40${pwHead}%2F${pwTail}%3Fx%23y@db.example:5432/app` });
  expect(exitCode).toBe(0);
  expect(out).toContain("Monitoring target 'db-example-app' added");
  expect(instances()).toContain("name: db-example-app");
  expect(instances()).toContain("node_name: db-example-app");
  expect(out).not.toContain(pwHead);
  expect(out).not.toContain(pwTail);
});

// A raw '/', '?' or '#' in the password ends the host early: WHATWG then reads
// part of the password as the host, and the real '@host' comes after it. pgx
// (Go's net/url) refuses a raw space, quote, non-ASCII character or a bad '%'
// in the user info. Each is refused, so no part of the password is in the
// output, a name or node_name.
test.each([
  [`postgresql://monitor:@@${pwHead}/${pwTail}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab@:@${pwHead}/${pwTail}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab@${pwHead}/${pwTail}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor@db.example/app?password=Ab:${pwHead}@${pwTail}/x`, ENCODE],
  [`postgresql://monitor:Ab@${pwHead}/${pwTail}?x@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab@${pwHead}/${pwTail}?k&x@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab@${pwHead}/${pwTail}#x@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab?${pwHead}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab#${pwHead}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab ${pwHead}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab"${pwHead}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab\\${pwHead}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab[${pwHead}]@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Abé${pwHead}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab%zz${pwHead}@db.example:5432/app`, ENCODE],
  [`postgresql://monitor:Ab%4${pwHead}@db.example:5432/app`, ENCODE],
  // Not the user:password@host[:port]/db form either.
  [`postgresql://monitor:${pwHead}@db.example:5432`, FORMAT],
  [`postgresql://:${pwHead}@db.example:5432/app`, FORMAT],
  ["postgresql://db.example:5432/app", FORMAT],
  ["postgresql://monitor@db.example:5432/app", NO_PASSWORD],
  [`postgresql://monitor@db.example:5432/app?password=${pwHead}`, NO_PASSWORD],
  [`postgresql://monitor:${pwHead}@[::1]:5432/app`, IPV6],
  // user:password encoded as one unit (quote(f"{user}:{pw}", safe="")): WHATWG
  // and pgx read it all as the user name, which decodes to monitor:<password>.
  [`postgresql://monitor%3A${pwHead}@db.example:5432/app`, USER_COLON],
  [`postgresql://monitor%3aAb%40${pwHead}@db.example:5432/app`, USER_COLON],
  [`postgresql://monitor%3A${pwHead}:x@db.example:5432/app`, USER_COLON],
  // WHATWG drops these, pgx refuses the URL: pgwatch would not collect.
  [`postgresql://monitor:${pwHead}@db.example:5432/app\r`, CONTROL],
  [`postgresql://monitor:${pwHead}@db.example:5432/app?sslmode=disable\r`, CONTROL],
  [`postgresql://monitor:Ab\t${pwHead}@db.example:5432/app`, CONTROL],
  [`postgresql://monitor:${pwHead}@db.example:5432/a\x7fpp`, CONTROL],
])("a PGAI_DB_URL that is not user:password@host[:port]/db is refused: %p", (dbUrl, error) => {
  const { exitCode, out } = run(["add"], { PGAI_DB_URL: dbUrl });
  expect(exitCode).toBe(1);
  expect(out.split("\n")).toContain(error);
  // The source is named; the user only once the URL is accepted.
  expect(out.split("\n")).toContain("Using PGAI_DB_URL");
  expect(out).not.toContain(pwHead);
  expect(out).not.toContain(pwTail);
  expect(instances()).toBe("");
});

// The database name stays as WHATWG reads it, percent-encoded.
test.each([
  ["postgresql://monitor:pw@db.example:5432/my%40db", "db-example-my-40db"],
  ["postgresql://monitor:pw@db.example:5432/база", "db-example--D0-B1-D0-B0-D0-B7-D0-B0"],
  ["postgresql://monitor:pw@db.example:5432/%D0%B1%D0%B0%D0%B7%D0%B0", "db-example--D0-B1-D0-B0-D0-B7-D0-B0"],
])("the default name of %p is %p", (dbUrl, name) => {
  const { exitCode, out } = run(["add"], { PGAI_DB_URL: dbUrl });
  expect(exitCode).toBe(0);
  expect(out).toContain("Using PGAI_DB_URL (user monitor)");
  expect(out).toContain(`Monitoring target '${name}' added`);
  expect(instances()).toContain(`name: ${name}`);
});

// An '@' after the host may be a raw '@' in a password: refused, with a name
// given too, and the error says where to percent-encode it.
test.each([
  ["postgresql://monitor:pw@db.example:5432/app?application_name=ops@team", "postgresql://monitor:pw@db.example:5432/app?application_name=ops%40team"],
  ["postgresql://monitor:pw@db.example:5432/my@db", "postgresql://monitor:pw@db.example:5432/my%40db"],
  ["postgresql://monitor:pw@db.example:5432/app?sslrootcert=/home/john@corp.example/root.crt", "postgresql://monitor:pw@db.example:5432/app?sslrootcert=/home/john%40corp.example/root.crt"],
])("an '@' in the database name or the query is refused raw and taken percent-encoded: %p", (raw, encoded) => {
  const refused = run(["add", raw, "svc"]);
  expect(refused.exitCode).toBe(1);
  expect(refused.out.split("\n")).toContain(ENCODE);
  expect(instances()).toBe("");
  const added = run(["add", encoded, "svc"]);
  expect(added.exitCode).toBe(0);
  expect(added.out).toContain("Monitoring target 'svc' added");
  // A long conn_str is folded onto the next line.
  expect(instances()).toContain(encoded);
});

test("targets add --help names PGAI_DB_URL, which automation probes for, and gives it to sudo on stdin", () => {
  const { exitCode, out } = run(["add", "--help"]);
  expect(exitCode).toBe(0);
  expect(out).toContain("PGAI_DB_URL");
  expect(out).toContain(`printf '%s\\n' "$URL" | sudo sh -c \\\n`);
  expect(out).toContain(`'IFS= read -r PGAI_DB_URL; export PGAI_DB_URL; exec postgres-ai mon targets add my-db'`);
  // sudo logs a variable kept with --preserve-env (ENV=PGAI_DB_URL=<the URL>).
  expect(out).not.toContain("--preserve-env=");
  // Not exported in the caller's shell: a later `mon local-install` or `prepare-db` would read it.
  expect(out).not.toContain("export PGAI_DB_URL=");
  // sudo I/O logging records stdin: the URL is then read from a file in the root shell.
  expect(out).toContain("log_input");
  expect(out).toContain(`'IFS= read -r PGAI_DB_URL < /path/to/db-url; export PGAI_DB_URL; exec postgres-ai mon targets add my-db'`);
});
