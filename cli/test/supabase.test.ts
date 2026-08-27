import { describe, expect, test, beforeEach, afterEach, mock } from "bun:test";
import {
  resolveSupabaseConfig,
  extractProjectRefFromUrl,
  fetchPoolerDatabaseUrl,
  SupabaseClient,
  applyInitPlanViaSupabase,
  verifyInitSetupViaSupabase,
  type PgCompatibleError,
} from "../lib/supabase";

// Valid project ref for tests (10-30 alphanumeric chars)
const VALID_PROJECT_REF = "abcdefghij1234567890";

describe("Supabase module", () => {
  describe("extractProjectRefFromUrl", () => {
    test("extracts project ref from standard Supabase URL", () => {
      const url =
        "postgresql://postgres:password@db.abcdefghij.supabase.co:5432/postgres";
      expect(extractProjectRefFromUrl(url)).toBe("abcdefghij");
    });

    test("extracts project ref from Supabase URL without db. prefix", () => {
      const url =
        "postgresql://postgres:password@abcdefghij.supabase.co:5432/postgres";
      expect(extractProjectRefFromUrl(url)).toBe("abcdefghij");
    });

    test("extracts project ref from legacy pooler URL", () => {
      const url =
        "postgresql://postgres:password@abcdefghij.pooler.supabase.com:6543/postgres";
      expect(extractProjectRefFromUrl(url)).toBe("abcdefghij");
    });

    test("extracts project ref from modern AWS pooler URL (username format)", () => {
      const url =
        "postgresql://postgres.abcdefghij:password@aws-0-us-east-1.pooler.supabase.com:6543/postgres";
      expect(extractProjectRefFromUrl(url)).toBe("abcdefghij");
    });

    test("returns undefined for non-Supabase URL", () => {
      const url = "postgresql://postgres:password@localhost:5432/postgres";
      expect(extractProjectRefFromUrl(url)).toBeUndefined();
    });

    test("returns undefined for RDS URL", () => {
      const url =
        "postgresql://postgres:password@mydb.cluster-xyz.us-east-1.rds.amazonaws.com:5432/postgres";
      expect(extractProjectRefFromUrl(url)).toBeUndefined();
    });

    test("returns undefined for invalid URL", () => {
      const url = "not a valid url";
      expect(extractProjectRefFromUrl(url)).toBeUndefined();
    });

    test("handles URL with special characters in password", () => {
      const url =
        "postgresql://postgres:p%40ss%2Fw0rd@db.myprojectref.supabase.co:5432/postgres";
      expect(extractProjectRefFromUrl(url)).toBe("myprojectref");
    });

    test("returns undefined for AWS regional pooler without username ref", () => {
      // AWS regional URLs without postgres.<ref> format should not extract region as ref
      const url =
        "postgresql://postgres:password@aws-0-us-east-1.pooler.supabase.com:6543/postgres";
      expect(extractProjectRefFromUrl(url)).toBeUndefined();
    });
  });

  describe("resolveSupabaseConfig", () => {
    const originalEnv = process.env;

    beforeEach(() => {
      // Reset env before each test
      process.env = { ...originalEnv };
      delete process.env.SUPABASE_ACCESS_TOKEN;
      delete process.env.SUPABASE_PROJECT_REF;
    });

    afterEach(() => {
      process.env = originalEnv;
    });

    test("resolves config from options", () => {
      const config = resolveSupabaseConfig({
        accessToken: "my-token",
        projectRef: "myprojectref12",
      });
      expect(config.accessToken).toBe("my-token");
      expect(config.projectRef).toBe("myprojectref12");
    });

    test("resolves config from environment variables", () => {
      process.env.SUPABASE_ACCESS_TOKEN = "env-token";
      process.env.SUPABASE_PROJECT_REF = "envprojectref1";

      const config = resolveSupabaseConfig({});
      expect(config.accessToken).toBe("env-token");
      expect(config.projectRef).toBe("envprojectref1");
    });

    test("options take precedence over environment variables", () => {
      process.env.SUPABASE_ACCESS_TOKEN = "env-token";
      process.env.SUPABASE_PROJECT_REF = "envprojectref1";

      const config = resolveSupabaseConfig({
        accessToken: "opts-token",
        projectRef: "optsprojectref",
      });
      expect(config.accessToken).toBe("opts-token");
      expect(config.projectRef).toBe("optsprojectref");
    });

    test("throws error when access token is missing", () => {
      expect(() =>
        resolveSupabaseConfig({
          projectRef: "myprojectref12",
        })
      ).toThrow(/access token is required/i);
    });

    test("throws error when project ref is missing", () => {
      expect(() =>
        resolveSupabaseConfig({
          accessToken: "my-token",
        })
      ).toThrow(/project reference is required/i);
    });

    test("trims whitespace from values", () => {
      const config = resolveSupabaseConfig({
        accessToken: "  my-token  ",
        projectRef: "  myprojectref12  ",
      });
      expect(config.accessToken).toBe("my-token");
      expect(config.projectRef).toBe("myprojectref12");
    });
  });

  describe("fetchPoolerDatabaseUrl", () => {
    const originalFetch = globalThis.fetch;

    afterEach(() => {
      globalThis.fetch = originalFetch;
    });

    test("returns pooler db url with username including project ref (db_host/db_port/db_name response)", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(
            JSON.stringify([
              {
                db_host: "aws-1-eu-west-1.pooler.supabase.com",
                db_port: 6543,
                db_name: "postgres",
              },
            ]),
            { status: 200 }
          )
        )
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(url).toBe(
        "postgresql://postgres_ai_mon.xhaqmsvczjkkvkgdyast@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"
      );
    });

    test("does not double-append project ref if username already has it", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(
            JSON.stringify([
              {
                db_host: "aws-1-eu-west-1.pooler.supabase.com",
                db_port: 6543,
                db_name: "postgres",
              },
            ]),
            { status: 200 }
          )
        )
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon.xhaqmsvczjkkvkgdyast"
      );
      expect(url).toBe(
        "postgresql://postgres_ai_mon.xhaqmsvczjkkvkgdyast@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"
      );
    });

    test("returns pooler db url via connection_string fallback path", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(
            JSON.stringify([
              {
                // No db_host/db_port/db_name - uses connection_string fallback
                connection_string:
                  "postgresql://ignored@aws-1-eu-west-1.pooler.supabase.com:6543/postgres",
              },
            ]),
            { status: 200 }
          )
        )
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(url).toBe(
        "postgresql://postgres_ai_mon.xhaqmsvczjkkvkgdyast@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"
      );
    });

    test("returns null for invalid connection_string URL", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(
            JSON.stringify([
              {
                connection_string: "not-a-valid-url",
              },
            ]),
            { status: 200 }
          )
        )
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(url).toBeNull();
    });

    test("returns null for empty API response", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(url).toBeNull();
    });

    test("returns null for API error response", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(new Response("Unauthorized", { status: 401 }))
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(url).toBeNull();
    });

    test("rewrites the transaction-mode pooler port to session mode", async () => {
      // The Supabase Management API reports the TRANSACTION-mode pooler port
      // (6543). pgwatch connects with pgx, which uses server-side prepared
      // statements; under transaction pooling those collide across the shared
      // backends with `prepared statement "stmtcache_<hash>" already exists`
      // (42P05) and metric collection silently stops.
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(
            JSON.stringify([
              {
                db_host: "aws-1-eu-west-1.pooler.supabase.com",
                db_port: 6543,
                db_name: "postgres",
              },
            ]),
            { status: 200 }
          )
        )
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(new URL(url!).port).toBe("5432");
      // Same host — the pooler serves session mode on 5432. Switching to the
      // direct `db.<ref>.supabase.co` host would require IPv6.
      expect(new URL(url!).hostname).toBe("aws-1-eu-west-1.pooler.supabase.com");
    });

    test("leaves a session-mode pooler port untouched", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(
            JSON.stringify([
              {
                db_host: "aws-1-eu-west-1.pooler.supabase.com",
                db_port: 5432,
                db_name: "postgres",
              },
            ]),
            { status: 200 }
          )
        )
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(new URL(url!).port).toBe("5432");
    });

    test("leaves a direct (non-pooler) host untouched", async () => {
      // A direct host never runs a transaction pooler, so its port is
      // authoritative and must be passed through verbatim.
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(
            JSON.stringify([
              {
                db_host: "db.xhaqmsvczjkkvkgdyast.supabase.co",
                db_port: 6543,
                db_name: "postgres",
              },
            ]),
            { status: 200 }
          )
        )
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(new URL(url!).hostname).toBe("db.xhaqmsvczjkkvkgdyast.supabase.co");
      expect(new URL(url!).port).toBe("6543");
    });

    test("returns null when fetch throws network error", async () => {
      globalThis.fetch = mock(() =>
        Promise.reject(new Error("Network error"))
      ) as unknown as typeof fetch;

      const url = await fetchPoolerDatabaseUrl(
        { projectRef: "xhaqmsvczjkkvkgdyast", accessToken: "token" },
        "postgres_ai_mon"
      );
      expect(url).toBeNull();
    });
  });

  describe("SupabaseClient", () => {
    test("throws error when project ref is empty", () => {
      expect(() => new SupabaseClient({ projectRef: "", accessToken: "token" })).toThrow(
        /project reference is required/i
      );
    });

    test("throws error when access token is empty", () => {
      expect(() => new SupabaseClient({ projectRef: VALID_PROJECT_REF, accessToken: "" })).toThrow(
        /access token is required/i
      );
    });

    test("throws error for invalid project ref format (too short)", () => {
      expect(() => new SupabaseClient({ projectRef: "short", accessToken: "token" })).toThrow(
        /invalid supabase project reference format/i
      );
    });

    test("throws error for invalid project ref format (special chars)", () => {
      expect(() => new SupabaseClient({ projectRef: "../admin/hack", accessToken: "token" })).toThrow(
        /invalid supabase project reference format/i
      );
    });

    test("accepts valid project ref format", () => {
      const client = new SupabaseClient({ projectRef: VALID_PROJECT_REF, accessToken: "token" });
      expect(client).toBeDefined();
    });

    describe("query method", () => {
      const originalFetch = globalThis.fetch;
      let mockFetch: ReturnType<typeof mock>;

      beforeEach(() => {
        mockFetch = mock(() =>
          Promise.resolve(new Response(JSON.stringify([{ db: "postgres" }]), { status: 200 }))
        );
        globalThis.fetch = mockFetch as unknown as typeof fetch;
      });

      afterEach(() => {
        globalThis.fetch = originalFetch;
      });

      test("makes correct API request", async () => {
        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        await client.query("SELECT 1", true);

        expect(mockFetch).toHaveBeenCalledTimes(1);
        const [url, options] = mockFetch.mock.calls[0] as [string, RequestInit];
        expect(url).toBe(
          `https://api.supabase.com/v1/projects/${VALID_PROJECT_REF}/database/query`
        );
        expect(options.method).toBe("POST");
        expect(options.headers).toEqual({
          "Content-Type": "application/json",
          Authorization: "Bearer mytoken",
        });
        const body = JSON.parse(options.body as string);
        expect(body.query).toBe("SELECT 1");
        expect(body.read_only).toBe(true);
      });

      test("returns rows from successful response", async () => {
        mockFetch = mock(() =>
          Promise.resolve(
            new Response(JSON.stringify([{ id: 1, name: "test" }]), { status: 200 })
          )
        );
        globalThis.fetch = mockFetch as unknown as typeof fetch;

        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        const result = await client.query("SELECT * FROM test");
        expect(result.rows).toEqual([{ id: 1, name: "test" }]);
        expect(result.rowCount).toBe(1);
      });

      test("handles empty result", async () => {
        mockFetch = mock(() =>
          Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
        );
        globalThis.fetch = mockFetch as unknown as typeof fetch;

        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        const result = await client.query("SELECT * FROM empty_table");
        expect(result.rows).toEqual([]);
        expect(result.rowCount).toBe(0);
      });

      test("throws PgCompatibleError on HTTP error", async () => {
        mockFetch = mock(() =>
          Promise.resolve(
            new Response(
              JSON.stringify({
                error: {
                  code: "42501",
                  message: "permission denied",
                  details: "Not authorized",
                },
              }),
              { status: 403 }
            )
          )
        );
        globalThis.fetch = mockFetch as unknown as typeof fetch;

        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        try {
          await client.query("SELECT * FROM secret_table");
          throw new Error("Expected query to throw");
        } catch (e) {
          const err = e as PgCompatibleError;
          expect(err.message).toBe("permission denied");
          expect(err.code).toBe("42501");
          expect(err.detail).toBe("Not authorized");
          expect(err.httpStatus).toBe(403);
        }
      });

      test("throws error on non-JSON response", async () => {
        mockFetch = mock(() =>
          Promise.resolve(new Response("Internal Server Error", { status: 500 }))
        );
        globalThis.fetch = mockFetch as unknown as typeof fetch;

        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        try {
          await client.query("SELECT 1");
          throw new Error("Expected query to throw");
        } catch (e) {
          const err = e as PgCompatibleError;
          expect(err.message).toContain("non-JSON response");
          expect(err.httpStatus).toBe(500);
        }
      });

      test("maps Supabase error codes to PostgreSQL codes", async () => {
        mockFetch = mock(() =>
          Promise.resolve(
            new Response(
              JSON.stringify({
                error: {
                  code: "PGRST200",
                  message: "table not found",
                },
              }),
              { status: 404 }
            )
          )
        );
        globalThis.fetch = mockFetch as unknown as typeof fetch;

        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        try {
          await client.query("SELECT * FROM nonexistent");
          throw new Error("Expected query to throw");
        } catch (e) {
          const err = e as PgCompatibleError;
          expect(err.code).toBe("42P01"); // undefined_table
        }
      });
    });

    describe("testConnection method", () => {
      const originalFetch = globalThis.fetch;

      afterEach(() => {
        globalThis.fetch = originalFetch;
      });

      test("returns database and version info", async () => {
        globalThis.fetch = mock(() =>
          Promise.resolve(
            new Response(
              JSON.stringify([
                { db: "postgres", version: "PostgreSQL 15.1" },
              ]),
              { status: 200 }
            )
          )
        ) as unknown as typeof fetch;

        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        const result = await client.testConnection();
        expect(result.database).toBe("postgres");
        expect(result.version).toBe("PostgreSQL 15.1");
      });

      test("returns empty strings for empty response", async () => {
        globalThis.fetch = mock(() =>
          Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
        ) as unknown as typeof fetch;

        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        const result = await client.testConnection();
        expect(result.database).toBe("");
        expect(result.version).toBe("");
      });
    });

    describe("getCurrentDatabase method", () => {
      const originalFetch = globalThis.fetch;

      afterEach(() => {
        globalThis.fetch = originalFetch;
      });

      test("returns current database name", async () => {
        globalThis.fetch = mock(() =>
          Promise.resolve(
            new Response(JSON.stringify([{ db: "mydb" }]), { status: 200 })
          )
        ) as unknown as typeof fetch;

        const client = new SupabaseClient({
          projectRef: VALID_PROJECT_REF,
          accessToken: "mytoken",
        });

        const result = await client.getCurrentDatabase();
        expect(result).toBe("mydb");
      });
    });
  });

  describe("applyInitPlanViaSupabase", () => {
    const originalFetch = globalThis.fetch;

    afterEach(() => {
      globalThis.fetch = originalFetch;
    });

    test("applies all non-optional steps and returns applied list", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      ) as unknown as typeof fetch;

      const client = new SupabaseClient({
        projectRef: VALID_PROJECT_REF,
        accessToken: "mytoken",
      });

      const result = await applyInitPlanViaSupabase({
        client,
        plan: {
          monitoringUser: "test_user",
          database: "testdb",
          steps: [
            { name: "step1", sql: "SELECT 1" },
            { name: "step2", sql: "SELECT 2" },
          ],
        },
      });

      expect(result.applied).toEqual(["step1", "step2"]);
      expect(result.skippedOptional).toEqual([]);
    });

    test("skips failing optional steps", async () => {
      let callCount = 0;
      globalThis.fetch = mock(() => {
        callCount++;
        if (callCount === 2) {
          // Second call (optional step) fails
          return Promise.resolve(
            new Response(JSON.stringify({ error: { message: "failed" } }), { status: 500 })
          );
        }
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
      }) as unknown as typeof fetch;

      const client = new SupabaseClient({
        projectRef: VALID_PROJECT_REF,
        accessToken: "mytoken",
      });

      const result = await applyInitPlanViaSupabase({
        client,
        plan: {
          monitoringUser: "test_user",
          database: "testdb",
          steps: [
            { name: "required1", sql: "SELECT 1" },
            { name: "optional1", sql: "SELECT 2", optional: true },
          ],
        },
      });

      expect(result.applied).toEqual(["required1"]);
      expect(result.skippedOptional).toEqual(["optional1"]);
    });

    test("throws on failing required step with preserved error fields", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({ error: { code: "42501", message: "permission denied" } }),
            { status: 403 }
          )
        )
      ) as unknown as typeof fetch;

      const client = new SupabaseClient({
        projectRef: VALID_PROJECT_REF,
        accessToken: "mytoken",
      });

      try {
        await applyInitPlanViaSupabase({
          client,
          plan: {
            monitoringUser: "test_user",
            database: "testdb",
            steps: [{ name: "create_role", sql: "CREATE ROLE test" }],
          },
        });
        throw new Error("Expected to throw");
      } catch (e) {
        const err = e as PgCompatibleError;
        expect(err.message).toContain('Failed at step "create_role"');
        expect(err.code).toBe("42501");
      }
    });
  });

  describe("verifyInitSetupViaSupabase", () => {
    const originalFetch = globalThis.fetch;

    afterEach(() => {
      globalThis.fetch = originalFetch;
    });

    test("throws error for invalid monitoring user name", async () => {
      const client = new SupabaseClient({
        projectRef: VALID_PROJECT_REF,
        accessToken: "mytoken",
      });

      try {
        await verifyInitSetupViaSupabase({
          client,
          database: "testdb",
          monitoringUser: "invalid-user-name!", // Invalid: contains hyphen and exclamation
          includeOptionalPermissions: false,
        });
        throw new Error("Expected to throw");
      } catch (e) {
        expect((e as Error).message).toContain("Invalid monitoring user name");
      }
    });

    test("throws error for database name with null bytes (SQL injection prevention)", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(new Response(JSON.stringify([{ rolname: "postgres_ai_mon" }]), { status: 200 }))
      ) as unknown as typeof fetch;

      const client = new SupabaseClient({
        projectRef: VALID_PROJECT_REF,
        accessToken: "mytoken",
      });

      try {
        await verifyInitSetupViaSupabase({
          client,
          database: "test\0db", // null byte should be rejected
          monitoringUser: "postgres_ai_mon",
          includeOptionalPermissions: false,
        });
        throw new Error("Expected to throw");
      } catch (e) {
        expect((e as Error).message).toContain("null bytes");
      }
    });

    test("returns missing role when role does not exist", async () => {
      globalThis.fetch = mock(() =>
        Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      ) as unknown as typeof fetch;

      const client = new SupabaseClient({
        projectRef: VALID_PROJECT_REF,
        accessToken: "mytoken",
      });

      const result = await verifyInitSetupViaSupabase({
        client,
        database: "testdb",
        monitoringUser: "postgres_ai_mon",
        includeOptionalPermissions: false,
      });

      expect(result.ok).toBe(false);
      expect(result.missingRequired).toContain('role "postgres_ai_mon" does not exist');
    });
  });
});
