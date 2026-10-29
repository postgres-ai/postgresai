/**
 * `pgai billing status` (postgres-ai/postgresai#462): whether the organization
 * has a payment method a deploy accepts. Read-only and safe to poll: it checks
 * no promo code and connects to no database, so an agent can watch for the
 * card the user is adding without spending the platform's promo budget.
 */

import type { Command } from "commander";
import { callDeployRpc, type ApiParams, type DeployOptions } from "./deploy";
import { billingPage, jsonOutput } from "./deploy-surface";
import type { DeployCliDeps } from "./deploy-commands";

export interface BillingStatus {
  org: string | null;
  /** null: the platform could not read Stripe just now. */
  has_payment_method: boolean | null;
  /** null when the platform named no org alias to link to. */
  billing_url: string | null;
}

/** 0 a payment method is on file, 3 the user must add one (as deploy's action_required), 1 not known. */
export function billingExit(s: BillingStatus): number {
  if (s.has_payment_method === true) return 0;
  if (s.has_payment_method === false) return 3;
  return 1;
}

/**
 * The payment method check the deploys use (org_payment_method_check), as
 * dblab_cloud_deploy_options reports it to any org member.
 */
export async function billingStatus(api: ApiParams, uiBaseUrl: string | undefined): Promise<BillingStatus> {
  const options = await callDeployRpc<DeployOptions>(api, "dblab_cloud_deploy_options", {}, "Payment method check");
  const org = options.org_alias ?? null;
  const has = options.has_payment_method;
  return {
    org,
    has_payment_method: typeof has === "boolean" ? has : null,
    billing_url: org ? billingPage(uiBaseUrl, org) : null,
  };
}

export function billingStatusLine(s: BillingStatus, uiBaseUrl?: string): string {
  const org = s.org ?? "this organization";
  const page = s.billing_url ?? billingPage(uiBaseUrl, null);
  if (s.has_payment_method === true) return `Payment method on file for ${org}.`;
  if (s.has_payment_method === false) return `No payment method on file for ${org}. Add one at ${page}`;
  return `Could not check whether ${org} has a payment method on file. Try again in a minute; `
    + `if this keeps happening, open ${page} or contact support@postgres.ai.`;
}

export function registerBillingCommands(program: Command, deps: DeployCliDeps): void {
  const billing = program.command("billing").description("check whether the organization has a payment method on file (billing status)");

  deps.withOrgOptions(billing.command("status"))
    .description("whether the organization has a payment method on file (read-only; exit 0 yes, 3 no, 1 error or could not check)")
    .option("--json", "JSON output")
    .option("--debug", "print HTTP requests (secrets masked)")
    .action(async (opts: { debug?: boolean; json?: boolean }) => {
      try {
        const ui = deps.uiBaseUrl?.();
        const status = await billingStatus(deps.resolveApi(!!opts.debug), ui);
        if (jsonOutput(opts.json)) deps.printResult(status, true);
        else console.log(billingStatusLine(status, ui));
        process.exitCode = billingExit(status);
      } catch (err) {
        // Every refusal is an error here, a 402 too: it can mean a suspended
        // organization, not a missing payment method.
        console.error(err instanceof Error ? err.message : String(err));
        process.exitCode = 1;
      }
    });
}
