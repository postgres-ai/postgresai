import { describe, test, expect } from "bun:test";
import { parseFaultyContainers, collectFaultyContainers } from "../lib/collectors/containers";

describe("parseFaultyContainers", () => {
  test("returns [] on empty input", () => {
    expect(parseFaultyContainers("")).toEqual([]);
  });

  test("flags exited containers", () => {
    const dump = [
      JSON.stringify({ Names: "postgres", Status: "Up 3 hours", State: "running" }),
      JSON.stringify({ Names: "cadvisor", Status: "Exited (137) 5 minutes ago", State: "exited" }),
    ].join("\n");
    expect(parseFaultyContainers(dump)).toEqual(["cadvisor"]);
  });

  test("flags unhealthy containers via Status string", () => {
    const dump = [
      JSON.stringify({ Names: "postgres", Status: "Up 3 hours (unhealthy)", State: "running" }),
      JSON.stringify({ Names: "node-exporter", Status: "Up 1 day (healthy)", State: "running" }),
    ].join("\n");
    expect(parseFaultyContainers(dump)).toEqual(["postgres"]);
  });

  test("flags unhealthy containers via Health field (newer docker)", () => {
    const dump = [
      JSON.stringify({ Names: "redis", Status: "Up 30 minutes", State: "running", Health: "unhealthy" }),
      JSON.stringify({ Names: "nginx", Status: "Up 2 hours", State: "running", Health: "healthy" }),
    ].join("\n");
    expect(parseFaultyContainers(dump)).toEqual(["redis"]);
  });

  test("flags Restarting containers (flapping)", () => {
    const dump = [
      JSON.stringify({ Names: "flappy", Status: "Restarting (1) 10 seconds ago", State: "restarting" }),
    ].join("\n");
    expect(parseFaultyContainers(dump)).toEqual(["flappy"]);
  });

  test("dedups when the same name appears more than once", () => {
    const dump = [
      JSON.stringify({ Names: "dup", Status: "Exited (1) 10 seconds ago", State: "exited" }),
      JSON.stringify({ Names: "dup", Status: "Restarting", State: "restarting" }),
    ].join("\n");
    expect(parseFaultyContainers(dump)).toEqual(["dup"]);
  });

  test("ignores malformed JSON lines without throwing", () => {
    const dump = [
      "not-json-at-all",
      JSON.stringify({ Names: "c1", Status: "Exited (0) 1 hour ago", State: "exited" }),
    ].join("\n");
    expect(parseFaultyContainers(dump)).toEqual(["c1"]);
  });

  test("returns [] when all containers are healthy", () => {
    const dump = [
      JSON.stringify({ Names: "a", Status: "Up 1 hour", State: "running" }),
      JSON.stringify({ Names: "b", Status: "Up 2 hours (healthy)", State: "running" }),
    ].join("\n");
    expect(parseFaultyContainers(dump)).toEqual([]);
  });
});

describe("collectFaultyContainers", () => {
  test("invokes docker ps -a --format json and parses output", async () => {
    let receivedArgs: string[] = [];
    const exec = async (args: string[]) => {
      receivedArgs = args;
      return JSON.stringify({ Names: "broken", Status: "Exited (1) 1 hour ago", State: "exited" });
    };
    const result = await collectFaultyContainers({ exec });
    expect(result).toEqual(["broken"]);
    expect(receivedArgs).toEqual(["ps", "-a", "--format", "{{json .}}"]);
  });

  test("returns [] when docker exec rejects", async () => {
    const exec = async () => {
      throw new Error("docker: not found");
    };
    const result = await collectFaultyContainers({ exec });
    expect(result).toEqual([]);
  });
});
