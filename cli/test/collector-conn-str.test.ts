import { expect, test } from "bun:test";
import * as instances from "../lib/instances";

for (const scheme of ["postgres", "postgresql"]) {
  const base = `${scheme}://u:p%40ss@EXAMPLE.com:5432/db`;
  test.each([
    ["", "", false],
    ["?sslmode=require", "?sslmode=require", false],
    ["?channel_binding=require", "", true],
    ["?channel_binding=require&sslmode=require", "?sslmode=require", true],
    ["?sslmode=require&channel_binding=require", "?sslmode=require", true],
    ["?sslmode=require&channel_binding=require&application_name=a%20b", "?sslmode=require&application_name=a%20b", true],
    ["?application_name=a+b%2f~&channel_binding=require#fragment", "?application_name=a+b%2f~#fragment", true],
    ["?channel_binding=&sslmode=require&channel_binding=prefer", "?sslmode=require", true],
    ["?channel%5Fbinding=require&sslmode=require", "?sslmode=require", true],
    ["?application_name=channel_binding%3Drequire", "?application_name=channel_binding%3Drequire", false],
  ])(`${scheme} collector URL: %s`, (query, expected, dropped) => {
    expect(instances.collectorConnStr(base + query)).toEqual({
      connStr: base + expected,
      droppedChannelBinding: dropped,
    });
  });
}

test.each(["not a URL", "postgres://u:p@[bad/db?channel_binding=require"])(
  "malformed collector URL is unchanged: %s", (connStr) => {
    expect(instances.collectorConnStr(connStr)).toEqual({ connStr, droppedChannelBinding: false });
  },
);
