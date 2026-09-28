// Strips the recording account's identifiers from fixture text. The poller
// reads none of these fields, and the fixtures live in a public repository.
export function redact(json: string): string {
  return json
    .replace(/"Address": "[^"]+"/g, '"Address": "redacted"')
    .replace(/(arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:)\d{12}/g, '$1123456789012')
    .replace(/\b(vpc|subnet|sg)-[0-9a-f]{8,17}\b/g, '$1-redacted')
    .replace(/(key\/)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g, '$1redacted')
}
