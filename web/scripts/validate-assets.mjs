import { existsSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const casesPath = path.join(root, 'src', 'data', 'cases.json');
const cases = JSON.parse(readFileSync(casesPath, 'utf8'));
const errors = [];

if (!Array.isArray(cases) || cases.length !== 7) {
  errors.push(`Expected exactly 7 showcase QC cases, found ${Array.isArray(cases) ? cases.length : 'invalid JSON'}.`);
}

for (const record of cases) {
  const publicPath = record?.assets?.contactSheet;
  if (typeof publicPath !== 'string' || !publicPath.startsWith('/lidc/')) {
    errors.push(`${record?.caseId ?? 'unknown case'}: invalid contactSheet path.`);
    continue;
  }

  const diskPath = path.join(root, 'public', publicPath.slice(1));
  if (!existsSync(diskPath)) {
    errors.push(`${record.caseId}: missing ${publicPath}.`);
    continue;
  }
  if (statSync(diskPath).size < 10_000) {
    errors.push(`${record.caseId}: ${publicPath} is unexpectedly small.`);
  }
}

const provenancePath = path.join(root, 'public', 'lidc', 'provenance.json');
if (!existsSync(provenancePath)) errors.push('Missing /lidc/provenance.json.');

if (errors.length) {
  console.error('Showcase asset validation failed:');
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`Validated ${cases.length} real QC contact sheets and provenance metadata.`);
