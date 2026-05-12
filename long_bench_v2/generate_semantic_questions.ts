#!/usr/bin/env node
import { Codex, type ModelReasoningEffort, type ThreadOptions } from "@openai/codex-sdk";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");

const DEFAULT_SOURCE_CSV = path.join(REPO_ROOT, "benchmark_data/long_bench_v2/data.csv");
const DEFAULT_OUTPUT_CSV = path.join(REPO_ROOT, "benchmark_data/long_bench_v2/data_semantic_codex.csv");
const DEFAULT_AUDIT_PATH = path.join(REPO_ROOT, "benchmark_data/long_bench_v2/data_semantic_codex_audit.jsonl");
const REASONING_EFFORTS: ModelReasoningEffort[] = ["minimal", "low", "medium", "high", "xhigh"];
const CSV_COLUMNS = [
  "case_id",
  "source_id",
  "row_type",
  "is_scored",
  "setup_case_id",
  "context_id",
  "expected_cache_type",
  "expected_from_cache",
  "depends_on_case_id",
  "domain",
  "sub_domain",
  "difficulty",
  "length",
  "question",
  "choice_A",
  "choice_B",
  "choice_C",
  "choice_D",
  "answer",
];

const REWRITER_INSTRUCTIONS = `You are rewriting benchmark questions for LongBench-v2 semantic cache evaluation.

Your task is to rewrite only the question text so it asks for the same answer as the original question, but with different wording.

Rules:
- Preserve the exact meaning, required reasoning, and correct answer.
- Preserve all names, dates, numbers, quoted strings, formulas, code identifiers, file names, legal terms, technical terms, and target sentences unless changing them is required only for grammar.
- If the question refers to multiple-choice options, keep that reference in the question text, but do not include the answer choices themselves.
- Do not make the question easier or harder.
- Do not add facts that are not in the original question.
- Do not reveal or hint at the correct answer.
- Do not include "Answer choices", "A.", "B.", "C.", "D.", or any answer-option text in rewritten_question.
- Do not read files, write files, run commands, or inspect the repository.
- Do not explain your rewrite.
- If a safe rewrite is not possible, return the original question unchanged.

Return strict JSON only:
{
  "rewritten_question": "...",
  "status": "ok" | "unchanged",
  "reason": "short reason if unchanged"
}`;

type CsvRow = Record<string, string>;

type Args = {
  sourceCsv: string;
  outputCsv: string;
  auditPath: string;
  model: string | null;
  reasoningEffort: ModelReasoningEffort | null;
  startIndex: number;
  limit: number | null;
  delayMs: number;
  maxRetries: number;
  force: boolean;
  keepExistingOnError: boolean;
};

type CsvData = {
  header: string[];
  rows: CsvRow[];
};

type RewriteResult = {
  question: string;
  status: string;
  reason: string;
  rawResponse: string;
  warnings: string[];
};

function usage(): string {
  return `Generate Codex SDK semantic rewrites for LongBench-v2 CSV rows.

Usage:
  npm run generate-semantic-questions -- [options]

Options:
  --source-csv PATH             CSV containing original rows keyed by source_id.
                                Default: ${relative(DEFAULT_SOURCE_CSV)}
  --output-csv PATH             Output CSV to write.
                                Default: ${relative(DEFAULT_OUTPUT_CSV)}
  --audit-path PATH             JSONL audit log path.
                                Default: ${relative(DEFAULT_AUDIT_PATH)}
  --model MODEL                 Codex model to use. Default: Codex SDK/CLI default.
  --reasoning-effort EFFORT     Reasoning effort: ${REASONING_EFFORTS.join(", ")}.
                                Default: Codex SDK/CLI default.
  --start-index N               Zero-based input row index to start at. Default: 0
  --limit N                     Maximum number of semantic rows to request from Codex.
  --delay-ms N                  Delay between Codex calls. Default: 0
  --max-retries N               Retries per row after a parse/validation error. Default: 1
  --force                       Overwrite output/audit files if they already exist.
  --no-keep-existing-on-error   Fail the run instead of keeping the original question on row errors.
  --help                        Show this help.
`;
}

function relative(filePath: string): string {
  return path.relative(REPO_ROOT, filePath);
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    sourceCsv: DEFAULT_SOURCE_CSV,
    outputCsv: DEFAULT_OUTPUT_CSV,
    auditPath: DEFAULT_AUDIT_PATH,
    model: null,
    reasoningEffort: null,
    startIndex: 0,
    limit: null,
    delayMs: 0,
    maxRetries: 1,
    force: false,
    keepExistingOnError: true,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const next = () => {
      const value = argv[i + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error(`Missing value for ${flag}`);
      }
      i += 1;
      return value;
    };

    if (flag === "--help") {
      console.log(usage());
      process.exit(0);
    } else if (flag === "--source-csv") {
      args.sourceCsv = path.resolve(next());
    } else if (flag === "--output-csv") {
      args.outputCsv = path.resolve(next());
    } else if (flag === "--audit-path") {
      args.auditPath = path.resolve(next());
    } else if (flag === "--model") {
      args.model = next().trim();
      if (!args.model) {
        throw new Error("--model cannot be empty");
      }
    } else if (flag === "--reasoning-effort") {
      args.reasoningEffort = parseReasoningEffort(next());
    } else if (flag === "--start-index") {
      args.startIndex = parseNonNegativeInt(next(), flag);
    } else if (flag === "--limit") {
      args.limit = parsePositiveInt(next(), flag);
    } else if (flag === "--delay-ms") {
      args.delayMs = parseNonNegativeInt(next(), flag);
    } else if (flag === "--max-retries") {
      args.maxRetries = parseNonNegativeInt(next(), flag);
    } else if (flag === "--force") {
      args.force = true;
    } else if (flag === "--no-keep-existing-on-error") {
      args.keepExistingOnError = false;
    } else {
      throw new Error(`Unknown option: ${flag}`);
    }
  }

  return args;
}

function parseReasoningEffort(value: string): ModelReasoningEffort {
  if (REASONING_EFFORTS.includes(value as ModelReasoningEffort)) {
    return value as ModelReasoningEffort;
  }
  throw new Error(`--reasoning-effort must be one of: ${REASONING_EFFORTS.join(", ")}`);
}

function parseNonNegativeInt(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${flag} must be a non-negative integer`);
  }
  return parsed;
}

function parsePositiveInt(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${flag} must be a positive integer`);
  }
  return parsed;
}

async function assertCanWrite(filePath: string, force: boolean): Promise<void> {
  if (force) {
    return;
  }
  try {
    await access(filePath);
  } catch {
    return;
  }
  throw new Error(`${relative(filePath)} already exists. Use --force to overwrite it.`);
}

async function loadCsv(filePath: string): Promise<CsvData> {
  const text = await readFile(filePath, "utf-8");
  const rows = parseCsv(text.replace(/^\uFEFF/, ""));
  if (rows.length === 0) {
    throw new Error(`${relative(filePath)} is empty`);
  }
  const [header, ...records] = rows;
  validateColumns(header, filePath);

  return {
    header,
    rows: records.map((record, index) => {
      if (record.length !== header.length) {
        throw new Error(
          `${relative(filePath)} row ${index + 2} has ${record.length} columns; expected ${header.length}`,
        );
      }
      return normalizeRow(Object.fromEntries(header.map((column, columnIndex) => [column, record[columnIndex] ?? ""])));
    }),
  };
}

function validateColumns(header: string[], filePath: string): void {
  for (const column of ["case_id", "source_id", "row_type", "question", "choice_A", "choice_B", "choice_C", "choice_D", "answer"]) {
    if (!header.includes(column)) {
      throw new Error(`${relative(filePath)} is missing required column: ${column}`);
    }
  }
}

function normalizeRow(row: CsvRow): CsvRow {
  const normalized = { ...row };
  normalized.is_scored ||= normalized.row_type === "setup" ? "false" : "true";
  normalized.setup_case_id ||= "";
  return normalized;
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const nextChar = text[i + 1];

    if (inQuotes) {
      if (char === '"' && nextChar === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char === "\r") {
      if (nextChar === "\n") {
        i += 1;
      }
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (inQuotes) {
    throw new Error("CSV ended inside a quoted field");
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function stringifyCsv(header: string[], rows: CsvRow[]): string {
  const lines = [header.map(escapeCsv).join(",")];
  for (const row of rows) {
    lines.push(header.map((column) => escapeCsv(row[column] ?? "")).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function escapeCsv(value: string): string {
  if (/[",\r\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function buildOriginalBySourceId(rows: CsvRow[]): Map<string, CsvRow> {
  const originals = new Map<string, CsvRow>();
  for (const row of rows) {
    const sourceId = row.source_id;
    if (!sourceId) {
      continue;
    }
    if (row.row_type === "original" || !originals.has(sourceId)) {
      originals.set(sourceId, row);
    }
  }
  return originals;
}

function buildSemanticRows(rows: CsvRow[]): CsvRow[] {
  const semanticRows = rows.filter((row) => row.row_type === "semantic");
  if (semanticRows.length > 0) {
    return semanticRows.map((row) => ({ ...row }));
  }

  const originalRows = rows.filter((row) => row.row_type === "original");
  if (originalRows.length === 0) {
    throw new Error("Input CSV must contain either semantic rows or original rows");
  }

  return originalRows.map((row) => ({
    ...row,
    case_id: `${row.source_id}__semantic`,
    row_type: "semantic",
    is_scored: "true",
    setup_case_id: "",
    expected_cache_type: "semantic",
    expected_from_cache: "true",
    depends_on_case_id: `${row.source_id}__original`,
  }));
}

function buildPrompt(original: CsvRow): string {
  return `${REWRITER_INSTRUCTIONS}

Original question:
${JSON.stringify(original.question)}

Answer choices for preservation only. Do not include these choices in rewritten_question:
A. ${original.choice_A}
B. ${original.choice_B}
C. ${original.choice_C}
D. ${original.choice_D}

Correct answer label from the dataset, for preservation only:
${JSON.stringify(original.answer)}

Rewrite the original question now.`;
}

async function rewriteWithCodex(codex: Codex, prompt: string, threadOptions: ThreadOptions): Promise<RewriteResult> {
  const thread = codex.startThread(threadOptions);
  const result = await thread.run(prompt);
  const rawResponse = extractText(result);
  const parsed = parseJsonObject(rawResponse);
  const rewritten = String(parsed.rewritten_question ?? "").trim();
  const status = String(parsed.status ?? "ok").trim() || "ok";
  const reason = String(parsed.reason ?? "").trim();

  if (!rewritten) {
    throw new Error("Codex returned an empty rewritten_question");
  }

  return {
    question: rewritten,
    status,
    reason,
    rawResponse,
    warnings: [],
  };
}

function extractText(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value === null || value === undefined) {
    return "";
  }
  if (Array.isArray(value)) {
    return value.map(extractText).filter(Boolean).join("\n");
  }
  if (typeof value !== "object") {
    return String(value);
  }

  const record = value as Record<string, unknown>;
  for (const key of [
    "finalResponse",
    "final_response",
    "outputText",
    "output_text",
    "response",
    "text",
    "message",
    "content",
  ]) {
    if (key in record) {
      const extracted = extractText(record[key]);
      if (extracted.trim()) {
        return extracted;
      }
    }
  }

  if (Array.isArray(record.messages) && record.messages.length > 0) {
    return extractText(record.messages[record.messages.length - 1]);
  }

  return JSON.stringify(value);
}

function parseJsonObject(rawResponse: string): Record<string, unknown> {
  const trimmed = rawResponse.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "");
  try {
    return JSON.parse(trimmed);
  } catch {
    const start = trimmed.indexOf("{");
    const end = trimmed.lastIndexOf("}");
    if (start >= 0 && end > start) {
      return JSON.parse(trimmed.slice(start, end + 1));
    }
    throw new Error(`Could not parse Codex response as JSON: ${trimmed.slice(0, 200)}`);
  }
}

function validateRewrite(originalQuestion: string, rewrittenQuestion: string): string[] {
  const warnings: string[] = [];
  if (rewrittenQuestion === originalQuestion) {
    warnings.push("rewritten question is identical to the original question");
  }
  for (const token of importantTokens(originalQuestion)) {
    if (!rewrittenQuestion.includes(token)) {
      warnings.push(`missing preserved token: ${token}`);
    }
  }
  return warnings;
}

function validateRewriteForRetry(original: CsvRow, rewrittenQuestion: string): string[] {
  const errors: string[] = [];
  if (/\banswer choices?\s*:/i.test(rewrittenQuestion)) {
    errors.push("rewritten question includes an answer choices heading");
  }
  const answerChoiceLabels = findAnswerChoiceLabels(rewrittenQuestion);
  if (new Set(answerChoiceLabels).size >= 2) {
    errors.push(`rewritten question includes answer-choice labels: ${answerChoiceLabels.join(", ")}`);
  }
  if (/\b(?:correct\s+)?answer\s*(?:is|:)\s*[A-D]\b/i.test(rewrittenQuestion)) {
    errors.push("rewritten question reveals an answer label");
  }
  for (const column of ["choice_A", "choice_B", "choice_C", "choice_D"]) {
    const choice = (original[column] ?? "").trim();
    if (choice && choice.length > 20 && rewrittenQuestion.includes(choice)) {
      errors.push(`rewritten question includes ${column}`);
    }
  }
  return errors;
}

function findAnswerChoiceLabels(text: string): string[] {
  const labels: string[] = [];
  const labelPattern = /(?:^|[\n\r\t (])([A-D])[\.\):]\s+\S/g;
  for (const match of text.matchAll(labelPattern)) {
    labels.push(match[1]);
  }
  return labels;
}

function importantTokens(question: string): string[] {
  const tokens = new Set<string>();
  for (const match of question.matchAll(/"([^"]+)"/g)) {
    if (match[1].trim().length > 1) {
      tokens.add(match[1].trim());
    }
  }
  for (const match of question.matchAll(/\b\d[\w./:-]*\b/g)) {
    tokens.add(match[0]);
  }
  return [...tokens];
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function appendAudit(auditPath: string, records: unknown[]): Promise<void> {
  if (records.length === 0) {
    return;
  }
  await mkdir(path.dirname(auditPath), { recursive: true });
  await writeFile(auditPath, records.map((record) => JSON.stringify(record)).join("\n") + "\n", "utf-8");
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  await assertCanWrite(args.outputCsv, args.force);
  await assertCanWrite(args.auditPath, args.force);

  const source = await loadCsv(args.sourceCsv);
  const originalsBySourceId = buildOriginalBySourceId(source.rows);
  const outputRows = buildSemanticRows(source.rows);
  const writtenRows: CsvRow[] = [];
  const auditRecords: unknown[] = [];
  const codex = new Codex();
  const threadOptions: ThreadOptions = {};
  if (args.model) {
    threadOptions.model = args.model;
  }
  if (args.reasoningEffort) {
    threadOptions.modelReasoningEffort = args.reasoningEffort;
  }

  let generatedCount = 0;
  for (let rowIndex = 0; rowIndex < outputRows.length; rowIndex += 1) {
    const row = outputRows[rowIndex];
    if (rowIndex < args.startIndex || row.row_type !== "semantic") {
      continue;
    }
    if (args.limit !== null && generatedCount >= args.limit) {
      break;
    }

    const original = originalsBySourceId.get(row.source_id);
    if (!original) {
      throw new Error(`No original row found for source_id=${row.source_id}`);
    }

    const prompt = buildPrompt(original);
    let result: RewriteResult | null = null;
    let errorMessage = "";
    for (let attempt = 0; attempt <= args.maxRetries; attempt += 1) {
      try {
        result = await rewriteWithCodex(codex, prompt, threadOptions);
        const validationErrors = validateRewriteForRetry(original, result.question);
        if (validationErrors.length > 0) {
          throw new Error(validationErrors.join("; "));
        }
        result.warnings.push(...validateRewrite(original.question, result.question));
        break;
      } catch (error) {
        errorMessage = error instanceof Error ? error.message : String(error);
        if (attempt < args.maxRetries) {
          await sleep(Math.max(args.delayMs, 250));
        }
      }
    }

    if (!result) {
      if (!args.keepExistingOnError) {
        throw new Error(`Failed to rewrite row ${rowIndex} (${row.case_id}): ${errorMessage}`);
      }
      auditRecords.push({
        row_index: rowIndex,
        case_id: row.case_id,
        source_id: row.source_id,
        status: "error_keep_existing",
        error: errorMessage,
        original_question: original.question,
        model: args.model,
        reasoning_effort: args.reasoningEffort,
      });
    } else {
      const codexKeptOriginal = result.question === original.question || result.status === "unchanged";
      row.question = result.question;
      auditRecords.push({
        row_index: rowIndex,
        case_id: row.case_id,
        source_id: row.source_id,
        status: codexKeptOriginal ? "codex_unchanged" : result.status,
        reason: result.reason,
        warnings: result.warnings,
        original_question: original.question,
        rewritten_question: row.question,
        model: args.model,
        reasoning_effort: args.reasoningEffort,
        raw_response: result.rawResponse,
      });
    }

    generatedCount += 1;
    writtenRows.push(row);
    console.log(`[LONGBENCH-V2] ${generatedCount} semantic rewrite(s): row ${rowIndex} ${row.case_id}`);
    if (args.delayMs > 0) {
      await sleep(args.delayMs);
    }
  }

  await mkdir(path.dirname(args.outputCsv), { recursive: true });
  await writeFile(args.outputCsv, stringifyCsv(CSV_COLUMNS, writtenRows), "utf-8");
  await appendAudit(args.auditPath, auditRecords);

  console.log(`[LONGBENCH-V2] Wrote ${writtenRows.length} rows to ${relative(args.outputCsv)}`);
  console.log(`[LONGBENCH-V2] Wrote ${auditRecords.length} audit rows to ${relative(args.auditPath)}`);
}

main().catch((error) => {
  console.error(`[LONGBENCH-V2] ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
