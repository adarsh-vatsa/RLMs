#!/usr/bin/env node
import { Codex, type ModelReasoningEffort, type ThreadOptions } from "@openai/codex-sdk";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");

const DEFAULT_SOURCE_CSV = path.join(REPO_ROOT, "benchmark_data/long_bench_v2/data.csv");
const DEFAULT_OUTPUT_CSV = path.join(REPO_ROOT, "benchmark_data/long_bench_v2/data_knowledge_codex.csv");
const DEFAULT_AUDIT_PATH = path.join(REPO_ROOT, "benchmark_data/long_bench_v2/data_knowledge_codex_audit.jsonl");
const REASONING_EFFORTS: ModelReasoningEffort[] = ["minimal", "low", "medium", "high", "xhigh"];
const CSV_COLUMNS = [
  "case_id",
  "source_id",
  "row_type",
  "is_scored",
  "setup_case_id",
  "context_id",
  "token_count",
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

const SETUP_INSTRUCTIONS = `You are generating one setup question for a LongBench-v2 cache workload.

The setup question will be asked before scored questions that use the same hidden long context. Its job is to make the system extract reusable knowledge from that context. You are not answering the setup question.

Rules:
- Write exactly one open-ended setup question.
- The setup question must be useful for answering later detailed questions from the same context.
- Do not include answer choices, labels like "A.", "B.", "C.", "D.", or a correct answer.
- Do not mention cache, benchmark, row_type, setup row, or metadata.
- Do not ask a yes/no question.
- Do not ask for every detail in the context; ask for compact reusable facts, rules, entities, reasoning state, or task patterns.
- Use the domain, subdomain, and representative questions to tailor the setup question.
- Do not read files, write files, run commands, or inspect the repository.

Return strict JSON only:
{
  "setup_question": "...",
  "status": "ok",
  "reason": ""
}`;

type CsvRow = Record<string, string>;

type Args = {
  sourceCsv: string;
  outputCsv: string;
  auditPath: string;
  model: string | null;
  reasoningEffort: ModelReasoningEffort | null;
  limitContexts: number | null;
  delayMs: number;
  maxRetries: number;
  force: boolean;
};

type CsvData = {
  header: string[];
  rows: CsvRow[];
};

type ContextGroup = {
  contextId: string;
  rows: CsvRow[];
};

type SetupResult = {
  question: string;
  status: string;
  reason: string;
  rawResponse: string;
};

function usage(): string {
  return `Generate Codex SDK knowledge setup rows for LongBench-v2.

Usage:
  npm run generate-knowledge-rows -- [options]

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
  --limit-contexts N            Generate only the first N unique original contexts.
  --delay-ms N                  Delay between Codex calls. Default: 0
  --max-retries N               Retries per context after parse/validation errors. Default: 1
  --force                       Overwrite output/audit files if they already exist.
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
    limitContexts: null,
    delayMs: 0,
    maxRetries: 1,
    force: false,
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
    } else if (flag === "--limit-contexts") {
      args.limitContexts = parsePositiveInt(next(), flag);
    } else if (flag === "--delay-ms") {
      args.delayMs = parseNonNegativeInt(next(), flag);
    } else if (flag === "--max-retries") {
      args.maxRetries = parseNonNegativeInt(next(), flag);
    } else if (flag === "--force") {
      args.force = true;
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
  for (const column of ["case_id", "source_id", "row_type", "context_id", "question", "choice_A", "choice_B", "choice_C", "choice_D", "answer"]) {
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

function originalContextGroups(rows: CsvRow[], limitContexts: number | null): ContextGroup[] {
  const groups = new Map<string, CsvRow[]>();
  for (const row of rows) {
    if (row.row_type !== "original") {
      continue;
    }
    const contextId = row.context_id;
    if (!contextId) {
      throw new Error(`Original row ${row.case_id} is missing context_id`);
    }
    if (!groups.has(contextId)) {
      if (limitContexts !== null && groups.size >= limitContexts) {
        continue;
      }
      groups.set(contextId, []);
    }
    groups.get(contextId)?.push(row);
  }
  return [...groups.entries()].map(([contextId, groupRows]) => ({ contextId, rows: groupRows }));
}

function guidanceFor(domain: string, subdomain: string): string {
  const label = `${domain} ${subdomain}`.toLowerCase();
  if (label.includes("detective")) {
    return "Ask for a compact inferred true-story summary, including culprit, motive, method, timeline, alibi, and physical clues.";
  }
  if (label.includes("translation") || label.includes("language")) {
    return "Ask for reusable vocabulary, grammar rules, sentence patterns, and translation examples needed for later translations.";
  }
  if (label.includes("code")) {
    return "Ask for repository structure, relevant files, APIs, functions, control flow, and dependencies needed for implementation questions.";
  }
  if (label.includes("agent history") || label.includes("dialogue")) {
    return "Ask for user preferences, entities, events, constraints, unresolved facts, and timeline details needed for later questions.";
  }
  if (label.includes("knowledge graph") || label.includes("structured data")) {
    return "Ask for entities, relationships, schema patterns, constraints, and reasoning rules needed for later structured-data questions.";
  }
  if (label.includes("many-shot") || label.includes("in-context")) {
    return "Ask for the task pattern shown by examples, including input format, output format, labels, decision rules, and edge cases.";
  }
  return "Ask for key facts, entities, relationships, constraints, and reasoning steps needed to answer later detailed questions.";
}

function buildPrompt(group: ContextGroup): string {
  const first = group.rows[0];
  const representativeQuestions = group.rows.slice(0, 4).map((row, index) => {
    return `${index + 1}. ${truncate(row.question, 500)}`;
  });
  return `${SETUP_INSTRUCTIONS}

Domain: ${first.domain}
Subdomain: ${first.sub_domain}
Difficulty: ${first.difficulty}
Length bucket: ${first.length}
Number of scored questions sharing this context: ${group.rows.length}

Domain-specific guidance:
${guidanceFor(first.domain, first.sub_domain)}

Representative scored questions from this context:
${representativeQuestions.join("\n")}

Generate the setup question now.`;
}

function truncate(value: string, maxLength: number): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength - 3)}...`;
}

async function generateSetupQuestion(codex: Codex, prompt: string, threadOptions: ThreadOptions): Promise<SetupResult> {
  const thread = codex.startThread(threadOptions);
  const result = await thread.run(prompt);
  const rawResponse = extractText(result);
  const parsed = parseJsonObject(rawResponse);
  const question = String(parsed.setup_question ?? "").trim();
  const status = String(parsed.status ?? "ok").trim() || "ok";
  const reason = String(parsed.reason ?? "").trim();
  if (!question) {
    throw new Error("Codex returned an empty setup_question");
  }
  return { question, status, reason, rawResponse };
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
  for (const key of ["finalResponse", "final_response", "outputText", "output_text", "response", "text", "message", "content"]) {
    if (key in record) {
      const extracted = extractText(record[key]);
      if (extracted.trim()) {
        return extracted;
      }
    }
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

function validateSetupQuestion(question: string): string[] {
  const errors: string[] = [];
  if (/\banswer choices?\s*:/i.test(question)) {
    errors.push("setup question includes an answer choices heading");
  }
  if (new Set(findAnswerChoiceLabels(question)).size >= 2) {
    errors.push("setup question includes answer-choice labels");
  }
  if (/\b(?:correct\s+)?answer\s*(?:is|:)\s*[A-D]\b/i.test(question)) {
    errors.push("setup question reveals an answer label");
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

function setupRow(group: ContextGroup, setupQuestion: string): CsvRow {
  const first = group.rows[0];
  const setupCaseId = `${group.contextId}__setup`;
  return {
    case_id: setupCaseId,
    source_id: setupCaseId,
    row_type: "setup",
    is_scored: "false",
    setup_case_id: "",
    context_id: group.contextId,
    expected_cache_type: "miss",
    expected_from_cache: "false",
    depends_on_case_id: "",
    domain: first.domain,
    sub_domain: first.sub_domain,
    difficulty: first.difficulty,
    length: first.length,
    question: setupQuestion,
    choice_A: "",
    choice_B: "",
    choice_C: "",
    choice_D: "",
    answer: "",
  };
}

function knowledgeRow(row: CsvRow): CsvRow {
  return {
    ...row,
    case_id: `${row.source_id}__knowledge`,
    row_type: "knowledge",
    is_scored: "true",
    setup_case_id: `${row.context_id}__setup`,
    expected_cache_type: "knowledge",
    expected_from_cache: "true",
    depends_on_case_id: `${row.context_id}__setup`,
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function writeAudit(auditPath: string, records: unknown[]): Promise<void> {
  await mkdir(path.dirname(auditPath), { recursive: true });
  await writeFile(auditPath, records.map((record) => JSON.stringify(record)).join("\n") + "\n", "utf-8");
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  await assertCanWrite(args.outputCsv, args.force);
  await assertCanWrite(args.auditPath, args.force);

  const source = await loadCsv(args.sourceCsv);
  const groups = originalContextGroups(source.rows, args.limitContexts);
  const outputRows: CsvRow[] = [];
  const auditRecords: unknown[] = [];
  const codex = new Codex();
  const threadOptions: ThreadOptions = {};
  if (args.model) {
    threadOptions.model = args.model;
  }
  if (args.reasoningEffort) {
    threadOptions.modelReasoningEffort = args.reasoningEffort;
  }

  for (let groupIndex = 0; groupIndex < groups.length; groupIndex += 1) {
    const group = groups[groupIndex];
    const prompt = buildPrompt(group);
    let result: SetupResult | null = null;
    let errorMessage = "";
    for (let attempt = 0; attempt <= args.maxRetries; attempt += 1) {
      try {
        result = await generateSetupQuestion(codex, prompt, threadOptions);
        const validationErrors = validateSetupQuestion(result.question);
        if (validationErrors.length > 0) {
          throw new Error(validationErrors.join("; "));
        }
        break;
      } catch (error) {
        errorMessage = error instanceof Error ? error.message : String(error);
        if (attempt < args.maxRetries) {
          await sleep(Math.max(args.delayMs, 250));
        }
      }
    }
    if (!result) {
      throw new Error(`Failed to generate setup question for context ${group.contextId}: ${errorMessage}`);
    }

    outputRows.push(setupRow(group, result.question));
    for (const row of group.rows) {
      outputRows.push(knowledgeRow(row));
    }
    auditRecords.push({
      group_index: groupIndex,
      context_id: group.contextId,
      setup_case_id: `${group.contextId}__setup`,
      source_ids: group.rows.map((row) => row.source_id),
      domain: group.rows[0].domain,
      sub_domain: group.rows[0].sub_domain,
      status: result.status,
      reason: result.reason,
      setup_question: result.question,
      model: args.model,
      reasoning_effort: args.reasoningEffort,
      raw_response: result.rawResponse,
    });

    console.log(`[LONGBENCH-V2] ${groupIndex + 1} setup context(s): ${group.contextId} (${group.rows.length} knowledge row(s))`);
    if (args.delayMs > 0) {
      await sleep(args.delayMs);
    }
  }

  await mkdir(path.dirname(args.outputCsv), { recursive: true });
  await writeFile(args.outputCsv, stringifyCsv(CSV_COLUMNS, outputRows), "utf-8");
  await writeAudit(args.auditPath, auditRecords);

  console.log(`[LONGBENCH-V2] Wrote ${outputRows.length} rows to ${relative(args.outputCsv)}`);
  console.log(`[LONGBENCH-V2] Wrote ${auditRecords.length} audit rows to ${relative(args.auditPath)}`);
}

main().catch((error) => {
  console.error(`[LONGBENCH-V2] ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
