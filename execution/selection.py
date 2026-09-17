from execution.tokens import chat_token_count


def add_source_arguments(parser):
    parser.add_argument("--min-source-tokens", type=int)
    parser.add_argument("--max-source-tokens", type=int)


def filter_sources(rows, args, tokenizer, task_factory):
    minimum, maximum = args.min_source_tokens, args.max_source_tokens
    if minimum is None and maximum is None:
        return rows
    if minimum is None or maximum is None or not 0 < minimum <= maximum:
        raise ValueError("Source bounds require positive minimum <= maximum")
    selected = [row for row in rows if minimum <= chat_token_count(tokenizer, task_factory(row).render(None)) <= maximum]
    if args.max_rows:
        selected = selected[:args.max_rows]
    if not selected:
        raise ValueError("No examples matched the requested source bounds")
    return selected


def selection_limit(args):
    return 0 if (getattr(args, "min_source_tokens", None) is not None
                 or getattr(args, "max_source_tokens", None) is not None) else args.max_rows
