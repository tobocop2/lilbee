# Crawl parity harness

This tool decides, by measurement, whether lilbee's crawler on a candidate backend is at least as good as it is on the oracle backend. It also measures each side against the text a reader sees in a browser, so it keeps working when no oracle is left.

It is a developer tool. It is not part of the lilbee package and it adds no dependency to lilbee.

## Status

The pipeline runs end to end on the synthetic corpus. These parts are not built yet: the recorder and the recorded corpus, the page generator and the minimiser, the `env` command that installs an unreleased crawlberg, and the CI job.

## Set up

Create the harness's own environment. It holds no crawler and no lilbee.

```bash
uv venv .venv-parity
uv pip install --python .venv-parity -r tools/qa/crawl_parity/requirements.txt
```

The ground truth needs the Chromium build of Playwright 1.61.0. If it is not installed, run `.venv-parity/bin/playwright install chromium`.

Write a `sides.toml`. Each side names one Python interpreter for each layer it can run. A layer with no interpreter is not run.

```toml
[oracle]
label = "lilbee on crawl4ai 0.9.2"
lilbee = "/path/to/lilbee-oracle/.venv/bin/python"     # a lilbee checkout with crawl4ai
crawler = "/path/to/lilbee-oracle/.venv/bin/python"    # any environment with crawl4ai
converter = "/path/to/lilbee-oracle/.venv/bin/python"
crawler_driver = "crawl4ai_crawl.py"
converter_driver = "crawl4ai_convert.py"

[candidate]
label = "lilbee on crawlberg 1.10.2"
lilbee = "/path/to/lilbee-candidate/.venv/bin/python"  # a lilbee checkout with crawlberg
crawler = "/path/to/crawlberg-env/bin/python"          # crawlberg alone
converter = "/path/to/h2m-env/bin/python"              # html-to-markdown alone, at crawlberg's version
crawler_driver = "crawlberg_crawl.py"
converter_driver = "h2m_convert.py"
chrome = "/path/to/chrome-headless-shell"              # crawlberg alone needs it in browser mode

[retrieval]
python = "/path/to/lilbee-candidate/.venv/bin/python"  # the lilbee that indexes both sides
embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
```

Remove the `[oracle]` table to judge the candidate against the ground truth alone.

## Commands

Run every command from the repository root with the harness's interpreter.

```bash
python -m tools.qa.crawl_parity selftest --sides sides.toml --work /tmp/parity-selftest
python -m tools.qa.crawl_parity run --sides sides.toml --work /tmp/parity-work --report /tmp/parity-report
python -m tools.qa.crawl_parity judge --report /tmp/parity-report
```

`run` prints one line that starts with `PASS` or `FAIL` and exits 0 or 1. It writes `report.md`, `differences.jsonl`, `pages.tsv` and `coverage.json` into the report directory.

`judge` sets a saved run against `expected.toml` again and writes `verdict.md` beside the report. Use it after you add or remove an entry: it makes no crawl.

| option of `run` | meaning |
|---|---|
| `--modes http,browser` | the render modes to crawl in |
| `--seeds f,b` | the seeds to crawl; the default is every seed |
| `--skip speed,retrieval` | yardsticks to leave out: `parity`, `retrieval`, `left-behind`, `speed` |
| `--browser firefox` | the ground truth browser, when the check must not share Chromium with both crawlers |
| `--thresholds`, `--expected` | other files than the two in this directory |
| `--sample-seed 7` | the seed of the question sample for retrieval |

## How a run works

1. A replay server on 127.0.0.1 serves the corpus. No crawler reaches a real site. A request for an address that is not in the corpus gets status 599, and the report counts it.
2. A real browser loads each replayed page. Its rendered text is the ground truth.
3. Each side crawls each seed at each layer, in its own process, with its own temp directory.
4. The yardsticks turn the output into differences. Each difference has a signature.
5. The verdict sets each signature against `expected.toml` and `thresholds.toml`.

## Layers

A difference is reported at the lowest layer that shows it, so the report says whose defect it is.

| layer | oracle | candidate | input |
|---|---|---|---|
| `converter` | crawl4ai's HTML cleaning and markdown generator | html-to-markdown | HTML text |
| `crawler` | crawl4ai alone | crawlberg alone | a replay URL |
| `lilbee` | `lilbee add <url> --crawl` | the same | a replay URL |

The `lilbee` driver makes two changes on both sides: it allows loopback addresses, and it replaces the sync after the crawl with one that does nothing. The `crawler` drivers use the settings that lilbee's adapter passes.

The verdict is about the highest layer that both sides have.

## Yardsticks

**A, parity.** For the page set: each page that only one side saved, with the reason. For each page both saved: word tokens lost and added, symbol tokens lost and added, and the structure signature (headings, links with targets, code blocks, tables with row and cell counts, list items).

**B, ground truth.** For each side: the share of the visible words that its markdown holds, and the words it holds that the page does not show (hidden in the page, attribute text, or unexplained). B decides which side is wrong when the two differ:

- A lost word counts when the page shows it more often than the candidate holds it.
- A lost word that the page does not show is reported as `NOT-COUNTED`, unless `invisible_text_lost_counts` is true.
- An added word does not count when the page shows it more often than the oracle holds it. Every other added word counts.
- A page only the oracle saved does not count when a reader does not get that page (an error status, no visible text).

**C, retrieval.** The harness samples visible sentences that occur on one page only. lilbee indexes each side's markdown under equal file names and searches for each sentence. The result is recall at k for each side. One page is 1 question in `sample_size` at most, so retrieval sees a lost page only when the page holds more than `recall_drop_max` of the questions; parity is the yardstick that sees one lost page.

**Left behind.** After each crawl process ends, the harness counts the processes of that run that still live, their listening sockets, the entries in the run's temp directory, and the Python threads alive in the driver after the crawl closed. A process is the run's own only if it did not exist before the run and it is a descendant of the run, or inherited the run's temp variable, or names the run's temp directory in its command line. The harness then stops those processes and no others.

**Speed.** Pages per second and time to the first page, for `repeats` runs of each side and mode. Each number is printed with the load of the machine. If one sample was taken above `load_max`, the harness makes no comparison and the run fails as not measured.

## Normalisations

These are all the ways two texts are made comparable. The report names them, and each has a self-test that shows a real deletion still gets through.

| name | what it does |
|---|---|
| `markdown-syntax` | Markdown is parsed. Markers and link targets are not tokens. Symbol runs made only of `\|`, `:` and `-` are not tokens. |
| `unicode-nfc` | Tokens are compared in Unicode NFC. |
| `whitespace` | White space separates tokens and is not a token. |
| `replay-origin` | The origin of the replay server is written as `http://replay.test`. The port changes with each run, and a signature must not. |
| `visible-case` | Yardstick B only: a visible word held in another letter case counts as held, because a style sheet can change the case a reader sees. |

Letter case (in yardstick A), punctuation and digits are not normalised. A word is a run of letters and digits; an underscore is a symbol, because markdown uses it as a marker. The alternative text of an image is its own words, whatever touches the image. Text tokens are read without the table extension, because a table parser drops the cells of a row that is longer than its header.

## Signatures, `expected.toml` and the verdict

A signature is `<kind>/<mode>/<layer>/<feature>`, for example `text-lost/http/converter/svg>text`.

- For lost and added text, the feature is the place of the words in the rendered page: the two nearest elements, with `hidden:` in front if the page does not show them, `attribute` for attribute text, and `not-in-page:` plus up to three of the words when the page does not hold them. `repeat:` marks words that the candidate holds more often than the page shows.
- For a lost page, the feature is the failure text with addresses and numbers taken out.
- For structure, the feature is the element.

`expected.toml` holds the differences that are already filed. Each entry is a glob over signatures, an issue, and optionally a glob over pages. For each difference, the first entry that matches wins. A signature is `KNOWN` only when every one of its differences that counts has an entry.

| status | meaning | fails the run |
|---|---|---|
| `NEW` | no entry matches | yes |
| `KNOWN` | an entry matches | yes, until the difference is gone |
| `ACCEPTED` | an entry with `accepted = true` matches; only the owner sets this | no |
| `NOT-COUNTED` | the ground truth says the candidate matches what a reader sees | no |
| `FIXED` | an entry that no difference matched, after a run of the whole corpus that measured its kind and mode; remove the entry | no |
| `NOT-RUN` | an entry whose kind or mode the run did not measure | no |

A run also fails when a requested yardstick gave no measurement.

`thresholds.toml` holds every limit, each with a comment. Every value is marked PROPOSED until the owner sets it.

## Self-tests

`selftest` runs the reference side (the oracle, or the candidate if there is no oracle) twice on the `f` seed and plants known defects in that real output.

- Controls: the reference against its own second run gives no difference on parity, left-behind and speed; markdown made of the visible text scores recall 1.0; two indexes of the same pages give equal recall.
- Plants, each named in the output: one sentence deleted, one sentence duplicated, one page dropped, one table flattened, a visible sentence missing, words the page does not show, a process left running with a listening socket, a temp directory and a thread, a 20 percent slowdown (a real delay in the replay server), pages dropped from the index.
- One test for each normalisation: silent on the variant, loud on a deletion inside it.

Without a `[retrieval]` table, the retrieval self-tests print `SKIPPED`.

## The corpus

`corpus/synthetic/` holds one small page for each page feature, with `corpus.toml`: one record for each path (status, headers, body file, delay, an alternate answer for a missing cookie or for the first requests). A body file holds the bytes exactly as they go on the wire. `{{ORIGIN}}` in a header is the replay origin; `{{OTHER_ORIGIN}}` stands for another host.

Replay does not use DNS. The site under crawl is `127.0.0.1:<port>` and every other host is `localhost:<port>`, so a link to another host is out of scope for both crawlers.

## What the ground truth is

- The browser reads the pages as one reader does: the seeds first, in one browser context, so a cookie that a seed sets is sent to the pages after it.
- For a record that gives another answer to its first requests (a 429, then the page), the reader gets the page.
- A page is one that a reader gets when its status is 200, the browser renders it as HTML or plain text, and it shows at least one word.
- The visible text is the browser's rendered text (`innerText`) 1 second after the network goes idle.

## Limits

- Text that arrives more than 1 second after the network goes idle is not in the ground truth.
- The ground truth does not include text in a closed shadow root.
- Text that a style makes unreadable without hiding it (font size zero, text colour equal to the background, a position off the screen) counts as visible.
- A captured page is cached by the content of the whole corpus. Use a new `--work` directory when the pages depend on something else.
- Native threads are recorded in each run's `run.json` and are not asserted on: the system starts and ends worker threads on its own, so two equal runs give two counts.
- The converter layer of the candidate is the html-to-markdown package. Install the version that the candidate's crawlberg was built with.

## Tests

```bash
python -m pytest -c tools/qa/crawl_parity/pytest.ini tools/qa/crawl_parity/tests
```

The tests use stand-in crawlers and need no crawler, no lilbee and no browser.
