# Test_v1 labeling brief

You are one of four independent labelers (A/B/C/D). Your task: produce
verbatim PII span annotations for ~20 Swiss-domain text chunks. Your
labels will be combined with three other labelers' independent labels;
spans where ≥3 labelers agree become gold; disagreements get
hand-resolved.

**Independence is what makes the ensemble valuable.** Don't try to guess
what the other labelers will say. Apply the rules below consistently.

---

## Categories (gheim's 8)

Apply these EXACT label strings — case sensitive, no synonyms:

| label | what counts | examples |
|---|---|---|
| `private_person` | A specific named individual | "Anna Müller", "Dr. Hans Schmid", "M. Dupont", ALL-CAPS legal-citation surnames |
| `private_address` | A street + number, or a residential / business address. PLZ + city only counts when joined to a street | "Bahnhofstrasse 12, 8001 Zürich", "Rue de Lausanne 5" |
| `private_date` | A specific calendar date with day-month-year structure | "12.03.1985", "31. Juli 2013", "29 ottobre 2001", "2 juin 2010". NOT bare years. NOT journal refs like "11/1993". |
| `private_email` | An email address | "anna.mueller@example.ch" |
| `private_phone` | A phone number, Swiss or international format | "044 668 18 00", "+41 79 123 45 67" |
| `private_url` | A URL or domain | "https://example.ch", "www.bag.admin.ch" |
| `account_number` | IBAN, AHV, VAT, credit-card, account number | "CH93 0076 2011 6238 5295 7", "756.1234.5678.97", "CHE-123.456.789 MWST" |
| `secret` | API keys, tokens, passwords | "sk-proj-…", "ghp_…", "AKIA…" |

---

## Rules

1. **Spans must be verbatim substrings.** No paraphrasing. Copy the exact
   characters from the chunk including punctuation and capitalisation.

2. **Boundaries matter.**
   - **Person**: include the full name as it appears, including titles
     (Dr., Herr, Mme) ONLY if they're directly attached. "Dr. Anna
     Müller" → `Dr. Anna Müller`. "Dr. med. Anna Müller" → `Dr. med.
     Anna Müller`. "der Patient Anna Müller" → just `Anna Müller`.
   - **Address**: include the street, number, optional comma, PLZ, city
     in one span if they're contiguous. "Bahnhofstrasse 12, 8001 Zürich"
     → ONE address span. If only the street is present, label just the
     street. If only PLZ+city without street, do NOT label.
   - **Date**: include only the date itself. "geboren am 12. März 1985"
     → just `12. März 1985`. Strip leading "am ", "le ", "il ".

3. **Place names alone are NOT addresses.** "Zürich" or "Genève" by
   themselves are place names, not residential addresses — DO NOT label.
   "Bahnhofstrasse 12, 8001 Zürich" IS an address.

4. **Role nouns / job titles are NOT persons.** "Anwalt", "Politiker",
   "Beschwerdeführer", "président" — DO NOT label. Only specific named
   individuals.

5. **Institution acronyms that look like ALL-CAPS surnames are NOT
   persons.** Do NOT label: BGE, AHV, IBAN, MWST, COVID, NATO, SUVA,
   SBB, EMRK, OFJ, OFAS, BAG, BGH, OLG, EGMR, ZGB, OR, ZPO, StPO, BV,
   SCHKG, … (Swiss/EU institutional codes, journal abbreviations like
   AJP/PJA/RNRF/ZBGR, court abbreviations).

6. **ALL-CAPS academic citation surnames ARE persons.** "ROLAND
   PFÄFFLI", "STAEHELIN", "AEBI-MÜLLER", "VAN DEN BERG" — DO label
   these. Slash-separated co-authors split into separate spans:
   "STAEHELIN/GROLIMUND" → two spans `STAEHELIN` and `GROLIMUND`.

7. **Year references and journal-issue refs are NOT dates.** "2018"
   alone, "AJP 11/1993", "BN 1992 S. 455" — these are NOT dates. A real
   date needs day + month (named or numeric) + year.

8. **For `control` chunks, your job is to verify there's no PII.**
   Often there will genuinely be 0 spans. Don't invent labels. But if
   you spot something the prefilter missed, label it — the control
   designation is a hint, not a constraint.

---

## Output format

For each chunk, output ONE LINE of JSON with `chunk_id`, the labeler
letter (passed in the invocation), and a `spans` array. The spans
array contains `{"label": "...", "value": "..."}` records — use the
verbatim surface text as `value`, NOT character offsets (the
aggregator will re-locate them via `str.find`).

```jsonl
{"chunk_id": "T000001", "labeler": "A", "spans": [{"label": "private_person", "value": "Anna Müller"}, {"label": "private_address", "value": "Bahnhofstrasse 12, 8001 Zürich"}]}
{"chunk_id": "T000002", "labeler": "A", "spans": []}
{"chunk_id": "T000003", "labeler": "A", "spans": [{"label": "private_email", "value": "info@example.ch"}]}
```

**Empty `spans: []` is fine** — many chunks have no PII, especially
controls.

**Output ALL chunks in your batch in the same JSONL file**, one line
per chunk, in the order they appear in the input.

---

## Quality gate

- Every `value` you emit MUST be a verbatim substring of the chunk
  text. The aggregator drops claims that can't be located.
- Every `label` MUST be one of the 8 strings listed above. Misspellings
  (e.g. "name" instead of "private_person") are dropped.
- If you're unsure about a span, leave it OUT. The 4-way agreement
  design assumes some labelers will miss things; the gold set is
  built from spans where ≥3 of 4 labelers agree, so over-labeling
  hurts quality more than under-labeling.

---

## Example: full chunk → labels

INPUT chunk (T000123):
> Anna Müller (geb. 12.03.1985) wohnt an der Bahnhofstrasse 12, 8001
> Zürich. Ihre IBAN ist CH93 0076 2011 6238 5295 7. Die
> Beschwerdeführerin hat das Bundesgericht (BGE 142 III 433) angerufen.

CORRECT output line:
```json
{"chunk_id": "T000123", "labeler": "A", "spans": [{"label": "private_person", "value": "Anna Müller"}, {"label": "private_date", "value": "12.03.1985"}, {"label": "private_address", "value": "Bahnhofstrasse 12, 8001 Zürich"}, {"label": "account_number", "value": "CH93 0076 2011 6238 5295 7"}]}
```

Things this example does NOT label (correctly):
- "Beschwerdeführerin" → role noun, not a person.
- "Bundesgericht" → institution, not a person.
- "BGE 142 III 433" → court ruling reference, not a date or person.
- "geb." → not a span; metadata word.
