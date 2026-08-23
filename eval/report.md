# RAG Evaluation Report

Scored 8 questions across 2 categories.

## Summary by category

| Category | N | recall@20 | struct_ok | cites_ok | no_bulk | faith | compl | clarity | hallucin | overcommit |
|---|---|---|---|---|---|---|---|---|---|---|
| factual | 6 | 75% | 50% | 100% | 100% | 0.96 | 0.83 | 5.00 | 0% | 17% |
| biographical | 2 | 67% | 100% | 100% | 100% | 0.88 | 0.71 | 4.00 | 0% | 100% |

## Per-question detail

### factual_oswald_marine_clearance — factual

**Q:** What level of security clearance did Lee Harvey Oswald hold during his Marine Corps service?

- recall@20: 50%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.98   completeness: 1.0   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The answer is fully grounded in source [1], and its extra note about the unit’s clearance is also supported by source [2].
- missed reference evidence: 180-10078-10492.pdf p.5

### factual_ruby_real_name — factual

**Q:** What was Jack Ruby's original family name before he changed it?

- recall@20: 100%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.98   completeness: 1.0   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The answer is fully grounded by the retrieved interview transcript, and the CIA memo also supports the Rubenstein alias; no substantive claim is unsupported.

### factual_oswald_mexico_cuban_embassy — factual

**Q:** On what date did Oswald visit the Cuban Embassy in Mexico City to request a transit visa?

- recall@20: 50%   structure_ok: False   no_bulk_cite: True
- faithfulness: 1.0   completeness: 0.75   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The answer’s date and visa-request claim are directly supported by source [1]; it omits the Soviet Embassy visit/refusal detail from the reference, but it does not introduce any unsupported factual claim.
- missed reference evidence: 144-10001-10125.pdf p.18

### factual_ruby_fbi_flynn_1959 — factual

**Q:** Which FBI agent contacted Jack Ruby in 1959, and on what date?

- recall@20: 100%   structure_ok: False   no_bulk_cite: True
- faithfulness: 0.8   completeness: 0.5   clarity: 5   hallucin: False   overcommit: True
- judge rationale: The date and Dallas Office contact are supported by source [1], but the agent's middle initial is given as 'N' in the answer while the source text shows 'W.' and the added purpose of determining criminal-element knowledge is grounded.

### factual_oswald_marine_discharge_1959 — factual

**Q:** When did Oswald's Marine Corps hardship or dependency discharge board convene to consider his case?

- recall@20: 100%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.98   completeness: 1.0   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The answer is fully supported by source [1], including the 27 August 1959 date, 1530 time, and Oswald’s rank; the extra note about the prior notice is also grounded in source [2].

### factual_cia_mexico_city_cable_date — factual

**Q:** When did CIA Headquarters send the Mexico City Station information responding to Oswald's initial reported contact?

- recall@20: 50%   structure_ok: False   no_bulk_cite: True
- faithfulness: 1.0   completeness: 0.75   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The answer is fully supported by source [1] for the date and the fact that CIA Headquarters responded to Mexico City Station, but it omits the detail that Headquarters suggested Oswald might be 'Lee Henry (sic) Oswald' and the later P-file opening.
- missed reference evidence: 180-10110-10484.pdf p.172

### biographical_jack_ruby — biographical

**Q:** Who was Jack Ruby?

- recall@20: 100%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.86   completeness: 0.67   clarity: 4   hallucin: False   overcommit: True
- judge rationale: The answer is mostly grounded by the retrieved FBI memo and other documents, but it overstates contested allegations about Mafia ties and Oswald-shooting as settled fact, while omitting the corpus-grounded detail that Ruby's real name was Rubenstein.

### biographical_oswald — biographical

**Q:** Who was Lee Harvey Oswald?

- recall@20: 33%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.9   completeness: 0.75   clarity: 4   hallucin: False   overcommit: True
- judge rationale: Most cited claims are supported by the retrieved records, but the opening frames Oswald as only a 'suspect' rather than the Warren Commission's identified assassin, and the answer omits the Soviet defection/return and Mexico City embassy contacts.
- missed reference evidence: 104-10332-10014.pdf p.56, 180-10072-10186.pdf p.46
