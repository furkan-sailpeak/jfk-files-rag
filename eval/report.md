# RAG Evaluation Report

Scored 30 questions across 5 categories.

## Summary by category

| Category | N | recall@20 | precision | struct_ok | cites_ok | no_bulk | faith | compl | clarity | hallucin | overcommit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| factual | 6 | 42% | 4% | 100% | 100% | 100% | 0.62 | 0.57 | 4.50 | 0% | 0% |
| biographical | 6 | 33% | 6% | 100% | 100% | 100% | 0.83 | 0.54 | 4.33 | 0% | 83% |
| analytical | 6 | 19% | 3% | 100% | 100% | 100% | 0.75 | 0.57 | 4.17 | 17% | 67% |
| partial_evidence | 6 | 8% | 1% | 100% | 100% | 67% | 0.74 | 0.56 | 4.33 | 17% | 67% |
| out_of_scope | 6 | — | — | 100% | 0% | 100% | — | — | — | — | — |

**Refusal accuracy:** 0/6 out-of-scope queries correctly refused.


## Per-question detail

### factual_oswald_marine_clearance — factual

**Q:** What level of security clearance did Lee Harvey Oswald hold during his Marine Corps service?

- recall@20: 50%   precision: 7%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.83   completeness: 0.67   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The core claim that Oswald held a "confidential" clearance is supported by source [1], and the Security Termination Statement date is also grounded, but the added radar-duty/security-clearance explanations rely on source [15] rather than the cited [1].
- missed reference evidence: 180-10078-10492.pdf p.5

### factual_ruby_real_name — factual

**Q:** What was Jack Ruby's original family name before he changed it?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.1   completeness: 0.0   clarity: 4   hallucin: False   overcommit: False
- judge rationale: The answer is largely unsupported by the retrieved sources, but it does not invent a Jack Ruby surname; it simply says the records do not contain it.
- missed reference evidence: 135-10001-10288.pdf p.18

### factual_oswald_mexico_cuban_embassy — factual

**Q:** On what date did Oswald visit the Cuban Embassy in Mexico City to request a transit visa?

- recall@20: 50%   precision: 7%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.95   completeness: 1.0   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The date and core facts are supported by source [5] and [6]; the extra mention of Kostikov/Sylvia Duran is also grounded in [5], so no substantive unsupported claim appears.
- missed reference evidence: 144-10001-10125.pdf p.18

### factual_ruby_fbi_flynn_1959 — factual

**Q:** Which FBI agent contacted Jack Ruby in 1959, and on what date?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.12   completeness: 0.0   clarity: 4   hallucin: False   overcommit: False
- judge rationale: The answer is faithful to the retrieved sources in saying they do not contain the Jack Ruby fact, but it fails to answer the question and does not recover the grounded fact that Charles W. Flynn contacted Ruby on March 11, 1959.
- missed reference evidence: 157-10014-10242.pdf p.441

### factual_oswald_marine_discharge_1959 — factual

**Q:** When did Oswald's Marine Corps hardship or dependency discharge board convene to consider his case?

- recall@20: 100%   precision: 7%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.76   completeness: 0.75   clarity: 4   hallucin: False   overcommit: False
- judge rationale: The core date/time claim is supported by source [1], but several added details (meeting members, recommendation timing, and the statement about records not providing further details) are not established by the cited sources.

### factual_cia_mexico_city_cable_date — factual

**Q:** When did CIA Headquarters send the Mexico City Station information responding to Oswald's initial reported contact?

- recall@20: 50%   precision: 7%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.96   completeness: 1.0   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The answer is well supported by source [1] and correctly gives October 11, 1963; it adds some extra detail about Oswald's biography that is also grounded, with no unsupported substantive claim.
- missed reference evidence: 180-10110-10484.pdf p.172

### biographical_jack_ruby — biographical

**Q:** Who was Jack Ruby?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.44   completeness: 0.0   clarity: 4   hallucin: False   overcommit: True
- judge rationale: The answer is only partly grounded: the sources support that Ruby was linked to Mafia allegations and visited Dallas offices on Nov. 21, 1963, but they do not support the broad settled claims about Mafia/CIA involvement or the assassination plot, while it also misses the reference facts that his real name was Rubenstein, he was a Dallas nightclub operator, and an FBI agent contacted him on March 11, 1959.
- missed reference evidence: 135-10001-10288.pdf p.18, 157-10014-10242.pdf p.441

### biographical_oswald — biographical

**Q:** Who was Lee Harvey Oswald?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.86   completeness: 0.5   clarity: 4   hallucin: False   overcommit: True
- judge rationale: Most cited claims are grounded in the retrieved HSCA/Warren materials, but the opening describes Oswald too narrowly and as if his Mexico City activities are settled investigation findings; it omits the core biography facts that he was identified by the Warren Commission as Kennedy's assassin, served in the Marine Corps, and defected to the Soviet Union and returned.
- missed reference evidence: 104-10332-10014.pdf p.56, 180-10072-10186.pdf p.46, 180-10104-10354.pdf p.13

### biographical_david_ferrie — biographical

**Q:** Who was David Ferrie?

- recall@20: 100%   precision: 13%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.98   completeness: 0.67   clarity: 5   hallucin: False   overcommit: True
- judge rationale: Most claims are supported by the retrieved sources, but the assertions that Ferrie was employed by Carlos Marcello and that he was linked to Clay Shaw as a factual tie are not directly established in the cited text and are stated too definitively.

### biographical_silvia_duran — biographical

**Q:** Who was Silvia Duran?

- recall@20: 33%   precision: 7%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.86   completeness: 0.8   clarity: 4   hallucin: False   overcommit: True
- judge rationale: Most claims are grounded in the retrieved sources, but the answer overstates as fact that she was a "Mexican national" and that CIA officers considered her a Cuban agent, while the core facts about her Cuban Consulate work and interactions with Oswald are supported.
- missed reference evidence: 144-10001-10099.pdf p.2, 180-10110-10484.pdf p.306

### biographical_clay_shaw — biographical

**Q:** Who was Clay Shaw?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.96   completeness: 0.67   clarity: 5   hallucin: False   overcommit: False
- judge rationale: Most claims are grounded in the retrieved sources, including Shaw’s role as a former International Trade Mart director and Garrison’s JFK investigation, but the answer omits the key Ferrie/plot allegation and the World Trade Center/Permindex linkage from the reference.
- missed reference evidence: 104-10330-10081.pdf p.15, 104-10337-10001.pdf p.195

### biographical_jim_garrison — biographical

**Q:** Who was Jim Garrison?

- recall@20: 67%   precision: 13%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.88   completeness: 0.6   clarity: 4   hallucin: False   overcommit: True
- judge rationale: Most claims are grounded in the retrieved FBI and press sources, but the answer overstates uncertain or editorial material as fact and misses key reference facts like the December 1966 start date and the Gurvich 'baseless fraud' assessment.
- missed reference evidence: 104-10337-10001.pdf p.195

### analytical_cia_surveillance_oswald — analytical

**Q:** What did the HSCA conclude about the CIA's photographic and electronic surveillance of Oswald in Mexico City, and how forthcoming was the CIA with materials?

- recall@20: 67%   precision: 10%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.86   completeness: 0.9   clarity: 5   hallucin: False   overcommit: True
- judge rationale: The main HSCA conclusions are supported by the sources, but the answer overstates certainty in places by saying CIA 'withheld' materials and 'prevented' determination where the record only shows incomplete production and inability to rule out a photo.
- missed reference evidence: 180-10110-10484.pdf p.35

### analytical_warren_ruby_hunts — analytical

**Q:** What links do the retrieved documents draw between Jack Ruby's circle, the Hunt family in Dallas, and the events surrounding the assassination?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.82   completeness: 0.56   clarity: 4   hallucin: True   overcommit: True
- judge rationale: Most claims about Braden, Jack Ruby, and the Hunt Oil visit are grounded in the sources, but the answer invents a direct link to Lamar Hunt and overstates Braden-Ruby linkage as a possible meeting rather than simply shared presence at the Hunt offices.
- missed reference evidence: 157-10014-10242.pdf p.44, 157-10014-10242.pdf p.384

### analytical_kgb_oswald_conclusion — analytical

**Q:** What did U.S. investigators ultimately conclude about whether Lee Harvey Oswald was a Soviet (KGB) agent?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.72   completeness: 0.28   clarity: 4   hallucin: False   overcommit: False
- judge rationale: The answer is mostly grounded in the retrieved CIA/Mexico City materials, but it misses the key Nosenko/Scelso evidence from the reference and instead adds extra theories not central to the question; no cited claim is clearly unsupported by the sources.
- missed reference evidence: 104-10332-10014.pdf p.56

### analytical_garrison_theory — analytical

**Q:** How is Jim Garrison's conspiracy theory characterized in the retrieved documents, both by himself and by his critics?

- recall@20: 50%   precision: 11%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.84   completeness: 0.8   clarity: 4   hallucin: False   overcommit: True
- judge rationale: Most claims are grounded in the retrieved documents, but the answer overstates unsupported details like the FBI finding "no evidence" for specific witness credibility issues and describes speculative points as settled fact rather than attributed criticism.
- missed reference evidence: 104-10305-10000.pdf p.114, 104-10337-10001.pdf p.195

### analytical_oswald_cuba_connections — analytical

**Q:** What connections between Oswald and Cuban-related activities do the retrieved documents establish?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.82   completeness: 0.75   clarity: 4   hallucin: False   overcommit: True
- judge rationale: Most claims are supported by the retrieved Mexico City/Cuban Consulate material, but the answer overstates uncertain details as settled fact and does not fully cover the Fair Play for Cuba Committee/CIA disinformation thread or the Bringuier/Blalock HSCA linkage.
- missed reference evidence: 104-10337-10001.pdf p.162, 104-10413-10306.pdf p.35, 157-10014-10242.pdf p.165, 180-10106-10008.pdf p.1

### analytical_hsca_ruby — analytical

**Q:** What investigative findings do the retrieved documents offer regarding the HSCA's examination of Jack Ruby?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.42   completeness: 0.1   clarity: 4   hallucin: False   overcommit: False
- judge rationale: The answer is mostly generic and grounded in the sequestered-collection descriptions, but it omits the specific Ruby findings in the corpus (Flynn's 1959 contact, the 1964 report to the Commission, and the Olsen/Sturgis interview); its claim that Ruby was 'the individual who killed Lee Harvey Oswald' is supported by the sources.
- missed reference evidence: 135-10001-10288.pdf p.18, 157-10014-10242.pdf p.441

### partial_evidence_oswald_motivations — partial_evidence

**Q:** What do the archival documents reveal about Lee Harvey Oswald's personal motivations for assassinating President Kennedy?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: False
- faithfulness: 0.83   completeness: 0.5   clarity: 4   hallucin: False   overcommit: True
- judge rationale: The answer is mostly grounded on Oswald's defection and post-return file history, but it overstates that the documents 'likely' explained counterintelligence concerns and adds an unsupported claim about his contact with the American Embassy in Moscow.
- missed reference evidence: 104-10332-10014.pdf p.56, 180-10078-10492.pdf p.5, 180-10104-10354.pdf p.13

### partial_evidence_oswald_acted_alone — partial_evidence

**Q:** Do the retrieved documents support the conclusion that Oswald acted alone in the assassination?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.68   completeness: 0.55   clarity: 4   hallucin: False   overcommit: False
- judge rationale: Most cited claims are grounded in the retrieved files (Oswald's Mexico City contacts, small pre-assassination CIA file, later accumulation of records), but the answer misses the key contested-point evidence from the corpus and does not make any clearly unsupported substantive claim.
- missed reference evidence: 104-10332-10009.pdf p.68, 104-10433-10209.pdf p.255, 180-10110-10484.pdf p.313

### partial_evidence_cuban_involvement — partial_evidence

**Q:** Was the Cuban government involved in the assassination of President Kennedy?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.92   completeness: 0.72   clarity: 5   hallucin: False   overcommit: False
- judge rationale: The answer is well grounded in the retrieved sources on Cuban reactions, CIA plots against Castro, and absence of direct evidence, but it omits the specific LBJ/Castro-retaliation speculation that the reference answer highlights.
- missed reference evidence: 104-10333-10001.pdf p.114, 157-10014-10242.pdf p.462, 157-10014-10242.pdf p.463

### partial_evidence_ruby_state_of_mind — partial_evidence

**Q:** What do the documents indicate about Jack Ruby's mental state or psychological condition at the time of the assassination?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.42   completeness: 0.2   clarity: 4   hallucin: True   overcommit: True
- judge rationale: The answer correctly notes the lack of direct psychological evaluation, but it invents unsupported claims about Ruby acting alone, wanting to be a hero, and conspiracy/organized-crime conclusions that do not appear in the retrieved sources.
- missed reference evidence: 135-10001-10288.pdf p.18, 157-10014-10242.pdf p.441

### partial_evidence_cia_oswald_connection — partial_evidence

**Q:** Did Lee Harvey Oswald have a connection to the CIA?

- recall@20: 50%   precision: 5%   structure_ok: True   no_bulk_cite: True
- faithfulness: 0.92   completeness: 0.8   clarity: 5   hallucin: False   overcommit: True
- judge rationale: Most claims are grounded in the retrieved memos, but the answer overstates uncertainty by saying Oswald was 'never connected ... in any way whatsoever' even though the sources frame the issue as contradictory and enigmatic rather than conclusively settled.
- missed reference evidence: 104-10332-10014.pdf p.56

### partial_evidence_organized_crime_role — partial_evidence

**Q:** What role, if any, did organized crime play in the Kennedy assassination according to the retrieved documents?

- recall@20: 0%   precision: 0%   structure_ok: True   no_bulk_cite: False
- faithfulness: 0.67   completeness: 0.6   clarity: 4   hallucin: False   overcommit: True
- judge rationale: Most claims are grounded in the retrieved materials, but the statement that the Warren Commission found no evidence of organized crime involvement is unsupported here, while the answer also overstates an uncertain theory as a potential role rather than a documented hypothesis.
- missed reference evidence: 104-10333-10001.pdf p.107, 157-10014-10242.pdf p.44, 157-10014-10242.pdf p.462

### out_of_scope_world_series_1964 — out_of_scope

**Q:** Who won the 1964 World Series?

- recall@20: —   precision: —   structure_ok: True   no_bulk_cite: True
- correct_refusal: False

### out_of_scope_caffeine_formula — out_of_scope

**Q:** What is the chemical formula for caffeine?

- recall@20: —   precision: —   structure_ok: True   no_bulk_cite: True
- correct_refusal: False

### out_of_scope_ocean_poem — out_of_scope

**Q:** Write me a poem about the ocean.

- recall@20: —   precision: —   structure_ok: True   no_bulk_cite: True
- correct_refusal: False

### out_of_scope_change_tire — out_of_scope

**Q:** How do I change a car tire?

- recall@20: —   precision: —   structure_ok: True   no_bulk_cite: True
- correct_refusal: False

### out_of_scope_latvia_capital — out_of_scope

**Q:** What is the capital of Latvia?

- recall@20: —   precision: —   structure_ok: True   no_bulk_cite: True
- correct_refusal: False

### out_of_scope_hamlet_plot — out_of_scope

**Q:** Summarize the plot of Hamlet.

- recall@20: —   precision: —   structure_ok: True   no_bulk_cite: True
- correct_refusal: False
