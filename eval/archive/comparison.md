# RAG vs. GPT-5.4 Baseline — Comparison

Total questions scored: 30 RAG, 30 baseline.

## Headline: completeness, hallucination, over-commitment

| Category | N | Compl (RAG) | Compl (GPT) | Hallucin (RAG) | Hallucin (GPT) | Overcommit (RAG) | Overcommit (GPT) | Clarity (RAG) | Clarity (GPT) |
|---|---|---|---|---|---|---|---|---|---|
| factual | 6 | 0.57 | 0.46 | 0% | 50% | 0% | 50% | 4.50 | 5.00 |
| biographical | 6 | 0.54 | 0.73 | 0% | 50% | 83% | 33% | 4.33 | 4.83 |
| analytical | 6 | 0.57 | 0.40 | 17% | 83% | 67% | 67% | 4.17 | 4.17 |
| partial_evidence | 6 | 0.56 | 0.34 | 17% | 67% | 67% | 33% | 4.33 | 4.50 |

## Baseline-specific: does GPT-5.4 contradict the archival record?

| Category | N | Correctness vs. corpus | Acknowledges uncertainty |
|---|---|---|---|
| factual | 6 | 0.66 | 17% |
| biographical | 6 | 0.83 | 50% |
| analytical | 6 | 0.59 | 50% |
| partial_evidence | 6 | 0.70 | 100% |

## RAG-specific: faithfulness + retrieval

| Category | N | Recall@20 | Precision | Faithfulness |
|---|---|---|---|---|
| factual | 6 | 42% | 4% | 0.62 |
| biographical | 6 | 33% | 6% | 0.83 |
| analytical | 6 | 19% | 3% | 0.75 |
| partial_evidence | 6 | 8% | 1% | 0.74 |

> **Note:** `out_of_scope` questions are intentionally excluded from the comparison. The RAG system is designed to refuse; plain GPT is expected to answer. It is not a like-for-like axis. See `report.md` for RAG's refusal-accuracy figure.


## Per-question head-to-head

### factual_oswald_marine_clearance — factual

**Q:** What level of security clearance did Lee Harvey Oswald hold during his Marine Corps service?

- **RAG:**  compl=0.67  hallucin=False  overcommit=False  faith=0.83  clarity=5
  rationale: The core claim that Oswald held a "confidential" clearance is supported by source [1], and the Security Termination Statement date is also grounded, but the added radar-duty/security-clearance explanations rely on source [15] rather than the cited [1].
- **GPT:**  compl=0.67  hallucin=True  overcommit=False  corr_vs_corpus=0.95  clarity=5
  rationale: It correctly identifies Oswald's clearance as confidential, but omits the specific archival details about the 1978 fact sheet and the September 11, 1959 Security Termination Statement while adding unsupported radar-duty context.

### factual_ruby_real_name — factual

**Q:** What was Jack Ruby's original family name before he changed it?

- **RAG:**  compl=0.0  hallucin=False  overcommit=False  faith=0.1  clarity=4
  rationale: The answer is largely unsupported by the retrieved sources, but it does not invent a Jack Ruby surname; it simply says the records do not contain it.
- **GPT:**  compl=1.0  hallucin=False  overcommit=True  corr_vs_corpus=1.0  clarity=5
  rationale: The answer matches the corpus fact that Jack Ruby’s original family name was Rubenstein, though it states it categorically without citing the archival source.

### factual_oswald_mexico_cuban_embassy — factual

**Q:** On what date did Oswald visit the Cuban Embassy in Mexico City to request a transit visa?

- **RAG:**  compl=1.0  hallucin=False  overcommit=False  faith=0.95  clarity=5
  rationale: The date and core facts are supported by source [5] and [6]; the extra mention of Kostikov/Sylvia Duran is also grounded in [5], so no substantive unsupported claim appears.
- **GPT:**  compl=0.4  hallucin=False  overcommit=False  corr_vs_corpus=1.0  clarity=5
  rationale: It gives the correct date and embassy visit but omits the broader context that he was seeking a Cuba-bound transit visa en route to the USSR.

### factual_ruby_fbi_flynn_1959 — factual

**Q:** Which FBI agent contacted Jack Ruby in 1959, and on what date?

- **RAG:**  compl=0.0  hallucin=False  overcommit=False  faith=0.12  clarity=4
  rationale: The answer is faithful to the retrieved sources in saying they do not contain the Jack Ruby fact, but it fails to answer the question and does not recover the grounded fact that Charles W. Flynn contacted Ruby on March 11, 1959.
- **GPT:**  compl=0.67  hallucin=False  overcommit=False  corr_vs_corpus=1.0  clarity=5
  rationale: The answer correctly identifies Charles W. Flynn and the March 11, 1959 contact date, but omits the archival context that Ruby was contacted as a nightclub operator.

### factual_oswald_marine_discharge_1959 — factual

**Q:** When did Oswald's Marine Corps hardship or dependency discharge board convene to consider his case?

- **RAG:**  compl=0.75  hallucin=False  overcommit=False  faith=0.76  clarity=4
  rationale: The core date/time claim is supported by source [1], but several added details (meeting members, recommendation timing, and the statement about records not providing further details) are not established by the cited sources.
- **GPT:**  compl=0.0  hallucin=True  overcommit=True  corr_vs_corpus=0.0  clarity=5
  rationale: The answer gives the wrong date and omits the board time and archival specifics, directly contradicting the corpus record.

### factual_cia_mexico_city_cable_date — factual

**Q:** When did CIA Headquarters send the Mexico City Station information responding to Oswald's initial reported contact?

- **RAG:**  compl=1.0  hallucin=False  overcommit=False  faith=0.96  clarity=5
  rationale: The answer is well supported by source [1] and correctly gives October 11, 1963; it adds some extra detail about Oswald's biography that is also grounded, with no unsupported substantive claim.
- **GPT:**  compl=0.0  hallucin=True  overcommit=True  corr_vs_corpus=0.0  clarity=5
  rationale: The LLM gives the wrong date and states it confidently, contradicting the archival record that places the cable on October 11, 1963.

### biographical_jack_ruby — biographical

**Q:** Who was Jack Ruby?

- **RAG:**  compl=0.0  hallucin=False  overcommit=True  faith=0.44  clarity=4
  rationale: The answer is only partly grounded: the sources support that Ruby was linked to Mafia allegations and visited Dallas offices on Nov. 21, 1963, but they do not support the broad settled claims about Mafia/CIA involvement or the assassination plot, while it also misses the reference facts that his real name was Rubenstein, he was a Dallas nightclub operator, and an FBI agent contacted him on March 11, 1959.
- **GPT:**  compl=0.67  hallucin=True  overcommit=True  corr_vs_corpus=0.48  clarity=5
  rationale: It correctly identifies Ruby as a Dallas nightclub operator and uses his real name, but adds many unsupported biographical and post-1963 details that go beyond the archival reference.

### biographical_oswald — biographical

**Q:** Who was Lee Harvey Oswald?

- **RAG:**  compl=0.5  hallucin=False  overcommit=True  faith=0.86  clarity=4
  rationale: Most cited claims are grounded in the retrieved HSCA/Warren materials, but the opening describes Oswald too narrowly and as if his Mexico City activities are settled investigation findings; it omits the core biography facts that he was identified by the Warren Commission as Kennedy's assassin, served in the Marine Corps, and defected to the Soviet Union and returned.
- **GPT:**  compl=0.75  hallucin=False  overcommit=False  corr_vs_corpus=1.0  clarity=5
  rationale: It correctly identifies Oswald as Kennedy’s assassin, notes his Marine service and Soviet defection/return, but omits the Mexico City embassy contacts.

### biographical_david_ferrie — biographical

**Q:** Who was David Ferrie?

- **RAG:**  compl=0.67  hallucin=False  overcommit=True  faith=0.98  clarity=5
  rationale: Most claims are supported by the retrieved sources, but the assertions that Ferrie was employed by Carlos Marcello and that he was linked to Clay Shaw as a factual tie are not directly established in the cited text and are stated too definitively.
- **GPT:**  compl=0.95  hallucin=True  overcommit=False  corr_vs_corpus=0.88  clarity=5
  rationale: It correctly identifies Ferrie as a former airline pilot and a central figure in Garrison’s JFK investigation, but adds several extra biographical and evidentiary details not supported by the reference corpus.

### biographical_silvia_duran — biographical

**Q:** Who was Silvia Duran?

- **RAG:**  compl=0.8  hallucin=False  overcommit=True  faith=0.86  clarity=4
  rationale: Most claims are grounded in the retrieved sources, but the answer overstates as fact that she was a "Mexican national" and that CIA officers considered her a Cuban agent, while the core facts about her Cuban Consulate work and interactions with Oswald are supported.
- **GPT:**  compl=0.75  hallucin=False  overcommit=False  corr_vs_corpus=0.92  clarity=4
  rationale: The answer captures her role at the Cuban consulate and her connection to Oswald, but omits her marriage to Horatio Duran and the Cuban diplomatic protest over her treatment.

### biographical_clay_shaw — biographical

**Q:** Who was Clay Shaw?

- **RAG:**  compl=0.67  hallucin=False  overcommit=False  faith=0.96  clarity=5
  rationale: Most claims are grounded in the retrieved sources, including Shaw’s role as a former International Trade Mart director and Garrison’s JFK investigation, but the answer omits the key Ferrie/plot allegation and the World Trade Center/Permindex linkage from the reference.
- **GPT:**  compl=0.67  hallucin=True  overcommit=True  corr_vs_corpus=0.78  clarity=5
  rationale: It correctly identifies Shaw as Garrison’s JFK-assassination defendant, but adds extra biographical and case details not supported by the reference corpus.

### biographical_jim_garrison — biographical

**Q:** Who was Jim Garrison?

- **RAG:**  compl=0.6  hallucin=False  overcommit=True  faith=0.88  clarity=4
  rationale: Most claims are grounded in the retrieved FBI and press sources, but the answer overstates uncertain or editorial material as fact and misses key reference facts like the December 1966 start date and the Gurvich 'baseless fraud' assessment.
- **GPT:**  compl=0.6  hallucin=False  overcommit=False  corr_vs_corpus=0.93  clarity=5
  rationale: The answer correctly identifies Garrison as the New Orleans district attorney tied to the JFK assassination investigation, but it omits the corpus-specific details about the December 1966 start, the full alleged conspiracy cast, and Gurvich's later criticism.

### analytical_cia_surveillance_oswald — analytical

**Q:** What did the HSCA conclude about the CIA's photographic and electronic surveillance of Oswald in Mexico City, and how forthcoming was the CIA with materials?

- **RAG:**  compl=0.9  hallucin=False  overcommit=True  faith=0.86  clarity=5
  rationale: The main HSCA conclusions are supported by the sources, but the answer overstates certainty in places by saying CIA 'withheld' materials and 'prevented' determination where the record only shows incomplete production and inability to rule out a photo.
- **GPT:**  compl=0.85  hallucin=True  overcommit=True  corr_vs_corpus=0.78  clarity=4
  rationale: It captures the existence of CIA surveillance and criticism of CIA candor, but it overstates gaps and omits the specific HSCA conclusion that Oswald was probably photographed and that non-sensitive Mexico information was generally relayed accurately.

### analytical_warren_ruby_hunts — analytical

**Q:** What links do the retrieved documents draw between Jack Ruby's circle, the Hunt family in Dallas, and the events surrounding the assassination?

- **RAG:**  compl=0.56  hallucin=True  overcommit=True  faith=0.82  clarity=4
  rationale: Most claims about Braden, Jack Ruby, and the Hunt Oil visit are grounded in the sources, but the answer invents a direct link to Lamar Hunt and overstates Braden-Ruby linkage as a possible meeting rather than simply shared presence at the Hunt offices.
- **GPT:**  compl=0.1  hallucin=False  overcommit=False  corr_vs_corpus=0.92  clarity=4
  rationale: It correctly avoids inventing document details and acknowledges uncertainty, but it does not answer the corpus-specific Hunt/Rothermel/Braden linkage described in the reference.

### analytical_kgb_oswald_conclusion — analytical

**Q:** What did U.S. investigators ultimately conclude about whether Lee Harvey Oswald was a Soviet (KGB) agent?

- **RAG:**  compl=0.28  hallucin=False  overcommit=False  faith=0.72  clarity=4
  rationale: The answer is mostly grounded in the retrieved CIA/Mexico City materials, but it misses the key Nosenko/Scelso evidence from the reference and instead adds extra theories not central to the question; no cited claim is clearly unsupported by the sources.
- **GPT:**  compl=0.67  hallucin=True  overcommit=True  corr_vs_corpus=0.58  clarity=5
  rationale: It captures the broad no-evidence takeaway, but it overstates a definitive conclusion and adds unsupported specifics like the Warren Commission and HSCA as settled findings, whereas the corpus shows a divided or unresolved intelligence assessment centered on Nosenko's disputed credibility.

### analytical_garrison_theory — analytical

**Q:** How is Jim Garrison's conspiracy theory characterized in the retrieved documents, both by himself and by his critics?

- **RAG:**  compl=0.8  hallucin=False  overcommit=True  faith=0.84  clarity=4
  rationale: Most claims are grounded in the retrieved documents, but the answer overstates unsupported details like the FBI finding "no evidence" for specific witness credibility issues and describes speculative points as settled fact rather than attributed criticism.
- **GPT:**  compl=0.2  hallucin=True  overcommit=True  corr_vs_corpus=0.18  clarity=4
  rationale: The answer is broadly generic and misses the archive-specific claims that Garrison linked Ferrie, Shaw, Oswald, and anti-Castro Cubans to the assassination and that critics called his case a baseless fraud.

### analytical_oswald_cuba_connections — analytical

**Q:** What connections between Oswald and Cuban-related activities do the retrieved documents establish?

- **RAG:**  compl=0.75  hallucin=False  overcommit=True  faith=0.82  clarity=4
  rationale: Most claims are supported by the retrieved Mexico City/Cuban Consulate material, but the answer overstates uncertain details as settled fact and does not fully cover the Fair Play for Cuba Committee/CIA disinformation thread or the Bringuier/Blalock HSCA linkage.
- **GPT:**  compl=0.5  hallucin=True  overcommit=False  corr_vs_corpus=0.78  clarity=4
  rationale: It captures Oswald’s FPCC activity and Mexico City Cuba-visa visit, but misses the CIA disinformation angle and the specific HSCA Bringuier/Blalock linkage while adding some unsupported detail.

### analytical_hsca_ruby — analytical

**Q:** What investigative findings do the retrieved documents offer regarding the HSCA's examination of Jack Ruby?

- **RAG:**  compl=0.1  hallucin=False  overcommit=False  faith=0.42  clarity=4
  rationale: The answer is mostly generic and grounded in the sequestered-collection descriptions, but it omits the specific Ruby findings in the corpus (Flynn's 1959 contact, the 1964 report to the Commission, and the Olsen/Sturgis interview); its claim that Ruby was 'the individual who killed Lee Harvey Oswald' is supported by the sources.
- **GPT:**  compl=0.1  hallucin=True  overcommit=True  corr_vs_corpus=0.3  clarity=4
  rationale: The answer is readable but largely invents broad HSCA conclusions and organized-crime themes not supported by the retrieved corpus, while missing the specific Ruby-related archival details about Flynn, the June 1964 report, and the Olsen-Sturgis interview.

### partial_evidence_oswald_motivations — partial_evidence

**Q:** What do the archival documents reveal about Lee Harvey Oswald's personal motivations for assassinating President Kennedy?

- **RAG:**  compl=0.5  hallucin=False  overcommit=True  faith=0.83  clarity=4
  rationale: The answer is mostly grounded on Oswald's defection and post-return file history, but it overstates that the documents 'likely' explained counterintelligence concerns and adds an unsupported claim about his contact with the American Embassy in Moscow.
- **GPT:**  compl=0.35  hallucin=True  overcommit=True  corr_vs_corpus=0.45  clarity=4
  rationale: It correctly says the archives do not provide a definitive motive, but it adds unsupported motive theories and document-based specifics not present in the reference.

### partial_evidence_oswald_acted_alone — partial_evidence

**Q:** Do the retrieved documents support the conclusion that Oswald acted alone in the assassination?

- **RAG:**  compl=0.55  hallucin=False  overcommit=False  faith=0.68  clarity=4
  rationale: Most cited claims are grounded in the retrieved files (Oswald's Mexico City contacts, small pre-assassination CIA file, later accumulation of records), but the answer misses the key contested-point evidence from the corpus and does not make any clearly unsupported substantive claim.
- **GPT:**  compl=0.1  hallucin=False  overcommit=False  corr_vs_corpus=0.95  clarity=5
  rationale: The response is cautious and broadly consistent, but it is generic and does not address the specific archival findings about the lack of direct lone-gunman endorsement or the cited contested evidence.

### partial_evidence_cuban_involvement — partial_evidence

**Q:** Was the Cuban government involved in the assassination of President Kennedy?

- **RAG:**  compl=0.72  hallucin=False  overcommit=False  faith=0.92  clarity=5
  rationale: The answer is well grounded in the retrieved sources on Cuban reactions, CIA plots against Castro, and absence of direct evidence, but it omits the specific LBJ/Castro-retaliation speculation that the reference answer highlights.
- **GPT:**  compl=0.4  hallucin=False  overcommit=False  corr_vs_corpus=0.86  clarity=5
  rationale: The answer reaches the same bottom line as the corpus but omits the key archival nuances about LBJ's Castro-retaliation remark and the CIA-Mafia plotting context.

### partial_evidence_ruby_state_of_mind — partial_evidence

**Q:** What do the documents indicate about Jack Ruby's mental state or psychological condition at the time of the assassination?

- **RAG:**  compl=0.2  hallucin=True  overcommit=True  faith=0.42  clarity=4
  rationale: The answer correctly notes the lack of direct psychological evaluation, but it invents unsupported claims about Ruby acting alone, wanting to be a hero, and conspiracy/organized-crime conclusions that do not appear in the retrieved sources.
- **GPT:**  compl=0.2  hallucin=True  overcommit=True  corr_vs_corpus=0.28  clarity=4
  rationale: The answer is readable and hedges somewhat, but it introduces unsupported psychiatric conclusions that go beyond the previews, which only indicate no direct evidence about Ruby's mental state and provide contextual facts about Ruby/Rubenstein and Flynn.

### partial_evidence_cia_oswald_connection — partial_evidence

**Q:** Did Lee Harvey Oswald have a connection to the CIA?

- **RAG:**  compl=0.8  hallucin=False  overcommit=True  faith=0.92  clarity=5
  rationale: Most claims are grounded in the retrieved memos, but the answer overstates uncertainty by saying Oswald was 'never connected ... in any way whatsoever' even though the sources frame the issue as contradictory and enigmatic rather than conclusively settled.
- **GPT:**  compl=0.6  hallucin=True  overcommit=False  corr_vs_corpus=0.9  clarity=5
  rationale: The answer is broadly consistent with the corpus on the lack of proven CIA connection, but it misses the specific 1998 memo evidence and adds unsupported detail about investigations and motives.

### partial_evidence_organized_crime_role — partial_evidence

**Q:** What role, if any, did organized crime play in the Kennedy assassination according to the retrieved documents?

- **RAG:**  compl=0.6  hallucin=False  overcommit=True  faith=0.67  clarity=4
  rationale: Most claims are grounded in the retrieved materials, but the statement that the Warren Commission found no evidence of organized crime involvement is unsupported here, while the answer also overstates an uncertain theory as a potential role rather than a documented hypothesis.
- **GPT:**  compl=0.4  hallucin=True  overcommit=False  corr_vs_corpus=0.78  clarity=4
  rationale: The answer is broadly consistent about uncertainty, but it omits several corpus-specific details and adds generic claims not grounded in the retrieved documents.
