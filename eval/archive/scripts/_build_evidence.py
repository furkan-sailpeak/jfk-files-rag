"""One-off: pull candidate evidence chunks for all eval questions from the DB.
Writes eval/_evidence_dump.json which the author then reads to write reference
answers in questions.yaml. Not used by the running eval."""
import json, os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")

QUERIES = [
    # (topic_id, ts_query, extra_filter_sql_or_none, limit)
    ("oswald_marine_security",        "Oswald Marine Corps security clearance", None, 5),
    ("ruby_real_name",                "Ruby Rubenstein real name", None, 5),
    ("oswald_mexico_cuban_embassy",   "Oswald Cuban Embassy Mexico City visa", None, 5),
    ("ruby_fbi_flynn_1959",           "Ruby Special Agent Flynn informant 1959", None, 5),
    ("oswald_marine_service_dates",   "Oswald Marine Corps enlisted discharge", None, 5),
    ("cia_mexico_city_oswald_cable",  "Oswald CIA cable Mexico City October 1963", None, 5),

    ("who_jack_ruby",                 "Jack Ruby nightclub Dallas", None, 8),
    ("who_oswald",                    "Lee Harvey Oswald defector Soviet Union Marine", None, 8),
    ("who_david_ferrie",              "David Ferrie Civil Air Patrol New Orleans", None, 8),
    ("who_silvia_duran",              "Silvia Duran Cuban Embassy Mexico City", None, 8),
    ("who_clay_shaw",                 "Clay Shaw New Orleans trade mart Garrison", None, 8),
    ("who_jim_garrison",              "Jim Garrison New Orleans District Attorney investigation", None, 8),

    ("cia_surveillance_oswald",       "CIA surveillance Oswald Mexico City photograph", None, 10),
    ("warren_ruby_hunts",             "Warren Commission Ruby Lamar Hunt notebook", None, 10),
    ("kgb_oswald_conclusion",         "KGB Oswald Minsk interest conclusion", None, 10),
    ("garrison_theory",               "Garrison conspiracy Ferrie Shaw anti-Castro", None, 10),
    ("oswald_cuba_connections",       "Oswald Fair Play for Cuba Committee", None, 10),
    ("hsca_ruby",                     "HSCA Ruby House Select Committee organized crime", None, 10),

    ("oswald_motivations",            "Oswald motivation psychology reasons", None, 10),
    ("oswald_acted_alone",            "Oswald acted alone lone gunman conspiracy", None, 10),
    ("cuban_involvement",             "Castro Cuba retaliation assassination Kennedy", None, 10),
    ("ruby_state_of_mind",            "Ruby emotional state mental Oswald shooting", None, 10),
    ("cia_oswald_connection",         "CIA Oswald agent contact pre-assassination", None, 10),
    ("organized_crime_role",          "organized crime mafia assassination Kennedy Marcello Trafficante", None, 10),
]

def main():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    out = {}
    for topic_id, tsq, _filter, limit in QUERIES:
        cur.execute("""
            SELECT filename, page_number,
              ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s), 2) AS r,
              substring(content, 1, 700) AS preview
            FROM jfk_pages
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            ORDER BY r DESC, length(content) ASC
            LIMIT %s
        """, (tsq, tsq, limit))
        rows = cur.fetchall()
        out[topic_id] = [
            {"filename": r[0], "page": r[1], "rank": float(r[2]), "preview": r[3]}
            for r in rows
        ]
    cur.close(); conn.close()
    (HERE / "_evidence_dump.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote _evidence_dump.json with {sum(len(v) for v in out.values())} rows across {len(out)} topics")

if __name__ == "__main__":
    main()
