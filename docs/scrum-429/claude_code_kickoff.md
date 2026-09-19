# SCRUM-429 kickoff for Claude Code

Persist the LeoLabs CDM to the ingest store on evaluate and link the decision
to it. Full spec in docs/scrum-429/implementation_plan.md. Read it first.

Branch: scrum-429-persist-leolabs-cdm off current main (5967af4).

Do:
1. Parser: add ParsedLeoLabsCDM.to_store_cdm_dict() in
   services/planner/common/leolabs_cdm_parser.py, emitting the flat
   OBJECTn_-prefixed CCSDS dict save_cdm_record expects (designators, TCA,
   MISS_DISTANCE, COLLISION_PROBABILITY, the six RTN position covariance
   elements per object), plus COMMENT_ID and COMMENT_EVENT_ID for dedup.
2. Ingest: give save_cdm_record a source parameter and make it return the row
   id; upsert/dedup on cdm_id (add nullable cdm_id/event_id columns with a
   guarded add-column migration; fall back to (primary,secondary,tca) if the
   column migration is not trivial on the live store, and flag it). Add POST
   /cdm/persist that saves with source='leolabs' and returns {id, created}.
   Route /cdm/inject through the same source parameter (source='real_cdm').
3. Planner: add _persist_leolabs_cdm(store_dict) -> Optional[int] in server.py,
   a guarded POST to /cdm/persist that returns the id or None on any failure.
   On the leolabs_used branch replace cdm_record_id = None with the persist
   call so the existing audit write links the decision. Do not change
   covariance_source or the covariance path.
4. Tests: the mapping, the upsert idempotency, and the evaluate-path threading
   plus the failure guard. Run the planner suite from the repo root.

Do not: add a scheduler or background poll, change how covariance is computed
or sourced, or let a store-write failure break /v1/evaluate. No AI attribution.
Do not merge; open the PR for John.

When done, report: files changed with line counts, the suite result on the
branch, and confirmation that a leolabs_used evaluate persists a source=leolabs
CDM and threads its cdm_record_id, idempotent on re-evaluate, with evaluate
unaffected when the persist fails.
