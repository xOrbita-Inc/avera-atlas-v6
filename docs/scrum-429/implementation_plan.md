# SCRUM-429 implementation plan

Persist the LeoLabs CDM to the ingest CDM store during /v1/evaluate and link
the decision to it. Persist-on-evaluate, additive, guarded, deduped, no
scheduler.

## The gap (confirmed in code, main 5967af4)

- On the LeoLabs live path, server.py sets cdm_record_id = None (the
  leolabs_used branch, around line 1522). The audit write is guarded by
  `if cdm_record_id is not None: _post_planner_output(...)` (around line 1750),
  so a LeoLabs-driven decision writes no stored CDM and no linked
  planner_output. The decision_log is written separately and carries no CDM
  link either.
- The only store writers today are POST /cdm/inject (the TIROS reference CDM)
  and the retired Space-Track poll. cdm_record_id is populated only from the
  store fetch GET /cdm/{primary}/{secondary} (server.py line 223), which the
  LeoLabs path deliberately skips because the parsed CDM already carries the
  real covariance in the request block.
- Net: with Space-Track retired, the store's only feeder is the injected TIROS
  reference CDM, so no real conjunction the system acts on lands in the durable
  record of truth. ADR-008 makes that store the record of truth, so this is a
  real audit hole.

## Design

When an evaluate resolves its conjunction from LeoLabs, persist the parsed
LeoLabs CDM to the store with source='leolabs', get the row id back, and thread
it as cdm_record_id so the existing audit and decision-log writes link the
decision to the exact stored CDM. Persist what the planner already has in hand,
no background fetch.

### 1. Parser: a to_store_cdm_dict on ParsedLeoLabsCDM

services/planner/common/leolabs_cdm_parser.py. Add a method that emits the flat
OBJECTn_-prefixed CCSDS dict that save_cdm_record expects, from the fields the
parser already validated:

- OBJECT1_OBJECT_DESIGNATOR, OBJECT2_OBJECT_DESIGNATOR (the primary/secondary
  NORAD ids).
- TCA (ISO), MISS_DISTANCE (m), COLLISION_PROBABILITY (the CDM Pc).
- The RTN position covariance per object, OBJECT1_CR_R, OBJECT1_CT_R,
  OBJECT1_CT_T, OBJECT1_CN_R, OBJECT1_CN_T, OBJECT1_CN_N and the OBJECT2_
  equivalents, mapped from the parsed per-object RTN covariance (the LeoLabs
  CDM's SATn_CR_R.. block). These are exactly the columns the store already
  holds for TIROS, so the mapping is a straight SATn_ to OBJECTn_ rename of the
  six position elements.
- COMMENT_ID (the LeoLabs cdm_id) and COMMENT_EVENT_ID (event_id), which the
  parser already keeps in metadata, carried through for dedup.

No re-parse and no new physics. This reuses the parsed data and produces the
store's existing input shape.

### 2. Ingest: a persist path that sets source, dedups, and returns the id

services/ingest/db.py and services/ingest/main.py.

- save_cdm_record(cdm, source="reference_cdm") gains a source parameter and
  RETURNS the row id (today it returns None and the source is hardcoded).
- Dedup and idempotency. Add nullable cdm_id and event_id columns to CdmRecord
  (a small additive change; guard startup with an ALTER TABLE ADD COLUMN IF NOT
  EXISTS style migration so an existing prod store file gains the columns
  without a rebuild). Upsert on cdm_id: if a row with this cdm_id already
  exists, return its id and do not insert; otherwise insert and return the new
  id. When cdm_id is absent, fall back to dedup on
  (primary_norad, secondary_norad, tca).
- New endpoint POST /cdm/persist: calls save_cdm_record(dict, source="leolabs")
  and returns {id, created}. Keep POST /cdm/inject as the TIROS path; route it
  through the same save_cdm_record(source="real_cdm") and drop its post-save
  source patch.

Note if the cdm_id column proves more than a trivial migration on the live
store, fall back to dedup on (primary_norad, secondary_norad, tca) with no
schema change and flag it. The audit win does not depend on the column, only
the idempotency key does.

### 3. Planner: persist and thread on the live path

services/planner/server.py.

- Add _persist_leolabs_cdm(store_dict) -> Optional[int]: POST
  parsed_ll.to_store_cdm_dict() to {_ingest_url()}/cdm/persist with a short
  timeout, return the returned id. Wrap the whole thing in try/except; on any
  failure log and return None, never raise.
- On the leolabs_used branch (around line 1516 to 1522), replace
  `cdm_record_id = None` with
  `cdm_record_id = _persist_leolabs_cdm(parsed_ll.to_store_cdm_dict())`.
  covariance_source stays real_cdm and the covariance still comes from the
  block; this only persists the CDM and captures its id.
- The existing audit write at around line 1750 then fires for a LeoLabs
  decision, linking planner_output to the stored CDM, and the decision_log and
  evidence records already thread cdm_record_id where they carry it.

### Keep intact and guard

- Additive and guarded: a store-write failure leaves cdm_record_id None and
  /v1/evaluate returns exactly as it does today. Same try/except discipline as
  the other audit writes.
- No scheduler and no background loop.
- source='leolabs' for these rows, distinct from reference_cdm and synthetic.
  The store view (/store/cdm_records) surfaces source, so leolabs rows read
  leolabs.

## Acceptance

- A LeoLabs-driven evaluate creates a stored CDM row with source='leolabs' and
  a planner_output plus decision-log record that reference its cdm_record_id,
  visible via VIEW CDM STORE and the decision-log lookup.
- Re-evaluating the same conjunction reuses the row (no duplicate) and returns
  the same cdm_record_id.
- A store-write failure does not break /v1/evaluate.
- Planner suite green, with new tests below.

## Tests

- to_store_cdm_dict maps the designators, TCA, miss, Pc, and the six RTN
  position covariance elements per object correctly, and carries cdm_id /
  event_id.
- The persist upsert is idempotent: two persists of the same cdm_id return the
  same id and leave one row; a different cdm_id inserts a second row.
- Evaluate-path test: a leolabs_used evaluate threads a non-None cdm_record_id
  into the audit write (mock the ingest persist to return an id), and a persist
  that raises leaves cdm_record_id None with evaluate still returning 200.

## Verification hints

- Idempotency: persist the same cdm_id twice, assert one row and the same id.
- Guard: make the persist raise, assert evaluate still returns 200 and no audit
  link is written.
- Browser: run a Live Asset evaluate, open VIEW CDM STORE and confirm a
  source=leolabs row for the scored pair, then the decision-log lookup for that
  decision carries its cdm_record_id.

## Files

- services/planner/common/leolabs_cdm_parser.py (to_store_cdm_dict)
- services/ingest/db.py (save_cdm_record source param, return id, dedup/upsert;
  CdmRecord cdm_id/event_id columns and the guarded add-column migration)
- services/ingest/main.py (POST /cdm/persist; route /cdm/inject through the
  source param)
- services/planner/server.py (_persist_leolabs_cdm and the one-line thread on
  the leolabs_used branch)
- tests alongside each
