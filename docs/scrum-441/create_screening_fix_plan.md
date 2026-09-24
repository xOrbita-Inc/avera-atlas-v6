# SCRUM-441 create-screening fix plan: the request is multipart form-data, not JSON

The first live submit rejected our body with `422 {"error": "Invalid Miss
Distance"}` on all three probes, including the one with the thresholds block
removed. The LeoLabs API reference (the "Create an On-Demand Screening" operation
at https://platform.leolabs.space/documentation/api, corroborated by the COLA Gap
Screenings article) shows why.

## Root cause

The create endpoint is multipart form-data with a file upload. The reference
states it directly: "Parameters must be sent as a multipart form." Our client
posts `application/json`, so the server's form parser saw no fields, and
`missDistance` is the first required field it validates. Every probe failed at the
same point because the body was never read as form data. This is not a field
rename, it is a transport change plus the correct field names.

## The confirmed request

    POST /catalog/conjunctions/screenings
    Content-Type: multipart/form-data

File part:

    file                     the SCRUM-440 ephemeris document serialised to JSON
                             bytes, uploaded as a file (LeoLabs .json is a
                             supported format). filename e.g. ephemeris.json,
                             content type application/json.

Form fields:

    primaryCatalogNumber     the LeoLabs catalog number, e.g. L3969. This is what
                             run_screening already passes as primary_object.
    missDistance             maximum miss distance in KILOMETRES, max 100.
                             policy.min_miss_distance_km (default 1.0) goes here
                             directly, no unit conversion.
    probabilityOfCollision   minimum PoC to return, as a string. Our
                             policy.pc_maneuver_threshold (1.0e-4).
    mahalanobisDistance      our policy.mahalanobis_screen_threshold (4.0). See
                             the open decision below on its meaning.
    primaryHardBodyRadius    the primary HBR override in METRES. See the open
                             decision below on the 15 m combined convention.

Omit for a whole-catalog secondary screen:

    secondaryCatalogNumber   omit. Its absence is what selects screen against all
    secondaryFile            objects. There is no screenAgainstAllObjects flag.

Omit to use the file's own covariance:

    radialUncertainty        omit all three. Their absence with covariance present
    inTrackUncertainty       in the file is what "Use File Uncertainty" means.
    crossTrackUncertainty    There is no useFileUncertainty flag.

Do not send the deprecated combined `hardBodyRadius`, `primaryObject`, an inline
`ephemeris`, `screenAgainstAllObjects`, `useFileUncertainty`, or a nested
`thresholds` object. None of those exist in the API.

## What changes in the code

1. Transport. LeoLabsClient.create_screening, and _request beneath it, must post
   multipart. Extend _request to accept optional `files` and `data` and, when
   present, pass them to requests instead of `json`. Keep the screening limiter,
   the org-wide limiter, and the retry and backoff exactly as they are. Do not
   set Content-Type by hand; requests sets the multipart boundary when `files` is
   passed.

2. The request builder. build_screening_request stops returning a JSON body and
   instead returns what create_screening needs for a multipart post: the flat
   form fields under the confirmed names, and the ephemeris as a file part
   (json.dumps of the SCRUM-440 dict, encoded to bytes). Keep every LeoLabs field
   name in this one function, as before.

3. run_screening. Unchanged in flow. It calls build_screening_request then
   create_screening then wait_for_screening then get_screening_cdms. Only the
   shape passed from the builder to the client changes.

## Two open decisions, implement a fail-safe default and flag them

Do not settle these silently. Implement the conservative default named here so a
submit can go through, and record both in live_verification.md for John or
Sreejit.

Hard body radius. Our convention is a single 15 m combined value, deliberately
inflated. The API no longer takes a combined value, only primaryHardBodyRadius
and secondaryHardBodyRadius, both metres, each required to be greater than zero.
Interim default: set primaryHardBodyRadius to the 15 m combined value from
conventions.combined_hbr_m and omit secondaryHardBodyRadius so each catalog
secondary keeps its own radius. That makes the effective combined radius per event
larger than 15 m, which raises Pc and returns more conjunctions, the safe
direction for a screen. Flag that this over-counts slightly and that the true
split is a physics call.

mahalanobisDistance. The reference describes it as "Supplied Mahalanobis distance
for ephemerides file", which does not clearly match our max_mahalanobis result
filter. Interim: send policy.mahalanobis_screen_threshold (4.0). If the corrected
submit returns a 422 naming this field, drop it and resubmit, then record what the
field actually wants.

## Tests

Offline, mocked client. Update the existing build_screening_request tests to
assert the multipart shape: primaryCatalogNumber equals primary_object,
missDistance equals the km threshold with no conversion, probabilityOfCollision is
the string PoC, primaryHardBodyRadius is 15.0, the ephemeris is present as a file
part and is the SCRUM-440 document, and none of secondaryCatalogNumber,
secondaryFile, the *Uncertainty fields, hardBodyRadius, screenAgainstAllObjects,
useFileUncertainty or a nested thresholds object is present. Add a client test that
create_screening posts multipart (files and data set, json not set) and still
passes through both limiters. Keep the real-422 fail-closed test; update its body
to the corrected shape or leave it asserting the historical raw 422 as a
regression guard, whichever is cleaner. Run python3 -m pytest services/planner and
confirm green.

## Live acceptance

One corrected submit on the stack, deliberately, respecting the 3-per-2-minutes
limit. Record in docs/scrum-441/live_verification.md: whether the multipart body
is accepted, the screening id and the terminal status enum observed (to confirm or
correct the injectable is_complete default), whether LeoLabs accepted the 865-state
300 s ephemeris and how the returned geometry looks at that step, and whether the
result is sane given the constant position-only covariance floor. Hand the
covariance-floor and HBR-split findings to John or Sreejit. Do not decide them.

## Commit

Same branch, scrum-441-ondemand-screening-client, updating PR #114. Include the
updated docs/scrum-441/. No attribution trailers, author John Avera
<javera@xorbita.com>. Report the new head SHA.

Then Cowork re-runs the verification-first review before John merges: it checks the
multipart body and the field names against the confirmed schema, runs the offline
suite, and reviews the corrected live submit and the three previously open
questions.
