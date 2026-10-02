"""tests/test_partial_warning.py

SCRUM-462: the partial-view warning as one toast plus one persistent chip.

partial_warning_harness.mjs extracts the real functions from index.html and drives
them against a stub DOM with controllable timers, so it can fire the auto-dismiss
and check the chip survives it. That is the property the redesign turns on: the
toast is allowed to disappear only because the chip is not.

This file runs that harness under pytest (skipped without node) and adds
template-wiring checks that need no node. The split is deliberate and was the point
SCRUM-461 made: a perfect render function that nothing calls still passes a pure
harness, and would ship a dashboard that stays silent on a truncated view.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).parent / "partial_warning_harness.mjs"
_TEMPLATE = Path(__file__).parent.parent / "app" / "templates" / "index.html"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available; the warning harness needs it")
def test_the_partial_warning_harness_passes():
    assert _HARNESS.exists(), f"missing harness at {_HARNESS}"
    proc = subprocess.run(
        ["node", str(_HARNESS)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        "the partial-warning harness failed:\n" + proc.stdout + "\n" + proc.stderr)
    assert "checks passed" in proc.stdout


@pytest.fixture(scope="module")
def html() -> str:
    return _TEMPLATE.read_text()


class TestTheOldBannersAreGone:
    """One signal, not three. The redesign must remove what it replaces."""

    @pytest.mark.parametrize("leftover", [
        "conjPartialBanner", "globePartialBanner", "partial-banner",
        "renderPartialBanner", "hidePartialBanner",
    ])
    def test_no_trace_of_the_scrum_461_banners(self, html, leftover):
        assert leftover not in html, (
            f"{leftover} survives; the old banner would show alongside the toast")


class TestSourceNeutralCopy:
    """The copy outlives today's provider, so it must not name it."""

    @pytest.fixture(scope="class")
    def copy_src(self, html) -> str:
        """The sentence builder, the copy wrapper and the head constant."""
        parts = []
        for fn in ("partialViewSentence", "partialViewCopy", "renderPartialChip"):
            m = re.search(r"^function " + fn + r"\(.*?^\}", html, re.S | re.M)
            assert m, f"{fn} is missing"
            parts.append(m.group(0))
        head = re.search(r"^const PARTIAL_HEAD = .*$", html, re.M)
        assert head
        parts.append(head.group(0))
        return "\n".join(parts)

    @pytest.mark.parametrize("name", ["LeoLabs", "leolabs", "LEOLABS", "Pulse"])
    def test_no_source_name_in_the_partial_copy(self, copy_src, name):
        assert name not in copy_src, (
            f"the partial-view copy names {name!r}; the warning will outlive the feed")

    def test_it_says_messages_are_not_in_risk_order(self, copy_src):
        assert "not in risk order" in copy_src

    def test_it_reads_only_the_three_backend_fields(self, copy_src):
        for field in ("cdms_pulled", "cdms_in_window", "truncated_by"):
            assert field in copy_src, f"{field} is not read"

    def test_both_truncation_reasons_are_translated(self, copy_src):
        assert "'deadline'" in copy_src and "time limit" in copy_src
        assert "'cap'" in copy_src and "size limit" in copy_src

    def test_the_head_names_the_risk(self, html):
        head = re.search(r"^const PARTIAL_HEAD = (.*)$", html, re.M).group(1)
        assert "PARTIAL VIEW" in head
        assert "WORST CONJUNCTION MAY NOT BE SHOWN" in head


class TestElementsAndStyle:
    def test_one_toast_and_one_chip_exist(self, html):
        assert html.count('id="partialToast"') == 1
        assert html.count('id="partialChip"') == 1

    def test_the_toast_is_app_level_not_per_panel(self, html):
        """One toast for the app: fixed to the viewport, declared once, at body top.

        Two toasts would let the table and the globe show different warnings about
        the same fetch, which is what the redesign exists to stop.
        """
        style = re.search(r"\.partial-toast\{([^}]*)\}", html).group(1)
        assert "position:fixed" in style
        body_at = html.index("<body>")
        toast_at = html.index('id="partialToast"')
        conj_at = html.index('id="conjToolbar"')
        assert body_at < toast_at < conj_at, "the toast is not at the top of the app"

    def test_the_chip_sits_with_the_pager(self, html):
        """The pager line is on screen in both the 2D and the globe view."""
        toolbar = html.split('id="conjToolbar"')[1].split("</div>\n        </div>")[0]
        assert 'id="partialChip"' in toolbar
        assert 'id="conjRange"' in toolbar

    def test_both_are_hidden_until_shown(self, html):
        toast = re.search(r"\.partial-toast\{([^}]*)\}", html).group(1)
        chip = re.search(r"\.partial-chip\{([^}]*)\}", html).group(1)
        # The toast is parked off-screen rather than display:none so it can slide.
        assert "translateY(-130%)" in toast
        assert "display:none" in chip
        assert ".partial-toast.show{" in html
        assert ".partial-chip.show{display:inline-flex}" in html

    def test_the_toast_has_a_close_control(self, html):
        copy_fn = re.search(r"^function partialViewCopy\(.*?^\}", html,
                            re.S | re.M).group(0)
        assert "dismissPartialToast()" in copy_fn
        assert "pt-close" in copy_fn


class TestTheChipIsTheDurableOne:
    """The structural half of the safety argument.

    The harness proves the behaviour; these prove the code is shaped so the
    behaviour cannot be undone by a later edit that looks harmless.
    """

    def test_dismissing_the_toast_does_not_touch_the_chip(self, html):
        fn = re.search(r"^function dismissPartialToast\(.*?^\}", html,
                       re.S | re.M).group(0)
        assert "partialChip" not in fn
        assert "renderPartialChip" not in fn
        assert "partialViewState" not in fn.replace("partialToastDismissed", "")

    def test_the_auto_dismiss_only_hides_the_toast(self, html):
        fn = re.search(r"^function showPartialToast\(.*?^\}", html,
                       re.S | re.M).group(0)
        timeout = fn[fn.index("setTimeout"):]
        assert "partialChip" not in timeout, (
            "the auto-dismiss reaches the chip; the condition would vanish with it")

    def test_the_chip_renders_from_the_state_not_from_the_toast(self, html):
        fn = re.search(r"^function renderPartialChip\(.*?^\}", html,
                       re.S | re.M).group(0)
        assert "partialViewState" in fn
        assert "partialToast" not in fn

    def test_the_chip_tooltip_carries_the_counts(self, html):
        fn = re.search(r"^function renderPartialChip\(.*?^\}", html,
                       re.S | re.M).group(0)
        assert "partialViewSentence" in fn
        assert ".title" in fn


class TestWiring:
    """Driven from the responses, and cleared on everything that changes the view."""

    def test_the_list_fetch_sets_it_from_its_response(self, html):
        fetch = html.split("async function fetchLiveConjunctions")[1].split("\n}")[0]
        assert "setPartialView(data)" in fetch

    def test_the_globe_fetch_sets_it_from_its_response(self, html):
        fetch = html.split("async function fetchLiveGlobeOrbits")[1].split("\n}")[0]
        assert "setPartialView(data)" in fetch

    @pytest.mark.parametrize("fn_name", [
        "async function fetchLiveConjunctions",
        "async function fetchLiveGlobeOrbits",
        "function clearConjunctionsForSourceSwitch",
        "function clearAll",
    ])
    def test_everything_that_changes_the_view_clears_it(self, html, fn_name):
        body = html.split(fn_name)[1].split("\n}")[0]
        assert "clearPartialView()" in body, f"{fn_name} does not clear the warning"

    def test_the_list_clears_before_it_fetches(self, html):
        """The error paths return early, so clearing has to come first."""
        fetch = html.split("async function fetchLiveConjunctions")[1].split("\n}")[0]
        assert fetch.index("clearPartialView()") < fetch.index("await fetch(")

    def test_the_globe_clears_before_it_fetches(self, html):
        fetch = html.split("async function fetchLiveGlobeOrbits")[1].split("\n}")[0]
        assert fetch.index("clearPartialView()") < fetch.index("await fetch(")

    def test_the_rows_are_not_gated_behind_the_warning(self, html):
        """The rows that came back are real; the warning sits with them."""
        fetch = html.split("async function fetchLiveConjunctions")[1].split("\n}")[0]
        table_at = fetch.index("updateConjunctionTable(rows)")
        warn_at = fetch.index("setPartialView(data)")
        assert table_at < warn_at
        assert "else" not in fetch[table_at:warn_at]

    def test_partial_is_read_as_an_explicit_flag(self, html):
        """SCRUM-461's invariant: absent fields mean complete, not partial."""
        fn = re.search(r"^function setPartialView\(.*?^\}", html, re.S | re.M).group(0)
        assert "data.partial === true" in fn
        assert "data.complete === false" in fn


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_481_policy_comparison_behaviour():
    # Controlled response fixtures exercise the actual dashboard JavaScript.
    html = _TEMPLATE.read_text(encoding="utf-8")
    start = html.index("// SCRUM-481: display backend comparisons")
    end = html.index("let policyDefaults=", start)
    functions = html[start:end]
    paging = re.search(
        r"^function conjPagingUrl\(.*?^\}", html, re.S | re.M,
    ).group(0)
    view = re.search(
        r"^function setView\(.*?^\}", html, re.S | re.M,
    ).group(0)

    setup = r"""
const assert=require('node:assert/strict');
const elements={};
function element(id){
  if(!elements[id]){
    const classes=new Set();
    elements[id]={
      innerHTML:'',textContent:'',style:{},disabled:false,
      classList:{
        add:name=>classes.add(name),
        remove:name=>classes.delete(name),
        toggle:(name,on)=>on?classes.add(name):classes.delete(name),
        contains:name=>classes.has(name)
      }
    };
  }
  return elements[id];
}
const document={
  getElementById:element,
  querySelector:()=>element('canvasWrap')
};
element('sliderLv').value='0.01';
element('sliderLl').value='0.005';
element('sliderDv').value='1';
element('pcActionSelect').value='aps';
let live=true;
let liveAssetNorad='36508';
const conjPaging={pageSize:100,inVolume:true};
function isLiveMode(){return live;}
let currentView='2d',cameraFollowActive=false;
let currentConj=null,currentPlannerResult=null;
let stopped=0;
function stopGlobeLoop(){stopped++;}
function drawEncounterEmpty(){}
function drawEncounter(){}
function resizeEncounter(){}
function requestAnimationFrame(fn){fn();}
"""

    functions += re.search(
        r"^let policyDefaults=.*$", html, re.M,
    ).group(0)
    functions += "\n" + re.search(
        r"^function onPolicyChange\(.*?^\}", html, re.S | re.M,
    ).group(0)
    checks = r"""
async function checks(){
  const modes=['aps','flight_rule_1e4','flight_rule_1e5'];
  const fixture={
    analysis_only:true,primary_norad:36508,
    complete:true,partial:false,partial_reasons:[],
    count:1,events_not_attempted:0,failures:[],
    window:{min_tca_utc:'2026-10-01',max_tca_utc:'2026-10-08'},
    cost:{processing_ms:10,total_ms:100,scoring_calls:1},
    modes:{
      aps:{maneuver_count:1,total_dv_m_s:0.5,known_dv_m_s:0.5,unpriced_maneuver_count:0},
      flight_rule_1e4:{maneuver_count:0,total_dv_m_s:0,known_dv_m_s:0,unpriced_maneuver_count:0},
      flight_rule_1e5:{maneuver_count:1,total_dv_m_s:0.5,known_dv_m_s:0.5,unpriced_maneuver_count:0}
    },
    conjunctions:[{
      event_id:'<unsafe>',secondary_norad:43476,tca_utc:'2026-10-02',
      pc_pre:0.00005,pc_source:'supplied',
      best_burn:{utility:2,utility_basis:'pc_traded'},
      modes:{
        aps:{maneuver_required:true,dv_m_s:0.5,pricing_available:true},
        flight_rule_1e4:{maneuver_required:false,dv_m_s:0,pricing_available:true},
        flight_rule_1e5:{maneuver_required:true,dv_m_s:0.5,pricing_available:true}
      }
    }]
  };
  const clone=()=>JSON.parse(JSON.stringify(fixture));
  renderPolicyPortfolio(fixture);
  assert.equal((element('policyAbRollup').innerHTML.match(/<h4>/g)||[]).length,3);
  assert.match(element('policyAbRollup').innerHTML,/Policy-derived \(APS\)/);
  assert.match(element('policyAbRollup').innerHTML,/Flight rule 1e-4/);
  assert.match(element('policyAbRollup').innerHTML,/Flight rule 1e-5/);
  assert.match(element('policyAbRollup').innerHTML,/0.5 m\/s/);
  assert.match(element('policyAbRows').innerHTML,/MANEUVER REQUIRED/);
  assert.match(element('policyAbRows').innerHTML,/NOT REQUIRED/);
  assert.match(element('policyAbRows').innerHTML,/&lt;unsafe&gt;/);
  assert.ok(!element('policyAbRows').innerHTML.includes('<unsafe>'));
  assert.equal(element('policyAbCoverage').style.display,'none');

  const partial=clone();
  partial.complete=false;partial.partial=true;
  partial.partial_reasons=['fetch_cap','processing_deadline','event_failures'];
  partial.events_not_attempted=3;partial.failures=[{reason:'controlled'}];
  renderPolicyPortfolio(partial);
  assert.equal(element('policyAbCoverage').style.display,'block');
  assert.match(element('policyAbCoverage').textContent,/PARTIAL WINDOW/);
  assert.match(element('policyAbCoverage').textContent,/fetch size limit/);
  assert.match(element('policyAbCoverage').textContent,/scoring time limit/);
  assert.match(element('policyAbCoverage').textContent,/Not attempted: 3/);
  assert.match(element('policyAbRollup').innerHTML,/analyzed events only/);
  assert.match(element('policyAbRows').innerHTML,/MANEUVER REQUIRED/);

  const unknown=clone();
  delete unknown.complete;delete unknown.partial;
  renderPolicyPortfolio(unknown);
  assert.equal(element('policyAbCoverage').style.display,'block');

  const unpriced=clone();
  unpriced.modes.flight_rule_1e4={
    maneuver_count:1,total_dv_m_s:null,known_dv_m_s:0,unpriced_maneuver_count:1
  };
  unpriced.conjunctions[0].modes.flight_rule_1e4={
    maneuver_required:true,dv_m_s:null,pricing_available:false
  };
  renderPolicyPortfolio(unpriced);
  assert.match(element('policyAbRollup').innerHTML,/Unavailable/);
  assert.match(element('policyAbRollup').innerHTML,/Known subtotal: 0 m\/s/);
  assert.match(element('policyAbRows').innerHTML,/Unavailable/);
  assert.equal(policyAbDv(null),'Unavailable');
  assert.equal(policyAbDv(0),'0 m/s');
  assert.notEqual(policyAbDv(0.00000001),'0 m/s');

  const empty=clone();
  empty.count=0;empty.conjunctions=[];
  modes.forEach(mode=>empty.modes[mode]={
    maneuver_count:0,total_dv_m_s:0,known_dv_m_s:0,unpriced_maneuver_count:0
  });
  renderPolicyPortfolio(empty);
  assert.match(element('policyAbRows').innerHTML,/No conjunctions/);
  empty.complete=false;empty.partial=true;
  renderPolicyPortfolio(empty);
  assert.match(element('policyAbRows').innerHTML,/not a confirmed empty window/);


  element('pcActionSelect').value='flight_rule_1e4';
  const evaluatedPolicy=getPolicy();
  currentPlannerResult={};
  recordEvaluatedPolicy(currentPlannerResult,evaluatedPolicy);
  onPolicyChange();
  assert.equal(element('replanBtn').disabled,true);
  element('pcActionSelect').value='aps';
  onPolicyChange();
  assert.equal(element('replanBtn').disabled,false);
  assert.match(element('decisionModeProvenance').textContent,/re-plan to apply/);
  recordEvaluatedPolicy(currentPlannerResult,getPolicy());
  onPolicyChange();
  assert.equal(element('replanBtn').disabled,true);
  assert.match(element('decisionModeProvenance').textContent,/Verdict driven by: Policy-derived/);
  currentPlannerResult=null;

  const key=policyPortfolioRequest().key;
  element('pcActionSelect').value='flight_rule_1e4';
  assert.equal(policyPortfolioRequest().key,key);
  assert.equal(getPolicy().decision_mode,'flight_rule_1e4');
  assert.equal(getPolicy().pc_maneuver_threshold,0.0001);
  updateDecisionModeProvenance({_decision_mode:'aps'});
  assert.match(element('decisionModeProvenance').textContent,/Verdict driven by: Policy-derived/);
  assert.match(element('decisionModeProvenance').textContent,/re-plan to apply/);
  updateDecisionModeProvenance({_decision_mode:'flight_rule_1e4'});
  assert.ok(!element('decisionModeProvenance').textContent.includes('re-plan'));
  updateDecisionModeProvenance(null);
  assert.match(element('decisionModeProvenance').textContent,/no evaluation yet/);

  let calls=0;
  global.fetch=async(url,options)=>{
    calls++;
    assert.ok(url.startsWith('/api/planner/portfolio?'));
    assert.ok(!url.includes('page_size'));
    assert.equal(options.method,'POST');
    const body=JSON.parse(options.body);
    assert.equal(body.satellite.v_remaining_m_s,25);
    assert.ok(!('authorization' in body));
    return {ok:true,json:async()=>fixture};
  };
  await refreshPolicyPortfolio();
  assert.equal(calls,1);
  assert.equal(policyAbState.data,fixture);
  element('pcActionSelect').value='flight_rule_1e5';
  synchronizePolicyPortfolio();
  assert.equal(calls,1);
  assert.match(element('policyAbHead').innerHTML,/<th class="active">Flight rule 1e-5/);

  element('sliderDv').value='0.5';
  synchronizePolicyPortfolio();
  assert.equal(policyAbState.data,null);
  assert.equal(element('policyAbRollup').innerHTML,'');

  let release;
  global.fetch=()=>new Promise(resolve=>release=resolve);
  const pending=refreshPolicyPortfolio();
  liveAssetNorad='43476';
  resetPolicyPortfolio();
  release({ok:true,json:async()=>fixture});
  await pending;
  assert.equal(policyAbState.data,null);
  assert.equal(element('policyAbRollup').innerHTML,'');

  global.fetch=async()=>({ok:false,status:503,json:async()=>({error:'controlled busy'})});
  await refreshPolicyPortfolio();
  assert.match(element('policyAbStatus').textContent,/Comparison unavailable/);
  assert.equal(element('policyAbRollup').innerHTML,'');
  assert.equal(element('policyAbRefresh').disabled,false);

  global.fetch=async()=>({ok:true,json:async()=>({})});
  await refreshPolicyPortfolio();
  assert.match(element('policyAbStatus').textContent,/Invalid portfolio response/);

  policyAbState.data=fixture;
  policyAbState.key=policyPortfolioRequest().key;
  setView('policy');
  assert.equal(element('policyAbWrap').style.display,'flex');
  assert.equal(element('canvasWrap').style.display,'none');
  assert.equal(element('globeWrap').style.display,'none');
  assert.ok(element('btnPolicyAb').classList.contains('active'));
  setView('2d');
  assert.equal(element('policyAbWrap').style.display,'none');
  assert.equal(element('canvasWrap').style.display,'');
  assert.ok(!element('btnPolicyAb').classList.contains('active'));
  assert.ok(stopped>0);

  live=false;
  resetPolicyPortfolio();
  assert.match(element('policyAbStatus').textContent,/Scenario mode has no live portfolio/);
  assert.equal(element('policyAbRefresh').disabled,true);
  console.log('Policy comparison checks passed');
}
checks().catch(error=>{console.error(error);process.exitCode=1;});
"""
    proc = subprocess.run(
        ["node", "-e", setup + paging + functions + view + checks],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "Policy comparison checks passed" in proc.stdout


@pytest.mark.parametrize("anchor", [
    'recordEvaluatedPolicy(result,request.policy);',
    'updateDecisionModeProvenance(result);',
    'updateDecisionModeProvenance(currentPlannerResult);',
    'onclick="setView(\'policy\')"',
    'onclick="refreshPolicyPortfolio()"',
])
def test_481_mode_and_comparison_are_wired(anchor):
    html = _TEMPLATE.read_text(encoding="utf-8")
    assert anchor in html


@pytest.mark.parametrize("name", [
    "clearConjunctionsForSourceSwitch", "clearAll", "toggleConjVolumeFilter",
])
def test_481_query_changes_clear_the_old_comparison(name):
    html = _TEMPLATE.read_text(encoding="utf-8")
    body = re.search(
        r"^function " + name + r"\(.*?^\}", html, re.S | re.M,
    ).group(0)
    assert "resetPolicyPortfolio();" in body
