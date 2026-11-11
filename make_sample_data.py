#!/usr/bin/env python3
"""Generate the synthetic evaluation dataset (15 files) + ground-truth manifest.

Dataset design (documented in dataset_manifest.json and EVALUATION.md):

  Ground truth planted in the corpus:
    * exact copy pair .......... submission_02.docx  <- submission_01.pdf  (peer)
    * paraphrase pair .......... submission_09.docx  <- submission_04.docx  (peer)
    * cross-modal copy ......... pitch_12.mp4 speaks a paragraph verbatim from submission_03.pdf
    * external (web-like) copy . submission_06.pdf lifts 3 paragraphs from reference_corpus/ev_charging_infrastructure.txt
    * AI-generated document .... submission_05.docx  (LLM-style text, heuristic-detectable)
    * benign originals ......... submission_07, 08, 15, pitch_14.mp4  (false-positive tests)
    * shared template header ... every analytical submission quotes the same ~110-word case prompt
                                 (Scenario 4: benign common formatting)
    * degraded files ........... submission_11.pdf (zero-byte), pitch_13.mp4 (silent, no audio)
    * low-content submission ... submission_10.docx

  All text is synthetic and written for this assessment - no copyrighted data.
  MP4 speech is synthesized with the offline Windows SAPI TTS voice and muxed
  with ffmpeg (bundled via imageio-ffmpeg); no network or copyrighted media.

Run:  python make_sample_data.py [--dir sample_submissions] [--force]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from integrity.utils import get_ffmpeg_exe, get_logger

logger = get_logger("make_data")

# --------------------------------------------------------------------------- #
# Shared template header (Scenario 4) - identical block quoted in submissions
# --------------------------------------------------------------------------- #
TEMPLATE_HEADER = (
    "Case Prompt (identical for all candidates - do not modify): You are a strategy "
    "consultant engaged by the City Development Authority. Rising demand on public "
    "services and tightening municipal budgets have created pressure to identify "
    "fundable, high-impact interventions. Your task is to evaluate the feasibility of "
    "two shortlisted infrastructure initiatives, quantify expected costs and benefits "
    "over a five-year horizon, and recommend a phased implementation plan. Your "
    "submission must address financing options, stakeholder resistance, and delivery "
    "risks, and must include an executive summary, a structured analysis, and a "
    "clearly justified final recommendation supported by evidence. Word limit: 600 "
    "words. Citations are permitted where they support the analysis. Late submissions "
    "are scored as incomplete, and all claims must be verifiable from the provided "
    "case pack data."
)

# --------------------------------------------------------------------------- #
# Submission bodies (synthetic, human-style unless noted)
# --------------------------------------------------------------------------- #
S01 = {
    "title": "Solar Microgrids for the Industrial Estate",
    "sections": [
        ("Executive Summary", [
            "Rooftop solar paired with neighbourhood batteries can cut the authority's peak load by roughly a fifth within four years, and it pays for itself before the five-year mark if we phase the rollout correctly.",
            "Our recommendation is to start with the two feeders that serve the industrial estate, where demand is heaviest and roof space is already leased.",
        ]),
        ("Analysis", [
            "The first option, a 4 MW ground-mounted array on the old landfill site, produces cheap daytime energy but needs a new substation spur, and the landfill cap complicates foundations.",
            "We estimate 6.8 million in capital cost and a 14-month build.",
            "The second option, distributed rooftop systems across 240 commercial roofs, costs about the same in aggregate but spreads delivery risk across many small contracts.",
            "It also sidesteps the substation spur because the network already has headroom on the affected feeders.",
            "Battery sizing is the awkward part; a 2 MWh store covers evening peaks in winter but sits idle most of the year.",
            "We sized batteries at 1.2 MWh and accepted a small amount of peak spillover in the worst fortnight.",
            "The finance model uses a municipal green bond; rate assumptions are conservative, a 4.9 percent coupon with a 25-year tenor.",
        ]),
        ("Recommendation", [
            "Phase one should cover the industrial estate feeders in year one and two, with rooftop installations contracted in tranches of sixty roofs so local electricians can compete.",
            "Phase two extends to schools and depots where roofs are simple and loads are predictable.",
            "The community opposed the landfill array at the last consultation, so we put it in phase three behind another traffic study.",
            "If grants from the state energy office land, the payback shortens by about eight months.",
        ]),
    ],
}

S03 = {
    "title": "Restructuring the Bus Network Before the Congestion Trial",
    "sections": [
        ("Executive Summary", [
            "The bus network restructure is fundable this year and the congestion pilot is not ready, so the sequencing question answers itself.",
            "We recommend fixing the airport corridor contract first, then holding the charging trial until after the council elections.",
        ]),
        ("Analysis", [
            "Operating margins for the city bus network contracted by 4.2 percent over the last three quarters because of rising driver wages and fuel costs, while ridership on the airport corridor grew by eleven percent after the express line opened.",
            "The depot electrification pilot reduced maintenance outlays by nine percent, but charging downtime still erodes those savings during peak service hours.",
            "Any restructuring of routes must therefore protect the airport corridor, which now generates a disproportionate share of network revenue.",
            "Congestion charging remains politically fragile.",
            "The trial zone cut weekday car entries by 17 percent, but two councillors face re-election next spring and will not endorse anything that smells like a commuter tax.",
            "The park-and-ride expansion at Hallam Road was cheap and it worked; occupancy hit 85 percent within six weeks.",
        ]),
        ("Recommendation", [
            "Fund the bus network restructure first, hold the congestion pilot until after the election cycle, and spend the remaining contingency on shelter upgrades along the eastern routes, which scored highest in the accessibility audit.",
            "Publish the corridor performance dashboard monthly so the panel can verify ridership claims without waiting for the annual report.",
        ]),
    ],
}

S04 = {
    "title": "Sensor Data as the Co-op's Next Revenue Line",
    "sections": [
        ("Executive Summary", [
            "Our co-op's soil sensors cut fertiliser spend by roughly a sixth last season, and the grain buyers started paying us a premium for traceability data.",
            "That's the kernel of this submission: the tech pays only when the data flows to someone who'll pay for it.",
        ]),
        ("Analysis", [
            "We looked at three options: expand the sensor grid across the southern paddocks, subscribe to the regional yield-model service, or buy the drone-scouting kit outright.",
            "Option three looks shiny, but the depreciation schedule is brutal and our agronomist team would spend half their week babysitting batteries.",
            "Option two is the sneaky-good deal; the service costs 12 grand a year and slots into the buyer's existing portal.",
            "We'd pair it with forty new sensors staged over two seasons.",
        ]),
        ("Recommendation", [
            "Go with option two plus 40 new sensors, with a fall-back clause if the buyer's premium dries up.",
            "The co-op board asked about drought resilience; the yield model doesn't forecast rainfall, it models soil moisture, which is what our insurance broker actually needs.",
        ]),
    ],
}

S05 = {  # LLM-style text (ground truth: AI-generated) - deliberately uniform
    # sentence lengths, dense discourse markers, zero contractions/first person.
    "title": "AI-Driven Crop Advisory Platform: A Phased Integration Strategy",
    "sections": [
        ("Executive Summary", [
            "This report evaluates the integration of an AI-driven crop advisory platform into the regional agricultural support framework.",
            "The proposed system leverages satellite imagery, historical yield data, and in-field sensor inputs to generate weekly operational recommendations.",
            "Furthermore, it outlines a phased implementation strategy designed to maximise adoption while minimising operational risk.",
            "Importantly, the platform is intended to complement, rather than replace, the professional judgement of local agronomists.",
        ]),
        ("Analysis", [
            "Consequently, all advisory outputs are presented as a structured decision-support layer.",
            "In addition, the implementation plan is organised across three distinct phases.",
            "Phase one establishes the underlying data infrastructure and integrates existing sensor networks.",
            "Phase two deploys the advisory service to a pilot cohort of two hundred registered growers.",
            "Phase three expands regional coverage and introduces multilingual support for all users.",
            "Moreover, a continuous evaluation framework will monitor adoption rates and recommendation accuracy over time.",
        ]),
        ("Recommendation", [
            "However, several material risks must be acknowledged and addressed comprehensively.",
            "These risks include data privacy concerns, rural connectivity limitations, and potential over-reliance on automated outputs.",
            "Nevertheless, each identified risk is mitigated through targeted and proportionate controls.",
            "In conclusion, the advisory platform offers a scalable and cost-effective modernisation pathway.",
            "Its phased deployment therefore ensures early value delivery while residual risks remain carefully managed.",
        ]),
    ],
}

S06 = {
    "title": "Public EV Charging: Network Design and Financing",
    "sections": [
        ("Executive Summary", [
            "This submission evaluates the city's shortlist for public charging infrastructure and recommends a financing structure for the first tranche of stations.",
        ]),
        ("Analysis", [
            "An electric vehicle charging network is an infrastructure system of publicly accessible charging stations deployed along roadways and in urban areas to support battery electric vehicles.",
            "Charging stations are typically categorised into three levels based on power output.",
            "Level 1 charging uses a standard household outlet and delivers the slowest charge.",
            "Level 2 charging uses dedicated equipment rated between 7 and 19 kilowatts and is the most common public option, while DC fast charging delivers 50 to 350 kilowatts and can recharge most vehicles in under an hour.",
            "Network operators generally adopt one of two business models.",
            "In the asset-owner model, the operator finances, builds and maintains the stations and earns revenue from electricity sales and usage fees.",
            "In the franchise model, site hosts such as retailers or fuel stations own the hardware and pay the network operator for software, billing services and roaming access.",
            "Revenue per station depends heavily on utilisation, and industry analysis suggests that most public stations require utilisation above fifteen percent to reach breakeven.",
            "Grid integration is a central planning concern.",
            "Unmanaged charging can coincide with evening demand peaks and force costly network upgrades, so utilities increasingly deploy smart-charging programmes that shift load to overnight periods or solar-rich hours.",
        ]),
        ("Recommendation", [
            "We recommend the franchise model for the first sixty stations, with the asset-owner model reserved for three highway plazas where utilisation risk is lowest.",
        ]),
    ],
}

S07 = {
    "title": "Water: Fix the Leaks Before the Brochures",
    "sections": [
        ("Executive Summary", [
            "Fixing the northern mains leak cluster is the highest-return water project on the shortlist.",
        ]),
        ("Analysis", [
            "Detectors logged 1.9 megalitres of monthly loss across eleven hydrants; repairing them costs about 740 thousand dollars and pays back in under three years at the current bulk price.",
            "Greywater recycling for the two aquatic centres looks attractive on brochures but the retrofit plumbing quotes came back at triple the feasibility study's estimate, so we parked it.",
            "Smart metering the commercial basin, about 4,100 accounts, is the sleeper: the trial found 6 percent of accounts with faulty old meters that under-billed, which is pure recovered revenue.",
        ]),
        ("Recommendation", [
            "Rainwater harvesting mandates for new builds don't move the needle much at city scale; the yield estimates are modest and enforcement would land on a building department that is already two years behind on inspections.",
            "We recommend the leak repairs and the commercial metering roll-out, sequenced so the recovered revenue part-funds the second stage.",
        ]),
    ],
}

S08 = {
    "title": "Merchant Credit That Uses the Bills People Already Pay",
    "sections": [
        ("Executive Summary", [
            "The municipal micro-loan fund has quietly become the city's best small-business programme, but it's buried under the general grants portal and half the eligible merchants never find it.",
        ]),
        ("Analysis", [
            "Our team mystery-shopped the application flow in February: eleven screens, three document uploads, and a median completion time of 34 minutes for the merchants we shadowed.",
            "Moving to a pre-filled application using rates-billing data would cut that to about 8 minutes for two-thirds of applicants.",
            "The bigger lever is credit scoring: the fund rejects 61 percent of first-time applicants, mostly for thin credit files, yet 83 percent of those same merchants repaid their rates and utility bills on time for the last three years.",
            "Cash-flow-based underwriting using that billing history would approve roughly 300 additional merchants a year without breaching the fund's loss appetite.",
        ]),
        ("Recommendation", [
            "We recommend the simplified application, the billing-data underwriting pilot, and a partnership with the two community banks that already run merchant accounts for these traders.",
        ]),
    ],
}

S09 = {  # paraphrase of S04 (Scenario 2) - same arguments, altered phrasing
    "title": "Turning Cooperative Sensor Data Into a Paying Product",
    "sections": [
        ("Executive Summary", [
            "Last season, sensor coverage across our cooperative reduced fertiliser costs by close to 17 percent, and grain purchasers began offering a premium tied to traceability information.",
            "That outcome is the central idea of this proposal: technology generates returns only when the resulting data reaches a party willing to pay for it.",
        ]),
        ("Analysis", [
            "Three pathways were assessed: extending the sensor network over the southern paddocks, purchasing a subscription to the regional yield-modelling service, and acquiring drone scouting hardware outright.",
            "The third route appears attractive, yet its depreciation schedule is punishing and our agronomy staff would lose much of their week maintaining batteries.",
            "The second option is quietly the strongest; it costs about 12 thousand dollars annually and plugs directly into the grain buyer's existing portal.",
            "Our preferred package combines option two with forty additional sensors rolled out across two seasons.",
        ]),
        ("Recommendation", [
            "The rollout should carry a fallback clause should the buyer's premium be withdrawn.",
            "On drought resilience, the board should note that the yield model does not forecast rainfall; it simulates soil moisture, which is precisely the variable our insurance broker requires.",
        ]),
    ],
}

S10 = {  # low-content submission
    "title": "Short Response",
    "sections": [
        ("Response", [
            "After reviewing the case pack, we believe the council should prioritise the option with the lowest delivery risk and the clearest funding path.",
            "We look forward to discussing the detailed model with the panel.",
        ]),
    ],
}

S15 = {
    "title": "Closing the After-Hours GP Gap on the Western Fringe",
    "sections": [
        ("Executive Summary", [
            "The after-hours GP gap on the western fringe is the costliest service failure in the case pack: 6,400 residents sit more than 25 minutes from any open clinic after 7pm.",
            "Ambulance data shows a spike in low-acuity callouts between 8pm and midnight.",
        ]),
        ("Analysis", [
            "A nurse-led telehealth hub co-located with the Hallam fire station answers that gap at a fraction of a new clinic build.",
            "Two nearby practices have already agreed to roster on-call GPs for escalation between 8pm and 1am, which covers 80 percent of the observed callout spike.",
            "The second initiative in the shortlist, mobile screening vans, performs well in publicity but poorly in the data: the trial van screened 9 residents per shift at a staffing cost of 610 dollars each, versus 38 dollars per screening at the hub.",
        ]),
        ("Recommendation", [
            "We recommend funding the hub for an 18-month trial and redirecting the van budget into weekend clinic hours at the two partner practices.",
        ]),
    ],
}

PITCH_12_UNIQUE_INTRO = (
    "Good afternoon, panel members. My name is Alex, and I am presenting our team's "
    "mobility proposal for the corridor review."
)
PITCH_12_SPOKEN = (
    "Operating margins for the city bus network contracted by 4.2 percent over the "
    "last three quarters because of rising driver wages and fuel costs, while "
    "ridership on the airport corridor grew by eleven percent after the express line "
    "opened. The depot electrification pilot reduced maintenance outlays by nine "
    "percent, but charging downtime still erodes those savings during peak service "
    "hours. Any restructuring of routes must therefore protect the airport corridor, "
    "which now generates a disproportionate share of network revenue."
)
PITCH_12_OUTRO = "Thank you for listening."

PITCH_14 = (
    "Hi everyone, I'm Priya. Our proposal upgrades the floodlighting at the three "
    "public sports fields to LED fittings with smart timers. The old metal halide "
    "lamps burn all evening even when the fields sit empty, and they use double the "
    "energy per lux. Metered trials at the Hallam field last winter show a 58 "
    "percent energy cut, and the light quality complaints stopped completely. "
    "Payback lands inside four years at current tariffs. We also suggest a small "
    "booking app so clubs can reserve slots, which trims the burning hours another "
    "ten percent. Simple, measured, and already trialled. Thanks."
)

# --------------------------------------------------------------------------- #
# Generators: PDF / DOCX / MP4
# --------------------------------------------------------------------------- #
def make_pdf(path: Path, title: str, sections) -> None:
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        PageTemplate,
        Paragraph,
        Spacer,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyBig", parent=styles["BodyText"], fontSize=13, leading=22,
        alignment=TA_JUSTIFY, spaceAfter=10,
    )
    h2 = ParagraphStyle("H2Big", parent=styles["Heading2"], fontSize=17,
                        leading=24, spaceBefore=14, spaceAfter=8)
    title_style = ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=22,
                                 leading=30, spaceAfter=12)
    story = [Paragraph(title, title_style), Spacer(1, 6 * mm)]
    for heading, paras in sections:
        story.append(Paragraph(heading, h2))
        for p in paras:
            story.append(Paragraph(p, body))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(TEMPLATE_HEADER, styles["Italic"]))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawString(
            20 * mm, A4[1] - 12 * mm, "City Development Authority Case Competition 2026"
        )
        canvas.drawCentredString(A4[0] / 2, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()

    frame = Frame(20 * mm, 20 * mm, A4[0] - 40 * mm, A4[1] - 40 * mm, id="body")
    doc = BaseDocTemplate(
        str(path), pagesize=A4,
        pageTemplates=[PageTemplate(id="main", frames=[frame], onPage=on_page)],
    )
    doc.build(story)


def make_docx(path: Path, title: str, sections) -> None:
    import docx

    document = docx.Document()
    document.add_heading(title, level=1)
    for heading, paras in sections:
        document.add_heading(heading, level=2)
        for p in paras:
            document.add_paragraph(p)
    document.add_paragraph(TEMPLATE_HEADER)
    document.save(str(path))


def tts_to_wav(text: str, wav_path: Path) -> None:
    """Synthesize speech with the offline Windows SAPI TTS engine."""
    ps1 = wav_path.with_suffix(".ps1")
    safe = text.replace("`", "``").replace('"', '`"').replace("$", "`$")
    script = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture -like 'en*' }}
if ($voices) {{ $s.SelectVoice($voices[0].VoiceInfo.Name) }}
$s.SetOutputToWaveFile('{wav_path}')
$s.Rate = 1
$s.Speak("{safe}")
$s.Dispose()
"""
    ps1.write_text(script, encoding="utf-8-sig")
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
        capture_output=True, text=True, timeout=180,
    )
    ps1.unlink(missing_ok=True)
    if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 1000:
        raise RuntimeError(f"SAPI TTS failed: {proc.stderr.strip()[:300]}")


def make_mp4(path: Path, spoken_text: str | None, silent_only: bool = False) -> None:
    """Create an MP4: solid-colour video with (optional) spoken audio track."""
    ffmpeg = get_ffmpeg_exe()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg unavailable (expected via imageio-ffmpeg).")
    with_tmp = Path("C:/Users/Lenovo/AppData/Local/Temp/opencode")
    with_tmp.mkdir(parents=True, exist_ok=True)
    if silent_only:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=0x22303F:s=480x270:r=10",
            "-t", "6", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", str(path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return
    wav = with_tmp / f"{path.stem}_tts.wav"
    tts_to_wav(spoken_text, wav)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(wav),
        "-f", "lavfi", "-i", "color=c=0x1B2A41:s=480x270:r=10",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    wav.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Manifest (ground truth)
# --------------------------------------------------------------------------- #
MANIFEST = {
    "dataset": "sample_submissions",
    "description": (
        "Synthetic multi-format submission corpus for the Unstop AI/ML Engineer "
        "assessment. All text is original synthetic content written for this "
        "assessment; MP4 speech is offline Windows SAPI TTS. Ground-truth "
        "relationships are listed below and used as evaluation labels in "
        "EVALUATION.md."
    ),
    "file_counts": {"pdf": 5, "docx": 7, "mp4": 3, "total": 15},
    "shared_template_header": {
        "text_snippet": TEMPLATE_HEADER[:120] + "...",
        "quoted_by": [
            "submission_01.pdf", "submission_02.docx", "submission_03.pdf",
            "submission_04.docx", "submission_05.docx", "submission_06.pdf",
            "submission_07.pdf", "submission_08.docx", "submission_09.docx",
            "submission_10.docx", "submission_15.docx",
        ],
        "expected_behavior": (
            "Scenario 4: the shared case prompt must be detected as cross-corpus "
            "boilerplate and discounted, NOT scored as plagiarism."
        ),
    },
    "plagiarism_relations": [
        {
            "query": "submission_02.docx",
            "source": "submission_01.pdf",
            "relation": "exact_copy",
            "signal": "peer",
            "note": "Body text is word-for-word identical; format differs.",
        },
        {
            "query": "submission_09.docx",
            "source": "submission_04.docx",
            "relation": "paraphrase",
            "signal": "peer",
            "note": "Synonym-swapped rewording; lexical overlap low, semantic overlap high.",
        },
        {
            "query": "pitch_12.mp4",
            "source": "submission_03.pdf",
            "relation": "verbatim_spoken",
            "signal": "peer / cross-modal",
            "note": "Transcript contains a paragraph of the PDF verbatim (Scenario 1).",
        },
        {
            "query": "submission_06.pdf",
            "source": "reference_corpus/ev_charging_infrastructure.txt",
            "relation": "external_copy",
            "signal": "external",
            "note": "Three paragraphs lifted near-verbatim from the open reference KB.",
        },
    ],
    "ai_generated": [
        {
            "file": "submission_05.docx",
            "note": "LLM-style prose (uniform sentence length, heavy discourse markers, no contractions). Ground truth for the AI-content signal.",
        }
    ],
    "independent_originals": [
        "submission_01.pdf",
        "submission_03.pdf",
        "submission_04.docx",
        "submission_07.pdf",
        "submission_08.docx",
        "submission_15.docx",
        "pitch_14.mp4",
    ],
    "degraded": [
        {
            "file": "submission_11.pdf",
            "issue": "zero-byte corrupted PDF (Scenario 3)",
        },
        {
            "file": "pitch_13.mp4",
            "issue": "silent MP4 with no audio track (Scenario 3)",
        },
    ],
    "low_content": ["submission_10.docx"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(ROOT / "sample_submissions"))
    ap.add_argument("--force", action="store_true", help="Regenerate even if files exist.")
    args = ap.parse_args()
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)

    plan = [
        # (file, kind, payload)
        ("submission_01.pdf", "pdf", S01),
        ("submission_02.docx", "docx", S01),          # exact copy of 01 body
        ("submission_03.pdf", "pdf", S03),
        ("submission_04.docx", "docx", S04),
        ("submission_05.docx", "docx", S05),          # LLM-style
        ("submission_06.pdf", "pdf", S06),            # external copy
        ("submission_07.pdf", "pdf", S07),
        ("submission_08.docx", "docx", S08),
        ("submission_09.docx", "docx", S09),          # paraphrase of 04
        ("submission_10.docx", "docx", S10),          # low content
        ("submission_15.docx", "docx", S15),
    ]
    for name, kind, payload in plan:
        path = out / name
        if path.exists() and not args.force:
            logger.info(f"exists, skip: {name}")
            continue
        if kind == "pdf":
            make_pdf(path, payload["title"], payload["sections"])
        else:
            make_docx(path, payload["title"], payload["sections"])
        logger.info(f"wrote {name}")

    # Degraded zero-byte PDF (Scenario 3).
    p11 = out / "submission_11.pdf"
    if args.force or not p11.exists():
        p11.write_bytes(b"")
        logger.info("wrote submission_11.pdf (zero-byte, corrupted)")

    # MP4 pitches.
    p12 = out / "pitch_12.mp4"
    if args.force or not p12.exists():
        spoken = f"{PITCH_12_UNIQUE_INTRO} {PITCH_12_SPOKEN} {PITCH_12_OUTRO}"
        make_mp4(p12, spoken)
        logger.info("wrote pitch_12.mp4 (cross-modal speech)")
    p13 = out / "pitch_13.mp4"
    if args.force or not p13.exists():
        make_mp4(p13, None, silent_only=True)
        logger.info("wrote pitch_13.mp4 (silent, no audio track)")
    p14 = out / "pitch_14.mp4"
    if args.force or not p14.exists():
        make_mp4(p14, PITCH_14)
        logger.info("wrote pitch_14.mp4 (independent original speech)")

    manifest_path = out / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
    logger.info(f"wrote {manifest_path.name}")

    files = sorted(p.name for p in out.iterdir() if p.is_file())
    logger.info(f"dataset complete: {len(files)} files in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
