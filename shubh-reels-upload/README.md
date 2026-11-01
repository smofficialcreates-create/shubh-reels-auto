# Daily AI reel bot

Makes and posts one Instagram Reel a day about **productivity, AI tools and career skills**,
starring one constant anime narrator (the character in `assets/`). Runs entirely on GitHub Actions.

```
Groq (free LLM)  ->  safety checks + reviewer pass  ->  Kokoro voice (free, CPU)
      ->  camera / captions / music edit (OpenCV + ffmpeg)  ->  Instagram Content Publishing API
```

## What it costs
| Piece | Tool | Cost |
|---|---|---|
| Scripts | Groq `gpt-oss-120b` (Gemini optional fallback) | free tier, about 1,000 requests and 200k tokens a day (third-party figures; one reel uses about 3 requests) |
| Voice | Kokoro-82M via kokoro-onnx, Apache-2.0 | free, no key, no quota |
| Video | the code in `reelbot/` | free |
| Compute | GitHub Actions on a **public** repo | free, unlimited minutes (a private repo gets 2,000 min/month, also enough) |
| Posting | Instagram Graph API | free |

## One-time setup (about 30 minutes)
1. Create a **public** GitHub repo and push this folder to it.
2. Get a free API key at console.groq.com and add it as the repo secret `GROQ_API_KEY`.
3. Instagram: switch the account to **Professional**; at developers.facebook.com create an app, add
   *Instagram API with Instagram Login*, request `instagram_business_basic` and
   `instagram_business_content_publish`, add your account as a tester/role, and generate a long-lived token.
   Add secrets `IG_USER_ID` and `IG_ACCESS_TOKEN`.
4. Optional: `GEMINI_API_KEY` (fallback LLM). `GH_ADMIN_TOKEN` = a fine-grained PAT with "Secrets: read and write"
   on this repo, so the weekly job can renew the 60-day Instagram token by itself.
5. Actions tab -> **daily-reel** -> *Run workflow* with `dry_run` ticked. Download the `reel` artifact and watch it.
6. Happy? Do nothing: it posts daily at 18:00 IST (edit the cron in `.github/workflows/daily-reel.yml`).

**Kill switch:** set repo variable `PUBLISH_MODE` to `dry` (it keeps rendering but never posts), or disable the workflow.

## Safety built in
- Every script must pass hard rules (no prices, stats, plan or version claims, income promises, health/legal/
  investing, politics, alcohol/tobacco) **and** a second LLM review. Any failure means no post that day.
- Hook quality is scored locally (instagram-agent-skill, MIT, vendored in `vendor/`); weak hooks are rejected.
- Topics are de-duplicated against `data/history.json`.
- The caption always says "AI-generated video."; hashtags are capped at 5.
- Secrets are only ever sent in headers, never printed.

## Honest limits
- **No lip-sync.** In this artwork his hand covers his mouth, so he narrates like a voice-over. He still
  breathes, sways, blinks, reacts to the voice and the camera cuts between shots. Lip-sync would need art
  with a visible mouth. `extras/kai_procedural.py` is a fully lip-synced original character if you want one.
- **Image rights.** Make sure you may use the artwork. A public repo makes `assets/` public too.
- The first real run exercises Kokoro, Groq and Instagram for the first time; check the first dry run.
- Instagram may rate-limit or label automated or AI content. Watch the first weeks.
- GitHub pauses scheduled workflows after 60 days without repo activity; the daily history commit prevents that.
- Voice is `am_adam` (your pick); change it with the repo variable or env `KOKORO_VOICE` (for example `am_onyx`).

## Local test (no network, placeholder voice)
`python -m reelbot.main --sample --fake-tts --no-publish --max-seconds 8`

## Look and motion
The mascot is the Shubh puppet (`reelbot/puppet.py`), cut from `assets/shubh_sheet.png` by `python -m reelbot.prep_puppet` into `assets/puppet/`.
Layers: head (hair, face, glasses, neck), eyelids, eyebrows, mouth, torso, pelvis, upper arms, forearms, hands, upper legs, lower legs, shoes.
Joints come from the pivot discs on the sheet: neck, shoulders, elbows, wrists, hips, knees, ankles. Hidden overlap discs sit under every joint so limbs never show gaps.
Hair and eyebrows move as masked displacement of the head layer (cutting them out would need repainting the skin under them). Eyelids and mouth are patches taken from your open-mouth and closed-eye drawings, so the face stays exactly on model.

`reelbot/animator.py` holds the animation presets (rest, wave, point_up, present_r, present_l, count, shrug, think, both_up, cheer, look_l, look_r, look_up, lean_in), a speed control and narration lip-sync.
`reelbot/editor.py` draws the motion graphics around him: top-left kinetic captions, UI-card props, circle wipes between paper/orange/dark scenes and synthesized SFX.

Use the puppet on its own, with a transparent background:

    python -m reelbot.puppet_cli --audio narration.wav --presets "wave@0.2,present_r@2.8,rest@4.8" --size 720x1280 --out out/shubh.webm
    python -m reelbot.puppet_cli --seconds 6 --png-dir out/frames --full-body       # PNG sequence, full body

    from reelbot.puppet import Shubh, Pose
    img = Shubh().render_rgba(Pose(r_sh=30, r_el=150, mouth_open=0.6))   # BGRA with real alpha

## Research and quality pipeline

Every day the bot chooses a topic from fresh sources rather than from memory:

1. **Discover** - `research.py` pulls Hacker News (Algolia) and the RSS feeds in `data/sources.json`, keeps items from the last 10 days, drops video and social hosts. Any feed that fails is skipped.
2. **Score** - `topic.py` has the LLM rate each shortlisted item on six criteria (0-10); the weighted total (20/20/20/15/15/10) is computed in code. Bar: 62. A category-rotation bonus avoids repeating the same kind of reel.
3. **Read** - the article text is fetched and extracted (`research.py`).
4. **Write** - `script_gen.py` asks for several hook options, a spoken script, on-screen key points, a takeaway, an optional chat demo, a cover line and a caption, with a verbatim evidence quote for every factual sentence.
5. **Verify** - `grounding.py` checks each quote exists in the article, that numbers appear in the source, and that facts are not invented. Digits and "free" pricing claims are blocked (TTS and accuracy reasons).
6. **Review** - a second LLM pass scores /100 (pass >= 80, accuracy >= 16, no unsupported claims). Fails closed.
7. If no source passes, an evergreen script is used (same gates, no source line).

Outputs per run: `reel.mp4`, `cover.jpg` (used as the Instagram cover), `caption.txt`, `research.json`.

Adopted from the master prompt: trend sourcing, scoring, hook options, evidence grounding, QC rubric, category diversification, cover, chat-demo scene, CTA types. Left out: unverifiable "viral prediction", scraping Instagram/TikTok (ToS), paid tools, long multi-file output format.
