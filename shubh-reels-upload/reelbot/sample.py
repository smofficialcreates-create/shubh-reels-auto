"""A built-in script used by --sample runs and tests (no LLM, no network)."""

SAMPLE = {
    "topic": "Stop switching tabs: the two-minute inbox rule",
    "pillar": "productivity",
    "category": "productivity_methods",
    "takeaway": "Pick one task with AI, then protect your focus for one hour.",
    "demo": {"prompt": "Here are my messy notes. Give me the three tasks that matter most today.",
             "result": "Task one: finish the draft. Task two: reply to the client. Task three: book the call."},
    "emphasis": [None, "switch", "fix", "goal", "notes", "task", "lunch", "currency", "steps"],
    "cover_text": "Stop losing forty minutes a day",
    "tag": "PRODUCTIVITY",
    "hook": "You lose forty minutes a day to tab switching.",
    "hook_focus": "forty",
    "sentences": [
        "Every switch costs your brain a few minutes to refocus.",
        "Here is the fix I use with any AI assistant.",
        "First, write one goal for the next hour.",
        "Second, paste your messy notes and ask for the three tasks that matter.",
        "Third, close everything else and work on task one only.",
        "Keep the other two for after lunch.",
        "Your focus is your real currency.",
    ],
    "key_points": [
        {"sentence": 3, "label": "Write one goal"},
        {"sentence": 4, "label": "Ask AI for top 3"},
        {"sentence": 5, "label": "Work on task one"},
    ],
    "cta": "Small steps beat a messy day.",
    "caption": "You lose focus every time you switch tabs. Try this one hour plan. AI-generated video. #productivity #aitools",
    "hashtags": ["#productivity", "#aitools", "#careertips"],
}
