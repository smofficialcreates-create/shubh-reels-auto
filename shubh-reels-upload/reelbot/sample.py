"""Built-in scripts used by --sample runs and tests (no LLM, no network).

SAMPLE is a full-length explainer in the new house format (analogy, stages, catch, recap, scenes).
SAMPLE_SHORT is the older 30 second format, which the renderer still supports.
"""

SAMPLE = {
    "topic": "How an AI assistant actually answers you",
    "pillar": "practical_ai_tutorials",
    "category": "practical_ai_tutorials",
    "tag": "AI EXPLAINED",
    "hook": "You are using AI the wrong way.",
    "hook_focus": "wrong",
    "sentences": [
        "It is a very well read autocomplete, and that changes how you should use it.",
        "Picture a librarian who has read almost everything but answers from memory, never from a shelf.",
        "First, your message is cut into small pieces called tokens.",
        "A token is often a whole word, or just part of one.",
        "Second, the assistant looks at everything in the chat at once, not only your last line.",
        "That window of text is its desk, and the desk has a limit.",
        "Third, it predicts the most likely next token.",
        "It adds that token to the pile and predicts again.",
        "It repeats this until the reply is done, and the words stream to you.",
        "Type, the capital of France is, and the likely next token is Paris.",
        "Nothing in that process checks a fact.",
        "So the catch is this: it sounds just as sure when it is wrong.",
        "Treat answers as a strong first draft, and check anything that matters.",
        "Give it your goal, your context and the format you want, and the draft gets much closer.",
        "Remember the three steps: pieces, desk, prediction.",
    ],
    "scenes": [
        {"from": 1, "to": 2, "label": "the idea", "visual": {
            "type": "definition", "term": "Autocomplete", "meaning": "A system that guesses the next word from everything it has read."}},
        {"from": 3, "to": 4, "label": "step one", "visual": {"type": "numeral", "n": 1, "total": 3, "title": "Cut into tokens"}},
        {"from": 5, "to": 6, "label": "step two", "visual": {
            "type": "compare",
            "left": {"title": "Off the desk", "rows": ["Oldest messages", "Early instructions", "Long old files"]},
            "right": {"title": "On the desk", "rows": ["Your latest message", "Recent chat", "Pasted notes"]}}},
        {"from": 7, "to": 9, "label": "step three", "visual": {
            "type": "flow", "nodes": ["Guess next token", "Add it to the pile", "Repeat, then stream"], "sync": [7, 8, 9]}},
        {"from": 10, "to": 10, "label": "an example", "visual": {
            "type": "chat", "prompt": "The capital of France is", "result": "Paris."}},
        {"from": 11, "to": 12, "label": "the catch", "visual": {
            "type": "compare",
            "left": {"title": "Trusting it", "rows": ["Sounds so sure", "Nothing checks facts", "Errors slip through"]},
            "right": {"title": "Checking it", "rows": ["Treat it as a draft", "Verify key facts", "Safe to use"]}}},
        {"from": 13, "to": 14, "label": "what to do", "visual": {
            "type": "checklist", "labels": ["Your goal", "Your context", "The format"]}},
        {"from": 15, "to": 15, "label": "the recap", "visual": {
            "type": "flow", "nodes": ["Tokens", "The desk", "Prediction"]}},
    ],
    "takeaway": "Treat an AI answer as a first draft and give it a goal, context and format.",
    "emphasis": ["wrong", "autocomplete", "memory", "tokens", "part", "everything", "limit", "predicts", "pile", "stream",
                 "Paris", "checks", "wrong", "draft", "context", "prediction", "prompt"],
    "cover_text": "You are using AI the wrong way",
    "key_points": [],
    "demo": None,
    "cta_type": "save",
    "cta": "Save this, and test it on your next prompt.",
    "caption": "You are using AI the wrong way.\nIt predicts the next token, so check what matters.\nai assistant, tokens, context window, prompting\nAI-generated video.",
    "hashtags": ["#aitools", "#prompting", "#productivity"],
}

SAMPLE_SHORT = {
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
