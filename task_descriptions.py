"""
task_descriptions.py
====================
Maps dataset names to task descriptions and one-shot examples used for
injecting context into model prompts across all pipelines.

ONE_SHOT_EXAMPLES format
------------------------
The examples use the same prompt format that all five pipelines produce:

  Setting A / B / C  :
      Dataset: <format description>
      --- DATA BEGIN ---
      <TSV lines>
      --- DATA END ---
      QUESTION: ...
      OUTPUT REQUIREMENT: ...
      ANSWER: ...

  Code Execution / ReAct  :
      The one_shot string is prepended as context BEFORE the actual data
      block, so the same TSV-based format works — the LLM reads it as an
      illustrative example rather than as the live data.

Targeted datasets (ste-sentiment_analysis, te-stance_detection):
      Data lines follow {target}\\t{entry} format (tab-separated).
      The example reflects this layout.

The examples intentionally do NOT mention the specific data source type
(tweets, posts, etc.) so the LLM focuses on the statistical task rather
than domain-specific assumptions.
"""

TASK_DESCRIPTIONS = {
    "te-sentiment_analysis":          "This dataset has been collected for analysing its sentiment, classifying it into three categories: positive, negative, and neutral.",
    "te-hate_speech_detection":       "This dataset has been collected for detecting hate speech, classifying it into two categories: hate speech and non-hate speech.",
    "te-offensive_language_detection":"This dataset has been collected for analysing offensive language, classifying it into two categories: offensive language and non-offensive language.",
    "ste-emotion":                    "This dataset has been collected for fine-grained emotion detection analysis, encompassing emotion types such as anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust, among others.",
    "ste-tweettopic":                 "This dataset has been collected for topic classification analysis, encompassing topic categories such as arts and culture, business and entrepreneurs, celebrity and pop culture, diaries and daily life, family, fashion and style, film/TV and video, fitness and health, food and dining, gaming, learning and educational, music, news and social concern, other hobbies, relationships, science and technology, sports, travel and adventure, and youth and student life.",
    "ste-tweethate":                  "This dataset has been collected for fine-grained hate speech detection analysis, encompassing hate speech types such as gender, race, sexuality, religion, origin, disability, age, and not hate.",
    "te-stance_detection":            "This dataset has been collected for stance detection, where each entry contains one target for detection and three categories: 'favor', 'against', and 'none'.",
    "ste-sentiment_analysis":         "This dataset has been collected for aspect-based sentiment analysis on a three-point scale, where each entry contains one target for detection and three categories: 'negative', 'neutral', and 'positive'.",
}


# =============================================================================
# ONE_SHOT_EXAMPLES
#
# Each example mirrors the exact prompt structure the pipelines produce:
#   - Data in TSV format inside --- DATA BEGIN --- / --- DATA END ---
#   - Section labels: QUESTION, OUTPUT REQUIREMENT, ANSWER
#   - No XML tags, no mention of specific data source types
#
# For targeted datasets the data lines are {target}\t{entry} (tab-separated).
# =============================================================================

ONE_SHOT_EXAMPLES = {

    "te-sentiment_analysis": """Example:

Dataset: Each line is a text entry, one record per line.

--- DATA BEGIN ---
This is awesome!
I had a bad day today.
Wow, that movie is absolutely terrible.
--- DATA END ---

QUESTION: Calculate the percentage of entries that express positive sentiment.

OUTPUT REQUIREMENT:
Respond with only a percentage in format "X%" (e.g., "85%").

ANSWER:
33.3%""",


    "te-hate_speech_detection": """Example:

Dataset: Each line is a text entry, one record per line.

--- DATA BEGIN ---
This is awesome!
I had a bad day today.
I HATE these people, they should not exist.
--- DATA END ---

QUESTION: Return the number of entries with hate speech expressions.

OUTPUT REQUIREMENT:
Respond with only a single integer number (e.g., "42").

ANSWER:
1""",


    "te-offensive_language_detection": """Example:

Dataset: Each line is a text entry, one record per line.

--- DATA BEGIN ---
This is awesome!
I had a bad day today.
We dont trust these nixxas, all these bxxes.
--- DATA END ---

QUESTION: Do the given entries contain offensive language?

OUTPUT REQUIREMENT:
Respond with exactly "yes" or "no" only.

ANSWER:
yes""",


    "ste-emotion": """Example:

Dataset: Each line is a text entry, one record per line.

--- DATA BEGIN ---
I feel so romantic and alive today.
I now feel compromised and deeply skeptical of everything.
We dont trust anyone around here anymore.
--- DATA END ---

QUESTION: Do the given entries contain fear emotion?

OUTPUT REQUIREMENT:
Respond with exactly "yes" or "no" only.

ANSWER:
no""",


    "ste-tweettopic": """Example:

Dataset: Each line is a text entry, one record per line.

--- DATA BEGIN ---
No player has more goal contributions than Marcus Rashford in the Champions League this season.
The Rockets are running the tallest starting lineup in NBA history on opening night.
There are no pro or max versions yet — for power users this is not the device to buy.
--- DATA END ---

QUESTION: Do all given entries discuss about sports topics?

OUTPUT REQUIREMENT:
Respond with exactly "yes" or "no" only.

ANSWER:
no""",


    "ste-tweethate": """Example:

Dataset: Each line is a text entry, one record per line.

--- DATA BEGIN ---
I hate older people just for their benefits, they have been useless.
I now feel compromised and skeptical of the value of every unit of work I put in.
I HATE this group of people, this is why they are discriminated against.
--- DATA END ---

QUESTION: How many entries contain hate speech targeting a racial group?

OUTPUT REQUIREMENT:
Respond with only a single integer number (e.g., "42").

ANSWER:
1""",


    "ste-sentiment_analysis": """Example:

Dataset: Each line follows the format: {target}\t{entry} (tab-separated, one record per line). The first column is the target word; the second is the text entry.

--- DATA BEGIN ---
microsoft\tWindows daily updates are completely broken. With all the resources you have, you still cannot get it right.
microsoft\tNow trying to reach support for the fifth time this week with no luck at all.
microsoft\tI am visiting with their engineers tomorrow at their invitation. Feeling very excited.
--- DATA END ---

QUESTION: Do all entries express negative attitude towards microsoft?

OUTPUT REQUIREMENT:
Respond with exactly "yes" or "no" only.

ANSWER:
no""",


    "te-stance_detection": """Example:

Dataset: Each line follows the format: {target}\t{entry} (tab-separated, one record per line). The first column is the target word; the second is the text entry.

--- DATA BEGIN ---
trump\tWho agrees that the current president looks far healthier than Donald Trump?
trump\tIn only nine months he managed to destroy the country's international standing.
trump\tIF YOU LOVE PRESIDENT TRUMP'S TEAM LEAVE A RED HEART.
--- DATA END ---

QUESTION: How many entries express a support stance towards Trump?

OUTPUT REQUIREMENT:
Respond with only a single integer number (e.g., "42").

ANSWER:
1""",

}