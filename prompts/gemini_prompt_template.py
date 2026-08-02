SYSTEM_PROMPT = """You will be given a set of descriptive attributes related to an audio sample of a person's speech.

These attributes include information about:

1. The speaker's identity:
   - gender: male, female
   - age_range: child, adolescent, young adult, middle_aged, mature, undefined
   - region_accent: northern dialect, central dialect, southern dialect

2. The speaker's voice characteristics:
   - pitch: very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch
   - timbre: dark, warm, neutral, bright
   - breathiness: clear, slightly breathy, breathy, very breathy
   - resonance: chest voice, balanced, forward, nasal

3. The speaking style and delivery:
   - speed: very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast
   - clarity: muffled, slightly clear, clear, very clear
   - intonation: very monotone, monotone, expressive, very expressive

4. The emotional expression:
   - emotion: neutral, happy, angry, surprised, fearful, sad, disgusted
   - energy: soft, moderate, energetic, powerful
   - expressiveness: flat, slightly expressive, expressive, very expressive

5. The technical quality of the recording:
   - recording_quality: very noisy, noisy, slightly noisy, balanced in clarity, slightly clean, clean, very clean
   - reverberation: very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding
   - stability: unstable, slightly unstable, stable, very stable

Your task is to generate ONE natural-sounding text description that accurately describes how the speaker sounds, based strictly on the provided attributes.
The description should function as a voice design brief, helping a listener or synthesis system imagine and recreate the voice, rather than as a technical annotation.

## Writing rules

Write one natural voice description of 2–3 sentences that flows like a short paragraph rather than a checklist.

The description should begin by introducing the speaker (gender, age if provided, accent) 
and describing their voice, clearly mentioning pitch and timbre, and including breathiness only if it is explicitly provided and fits naturally. 
You may merge multiple attributes that describe the same perceptual dimension into a single natural phrase, provided the meaning of each attribute is preserved.
It should then describe how the speaker talks, focusing on the most salient aspects of delivery such as speaking speed, intonation, or emotional tone, 
without forcing all attributes into the text or sounding repetitive or mechanical. A brief comment on recording quality may be included at the end, mentioning noise level and whether the voice sounds close or distant.

If the recording is very clean and very close-sounding, describe it as an excellent or studio-quality recording. 
If the recording is both noisy and distant-sounding, describe it as poor recording quality.


## Hard constraints (must follow)

- Use only the attributes explicitly provided in the input. Do NOT infer, normalize, or add any information that is not given.
- If intonation is "monotone" or "very monotone", the description MUST NOT include any words implying expressiveness or animation (e.g. expressive, animated, lively, dynamic), even if an emotion or expressiveness label is present.
- Do NOT describe both "monotone" (or "very monotone") and any form of "expressive" or "expressiveness" in the same description.
- If any attribute value is "undefined", DO NOT replace it with neutral or default descriptions.
- If intonation and expressiveness attributes conflict, prioritize intonation and adjust wording to avoid contradiction.
- It is acceptable to omit attributes that would make the description awkward or repetitive, as long as no new information is introduced.
- If an attribute represents a neutral or default state (e.g. moderate speed, balanced in clarity, slightly clean, neutral timbre, balanced resonance, stable, neutral emotion), do not use the label wording directly. Either omit the attribute or replace it with a natural, non-technical phrasing that does not add new information (e.g. steady, even, matter-of-fact).
- Avoid vague or unnatural modifiers such as "slightly clean" or redundant phrasing like "clear clarity".
- Technical attributes such as stability should only be mentioned if they strongly affect how the voice is perceived.
- The final description must be fluent, concise, internally consistent, and easy to understand.
- Treat "intonation" as the primary indicator of expressiveness. Use "expressiveness" only to reinforce intonation, never to contradict it.
- Do NOT add energy, clarity, or stability unless they are explicitly provided in the input.
- Describe breathiness only as a voice quality, not as a speaking action (avoid phrases like "speaks breathily").
- Use exact or very close wording for pitch labels (e.g. "moderately pitched", not "moderate-pitched").


## Examples

Input attributes:
female, young, northern dialect, very high-pitch voice, warm timbre, slightly breathy, balanced resonance, slightly slow speech, clear pronunciation, expressive intonation, happy emotion, moderate energy, very expressive, slightly clean recording, very close-sounding, stable

Valid descriptions:
-> A young woman with a warm, very high-pitched voice and a northern dialect speaks with expressive intonation, conveying a happy mood. Her voice is slightly breathy, and she speaks slightly slowly. The recording is slightly clean and very close-sounding.
-> A young woman with a warm, very high-pitched, slightly breathy voice and a northern dialect speaks slightly slowly with expressive intonation, giving a cheerful impression. The voice feels close and intimate due to the very close-sounding, clean recording.

Input attributes:
male, middle-aged, southern dialect, moderate pitch, neutral timbre, breathy, balanced resonance, moderate speed, clear speech, monotone intonation, neutral emotion, soft energy, slightly expressive, balanced clarity recording, slightly close-sounding, stable

Valid descriptions:
-> A middle-aged man with a southern dialect speaks in a monotone voice. His moderately pitched voice has a breathy quality. The recording is balanced in clarity and sounds slightly close.
-> A middle-aged man with a southern dialect speaks in a monotone manner, his moderately pitched voice carrying a breathy quality, with the recording sounding balanced in clarity and slightly close.

## Task

Given the following attributes:
{attributes}

Generate the corresponding description:
"""


USER_PROMPT_TEMPLATE = """Based on the following audio attributes, generate a natural language voice description:

{attributes_json}

Description:
"""


def format_record_for_prompt(record: dict) -> str:
    flat = {}
    
    # Identity
    if "identity" in record:
        flat.update(record["identity"])
    
    # Voice
    if "voice" in record:
        flat.update(record["voice"])
    
    # Speech
    if "speech" in record:
        flat.update(record["speech"])
    
    # Expression
    if "expression" in record:
        flat.update(record["expression"])
    
    # Technical
    if "technical" in record:
        flat.update(record["technical"])
    
    return flat


def build_prompt(record: dict) -> str:
    import json
    flat = format_record_for_prompt(record)
    return USER_PROMPT_TEMPLATE.format(attributes_json=json.dumps(flat, indent=2,sort_keys=True))
