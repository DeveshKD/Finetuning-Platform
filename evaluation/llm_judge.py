import os
import json
import time
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class JudgeVerdict:
    prompt: str
    candidate_response: str
    reference_response: str
    score: float
    rationale: str

class LLMJudge:
    def __init__(self, judge_model: str):
        self.judge_model = judge_model
        # Use GEMINI_API_KEY if testing with Gemini 2.5 Flash, or OPENAI_API_KEY
        api_key = os.getenv("JUDGE_MODEL_API_KEY", os.getenv("GEMINI_API_KEY"))
        
        # Gemini provides OpenAI compatible endpoints
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/" if "gemini" in judge_model else None
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def score(self, prompt: str, candidate: str, reference: str) -> JudgeVerdict | None:
        if not self.client.api_key:
            print("Warning: No JUDGE_MODEL_API_KEY provided. Skipping LLM Judge.")
            return None

        system_prompt = """You are an impartial AI judge. Evaluate the Candidate Response based on the User Prompt and the Reference Response.
        Score the candidate from 1 to 10. 10 is perfect.
        Return ONLY valid JSON in this exact format:
        {"score": 8.5, "rationale": "Your short explanation here"}"""

        user_content = f"User Prompt: {prompt}\n\nReference Answer: {reference}\n\nCandidate Answer: {candidate}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                result = json.loads(response.choices[0].message.content)
                return JudgeVerdict(
                    prompt=prompt,
                    candidate_response=candidate,
                    reference_response=reference,
                    score=float(result.get("score", 0.0)),
                    rationale=result.get("rationale", "No rationale provided")
                )
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "503" in error_str:
                    print(f"API Rate limited or unavailable. Sleeping 25s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(25)
                else:
                    print(f"LLM Judge failed: {e}")
                    return None
                    
        print("Failed to score after max retries due to rate limits.")
        return None