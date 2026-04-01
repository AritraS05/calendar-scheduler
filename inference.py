import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "your-fallback-key")

ENV_URL = "http://127.0.0.1:7860"
MAX_STEPS = 10

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

SYSTEM_PROMPT = """
You are an autonomous calendar scheduling agent. 
Your goal is to book a meeting based on user requirements without creating overlaps.

You must output your action as a raw JSON object matching this schema exactly:
{
    "action_type": "view_schedule" OR "book_slot",
    "start_time": "YYYY-MM-DDTHH:MM:SS" (Optional, use if booking),
    "end_time": "YYYY-MM-DDTHH:MM:SS" (Optional, use if booking),
    "guest_email": "string" (Optional, use if booking)
}
Do not include markdown blocks, just the raw JSON.
"""

def reset_env():
    """Resets the environment and returns the initial observation."""
    response = requests.post(f"{ENV_URL}/reset")
    response.raise_for_status()
    return response.json()

def step_env(action_dict):
    """Sends the chosen action to the environment."""
    response = requests.post(f"{ENV_URL}/step", json=action_dict)
    response.raise_for_status()
    return response.json()

def parse_llm_action(response_text):
    """Safely parses the LLM's JSON output."""
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        print(f"Failed to parse LLM output: {response_text}")
        return {"action_type": "view_schedule"}

def run_evaluation(task_description):
    """Runs a single episode loop for a specific task."""
    print(f"\n--- Starting Task: {task_description} ---")
    
    observation = reset_env()
    done = False
    step_count = 0
    total_reward = 0.0
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task_description}\nCurrent Observation: {json.dumps(observation)}"}
    ]

    while not done and step_count < MAX_STEPS:
        step_count += 1
        print(f"\n[Step {step_count}] Thinking...")

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1, 
            )
            llm_response = completion.choices[0].message.content
            
            action = parse_llm_action(llm_response)
            print(f"Action chosen: {action}")

            step_result = step_env(action)
            observation = step_result["observation"]
            reward = step_result["reward"]
            done = step_result["done"]
            
            total_reward = reward 
            
            print(f"Reward: {reward} | Done: {done} | Status: {observation['last_action_status']}")

            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user", 
                "content": f"Action result: {json.dumps(observation)}. Reward received: {reward}."
            })

        except Exception as e:
            print(f"Error during step: {e}")
            break

    print(f"\n--- Episode Complete ---")
    print(f"Total Steps: {step_count}")
    print(f"Final Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    try:
        requests.get(f"{ENV_URL}/")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the environment at {ENV_URL}.")
        print("Please ensure your FastAPI server is running in a separate terminal.")
        exit(1)

    tasks = [
        "Book a 30-minute meeting on 2026-04-02 starting at 10:00:00 for test@example.com.",
        "Check the schedule first, then book a 1-hour meeting on 2026-04-02 at a time that does not overlap with existing bookings.",
        "Evaluate the current schedule. If there is a conflict at 09:00:00, book an alternative time slot for VIP@example.com and maximize the schedule density."
    ]

    scores = []
    for i, task in enumerate(tasks, 1):
        score = run_evaluation(task)
        scores.append(score)
        
    print("\n==========================")
    print("      FINAL RESULTS       ")
    print("==========================")
    for i, score in enumerate(scores, 1):
        print(f"Task {i} Score: {score}")