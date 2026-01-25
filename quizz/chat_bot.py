import requests
import os
import json

API_KEY = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ---------------------------
# STUDY MATERIAL
# ---------------------------
text = """
Object-Oriented Programming (OOP) revolutionized software development in the latter half of the 20th century...
"""

# ---------------------------
# 1️⃣ GENERATE STRUCTURED QUIZ
# ---------------------------
quiz_payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are an expert quiz generator. "
                "Return ONLY valid JSON."
            )
        },
        {
            "role": "user",
            "content": f"""
Generate 5 multiple-choice questions from the text below.

Return JSON in this format:
[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "correct_answer": "A",
    "explanation": "3–5 sentence explanation"
  }}
]

Text:
{text}
"""
        }
    ],
    "temperature": 0.2
}

response = requests.post(url, headers=headers, json=quiz_payload)
quiz = json.loads(response.json()["choices"][0]["message"]["content"])

# ---------------------------
# 2️⃣ ASK QUESTIONS ONE BY ONE
# ---------------------------
results = []

print("\n📝 QUIZ STARTED\n")

for idx, q in enumerate(quiz, start=1):
    print(f"\nQuestion {idx}: {q['question']}")
    for key, value in q["options"].items():
        print(f"  {key}) {value}")

    user_ans = input("Your answer (A/B/C/D): ").strip().upper()

    is_correct = user_ans == q["correct_answer"]

    results.append({
        "question": q["question"],
        "user_answer": user_ans,
        "correct_answer": q["correct_answer"],
        "is_correct": is_correct,
        "explanation": q["explanation"]
    })

# ---------------------------
# 3️⃣ SCORE + PERFORMANCE LEVEL
# ---------------------------
score = sum(1 for r in results if r["is_correct"])
total = len(results)
percentage = (score / total) * 100

if percentage <= 40:
    learning_level = "fundamental"
elif percentage <= 80:
    learning_level = "intermediate"
else:
    learning_level = "advanced"

weak_questions = [r["question"] for r in results if not r["is_correct"]]

print(f"\n✅ Final Score: {score} / {total}")
print(f"📈 Learning Level: {learning_level.upper()}")

# ---------------------------
# 4️⃣ SHOW EXPLANATIONS
# ---------------------------
print("\n📊 QUIZ REVIEW\n")

for i, r in enumerate(results, start=1):
    print(f"\nQuestion {i}")
    print("Your answer:", r["user_answer"])
    print("Correct answer:", r["correct_answer"])
    print("Explanation:", r["explanation"])

# ---------------------------
# 5️⃣ START ADAPTIVE CHATBOT
# ---------------------------
print("\n🤖 Study Buddy is now active. Type 'bye' to exit.\n")

system_prompt = f"""
You are an intelligent diagnostic study tutor.

Student performance level: {learning_level}

Teaching behavior:

If level is FUNDAMENTAL:
- Start from basic definitions
- Use simple language and analogies
- Explain step-by-step
- Avoid jargon

If level is INTERMEDIATE:
- Clarify misconceptions
- Explain why concepts work
- Connect related ideas

If level is ADVANCED:
- Focus on deeper insights
- Discuss real-world applications
- Explore design trade-offs

General rules:
- Base explanations on quiz mistakes
- Be encouraging and supportive
- Adjust depth based on student confidence
- Do not repeat quiz questions verbatim
"""

chat_history = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": f"""
Study material:
{text}

Quiz results:
{json.dumps(results, indent=2)}

Weak areas:
{weak_questions}

Score: {score}/{total}

Start by summarizing my understanding level.
"""
    }
]

while True:
    user_question = input("\nYou: ")

    if user_question.lower() == "bye":
        print("\n👋 Study Buddy: Goodbye! Keep learning 🚀")
        break

    chat_history.append({
        "role": "user",
        "content": user_question
    })

    chat_payload = {
        "model": "llama-3.1-8b-instant",
        "messages": chat_history,
        "temperature": 0.4
    }

    response = requests.post(url, headers=headers, json=chat_payload)
    reply = response.json()["choices"][0]["message"]["content"]

    print("\n📘 Study Buddy:", reply)

    chat_history.append({
        "role": "assistant",
        "content": reply
    })
