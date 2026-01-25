import requests
import os

API_KEY = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

text = """
Object-Oriented Programming (OOP) revolutionized software development in the latter half of the 20th century, providing a way to model complex systems more naturally. At its core, OOP organizes programs around objects, which are self-contained entities that combine both data (attributes) and behavior (methods). This idea of encapsulation ensures that an object’s internal workings are hidden from the outside world, reducing errors and making code easier to maintain. Abstraction allows developers to interact with objects at a high level, focusing on what they do rather than how they do it, which simplifies the management of complex software systems. Inheritance provides a mechanism to create new classes based on existing ones, promoting code reuse and establishing a logical hierarchy of related objects. Meanwhile, polymorphism allows objects of different classes to be treated uniformly, letting the same operation behave differently depending on the object’s type, which increases flexibility and reduces the need for repetitive code. These concepts were first implemented in languages like Smalltalk and later popularized by C++, Java, and Python, enabling the creation of large-scale, maintainable applications. OOP’s influence extended beyond programming syntax; it shaped software engineering philosophies, inspiring design patterns, frameworks, and best practices that are still fundamental in modern development. By modeling real-world entities and their interactions, OOP bridged the gap between human conceptual thinking and machine-executable instructions, making software development more intuitive and scalable.
"""
payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a friendly study buddy and tutor. "
                "Explain concepts step by step with examples and clarity."
            )
        },
        {
            "role": "user",
            "content": user_question
        }
    ],
    "temperature": 0.4
}

payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are an expert educator and quiz designer. "
                "You explain concepts deeply, clearly, and with context."
            )
        },
        {
            "role": "user",
            "content": f"""
Generate 5 multiple-choice questions from the text below.

Rules:
- Each question must have exactly 4 options (A, B, C, D)
- Clearly indicate the correct answer
- Provide a detailed explanation for each answer
- Explanations must be 3–5 sentences
- Explanations must add context, significance, and consequences
- Do NOT repeat the text verbatim
- Write explanations as if teaching a student

Text:
{text}
"""
        }
    ],
    "temperature": 0.25
}


response = requests.post(url, headers=headers, json=payload)
data = response.json()

print(data["choices"][0]["message"]["content"])
