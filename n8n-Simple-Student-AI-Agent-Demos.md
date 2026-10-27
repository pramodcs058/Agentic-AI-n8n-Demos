# 5 Simple n8n Prompt-Based AI Agent Demos for Students

These demos are designed for beginners using the **New chat** prompt box in n8n. Copy one prompt, paste it into n8n, and submit it.

**Before the class:** If n8n asks for an AI model credential, your trainer must configure a supported model. Do not enter a made-up API key. Test the workflow with sample data only.

---

## Demo 1: Study Topic Explainer

### Prompt to paste into n8n

```text
Build a very simple student study helper in n8n.

The student types a topic in chat. Use an AI chat model to explain the topic in simple language and return the answer to the chat.

Instructions for the AI:
- Explain the topic for a beginner.
- Use no more than 5 bullet points.
- Give one simple example.
- Explain difficult words in plain English.
- If the topic is unclear, ask one short question.
- Do not use external tools, websites, databases, or APIs.
- Keep the workflow to the minimum number of nodes needed.
- Name the nodes clearly and tell me how to test the workflow.

Test with this topic: "What is cloud computing?"
```

**Try:** “What is artificial intelligence?” or “Explain APIs.”

---

## Demo 2: Quiz Question Generator

### Prompt to paste into n8n

```text
Build a simple quiz generator in n8n.

The student enters a subject or topic in chat. Use an AI chat model to create 5 multiple-choice questions about that topic and return them in the chat.

For each question:
- Provide 4 options labelled A, B, C, and D.
- Mark the correct answer after the options.
- Give a one-sentence explanation.
- Keep the questions suitable for beginners.
- Do not use external tools, websites, databases, or APIs.
- Keep the workflow simple and use the minimum number of nodes needed.
- Tell me how to test it.

Test with this topic: "Python basics".
```

**Try:** “Azure fundamentals,” “Cybersecurity basics,” or “Generative AI.”

---

## Demo 3: Text Summarizer

### Prompt to paste into n8n

```text
Build a simple text summarizer in n8n.

The student pastes a paragraph into chat. Use an AI chat model to summarize it and return the summary in the chat.

Instructions:
- Write 3 short bullet points.
- Keep the language simple.
- Include only information from the supplied paragraph.
- Do not add new facts.
- If the student provides no text, ask them to paste a paragraph.
- Do not use external tools, websites, databases, or APIs.
- Keep the workflow simple and use the minimum number of nodes needed.
- Tell me how to test it.

Test with this paragraph:
"Cloud computing lets people use computing resources over the internet. These resources can include servers, storage, databases, and software. Organizations can scale resources when demand changes and pay according to the services they use."
```

**Try:** Paste a paragraph from a textbook or your own notes.

---

## Demo 4: Sentiment Checker

### Prompt to paste into n8n

```text
Build a simple sentiment checker in n8n.

The student types a sentence in chat. Use an AI chat model to classify the sentence as Positive, Negative, or Neutral and explain the result in one short sentence.

Rules:
- Choose exactly one label: Positive, Negative, or Neutral.
- Give one short reason based on the sentence.
- Do not infer personal details about the writer.
- Do not use external tools, websites, databases, or APIs.
- Keep the workflow simple and use the minimum number of nodes needed.
- Tell me how to test it.

Test these sentences:
1. "I really enjoyed this course."
2. "The application keeps crashing."
3. "The class starts at 10 AM."
```

**Try:** Enter a review of a movie, course, or app.

---

## Demo 5: Programming Concept Helper

### Prompt to paste into n8n

```text
Build a simple programming learning helper in n8n.

The student enters a programming concept in chat. Use an AI chat model to explain it and give a tiny example.

Instructions:
- Explain the concept in beginner-friendly language.
- Give one small Python code example when appropriate.
- Explain what the example does in 2 or 3 sentences.
- Avoid advanced terminology unless you explain it.
- Do not run code or use external tools, websites, databases, or APIs.
- Keep the workflow simple and use the minimum number of nodes needed.
- Tell me how to test it.

Test with this topic: "What is a Python for loop?"
```

**Try:** “What is a variable?”, “Explain an if statement,” or “What is a function?”

---

## Suggested 30-minute student demo

| Time | Activity |
|---|---|
| 0–5 minutes | Show the n8n **New chat** prompt box. Explain that a prompt describes the workflow students want. |
| 5–12 minutes | Build the Study Topic Explainer together. |
| 12–18 minutes | Build the Text Summarizer. |
| 18–24 minutes | Build the Quiz Generator. |
| 24–30 minutes | Let students modify one prompt and test a new topic. |

## Student checklist

- [ ] Paste a prompt into n8n's **New chat** box.
- [ ] Review the workflow the builder creates.
- [ ] Configure the AI model credential if requested.
- [ ] Test with the sample topic or text.
- [ ] Change the input and run it again.
- [ ] Check whether the AI response follows the prompt.

## Important note

These prompts request simple chat-based workflows. The exact nodes generated depend on the n8n version and available AI integrations. Students should inspect the generated workflow and test it with harmless sample content. AI output can be incorrect, so verify answers before using them for assignments or decisions.
