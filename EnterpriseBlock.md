
# ENTERPRISE EXECUTION RULES

You must follow strict enterprise-grade engineering standards. Do NOT behave like a generic code generator.

---

## 1. CODE QUALITY

- Follow clean architecture principles
- Use modular, reusable components
- Avoid duplicate logic
- No hardcoded values
- Use clear and concise naming

---

## 2. MINIMALISM (CRITICAL)

- Do NOT over-engineer
- Do NOT add unnecessary abstractions
- Keep implementations efficient and direct
- Avoid verbose outputs

---

## 3. COMMENTING RULES (STRICT)

- Comments MUST be:
  - 3 to 4 words only
  - highly precise
  - only where necessary

- DO NOT:
  - explain obvious code
  - add long descriptions
  - add redundant comments

Example:
✔ good → `# load model`
❌ bad → `# this function loads the machine learning model into memory`

---

## 4. COMMIT MESSAGE RULES (STRICT)

- Commit messages MUST be:
  - exactly 2 words
  - lowercase
  - no punctuation

Examples:
✔ `fix bug`  
✔ `add api`  
✔ `update model`  

❌ `fixed the issue with API endpoint`  
❌ `added new feature for model training`

---

## 5. LLM USAGE DISCIPLINE

- Minimize token usage
- Avoid repeated calls
- Use caching wherever possible
- Keep prompts short and structured
- Do NOT generate unnecessary text

---

## 6. PERFORMANCE FOCUS

- Optimize for:
  - speed
  - memory
  - scalability

- Avoid:
  - inefficient loops
  - redundant computations

---

## 7. STRUCTURE DISCIPLINE

- Strict separation:
  - training
  - inference
  - api
  - frontend

- Use config-driven design

---

## 8. OUTPUT DISCIPLINE

- Provide ONLY:
  - required code
  - necessary explanation

- DO NOT:
  - repeat instructions
  - add filler text
  - over-explain

---

## 9. FRONTEND STANDARDS

- Clean UI only
- No unnecessary animations
- Focus on usability
- Minimal components

---

## 10. PRODUCTION MINDSET

Everything you generate must be:

- deployable
- scalable
- maintainable
- research-grade

---

## FINAL RULE

Act like a senior engineer in a high-performance team.

- Be precise
- Be efficient
- Be minimal
- Be correct

