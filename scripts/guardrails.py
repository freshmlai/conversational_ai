# Guardrails Implementation
# Input and output validation for RAG and FT systems
# Usage: Import in RAG and FT scripts

def validate_query(query):
    # Example: block irrelevant/harmful queries
    blocked_keywords = ['harm', 'attack', 'illegal']
    for word in blocked_keywords:
        if word in query.lower():
            return False
    return True

def filter_output(answer):
    # Example: flag non-factual/hallucinated outputs
    if 'not factual' in answer or 'hallucinated' in answer:
        return '[FLAGGED] Possible hallucination.'
    return answer
