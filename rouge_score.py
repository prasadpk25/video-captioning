from rouge_score import rouge_scorer

# ---------------------------------------------
# 1) Put your 10 original/reference texts here
# ---------------------------------------------
reference_texts = [
    "Reference text 1 ...",
    "Reference text 2 ...",
    "Reference text 3 ...",
    "Reference text 4 ...",
    "Reference text 5 ...",
    "Reference text 6 ...",
    "Reference text 7 ...",
    "Reference text 8 ...",
    "Reference text 9 ...",
    "Reference text 10 ..."
]

# ---------------------------------------------
# 2) Put your 10 generated summaries here
# ---------------------------------------------
generated_summaries = [
    "A group of people are playing badminton in the courtyard of an apartment building",# badminton 1min 26 sec 
    "The horse is being groomed by the groomer.", # from the data set 
    "person is playing with the ball",#volley ball(short)
    "Generated summary 4 ...",
    "Generated summary 5 ...",
    "Generated summary 6 ...",
    "Generated summary 7 ...",
    "Generated summary 8 ...",
    "Generated summary 9 ...",
    "Generated summary 10 ..."
]

# ------------------------------------------------
# 3) Initialize ROUGE scorer
# ------------------------------------------------
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], 
    use_stemmer=True
)

# ------------------------------------------------
# 4) Calculate ROUGE for all 10 samples
# ------------------------------------------------
all_scores = []

for i in range(10):
    ref = reference_texts[i]
    gen = generated_summaries[i]

    score = scorer.score(ref, gen)
    all_scores.append(score)

    print(f"\n==========================")
    print(f" Sample {i+1} ROUGE Scores")
    print("==========================")
    print("ROUGE-1:", score['rouge1'])
    print("ROUGE-2:", score['rouge2'])
    print("ROUGE-L:", score['rougeL'])


# ------------------------------------------------
# 5) Optional: Average ROUGE across all 10 samples
# ------------------------------------------------
avg_rouge1 = sum(score['rouge1'].fmeasure for score in all_scores) / 10
avg_rouge2 = sum(score['rouge2'].fmeasure for score in all_scores) / 10
avg_rougeL = sum(score['rougeL'].fmeasure for score in all_scores) / 10

print("\n\n============ AVERAGE ROUGE SCORES (10 Samples) ============")
print("Avg ROUGE-1 F1:", avg_rouge1)
print("Avg ROUGE-2 F1:", avg_rouge2)
print("Avg ROUGE-L F1:", avg_rougeL)
