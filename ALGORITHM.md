# How the Naive Bayes Spam Classifier Works

## The idea in one paragraph

Spam emails use certain words a lot ("free", "prize", "click"). Normal emails (ham) use other words ("meeting", "report"). To judge a new email, we ask: **"Does this email's wording look more like the spam we've seen, or more like the ham we've seen?"** Whichever class fits better wins.

It's called *naive* because it pretends every word is independent of the others. That isn't true in real language, but it works surprisingly well.

## Big picture

> **Colour key:** 🟩 green = ham (legitimate) · 🟥 red = spam · ⬜ grey = neutral steps shared by both. Colour is only used where it tells you which class something belongs to.

```mermaid
flowchart LR
    HT[Ham training emails] --> HM[(Ham model:<br/>prior + word probabilities)]
    ST[Spam training emails] --> SM[(Spam model:<br/>prior + word probabilities)]
    N[New email] --> E[Score the email<br/>against each model]
    HM --> E
    SM --> E
    E --> F{Higher score?}
    F -->|ham| G[Label: HAM]
    F -->|spam| H[Label: SPAM]
    class N,E,F plain
    class HT,HM,G ham
    class ST,SM,H spam
    classDef plain fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef ham fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef spam fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
```

## Step 1 – Clean the text

```mermaid
flowchart LR
    T["'Win FREE money now!'"] --> S[Split on spaces]
    S --> L[Lowercase]
    L --> O[Drop 1-letter tokens]
    O --> W[Optionally drop stop words<br/>like 'the', 'and', 'of']
    W --> R["['win', 'free', 'money', 'now!']"]
    class T,S,L,O,W,R plain
    classDef plain fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef ham fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef spam fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
```

Stop words are very common words that say little about spam vs. ham. The program runs twice: once removing them, once keeping them, so you can compare.

## Step 2 – Training (learn from examples)

**a) Priors – how common is each class?**

> prior(spam) = spam training emails ÷ all training emails
> prior(ham) = ham training emails ÷ all training emails

**b) Word probabilities – how likely is each word in each class?**

Paste all spam emails together and count each word; do the same for ham.

> P(word | class) = (times the word appears in the class + 1) ÷ (total words in the class + vocabulary size)

The **+1** is called *add-one (Laplace) smoothing*. Without it, a word never seen in spam would get probability 0 and wipe out the whole score. With it, unseen words just get a small non-zero probability.

Tiny example:

| Word | Count in spam | Count in ham |
|---|---|---|
| free | 5 | 0 |
| meeting | 0 | 4 |

"free" is far more likely in spam, "meeting" far more likely in ham.

## Step 3 – Classifying a new email

Start each class with its prior, then add up evidence from every word:

```mermaid
flowchart TD
    A[New email → list of words] --> B[spam score = log prior spam]
    A --> C[ham score = log prior ham]
    B --> D[for each word:<br/>add log P word given spam<br/>unseen word → use small default]
    C --> E[for each word:<br/>add log P word given ham<br/>unseen word → use small default]
    D --> F{spam score ≥ ham score?}
    E --> F
    F -->|yes| G[SPAM]
    F -->|no| H[HAM]
    class A,F plain
    class B,D,G spam
    class C,E,H ham
    classDef plain fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef ham fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef spam fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
```

**Why logs?** Multiplying hundreds of tiny probabilities (like 0.0003 × 0.0001 × …) gets so small a computer rounds it to zero. Adding their logarithms gives the same ranking with no precision problem. The code uses base-10 logs.

Worked example for the email "free prize meeting":

| | prior | free | prize | meeting | total (log) |
|---|---|---|---|---|---|
| spam | high | high | high | tiny | **larger** |
| ham | lower | tiny | tiny | high | smaller |

Spam wins → labelled **SPAM**.

## Step 4 – Measuring accuracy

Each test email's true label is its folder (ham test folder = truly ham). The program counts correct vs. wrong per class:

> accuracy = (ham correct + spam correct) ÷ all test emails × 100

## How it maps to the code

| Concept | Function in `src/naive_bayes.py` |
|---|---|
| Clean / tokenise text | `Test` |
| Read a folder of emails | `folderText`, `fnPrior` |
| Word probabilities (smoothed) | `spamConditionalProbablity` |
| Train one class | `Train` |
| Train both + priors | `TrainMultinomail` |
| Score an email | `_scores` |
| Classify ham / spam test folders | `hamTest`, `spamTest` |
| Run everything, print accuracy | `mainfn` |
