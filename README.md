# Naive Bayes Spam Classifier

A from-scratch **multinomial Naive Bayes** classifier in pure Python (standard library only) that labels emails as **spam** or **ham** (not spam). It was written for a university machine-learning text-classification assignment (UTD, 2017).

It trains on two folders of emails, classifies two folders of test emails, and prints how many it got right — once with stop words removed and once with them kept.

For how the math works, in plain English with diagrams, see **[ALGORITHM.md](ALGORITHM.md)**.

## Repository layout

```
.
├── src/
│   └── naive_bayes.py            # the classifier
├── data/
│   ├── stopwords.txt             # common words ignored in the "stop words removed" run
│   └── sample/                   # tiny demo dataset (3 ham + 3 spam train, 1 + 1 test)
│       ├── train/{ham,spam}/
│       └── test/{ham,spam}/
├── docs/
│   ├── Assignment_Instructions.docx   # original assignment run notes
│   └── Report.docx                    # original report (incl. Logistic Regression comparison)
├── ALGORITHM.md                  # visual explanation of the algorithm
└── README.md
```

## Requirements

- Python 3.x (tested on 3.13). No third-party packages.

## How to run

```bash
python3 src/naive_bayes.py <train-ham-dir> <train-spam-dir> <test-ham-dir> <test-spam-dir>
```

Try it on the bundled sample data:

```bash
python3 src/naive_bayes.py data/sample/train/ham data/sample/train/spam data/sample/test/ham data/sample/test/spam
```

Each directory must contain one plain-text file per email. The stop-word list is picked up automatically from `data/stopwords.txt`.

### Output

The script prints two result blocks:

1. `without stop words` – stop words are **removed** before classifying.
2. `with stop words` – stop words are **kept**.

Each block shows how many ham and spam test emails were classified correctly / incorrectly, plus the overall accuracy.

### Results from the original assignment

The original assignment used the Enron-style ham/spam dataset (not included here). On it, this code scored:

| Setting | Accuracy |
|---|---|
| Stop words removed | 94.35 % |
| Stop words kept | 94.98 % |

The report in `docs/` also compares against Logistic Regression (92–95 %); that code is not in this repository.

## Known quirks (kept to preserve the original results)

- Stop-word removal deletes from a list while looping over it, so a few stop words slip through.
- Tokenising only splits on spaces, so punctuation stays attached to words (`"free!"` ≠ `"free"`).
- Function names are historical: `spamConditionalProbablity` is used for **both** classes, and `hamConditionalProbablity` is unused.
