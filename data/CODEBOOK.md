# Codebook: `shared-custody-sus-experiment.csv`

Pseudonymized data from a between-subjects usability experiment (n = 67)
comparing single-signature (1-of-1) and multi-signature (2-of-2)
cryptocurrency spending flows using the Safe wallet on the Ethereum Sepolia
test network. Conducted 6, 7, and 8 November 2024 at the Faculty of
Informatics, National University of La Plata (FI-UNLP), Argentina.

The questionnaires were administered in Spanish; variable names and
categorical values are English translations of the original responses.
Free-text comments were removed for privacy and are available from the
authors on reasonable request. Timestamps were reduced to the experiment day,
and participants are identified only by a serial pseudonym assigned in order
of submission.

One row per participant, n = 67. Empty cells mean "not reported" or "not
answered".

| Column | Description |
|---|---|
| `participant_id` | Pseudonym P01 to P67, in order of submission |
| `day` | Experiment day: 1 = 2024-11-06, 2 = 2024-11-07, 3 = 2024-11-08 |
| `treatment` | 1 = single-sig mobile; 2 = 2-of-2 initiated on desktop, authorized on mobile; 3 = 2-of-2 initiated on mobile, authorized on desktop; 4 = single-sig desktop |
| `treatment_description` | Human-readable description of the treatment |
| `nationality` | Self-reported nationality |
| `age` | Age in years |
| `gender` | Self-reported gender identity |
| `education_level` | Highest completed: Primary / Secondary / Tertiary/University / Postgraduate |
| `occupation` | Self-reported occupation(s) |
| `it_experience_<role>` | Years of experience per IT role (software_developer, qa, architect, analyst, support, teacher, student, researcher): Less than 1 year / 1-3 years / 5-6 years / 7-9 years / 10+ years; empty = none reported |
| `fintech_wallet_use_frequency` | Use of traditional fintech wallet apps: Daily / A few times a week / A few times a month / Almost never / Never |
| `crypto_app_use_frequency` | Use of cryptocurrency mobile apps (wallet or exchange): same scale |
| `crypto_self_perceived_knowledge` | Self-perceived cryptocurrency knowledge, 1 (inexpert) to 4 (expert); empty = no experience reported |
| `sus_q1` to `sus_q10` | SUS item responses, Likert 1 to 5 (Brooke, 1996). Odd items are positively worded, even items negatively worded. Statements in the paper's appendix. |
| `sus_score` | SUS score 0 to 100: (sum(odd - 1) + sum(5 - even)) x 2.5 |
| `security_q1` to `security_q6` | Security-perception item responses, Likert 1 to 5. Odd items positively worded, even items negatively worded. Statements in the paper's appendix. Empty = not answered. |
| `security_items_answered` | Number of security items answered, 0 to 6 |
| `security_score` | Security-perception score 0 to 60: ((sum(odd) - 3) + (15 - sum(even))) x 2.5. Defined only for complete six-item responses; empty otherwise (seven participants answered no item and one, P53, answered five of six). |
| `feels_safer_with_devices` | Normalized answer to "Do you feel safer performing the operation from one or two devices?": one / two / unclear; empty = not answered |

## Scoring in the co-authors' spreadsheet

The spreadsheet used to produce the manuscript's Results section stored the
security score as a raw sum in the range -12 to 12 (column `seguridad sus`),
equal to `security_score / 2.5 - 12`. It stored 0 for the seven participants
who answered no item (the analysis notebooks exclude them explicitly) and -9
for P53, where the items give -6. See the README for the effect of this
difference.

## Groups used in the analysis

- Single-signature group: treatments 1 and 4 (n = 32).
- Multi-signature group: treatments 2 and 3 (n = 35).
- Security perception and correlation analyses use the 59 participants with
  `security_items_answered == 6` (T1 = 14, T2 = 13, T3 = 18, T4 = 14).
