## Test-set results (fake class, 20% stratified hold-out, threshold tuned on train CV)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | 0.951 | 0.896 | 0.923 | 0.952 |
| TF-IDF + scam features + LogReg | 0.949 | 0.867 | 0.906 | 0.959 |

Test set: 3,576 postings (173 fake). Final threshold: 0.72.

## Strongest fraud signals (model coefficients)

`using link`, `apply using`, `money`, `link`, `line requirements`, `receptionist`, `high school`, `data entry`, `administrative`, `earn`, `accion`, `clerk`, `administrative assistant`, `internet`, `entry`

## Strongest legitimate signals

`english`, `web`, `client`, `software`, `experience`, `growing`, `digital`, `companies`, `fun`, `php`, `team`, `pm`, `university`, `marketing`, `search`
