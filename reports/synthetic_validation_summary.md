# Synthetic Data Validation Summary

## Files
- Clean real data: `data/credit_risk_clean.csv`
- Added synthetic rows: `data/credit_risk_synthetic_added_rows.csv`
- Augmented data: `data/credit_risk_augmented.csv`
- Figures: `figures/`

## Row Counts
- Real clean rows: 28632
- Added synthetic rows: 16228
- Augmented rows: 44860

## Target Distribution
| loan_status | value |
| --- | --- |
| 0 | 0.7834 |
| 1 | 0.2166 |

| loan_status | value |
| --- | --- |
| 0 | 0.5 |
| 1 | 0.5 |

## Numeric KS Tests
| column | real_mean | synthetic_mean | real_median | synthetic_median | ks_statistic | ks_p_value |
| --- | --- | --- | --- | --- | --- | --- |
| person_age | 27.7121 | 27.6156 | 26.0 | 26.0 | 0.0095 | 0.0832 |
| person_income | 66426.5056 | 60485.6986 | 55900.0 | 50493.61 | 0.0776 | 0.0 |
| person_emp_length | 4.7803 | 4.5395 | 4.0 | 4.0 | 0.0319 | 0.0 |
| loan_amnt | 9655.3314 | 10148.4055 | 8000.0 | 8600.0 | 0.0349 | 0.0 |
| loan_int_rate | 11.0397 | 11.7934 | 10.99 | 11.83 | 0.0966 | 0.0 |
| loan_percent_income | 0.1695 | 0.1997 | 0.15 | 0.17 | 0.0783 | 0.0 |
| cb_person_cred_hist_length | 5.7936 | 5.7497 | 4.0 | 4.0 | 0.0086 | 0.1496 |

## Categorical Chi-Square Tests
| column | chi2_statistic | chi2_p_value | real_top | synthetic_top |
| --- | --- | --- | --- | --- |
| person_home_ownership | 473.2617 | 0.0 | RENT | RENT |
| loan_intent | 148.1313 | 0.0 | EDUCATION | MEDICAL |
| loan_grade | 1143.53 | 0.0 | A | B |
| cb_person_default_on_file | 250.4379 | 0.0 | N | N |
| loan_status | 5900.9103 | 0.0 | 0 | 0 |

## Added Synthetic Rows Fidelity
The added rows should be compared with real clean rows where `loan_status = 1`, because they were generated only for that class.

### Basic Rule Checks
| check | value |
| --- | --- |
| all_rows_are_target_1 | True |
| missing_values | 0 |
| duplicates_inside_added_rows | 0 |
| exact_duplicates_with_real_clean | 0 |
| invalid_home_ownership_categories | 0 |
| invalid_loan_intent_categories | 0 |
| invalid_loan_grade_categories | 0 |
| invalid_default_file_categories | 0 |
| non_positive_income | 0 |
| non_positive_loan_amount | 0 |
| non_positive_interest_rate | 0 |
| age_outside_real_clean_range | 0 |
| loan_percent_income_mismatch_gt_0_01 | 0 |

### Numeric Fidelity vs Real Target Class
| column | real_target_1_mean | added_mean | mean_delta | real_target_1_median | added_median | ks_statistic | ks_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| person_age | 27.4481 | 27.4452 | -0.0029 | 26.0 | 26.0 | 0.0001 | 1.0 |
| person_income | 50053.1051 | 50003.989 | -49.1161 | 42000.0 | 42000.0 | 0.0002 | 1.0 |
| person_emp_length | 4.1166 | 4.1145 | -0.0021 | 3.0 | 3.0 | 0.0002 | 1.0 |
| loan_amnt | 11019.4856 | 11018.3646 | -1.121 | 10000.0 | 10000.0 | 0.0002 | 1.0 |
| loan_int_rate | 13.1235 | 13.1233 | -0.0002 | 13.49 | 13.49 | 0.0002 | 1.0 |
| loan_percent_income | 0.2463 | 0.253 | 0.0068 | 0.24 | 0.21 | 0.0864 | 0.0 |
| cb_person_cred_hist_length | 5.674 | 5.6724 | -0.0016 | 4.0 | 4.0 | 0.0001 | 1.0 |

### Categorical Fidelity vs Real Target Class
| column | real_target_1_top | added_top | total_variation_distance | max_category_delta |
| --- | --- | --- | --- | --- |
| person_home_ownership | RENT | RENT | 0.0032 | 0.0031 |
| loan_intent | MEDICAL | MEDICAL | 0.0093 | 0.0085 |
| loan_grade | D | D | 0.0066 | 0.0051 |
| cb_person_default_on_file | N | N | 0.002 | 0.002 |

### Categorical Proportion Details

#### person_home_ownership
| person_home_ownership | real_target_1_proportion | added_proportion | absolute_delta |
| --- | --- | --- | --- |
| MORTGAGE | 0.2394 | 0.2408 | 0.0014 |
| OTHER | 0.0044 | 0.0043 | 0.0001 |
| OWN | 0.0235 | 0.0253 | 0.0018 |
| RENT | 0.7327 | 0.7296 | 0.0031 |

#### loan_intent
| loan_intent | real_target_1_proportion | added_proportion | absolute_delta |
| --- | --- | --- | --- |
| DEBTCONSOLIDATION | 0.209 | 0.2168 | 0.0078 |
| EDUCATION | 0.1566 | 0.1562 | 0.0004 |
| HOMEIMPROVEMENT | 0.1324 | 0.1326 | 0.0002 |
| MEDICAL | 0.2291 | 0.2286 | 0.0005 |
| PERSONAL | 0.1551 | 0.1467 | 0.0085 |
| VENTURE | 0.1179 | 0.1191 | 0.0012 |

#### loan_grade
| loan_grade | real_target_1_proportion | added_proportion | absolute_delta |
| --- | --- | --- | --- |
| A | 0.1458 | 0.1455 | 0.0003 |
| B | 0.2343 | 0.2371 | 0.0028 |
| C | 0.1866 | 0.188 | 0.0015 |
| D | 0.3099 | 0.3048 | 0.0051 |
| E | 0.0906 | 0.0917 | 0.0011 |
| F | 0.0235 | 0.0224 | 0.0012 |
| G | 0.0094 | 0.0106 | 0.0012 |

#### cb_person_default_on_file
| cb_person_default_on_file | real_target_1_proportion | added_proportion | absolute_delta |
| --- | --- | --- | --- |
| N | 0.6895 | 0.6875 | 0.002 |
| Y | 0.3105 | 0.3125 | 0.002 |
## Correlation Preservation
- Mean absolute off-diagonal correlation difference: 0.0210

## Risk by Group

### loan_grade
| loan_grade | real_risk | synthetic_risk |
| --- | --- | --- |
| A | 0.0962 | 0.2776 |
| B | 0.1588 | 0.4078 |
| C | 0.2031 | 0.481 |
| D | 0.5919 | 0.8383 |
| E | 0.646 | 0.8694 |
| F | 0.6986 | 0.8899 |
| G | 0.9831 | 0.9957 |

### loan_intent
| loan_intent | real_risk | synthetic_risk |
| --- | --- | --- |
| DEBTCONSOLIDATION | 0.2839 | 0.5956 |
| EDUCATION | 0.1703 | 0.4256 |
| HOMEIMPROVEMENT | 0.2567 | 0.5557 |
| MEDICAL | 0.2685 | 0.5699 |
| PERSONAL | 0.1973 | 0.4606 |
| VENTURE | 0.1462 | 0.3843 |

### person_home_ownership
| person_home_ownership | real_risk | synthetic_risk |
| --- | --- | --- |
| MORTGAGE | 0.1259 | 0.3434 |
| OTHER | 0.2872 | 0.589 |
| OWN | 0.0666 | 0.214 |
| RENT | 0.3123 | 0.6209 |

## Privacy Check
- Exact added synthetic rows copied from real clean data: 0

## Training Utility Evaluation
### Real train -> Real test
| accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- |
| 0.9352 | 0.956 | 0.7349 | 0.831 | 0.9357 |
- Confusion matrix: `[[4444, 42], [329, 912]]`

### Augmented train -> Real test
| accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- |
| 0.8977 | 0.7557 | 0.78 | 0.7676 | 0.9245 |
- Confusion matrix: `[[4173, 313], [273, 968]]`

## Notes
- The generator is Gaussian Copula, trained only on the minority `loan_status` class.
- The augmented dataset keeps all clean real rows and adds synthetic minority-class rows.
- `loan_percent_income` is recalculated after sampling as `loan_amnt / person_income`.
- Use this dataset when the goal is to reduce class imbalance, not to replace the real dataset.