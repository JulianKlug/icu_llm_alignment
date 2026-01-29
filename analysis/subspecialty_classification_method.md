# ICU Subspecialty Classification Method

## Overview

Questions are classified into ICU subspecialties using a keyword-based scoring algorithm. Each question is assigned to the subspecialty with the highest keyword match score.

## Algorithm

1. **Preprocessing**: Convert question text to lowercase
2. **Keyword Matching**: Search for each subspecialty's keywords in the question text
3. **Scoring**: Each keyword match adds to the subspecialty score
   - Single-word keywords: +1 point
   - Multi-word phrases: +N points (where N = number of words)
4. **Assignment**: Question assigned to subspecialty with highest score
5. **Default**: If no keywords match (score = 0), classified as "General ICU"

## Subspecialty Definitions

### Cardiovascular
Keywords: `cardiac arrest`, `cardiogenic shock`, `ecmo`, `va-ecmo`, `vv-ecmo`, `myocardial infarction`, `stemi`, `nstemi`, `heart`, `cardiac`, `swan-ganz`, `picco`, `hemodynamic`, `inotrope`, `vasopressor`, `dobutamine`, `norepinephrine`, `arrhythmia`, `fibrillation`, `tachycardia`, `bradycardia`, `pacemaker`, `cabg`, `valve`, `tamponade`, `pulmonary embolism`, `aortic`, `cardiac surgery`, `rosc`, `cpr`, `resuscitation`

Excludes: `septic shock` (to avoid misclassifying sepsis cases)

### Respiratory
Keywords: `ards`, `respiratory failure`, `mechanical ventilation`, `ventilator`, `intubation`, `extubation`, `tracheostomy`, `tracheotomie`, `weaning`, `oxygen`, `hypoxemia`, `hypoxia`, `pneumothorax`, `asthma`, `copd`, `bronchospasm`, `hemoptysis`, `prone position`, `peep`, `fio2`, `tidal volume`, `respiratory`, `lung`, `pulmonary edema`

Excludes: `pulmonary embolism` (classified under Cardiovascular)

### Sepsis & Infectious Disease
Keywords: `sepsis`, `septic shock`, `infection`, `antibiotic`, `antimicrobial`, `meningitis`, `pneumonia`, `bacteremia`, `fungal`, `viral`, `covid`, `influenza`, `malaria`, `abscess`, `cellulitis`, `endocarditis`, `osteomyelitis`, `necrotizing`, `fever`, `inflammatory`, `procalcitonin`, `lactate`, `source control`, `cultures`, `resistant`, `mrsa`, `pseudomonas`, `candida`, `pneumocystis`, `immunocompromised`

### Neurological
Keywords: `traumatic brain injury`, `tbi`, `stroke`, `subarachnoid`, `sah`, `intracranial`, `icp`, `cerebral`, `brain`, `neurological`, `neuro`, `seizure`, `epilepsy`, `encephalopathy`, `coma`, `glasgow`, `gcs`, `pupil`, `herniation`, `vasospasm`, `neuropronostication`, `brain death`, `eeg`, `ct scan brain`, `mri brain`, `epidural hematoma`, `subdural`, `contusion`

### Trauma & Burns
Keywords: `trauma`, `burn`, `fracture`, `hemorrhage`, `bleeding`, `transfusion`, `massive transfusion`, `coagulopathy`, `polytrauma`, `injury`, `accident`, `crash`, `fall`, `laceration`, `contusion`, `spleen`, `liver laceration`, `damage control`, `tourniquet`

Excludes: `subarachnoid hemorrhage`, `intracranial`, `brain injury` (classified under Neurological)

### Renal
Keywords: `acute kidney injury`, `aki`, `renal failure`, `dialysis`, `crrt`, `hemodialysis`, `hemofiltration`, `oliguria`, `anuria`, `uremia`, `creatinine`, `urea`, `electrolyte`, `hyperkalemia`, `hyponatremia`, `rhabdomyolysis`, `nephrotoxic`

### Toxicology
Keywords: `overdose`, `intoxication`, `poisoning`, `toxicity`, `toxic`, `paracetamol`, `acetaminophen`, `drug`, `suicidal attempt`, `ingestion`, `antidote`, `n-acetylcysteine`, `charcoal`, `methanol`, `ethylene glycol`, `opioid`, `benzodiazepine`

### Hepatic & GI
Keywords: `liver failure`, `hepatic`, `cirrhosis`, `encephalopathy hepatic`, `pancreatitis`, `gi bleeding`, `gastrointestinal`, `varices`, `ascites`, `hepatorenal`, `meld`, `child pugh`, `nash`, `transplant liver`, `biliary`, `cholangitis`

### Metabolic & Endocrine
Keywords: `diabetic ketoacidosis`, `dka`, `hypoglycemia`, `hyperglycemia`, `thyroid`, `adrenal`, `cortisol`, `insulin`, `glucose`, `acid-base`, `acidosis`, `alkalosis`, `electrolyte`, `hypercalcemia`, `hypocalcemia`, `hypomagnesemia`

Excludes: `lactic acidosis`, `septic` (to avoid misclassifying sepsis cases)

### Ethics & End-of-Life
Keywords: `withdrawal`, `end of life`, `palliative`, `comfort care`, `organ donation`, `brain death`, `prognosis`, `futility`, `family meeting`, `goals of care`, `do not resuscitate`, `dnr`, `advance directive`, `surrogate`, `ethics`

### Procedures & Monitoring
Keywords: `central line`, `central venous`, `arterial line`, `catheter`, `lumbar puncture`, `thoracentesis`, `paracentesis`, `bronchoscopy`, `ultrasound`, `echocardiography`, `monitoring`, `supervision`, `training`, `procedure`, `technique`, `insertion`

### General ICU
Default category for questions that do not match any subspecialty keywords (score = 0).

## Handling Overlapping Topics

Many ICU questions involve multiple organ systems or clinical domains. The algorithm handles this by:

1. **Scoring all matches**: A question about "septic shock with ARDS" would score points for both Sepsis and Respiratory
2. **Winner takes all**: The subspecialty with the highest total score is assigned
3. **Exclusion rules**: Some keywords are excluded from certain categories to reduce misclassification (e.g., "septic shock" is excluded from Cardiovascular)

## Confidence Score

The confidence score represents the total number of keyword match points. Higher scores indicate stronger classification confidence:

- Score 0: No matches (General ICU)
- Score 1-2: Weak match (single keyword)
- Score 3-5: Moderate match (multiple keywords or phrases)
- Score 6+: Strong match (multiple phrase matches)

## Limitations

1. **Rule-based**: May miss nuanced or context-dependent classifications
2. **Keyword overlap**: Some terms appear in multiple clinical contexts
3. **Single assignment**: Each question gets only one subspecialty, even if multiple apply
4. **Language-dependent**: Optimized for English medical terminology

## Output Files

- `question_classifications.csv`: Full classification with question text, subspecialty, and confidence score
- `subspecialty_summary.csv`: Aggregate statistics per subspecialty
- `subspecialty_analysis.csv`: Detailed metrics including dimension-level scores

## Potential Improvements

- Manual review and correction of edge cases
- LLM-based classification for improved accuracy
- Multi-label classification for questions spanning multiple subspecialties
- Validation against expert-assigned categories
