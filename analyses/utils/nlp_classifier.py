"""NLP-based classification for task types and subspecialties."""

import re
from typing import List, Tuple


# Task type keywords
TASK_TYPE_PATTERNS = {
    'Diagnosis': [
        r'\bdiagnos[ie]s?\b',
        r'\bdifferential\b',
        r'\blikely cause\b',
        r'\bwhat.*(?:cause|etiology|underlying)\b',
        r'\bidentif(?:y|ication)\b.*(?:condition|disease|disorder)\b',
        r'\bsuspect(?:ed|ing)?\b',
        r'\bwhat is (?:the|this)\b',
        r'\bpresentation suggest\b',
    ],
    'Prognosis': [
        r'\bprogno(?:sis|stic)\b',
        r'\boutcome[s]?\b',
        r'\bsurviv(?:al|e)\b',
        r'\bmortality\b',
        r'\bexpect(?:ed|ation)?\b.*(?:outcome|survival|recovery)\b',
        r'\blong.term\b',
        r'\brisk of death\b',
        r'\bchance[s]? of\b',
    ],
    'Treatment': [
        r'\btreat(?:ment|ed|ing)?\b',
        r'\bmanage(?:ment)?\b',
        r'\btherap(?:y|eutic|ies)\b',
        r'\bshould (?:I|we)\b',
        r'\bnext step\b',
        r'\binterven(?:tion|e)\b',
        r'\bmedication[s]?\b',
        r'\bdrug[s]?\b',
        r'\bdose\b',
        r'\badminister\b',
        r'\bprescri(?:be|ption)\b',
        r'\bstart(?:ing)?\b.*(?:on|with)\b',
        r'\binitiating?\b',
        r'\bwean(?:ing)?\b',
        r'\bextubat(?:e|ion)\b',
        r'\btracheostom(?:y|ie)\b',
        r'\bantibiotic[s]?\b',
        r'\bempiri[c]?\b',
        r'\bresuscitat(?:e|ion)\b',
        r'\bfluid[s]?\b',
        r'\bvasopressor[s]?\b',
    ],
    'Knowledge': [
        r'\bexplain\b',
        r'\bdefin(?:e|ition)\b',
        r'\bmechanism\b',
        r'\bpathophysiology\b',
        r'\bhow does\b',
        r'\bwhy (?:is|does|do)\b',
        r'\bwhat (?:is|are) the\b.*(?:criteria|guidelines|recommendations)\b',
        r'\bdescribe\b',
        r'\bwhat causes\b',
    ],
}

# Subspecialty keywords
SUBSPECIALTY_PATTERNS = {
    'Cardiovascular': [
        r'\bcardi(?:ac|ovascular|ology)\b',
        r'\bheart\b',
        r'\bmyocardial\b',
        r'\barrhythmi(?:a|as)\b',
        r'\batrial fibrillation\b',
        r'\bventricular\b',
        r'\bcoronary\b',
        r'\bami\b',
        r'\bstemi\b',
        r'\bnstemi\b',
        r'\bhypotension\b',
        r'\bshock\b',
        r'\bvasopressor\b',
        r'\btroponin\b',
        r'\becg\b',
        r'\bechocardiograph\b',
        r'\bejection fraction\b',
    ],
    'Respiratory': [
        r'\brespir(?:atory|ation)\b',
        r'\bpulmonary\b',
        r'\blung[s]?\b',
        r'\bpneumon(?:ia|itis)\b',
        r'\bards\b',
        r'\bintubat(?:e|ion|ed)\b',
        r'\bventilat(?:or|ion|ed)\b',
        r'\bextubat(?:e|ion)\b',
        r'\btracheostom(?:y|ie)\b',
        r'\bwean(?:ing)?\b',
        r'\boxygen(?:ation)?\b',
        r'\bhypox(?:ia|emia|ic)\b',
        r'\bdyspn(?:o|e)ea\b',
        r'\bcopd\b',
        r'\basthma\b',
        r'\bpleural\b',
        r'\bchest\b',
    ],
    'Neurological': [
        r'\bneuro(?:logical|logy)?\b',
        r'\bbrain\b',
        r'\bcerebr(?:al|ovascular)\b',
        r'\bstroke\b',
        r'\bseizure[s]?\b',
        r'\bepilepsy\b',
        r'\bencephalopath(?:y|ic)\b',
        r'\bcoma(?:tose)?\b',
        r'\bconscious(?:ness)?\b',
        r'\bgcs\b',
        r'\bglasgow\b',
        r'\bsedation\b',
        r'\bdelirium\b',
        r'\bmeningitis\b',
        r'\bintracranial\b',
        r'\bicp\b',
    ],
    'Infectious Disease': [
        r'\binfect(?:ion|ious|ed)\b',
        r'\bsepsis\b',
        r'\bseptic\b',
        r'\bbacteri(?:a|al|emia)\b',
        r'\bvir(?:us|al|emia)\b',
        r'\bfung(?:al|i|emia)\b',
        r'\bantibiotic[s]?\b',
        r'\bantimicrob(?:ial)?\b',
        r'\bfever\b',
        r'\bpyrexia\b',
        r'\bculture[s]?\b',
        r'\bpneumonia\b',
        r'\bmeningitis\b',
        r'\bendocarditis\b',
        r'\bcsf\b',
    ],
    'Renal': [
        r'\bren(?:al|o)\b',
        r'\bkidney[s]?\b',
        r'\bakut?e kidney\b',
        r'\baki\b',
        r'\bdialys(?:is|ed)\b',
        r'\bcrrt\b',
        r'\bhemodialysis\b',
        r'\boliguri(?:a|c)\b',
        r'\banuri(?:a|c)\b',
        r'\bcreatinine\b',
        r'\bgfr\b',
        r'\burea\b',
        r'\belectrolyte[s]?\b',
        r'\bhyperkal(?:emia|emic)\b',
        r'\bhyponatr(?:emia|emic)\b',
    ],
    'Trauma': [
        r'\btrauma(?:tic)?\b',
        r'\binjur(?:y|ies|ed)\b',
        r'\bfracture[s]?\b',
        r'\baccident\b',
        r'\bfall\b',
        r'\bhead injur(?:y|ies)\b',
        r'\btbi\b',
        r'\bhemorrhag(?:e|ic)\b',
        r'\bbleed(?:ing)?\b',
        r'\bsurger(?:y|ical)\b',
        r'\bpost.?op(?:erative)?\b',
    ],
    'Metabolic/Endocrine': [
        r'\bmetabol(?:ic|ism)\b',
        r'\bendocrin(?:e|ology)\b',
        r'\bdiabet(?:es|ic)\b',
        r'\bglucose\b',
        r'\bhyperglycemi(?:a|c)\b',
        r'\bhypoglycemi(?:a|c)\b',
        r'\binsulin\b',
        r'\bdka\b',
        r'\bketoacidosis\b',
        r'\bthyroid\b',
        r'\badrenal\b',
        r'\blactate\b',
        r'\bacidos(?:is|ic)\b',
        r'\balk(?:al)?os(?:is|ic)\b',
    ],
    'Gastrointestinal': [
        r'\bgastro(?:intestinal)?\b',
        r'\bgi\b(?! )', # GI but not followed by space (avoid GI Bill etc)
        r'\bliver\b',
        r'\bhepat(?:ic|itis|ology)\b',
        r'\bpancrea(?:s|tic|titis)\b',
        r'\bbowel\b',
        r'\bintestin(?:e|al)\b',
        r'\bbleed(?:ing)?.*(?:gi|gastro|upper|lower)\b',
        r'\bascites\b',
        r'\bcirrhosis\b',
        r'\bvarices\b',
    ],
    'Hematology': [
        r'\bhemato(?:logy|logical)\b',
        r'\bblood\b',
        r'\banemi(?:a|c)\b',
        r'\bthrombocytopeni(?:a|c)\b',
        r'\bcoagul(?:ation|opathy)\b',
        r'\bdic\b',
        r'\btransfus(?:ion|ed)\b',
        r'\bplatelet[s]?\b',
        r'\bhemoglobin\b',
        r'\bhematocrit\b',
    ],
}


def classify_task_type(question: str) -> str:
    """
    Classify a question into a task type based on keywords.

    Args:
        question: The question text

    Returns:
        Task type: 'Diagnosis', 'Prognosis', 'Treatment', 'Knowledge', or 'Other'
    """
    question_lower = question.lower()

    scores = {}
    for task_type, patterns in TASK_TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, question_lower):
                score += 1
        scores[task_type] = score

    # Return the task type with highest score, or 'Other' if no matches
    if max(scores.values()) == 0:
        return 'Other'

    return max(scores, key=scores.get)


def classify_subspecialty(question: str) -> str:
    """
    Classify a question into an ICU subspecialty based on keywords.

    Args:
        question: The question text

    Returns:
        Subspecialty name or 'General ICU'
    """
    question_lower = question.lower()

    scores = {}
    for subspecialty, patterns in SUBSPECIALTY_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, question_lower)
            score += len(matches)
        scores[subspecialty] = score

    # Return subspecialty with highest score, or 'General ICU' if no clear match
    if max(scores.values()) == 0:
        return 'General ICU'

    # Require at least 2 matches to assign a subspecialty
    max_score = max(scores.values())
    if max_score < 2:
        return 'General ICU'

    return max(scores, key=scores.get)


def classify_all_questions(questions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Classify all questions into task types and subspecialties.

    Args:
        questions: List of question texts

    Returns:
        Tuple of (task_types, subspecialties)
    """
    task_types = [classify_task_type(q) for q in questions]
    subspecialties = [classify_subspecialty(q) for q in questions]

    return task_types, subspecialties
