# KAN-175 recovered manuscript source

## Provenance

Recovered on 2026-10-16 from the user's prior ChatGPT conversation artifacts so the execution repository has the actual pre-KAN-175 manuscript package rather than reconstructing it from fragments.

Recovered source files:
- `Manuscript_JADR_full_revision_current_results.docx`
- `Cover_Letter_JADR_revised.docx`
- `Highlights_JADR.docx`

This file is a **text recovery snapshot only**. It is not the corrected final manuscript. All numerical/scientific claims below predate KAN-172/KAN-173 and must be replaced from the canonical corrected outputs before submission.

Canonical corrected numerical authority:
- `outputs/models/kan173_run_manifest.json`
- `outputs/tables_figures_manifest.json`

Accepted corrected asset commit before this recovery:
- `c2d901ac0db256a407a83c567729310f4f44e057`

---

# Recovered manuscript text

<PARSED TEXT FOR PAGE: 1 / 8>

Page 1
Nocturnal Rest-to-Activity Transition Probability and
Depressive Symptoms in NHANES 2013-2014: A Cross-Sectional
Actigraphy Study
Minjun Park
Department of Medicine, Sapporo Medical University, Sapporo, Japan
Corresponding author: Minjun Park | a25m080@sapmed.ac.jp | ORCID: 0009-0002-5305-3491
Highlights
 A transition-based actigraphy feature was associated with depressive symptom burden.
 The association persisted after demographic, socioeconomic, and sleep covariate adjustment.
 A conventional wake-after-sleep-onset proxy was not associated in the same model.
 Discrimination for PHQ-9 >= 10 improved only modestly after adding the transition feature.
 The results support association and complementary measurement, not clinical screening use.
Abstract
Background: Actigraphy studies commonly summarize nocturnal behavior using duration-based 
measures. Such summaries may not represent the temporal structure of transitions between relatively 
inactive and active states. We examined whether a nocturnal rest-to-activity transition probability (P01) 
was associated with depressive symptoms beyond conventional demographic and sleep-related 
variables.
Methods: This cross-sectional secondary analysis used NHANES 2013-2014 data. The final complete￾case analytic sample comprised 4,422 participants. Minute-level Monitor-Independent Movement 
Summary (MIMS) activity was aggregated into 10-minute blocks. For each participant, blocks were 
classified as lower-activity (rest) or higher-activity (active) relative to the participant-specific median 
activity level, and P01 was defined as the conditional probability of a rest-to-active transition during the 
fixed nocturnal window (22:00-06:00). The primary outcome was continuous PHQ-9 score. Ordinary least
squares models adjusted first for age, sex, and BMI and then additionally for Poverty Income Ratio, self￾reported usual weekday sleep duration, and an actigraphy-derived wake-after-sleep-onset (WASO) 
proxy. Exploratory ROC analyses evaluated PHQ-9 >= 10.
Results: Higher P01 was associated with higher PHQ-9 scores in the demographic-adjusted model (beta
= 10.707, SE = 1.277, p < 0.001) and in the extended model (beta = 9.105, SE = 1.350, p = 1.76 x 
10^-11). In the extended model, the WASO proxy was not associated with PHQ-9 score (p = 0.470). In 
secondary in-sample ROC analysis, P01 alone yielded an AUC of 0.590; a demographic model yielded 
an AUC of 0.624; adding P01 increased the AUC to 0.643.
Conclusions: Nocturnal rest-to-activity transition probability was associated with depressive symptom 
burden in this NHANES analytic sample and provided information not fully represented by the included 
duration-based sleep measures. The modest discrimination and cross-sectional design do not support 
diagnostic or screening claims. Replication using survey-design-aware analyses, prespecified transition 
definitions, and external cohorts is needed.
Keywords: actigraphy; depression; PHQ-9; rest-activity rhythm; transition probability; digital phenotyping;
NHANES
1. Introduction
Depressive symptoms are frequently accompanied by disturbances in sleep timing, continuity, and 
circadian rest-activity rhythms. Wrist-worn activity monitoring offers an objective and scalable way to 

<PARSED TEXT FOR PAGE: 2 / 8>

Page 2
characterize these behavioral patterns in naturalistic settings, complementing self-report measures. Prior 
actigraphy studies have identified altered sleep and 24-hour activity rhythms in people with depression, 
while more recent digital-phenotyping work has emphasized the need to move beyond simple time￾averaged features toward measures that retain temporal information.
Common sleep summaries such as sleep duration and wake after sleep onset (WASO) are useful 
because they are interpretable and clinically familiar. However, two individuals with similar cumulative 
wake time can have different temporal organizations of activity: one may have a few prolonged active 
episodes, whereas another may exhibit repeated switching between lower- and higher-activity states. A 
transition probability provides a compact description of this temporal organization without implying that 
actigraphy-derived states are equivalent to polysomnographic sleep stages.
Population datasets provide an opportunity to test whether transition-based features contribute 
information beyond conventional summaries. This question is especially relevant because recent 
NHANES research has linked accelerometer-derived sleep regularity to depression, while using survey￾weighted analyses to account for the complex sampling design. The present study focuses on a different 
feature: the conditional probability of transitioning from a lower-activity state to a higher-activity state 
during a fixed nocturnal window.
The primary objective was to examine whether nocturnal rest-to-activity transition probability (P01) was 
associated with continuous PHQ-9 depressive symptom scores after adjustment for demographic, 
socioeconomic, and sleep-related covariates. A secondary objective was to describe whether P01 added 
modest discriminative information for PHQ-9 >= 10 beyond age, sex, and BMI. These analyses were 
designed as association analyses rather than as development or validation of a clinical prediction tool.
2. Materials and Methods
2.1 Study design and data source
We conducted a cross-sectional secondary analysis of the National Health and Nutrition Examination 
Survey (NHANES) 2013-2014 cycle, combining physical activity monitor data with demographic, body￾measure, sleep-questionnaire, socioeconomic, and depression-questionnaire data. NHANES data are 
publicly available and de-identified. The parent NHANES protocol was approved by the National Center 
for Health Statistics Research Ethics Review Board, and participants provided informed consent.
The analytic cohort was restricted to participants with the data required for the prespecified regression 
models. The final complete-case sample for the extended continuous PHQ-9 model contained 4,422 
participants. Because the models reported here were not survey weighted, inferential statements are 
restricted to this analytic sample and should not be interpreted as nationally representative prevalence or 
population-average effect estimates.
2.2 Depressive symptoms
Depressive symptom burden was measured using the Patient Health Questionnaire-9 (PHQ-9), scored 
from 0 to 27. The continuous PHQ-9 score was the primary outcome. For secondary discrimination 
analyses, PHQ-9 >= 10 was used as a conventional threshold indicating clinically relevant depressive 
symptoms; this threshold was treated as a questionnaire-based classification rather than a formal 
psychiatric diagnosis.
2.3 Actigraphy-derived transition probability
Minute-level activity was obtained from the NHANES physical activity monitor files using the Monitor￾Independent Movement Summary (MIMS) measure. Minutes failing the available activity quality flag were
excluded. Activity values were summed into non-overlapping 10-minute blocks within each wear day. The
nocturnal window was defined a priori as 22:00-06:00.
For each participant, the median activity value across valid 10-minute blocks was used as a participant￾specific threshold. Blocks with activity at or below the threshold were labeled state 0 (lower activity/rest) 

<PARSED TEXT FOR PAGE: 3 / 8>

Page 3
and blocks above the threshold were labeled state 1 (higher activity/active). The nocturnal rest-to-activity 
transition probability was calculated as P01 = N(0->1) / [N(0->0) + N(0->1)], where N(i->j) is the number 
of observed transitions from state i to state j among ordered nocturnal blocks. P01 therefore quantifies 
how frequently a lower-activity block is followed by a higher-activity block, conditional on being in the 
lower-activity state. The metric is a coarse behavioral transition feature and is not intended to identify 
physiological sleep stages.
2.4 Covariates and conventional sleep measures
Model 1 included age, sex, and body mass index (BMI). Model 2 additionally included Poverty Income 
Ratio (PIR), self-reported usual weekday sleep duration from NHANES item SLD010H, and an 
actigraphy-derived WASO proxy. The WASO proxy was calculated as the mean number of wake￾classified minutes between the first and last sleep-classified minute within the nocturnal window across 
available wear nights. Continuous covariates (age, BMI, PIR, self-reported sleep duration, and the WASO
proxy) were standardized within the corresponding complete-case model sample. P01 was retained on its
native probability scale; sex was entered as an indicator variable with male as the reference category.
2.5 Statistical analysis
Ordinary least squares (OLS) regression was used for the primary continuous PHQ-9 analysis. Model 1 
estimated the association of P01 with PHQ-9 after adjustment for age, sex, and BMI. Model 2 additionally
adjusted for PIR, self-reported weekday sleep duration, and the actigraphy-derived WASO proxy. 
Regression coefficients are presented with standard errors and p-values. For P01, the coefficient 
corresponds to a one-unit change on the 0-1 probability scale; a 0.01 increase corresponds to one 
hundredth of the reported coefficient.
As a secondary descriptive analysis, receiver operating characteristic (ROC) curves were generated for 
PHQ-9 >= 10. We compared P01 alone, a demographic model containing age, sex, and BMI, and the 
demographic model with P01 added. These ROC analyses were not intended as model development or 
clinical validation and were interpreted conservatively because the displayed AUCs were estimated in the
analysis sample. Statistical significance was defined as two-sided p < 0.05. Analyses were conducted in 
Python using pandas, NumPy, SciPy, statsmodels, and scikit-learn.
3. Results
3.1 Association between P01 and PHQ-9 score
P01 was positively associated with depressive symptom burden in both hierarchical models. In Model 1, 
adjusted for age, sex, and BMI, the P01 coefficient was 10.707 (SE 1.277; p < 0.001). In Model 2, after 
additional adjustment for PIR, self-reported weekday sleep duration, and the WASO proxy, the coefficient
remained positive and statistically significant (beta = 9.105, SE = 1.350; p = 1.76 x 10^-11). On the native
P01 scale, a 0.01 increase in transition probability corresponds to an estimated 0.091-point higher PHQ-9
score in Model 2.
The extended model R-squared was 0.088. The limited explained variance is consistent with the 
multifactorial nature of depressive symptoms and reinforces that P01 should be interpreted as one 
behavioral correlate rather than as a stand-alone marker of depression.
3.2 Comparison with included sleep measures
In Model 2, the actigraphy-derived WASO proxy was not associated with PHQ-9 score (p = 0.470), 
whereas P01 remained associated. Self-reported weekday sleep duration and PIR were also associated 
with PHQ-9 score. The contrast between P01 and the WASO proxy suggests that transition structure and
cumulative wake duration are not interchangeable summaries of nocturnal behavior; it does not establish 
that one measure is clinically superior to the other.

<PARSED TEXT FOR PAGE: 4 / 8>

Page 4
Predictor Model 1 
beta SE p
Model 2 
beta SE p
Age -0.021 0.005 < 0.001 -0.012 0.005 0.015
Sex 
(Female)
0.852 0.17 < 0.001 0.893 0.178 < 0.001
BMI 0.115 0.012 < 0.001 0.121 0.013 < 0.001
Poverty 
Income 
Ratio (PIR)
--- --- --- -0.56 0.055 < 0.001
Self￾reported 
weekday 
sleep 
duration
--- --- --- -0.003 0.001 0.003
Actigraphy￾derived 
WASO 
proxy
--- --- --- 0.001 0.002 0.47
Rest-to￾active 
transition 
probability 
(P01)
10.707 1.277 < 0.001 9.105 1.35 < 0.001
Table 1. Multivariable OLS regression of PHQ-9 score. Continuous covariates other than P01 were z-scored within 
each complete-case model sample; P01 remained on the 0-1 probability scale. Female sex is compared with male 
reference. Model 2 used self-reported usual weekday sleep duration (SLD010H) and an actigraphy-derived WASO 
proxy.
3.3 Exploratory discrimination for PHQ-9 >= 10
P01 alone showed limited discrimination (AUC = 0.590). The demographic model containing age, sex, 
and BMI yielded an AUC of 0.624. Adding P01 increased the in-sample AUC to 0.643 (Figure 1). The 
absolute increase was modest, and no claim of clinical screening performance is made.

<PARSED TEXT FOR PAGE: 5 / 8>

Page 5
Figure 1. Exploratory ROC curves for PHQ-9 >= 10. P01 alone (AUC 0.590), age + sex + BMI (AUC 0.624), and age + 
sex + BMI + P01 (AUC 0.643). Values are descriptive in-sample discrimination estimates and should not be 
interpreted as externally validated prediction performance.
4. Discussion
4.1 Principal findings
In this cross-sectional NHANES analytic sample, a higher nocturnal rest-to-activity transition probability 
was associated with greater depressive symptom burden after adjustment for demographic, 
socioeconomic, and sleep-related covariates. The association remained detectable when an actigraphy￾derived WASO proxy and self-reported sleep duration were included in the same model. These results 
are consistent with the broader view that temporal organization of behavior can contain information that is
lost when high-resolution time series are reduced to cumulative duration measures.
The finding should be interpreted as complementary measurement rather than evidence that P01 is a 
depression biomarker or diagnostic test. P01 is a coarse transition feature defined from activity blocks, 
and the rest/active states are behavioral categories rather than physiological sleep stages. Its principal 
advantage is interpretability: it directly estimates the conditional frequency with which lower-activity 
periods are followed by higher-activity periods during a specified nocturnal window.

<PARSED TEXT FOR PAGE: 6 / 8>

Page 6
The results also fit with prior actigraphy work showing altered sleep and circadian rest-activity patterns in 
depression. A recent systematic review of digital phenotyping in depression noted that many studies rely 
on time-averaged features and suggested that direct analysis of temporal dynamics may provide useful 
information. Our analysis applies that general idea to a simple transition statistic that can be calculated 
from wearable activity data.
4.2 Relation to recent NHANES evidence
Recent NHANES work has reported an association between accelerometer-derived sleep irregularity and
depression using survey-weighted models. That work and the present analysis address related but 
distinct aspects of temporal organization: sleep regularity characterizes consistency of sleep patterns 
across time, whereas P01 characterizes within-window switching from lower to higher activity. The 
convergence of these approaches supports further study of temporal features, but it also highlights the 
need for replication using complex-survey methods and prespecified definitions.
4.3 Discrimination and clinical interpretation
The ROC results should remain secondary. Although adding P01 to age, sex, and BMI increased the 
AUC from 0.624 to 0.643, the resulting discrimination was still modest. This magnitude is insufficient to 
support individual-level screening, diagnosis, or risk stratification. The appropriate interpretation is that 
P01 contributed nonredundant information in the analysis sample. Future prediction studies would require
cross-validation or external validation, calibration assessment, clinically relevant decision thresholds, and
comparison against stronger baseline models.
4.4 Strengths and limitations
Strengths include the use of a large publicly available dataset, high-resolution wearable activity data, a 
transparent transition definition, and direct adjustment for both demographic factors and conventional 
sleep-related variables. Code availability can further support reproducibility.
Several limitations are important. First, the cross-sectional design precludes temporal or causal 
inference. Depressive symptoms may alter nocturnal activity patterns, nocturnal disruption may 
contribute to symptoms, or both may reflect shared factors. Second, PHQ-9 is a symptom questionnaire 
and PHQ-9 >= 10 is not equivalent to a clinician-established depressive disorder diagnosis.
Third, the actigraphy states used to compute P01 are relative activity states, not polysomnographic sleep 
and wake stages. The participant-specific median threshold is intentionally simple but may behave 
differently across activity distributions, wear patterns, or devices. The fixed 22:00-06:00 window also 
does not accommodate individual chronotype, shift work, or delayed sleep timing.
Fourth, the present regression estimates were unweighted. NHANES uses a complex multistage 
probability design; therefore, the current estimates should not be interpreted as nationally representative. 
A survey-weighted sensitivity analysis incorporating the relevant weights, strata, and primary sampling 
units would strengthen population-level inference and should be prioritized in replication or revision.
Fifth, the ROC analysis was exploratory and the displayed AUCs were estimated within the analysis 
sample. The modest AUC change may overestimate out-of-sample improvement. Finally, residual 
confounding is possible, including medication use, comorbid medical and psychiatric conditions, 
substance use, physical activity, work schedule, and other determinants of sleep and mood.
5. Conclusions
Nocturnal rest-to-activity transition probability was associated with depressive symptom burden in a 
cross-sectional NHANES 2013-2014 analytic sample after adjustment for demographic, socioeconomic, 
and sleep-related covariates. The transition feature may capture an aspect of nocturnal behavioral 
organization that is not represented by cumulative wake duration alone. The findings are hypothesis￾supporting rather than diagnostic: survey-design-aware replication, improved temporal segmentation, and
external validation are required before clinical use can be considered.

<PARSED TEXT FOR PAGE: 7 / 8>

Page 7
Declarations
Ethics approval and consent to participate: This study used publicly available, de-identified NHANES 
data. The NHANES protocol was approved by the National Center for Health Statistics Research Ethics 
Review Board, and participants provided informed consent. No additional participant contact occurred in 
this secondary analysis.
Consent for publication: Not applicable.
Funding: This research received no specific grant from any funding agency in the public, commercial, or 
not-for-profit sectors.
Competing interests: The author declares no competing interests.
Data and code availability: NHANES 2013-2014 data are publicly available from the U.S. Centers for 
Disease Control and Prevention. Analysis code is available at https://github.com/0ArchLinux0/Nocturnal￾Rest-Activity-Transition-Instability_NHANES-2013-2014.
Acknowledgements: I thank Professor Ayako Sumi at Sapporo Medical University for valuable insights 
and support. I also acknowledge NCHS/CDC for providing the NHANES data. The conclusions are those 
of the author and do not represent the official position of NCHS or CDC.
Use of artificial intelligence: Artificial intelligence tools were used for language editing, structural 
refinement, and consistency checking. The author retained responsibility for the research question, 
analysis, interpretation, and final manuscript content.
References
1. Kroenke K, Spitzer RL, Williams JB. The PHQ-9: validity of a brief depression severity measure. J Gen
Intern Med. 2001;16:606-613. doi:10.1046/j.1525-1497.2001.016009606.x.
2. Onnela JP, Rauch SL. Harnessing smartphone-based digital phenotyping to enhance behavioral and 
mental health. Neuropsychopharmacology. 2016;41:1691-1696.
3. Hori H, Koga N, Hidese S, et al. 24-h activity rhythm and sleep in depressed outpatients. J Psychiatr 
Res. 2016;77:27-34. doi:10.1016/j.jpsychires.2016.02.022.
4. Leaning IE, Ikani N, Savage HS, et al. From smartphone data to clinically relevant predictions: a 
systematic review of digital phenotyping methods in depression. Neurosci Biobehav Rev. 
2024;158:105541. doi:10.1016/j.neubiorev.2024.105541.
5. Zhao B, Huang X, Xing Y, et al. Sleep irregularity and depression among U.S. adults: an 
accelerometer-based cross-sectional study in NHANES. Sleep Health. 2026;12(4):612-618. 
doi:10.1016/j.sleh.2026.04.007.
6. Cole RJ, Kripke DF, Gruen W, Mullaney DJ, Gillin JC. Automatic sleep/wake identification from wrist 
activity. Sleep. 1992;15(5):461-469. doi:10.1093/sleep/15.5.461.
7. Centers for Disease Control and Prevention, National Center for Health Statistics. National Health and 
Nutrition Examination Survey, 2013-2014: questionnaires, examination data, and physical activity monitor
data. Accessed September 2026.
8. Centers for Disease Control and Prevention, National Center for Health Statistics. NHANES survey 
methods and analytic guidelines. Accessed September 2026.

<PARSED TEXT FOR PAGE: 8 / 8>

Page 8
Supplementary Methods Note: Temporal-resolution sensitivity analysis
The supplied sensitivity-analysis script recalculates P01 after aggregating minute-level MIMS activity into 
5-, 10-, 30-, and 60-minute blocks and compares the 60-minute estimate with finer resolutions. Because 
this analysis uses a separate sensitivity workflow and the current primary OLS models use P01 derived 
from 10-minute blocks, the resolution comparison is presented as supplementary material rather than as 
evidence that 1-minute and 60-minute estimates agree.
Supplementary Figure S1. Correlation of participant-level P01 estimates calculated after 5-, 10-, 30-, and 60-minute
aggregation. Pearson correlations with the 60-minute estimate were r = 0.632 for 5-minute, r = 0.690 for 10-minute, 
and r = 0.804 for 30-minute aggregation in the available sensitivity-analysis sample. The plot reflects correlation 
across temporal resolutions, not agreement with a native 1-minute P01 estimate.

---

# Recovered cover letter text

<PARSED TEXT FOR PAGE: 1 / 1>

Page 1
25 September 2026
Editorial Office
Journal of Affective Disorders Reports
Re: Original Research submission - “Nocturnal Rest-to-Activity Transition Probability and 
Depressive Symptoms in NHANES 2013-2014: A Cross-Sectional Actigraphy Study”
Dear Editors,
I am pleased to submit the enclosed manuscript for consideration as an Original Research article in 
Journal of Affective Disorders Reports. This cross-sectional secondary analysis examines whether a 
simple transition-based feature derived from nocturnal wearable activity data is associated with 
depressive symptom burden in NHANES 2013-2014.
The study focuses on a specific measurement question rather than proposing a diagnostic test. In an 
analytic sample of 4,422 participants, higher nocturnal rest-to-activity transition probability (P01) was 
associated with higher PHQ-9 scores after adjustment for age, sex, BMI, socioeconomic status, self￾reported sleep duration, and an actigraphy-derived wake-after-sleep-onset proxy. The corresponding 
WASO proxy was not associated with PHQ-9 score in the same extended model. Exploratory ROC 
analyses showed only a modest increase in discrimination when P01 was added to age, sex, and BMI 
(AUC 0.624 to 0.643), and the manuscript explicitly avoids clinical screening or diagnostic claims.
I believe the manuscript fits the journal because it addresses depressive symptomatology using an 
epidemiological and digital-behavioral measurement approach, while keeping the interpretation focused 
on affective-disorder research. The revised manuscript also distinguishes association from prediction, 
describes the transition construction directly, identifies the sleep-duration variable as self-reported rather 
than actigraphic, and treats the ROC analysis as secondary.
The work is original, is not under consideration elsewhere, and all data used are publicly available 
through NHANES. I am the sole author and have approved the submitted version. I declare no competing
interests and no specific funding for this study.
Thank you for considering this manuscript. I would be grateful for the opportunity to have the work 
evaluated by reviewers with expertise in affective-disorder epidemiology, actigraphy, or digital 
phenotyping.
Sincerely,
Minjun Park
Department of Medicine
Sapporo Medical University
Sapporo, Japan
a25m080@sapmed.ac.jp

---

# Recovered highlights text

<PARSED TEXT FOR PAGE: 1 / 1>

Page 1
Highlights
 Nocturnal rest-to-activity transitions were associated with PHQ-9 scores.
 The association persisted after demographic and sleep-related adjustment.
 An actigraphy-derived WASO proxy was not associated in the same model.
 Adding P01 to demographic variables increased AUC from 0.624 to 0.643.
 Findings support complementary measurement, not clinical screening.

---

## KAN-175 rewrite warning

Do not carry forward obsolete values or claims merely because they appear in the recovered source. In particular, revalidate or replace:
- N=4,422
- historical P01 beta/SE/p values
- in-sample AUC 0.624/0.643 framing
- unweighted-only limitation language
- any statement that WASO is validated rather than a proxy
- any implication that SLD010H is actigraphy-derived TST
- any screening/diagnostic/biomarker/clinical-utility claim
- any potential-landscape primary-story language
- any resolution-interchangeability/agreement claim
