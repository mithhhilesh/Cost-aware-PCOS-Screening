export const guideline = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10477934/';
type Reference = { text: string; url?: string; source?: string };
const med = (text: string, article: string): Reference => ({text, url: `https://medlineplus.gov/ency/article/${article}.htm`, source: 'MedlinePlus'});
const pcos = (text: string): Reference => ({text, url: guideline, source: '2023 PCOS guideline'});
export const clinicalReferences: Record<string, Reference> = {
  'Age (yrs)': {text: 'No normal-age cutoff; interpretation depends on life stage.'},
  'BMI': {text: 'Adults 20+: 18.5 to <25 kg/m² is the general healthy-weight category; not a PCOS cutoff.', url: 'https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html', source: 'CDC'},
  'Waist:Hip Ratio': {text: 'Abdominal fat-distribution measure; risk interpretation depends on sex and population.', url: 'https://www.who.int/publications-detail-redirect/9789241501491', source: 'WHO'},
  'Cycle(R/I)': pcos('More than 3 years after first period: <21 or >35 days between periods, or <8/year, counts as irregular.'),
  'Cycle length(days)': {text: 'Bleeding commonly lasts 2–7 days; this field is bleeding duration, not time between periods.', url: 'https://www.nhs.uk/conditions/periods/', source: 'NHS'},
  'Weight gain(Y/N)': {text: 'Reported Yes/No; no clinical normal range without amount and time interval.'},
  'hair growth(Y/N)': pcos('A Yes/No report cannot replace a clinical hirsutism assessment.'),
  'Skin darkening (Y/N)': {text: 'Reported Yes/No; cause and significance require clinical assessment.'},
  'Hair loss(Y/N)': pcos('Hair loss alone is a weak predictor of biochemical androgen excess.'),
  'Pimples(Y/N)': pcos('Acne alone is a weak predictor of biochemical androgen excess.'),
  'Fast food (Y/N)': {text: 'Reported Yes/No; this field has no validated clinical intake threshold.'},
  'Reg.Exercise(Y/N)': {text: 'Reported Yes/No; frequency, duration and intensity are not captured.'},
  'Vit D3 (ng/mL)': {text: 'If total 25(OH)D: ≥20 ng/mL is adequate for most; confirm the assay before applying this value.', url: 'https://ods.od.nih.gov/factsheets/VitaminD-HealthProfessional/', source: 'NIH ODS'},
  'FSH(mIU/mL)': med('Example while menstruating: 4.7–21.5 mIU/mL; use the lab’s cycle-phase interval.', '003710'),
  'LH(mIU/mL)': med('Example before menopause: 5–25 mIU/mL; levels rise around ovulation and vary by lab.', '003708'),
  'TSH (mIU/L)': med('Example nonpregnant reference: 0.4–4.8 mIU/L; pregnancy, age and lab can change it.', '003684'),
  'AMH(ng/mL)': pcos('No universal cutoff: use age- and assay-specific limits; AMH alone cannot diagnose PCOS.'),
  'PRL(ng/mL)': med('Example for nonpregnant women: <25 ng/mL; use the lab’s interval and clinical context.', '003718'),
  'PRG(ng/mL)': med('Before ovulation: typically <1 ng/mL; later phases and pregnancy need different intervals.', '003714'),
  'Follicle No. (L)': pcos('In adults, ≥20 follicles in either ovary can indicate PCOM with suitable ultrasound; not PCOS diagnosis alone.'),
  'Follicle No. (R)': pcos('In adults, ≥20 follicles in either ovary can indicate PCOM with suitable ultrasound; not PCOS diagnosis alone.'),
  'Avg. F size (L) (mm)': pcos('No universal normal average; the 2–9 mm PCOM counting criterion is not an average-size reference.'),
  'Avg. F size (R) (mm)': pcos('No universal normal average; the 2–9 mm PCOM counting criterion is not an average-size reference.'),
  'Endometrium (mm)': {text: 'No single normal thickness across the menstrual cycle; interpret with cycle phase and symptoms.'},
};
